"""The tool-use loop: a hand-rolled ReAct over the existing structured-output ladder.

The harness has no native tool-calling (its `Message` is role+content only), and
native tool-call wire formats differ per provider anyway. So the loop is a
JSON-action cycle over `complete_structured`: each turn the model returns an
`AgentAction` (a thought plus the tool to call), the loop runs the tool, appends
the result as an `Observation:` turn, and repeats until the model calls the
`finish` sentinel or the step cap is hit. This works on every backend the harness
supports (DeepSeek included) with zero adapter changes -- PLAN.md 4.5's "no
framework, hand-rolled orchestration," applied to an agent rather than a DAG.

The loop is deliberately dumb about *what* the tools do: it validates the action,
dispatches by name, accumulates whatever evidence the tools return, and never lets
a tool exception kill the run (a failure becomes an observation the agent can
react to). All the judgment lives in the model and all the guardrails live in the
tools and in `brief.py` -- not here.
"""

from __future__ import annotations

import json
import logging

from scout.agent.models import (
    FINISH_TOOL,
    AgentAction,
    AgentRun,
    AgentStep,
    Tool,
    ToolResult,
)
from scout.harness.cost import CostLedger
from scout.harness.protocol import Effort, LLMClient, Message
from scout.research.evidence import EvidencePack

logger = logging.getLogger(__name__)

# An observation shown back to the model is capped so one giant filing cannot blow
# the context or destabilise the cache. The FULL text still reaches the evidence
# pack (for verification) via ToolResult.filing_texts -- only the model's view is
# clipped, and tools are expected to return an excerpt as `output` regardless.
_MAX_OBSERVATION_CHARS = 6000


async def run_agent(
    client: LLMClient,
    *,
    goal: str,
    system_prompt: str,
    tools: list[Tool],
    entity_id: str | None = None,
    max_steps: int = 12,
    effort: Effort | None = None,
    temperature: float | None = 0.2,
    max_tokens: int | None = 1200,
    ledger: CostLedger | None = None,
) -> AgentRun:
    """Drive the agent to gather evidence for `goal`. Returns the raw run trace.

    Composition of the final brief (with its guardrails) is a separate step in
    `brief.py`; this function only runs the loop and collects what the tools
    found. `entity_id` names the evidence pack when the investigation centres on
    one known filer; it is optional for a free-text thesis.
    """
    tool_map = {tool.name: tool for tool in tools}
    run = AgentRun(goal=goal, evidence=EvidencePack(entity_id=entity_id or "investigation"))

    transcript = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=_opening_message(goal, tools)),
    ]

    for _ in range(max_steps):
        result = await complete_action(client, transcript, effort, temperature, max_tokens)
        if ledger is not None:
            ledger.record(result.response)
        action = result.value

        if action.tool == FINISH_TOOL:
            run.finished = True
            run.stop_reason = "the agent decided it had enough evidence."
            break

        observation, ok = await _run_tool(tool_map, action, run)
        run.steps.append(
            AgentStep(
                thought=action.thought,
                tool=action.tool,
                tool_input=action.tool_input,
                observation=observation,
                ok=ok,
            )
        )
        transcript.append(Message(role="assistant", content=_action_json(action)))
        transcript.append(
            Message(role="user", content=f"Observation: {_truncate(observation)}")
        )
    else:
        run.stop_reason = f"reached the {max_steps}-step limit before finishing."

    logger.info("agent run: %d step(s), %s", len(run.steps), run.stop_reason)
    return run


async def complete_action(
    client: LLMClient,
    transcript: list[Message],
    effort: Effort | None,
    temperature: float | None,
    max_tokens: int | None,
):  # type: ignore[no-untyped-def]
    """One structured decision from the model. Split out so a test can drive a
    single step, and so the effort/temperature wiring lives in exactly one place."""
    # Local import avoids a cycle: structured.py has no agent dependency, but
    # keeping the import here mirrors how the research stages call it.
    from scout.harness.structured import complete_structured

    return await complete_structured(
        client,
        transcript,
        AgentAction,
        effort=effort,
        temperature=temperature,
        max_tokens=max_tokens,
    )


async def _run_tool(
    tool_map: dict[str, Tool], action: AgentAction, run: AgentRun
) -> tuple[str, bool]:
    """Dispatch one tool call, fold its evidence into the run, return (obs, ok).

    A tool exception is caught and turned into an error observation on purpose:
    one bad tool call must not end an investigation the way it must not end a
    fan-out (`return_exceptions=True` is the same instinct)."""
    tool = tool_map.get(action.tool)
    if tool is None:
        available = ", ".join(sorted(tool_map)) or "(none)"
        return f"ERROR: unknown tool {action.tool!r}. Available tools: {available}, {FINISH_TOOL}.", False

    try:
        result = await tool.run(action.tool_input)
    except Exception as exc:  # a tool bug becomes an observation, never a crash
        logger.warning("tool %s raised: %s", action.tool, exc)
        result = ToolResult.error(f"{type(exc).__name__}: {exc}")

    if run.evidence is not None:
        run.evidence.texts_by_accession.update(result.filing_texts)
    run.web_sources.update(result.web_pages)
    if result.metrics_block:
        run.metrics_block = result.metrics_block
    return result.output, result.ok


def _opening_message(goal: str, tools: list[Tool]) -> str:
    return (
        f"Investigation task:\n{goal}\n\n"
        f"{_format_catalog(tools)}\n\n"
        "Work step by step. Each step, call ONE tool to gather evidence, or call "
        f"'{FINISH_TOOL}' when you have enough to write the brief. Prefer the "
        "code-computed metric tools for any number, and always keep the exact "
        "wording of anything you might quote. Respond with your first action."
    )


def _format_catalog(tools: list[Tool]) -> str:
    """The tool menu shown to the model -- prose, not JSON Schema, to keep the
    prompt prefix small and stable (which is what makes prompt caching hit)."""
    lines = ["You have these tools:"]
    for tool in tools:
        lines.append(f"- {tool.name}: {tool.summary} Arguments: {tool.parameters}")
    lines.append(
        f"- {FINISH_TOOL}: stop gathering and write the brief. Arguments: none."
    )
    return "\n".join(lines)


def _action_json(action: AgentAction) -> str:
    return json.dumps(
        {"thought": action.thought, "tool": action.tool, "tool_input": action.tool_input}
    )


def _truncate(text: str) -> str:
    if len(text) <= _MAX_OBSERVATION_CHARS:
        return text
    return text[:_MAX_OBSERVATION_CHARS] + f"\n… [truncated {len(text) - _MAX_OBSERVATION_CHARS} chars]"
