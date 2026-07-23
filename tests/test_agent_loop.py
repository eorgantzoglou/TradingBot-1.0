"""The tool-use loop, driven by a scripted FakeClient. No network.

Proves the loop mechanics and the two properties that keep it safe: a tool
exception (or an unknown tool) becomes an observation the agent can react to
rather than a crash, and evidence a tool returns is accumulated for later
verification.
"""

from __future__ import annotations

import json

from scout.agent.loop import run_agent
from scout.agent.models import Tool, ToolResult


def _action(tool: str, **tool_input) -> str:
    return json.dumps({"thought": "thinking", "tool": tool, "tool_input": tool_input})


async def _echo(inp: dict) -> ToolResult:
    return ToolResult(output=f"echoed {inp.get('msg', '')}")


async def _boom(inp: dict) -> ToolResult:
    raise RuntimeError("tool blew up")


async def _reads_filing(inp: dict) -> ToolResult:
    return ToolResult(
        output="excerpt: going concern...",
        filing_texts={"acc-1": "The full filing text mentions going concern doubt."},
    )


ECHO = Tool(name="echo", summary="Echo a message.", parameters="msg", run=_echo)
BOOM = Tool(name="boom", summary="Always fails.", parameters="none", run=_boom)
READER = Tool(name="read", summary="Read a filing.", parameters="acc", run=_reads_filing)


async def test_loop_runs_a_tool_then_finishes(make_client, all_modes):
    client = make_client(
        script=[_action("echo", msg="hi"), _action("finish")], capabilities=all_modes
    )
    run = await run_agent(client, goal="test", system_prompt="sys", tools=[ECHO])
    assert run.finished
    assert len(run.steps) == 1
    assert run.steps[0].tool == "echo"
    assert "echoed hi" in run.steps[0].observation
    assert run.steps[0].ok


async def test_tool_exception_becomes_an_observation_not_a_crash(make_client, all_modes):
    client = make_client(
        script=[_action("boom"), _action("finish")], capabilities=all_modes
    )
    run = await run_agent(client, goal="test", system_prompt="sys", tools=[BOOM])
    assert run.finished  # the run survived the tool blowing up
    assert run.steps[0].ok is False
    assert "tool blew up" in run.steps[0].observation


async def test_unknown_tool_is_reported_with_the_menu(make_client, all_modes):
    client = make_client(
        script=[_action("nonexistent"), _action("finish")], capabilities=all_modes
    )
    run = await run_agent(client, goal="test", system_prompt="sys", tools=[ECHO])
    assert run.steps[0].ok is False
    assert "unknown tool" in run.steps[0].observation
    assert "echo" in run.steps[0].observation  # the menu is offered back


async def test_filing_text_is_accumulated_into_the_evidence_pack(make_client, all_modes):
    client = make_client(
        script=[_action("read", acc="acc-1"), _action("finish")], capabilities=all_modes
    )
    run = await run_agent(client, goal="test", system_prompt="sys", tools=[READER])
    assert run.evidence is not None
    assert "acc-1" in run.evidence.texts_by_accession
    assert "going concern" in run.evidence.texts_by_accession["acc-1"]
    # The model only saw the short excerpt, not the full text.
    assert "excerpt" in run.steps[0].observation


async def test_step_cap_stops_an_agent_that_never_finishes(make_client, all_modes):
    # Never emits finish; the loop must stop at max_steps and say so.
    client = make_client(script=[_action("echo", msg="x")], capabilities=all_modes, repeat_last=True)
    run = await run_agent(
        client, goal="test", system_prompt="sys", tools=[ECHO], max_steps=3
    )
    assert not run.finished
    assert len(run.steps) == 3
    assert "3-step limit" in run.stop_reason
