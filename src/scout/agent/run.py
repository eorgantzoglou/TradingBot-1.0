"""Orchestration for `scout investigate`: open resources, run the loop, compose.

The one place the agent's moving parts are wired together -- the LLM client, the
shared HTTP client, the fundamentals store, the web provider -- opened for the
run and closed after, so the CLI stays a thin caller. Mirrors how
`research.pipeline.research_entities` owns its own resource lifecycle.
"""

from __future__ import annotations

import logging

from scout.agent.brief import compose_brief
from scout.agent.loop import run_agent
from scout.agent.models import FinishedBrief
from scout.agent.prompts import PROMPT_VERSION, SYSTEM_PROMPT
from scout.agent.tools import ToolContext, build_tools
from scout.config import Config
from scout.data.http import HttpClient
from scout.data.sources.web import build_web_provider
from scout.fundamentals.store import FundamentalsStore
from scout.harness.build import build_client
from scout.harness.cost import CostLedger

logger = logging.getLogger(__name__)


async def investigate(
    config: Config,
    *,
    subject: str,
    entity_id: str | None = None,
    max_steps: int = 12,
    use_cache: bool = True,
) -> tuple[FinishedBrief, CostLedger]:
    """Run one investigation end to end. Returns the guardrail-checked brief and
    the cost ledger.

    `entity_id` set means the investigation centres on one known, ideally
    ingested, filer (its numbers and the disciplined veto are available).
    `entity_id` None means a free-text thesis -- the agent discovers via search
    and the screen, and there is no code veto to anchor to.
    """
    client = build_client(config, use_cache=use_cache, prompt_version=PROMPT_VERSION)
    ledger = CostLedger()

    async with HttpClient(user_agent=config.user_agent) as http:
        store = _open_store(config)
        web_provider = build_web_provider(
            config.web_search_provider, tavily_api_key=config.credentials.tavily_api_key
        )
        if web_provider is None:
            logger.warning(
                "web search unavailable (provider=%s); the agent runs without web tools",
                config.web_search_provider,
            )
        try:
            ctx = ToolContext(
                config=config,
                http=http,
                client=client,
                store=store,
                web_provider=web_provider,
                effort=config.llm.effort,  # type: ignore[arg-type]
            )
            tools = build_tools(ctx)
            goal = _goal(subject, entity_id)

            with ledger.stage("investigate"):
                run = await run_agent(
                    client,
                    goal=goal,
                    system_prompt=SYSTEM_PROMPT,
                    tools=tools,
                    entity_id=entity_id,
                    max_steps=max_steps,
                    effort=config.llm.effort,  # type: ignore[arg-type]
                    ledger=ledger,
                )
                brief = await compose_brief(
                    client,
                    run,
                    subject=subject,
                    entity_id=entity_id,
                    config=config,
                    model=client.model,
                    effort=config.llm.effort,  # type: ignore[arg-type]
                    ledger=ledger,
                )
        finally:
            if store is not None:
                store.close()

    return brief, ledger


def resolve_subject(config: Config, entity_id: str) -> str:
    """A human name for a known entity, for the brief header. Falls back to the id."""
    store = _open_store(config)
    if store is None:
        return entity_id
    try:
        snapshot = store.latest_snapshot(entity_id)
        return (snapshot.entity.name if snapshot and snapshot.entity.name else None) or entity_id
    finally:
        store.close()


def _open_store(config: Config) -> FundamentalsStore | None:
    """The read-only fundamentals store, or None if nothing has been ingested."""
    if not config.db_path.exists():
        return None
    try:
        return FundamentalsStore(config.db_path, read_only=True)
    except Exception as exc:  # a missing/locked db just disables the metric tools
        logger.warning("could not open fundamentals store: %s", exc)
        return None


def _goal(subject: str, entity_id: str | None) -> str:
    if entity_id:
        return (
            f"Investigate the company '{subject}' (SEC entity id {entity_id}). Decide whether "
            "it is worth further work: assess its financial health, valuation and any red "
            "flags. Get its fundamentals and metrics, read its most recent filing, run "
            "deep_analyze for the disciplined verdict, and check the web for anything material."
        )
    return (
        f"{subject}\n\nUse the screen and filing/web search to surface the best one or two "
        "candidates for this thesis, then investigate each: fundamentals, metrics, filings, "
        "and web. Be concrete about which companies and why."
    )
