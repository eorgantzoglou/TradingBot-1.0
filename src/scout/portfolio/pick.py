"""`scout pick`: pre-register a batch of paper picks from one screen run.

The forward-evidence clock starts the day the screen works (PLAN.md 7), so this
command's job is to write down -- before any outcome is known -- what each
strategy would hold today. Every strategy selects from the *same* universe at the
*same* instant, so they can later be graded on the identical forward window and
the comparison is fair.

Deterministic by default (no model needed, mirroring how `harvest` never
requires an LLM): it records the screen, the equal-weight universe, the EV/EBIT
decile, and -- when there is enough history -- the GBDT baseline. Pass a research
client to also record the agent's book (the names it did not veto), which is the
one strategy whose cost the other four exist to judge.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date

from scout.config import Config
from scout.harness.protocol import Effort, LLMClient
from scout.metrics.base import MarketData
from scout.portfolio import baselines
from scout.portfolio.baselines import InsufficientHistory, LabeledObservation, Selection
from scout.portfolio.ledger import Ledger
from scout.portfolio.models import PaperPick, Strategy, finite_features
from scout.research.pipeline import research_entities
from scout.screen.models import ScoredCandidate
from scout.screen.screen import run_screen

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StrategyBatch:
    """What one strategy contributed to a pick run -- for the CLI summary."""

    strategy: Strategy
    n_picks: int
    n_priced: int
    """Picks that got a reference price, so are actually gradeable later."""

    note: str | None = None


@dataclass(slots=True)
class PickBatch:
    """The outcome of one `scout pick`: what was written, and the caveats."""

    run_id: str
    as_of: date
    universe_size: int
    strategies: list[StrategyBatch] = field(default_factory=list)
    total_written: int = 0
    notes: list[str] = field(default_factory=list)


def run_pick(
    config: Config,
    *,
    as_of: date,
    prices: dict[str, float] | None = None,
    top: int = 20,
    research_client: LLMClient | None = None,
    skeptic_client: LLMClient | None = None,
    effort: Effort | None = None,
    run_id: str | None = None,
    ledger: Ledger | None = None,
) -> PickBatch:
    """Screen the universe, build each strategy's book, and append it to the ledger.

    `prices` maps entity_id -> a manual reference price (no price feed yet,
    PLAN.md 1.3). A name with no price is still recorded, but as ungradeable
    (`reference_price=None`) -- an honest gap, not a fabricated entry. Pass a
    `research_client` to also record the agent strategy.
    """
    prices = prices or {}
    run_id = run_id or f"{as_of.isoformat()}-{uuid.uuid4().hex[:8]}"
    ledger = ledger or Ledger(config.ledger_path)

    market_data = {eid: MarketData(price=price) for eid, price in prices.items()}
    screen_result = run_screen(config, market_data=market_data)
    candidates = screen_result.ranked
    by_id = {c.entity_id: c for c in candidates}

    batch = PickBatch(run_id=run_id, as_of=as_of, universe_size=screen_result.universe_size)
    all_picks: list[PaperPick] = []

    # --- the deterministic strategies (no LLM) ---------------------------------
    deterministic: list[tuple[Strategy, list[Selection], str | None]] = [
        (Strategy.UNIVERSE_EW, baselines.equal_weight_universe(candidates), None),
        (Strategy.SCREEN, baselines.top_composite(candidates, top=top), None),
        (Strategy.EV_EBIT_DECILE, baselines.ev_ebit_decile(candidates), None),
    ]
    for strategy, selections, note in deterministic:
        picks = _picks_from_selections(strategy, selections, by_id, as_of, prices, run_id)
        all_picks.extend(picks)
        batch.strategies.append(_strategy_batch(strategy, picks, note))

    # EV/EBIT is empty without prices; say so rather than leaving a silent gap.
    if not prices:
        batch.notes.append(
            "No reference prices supplied: the EV/EBIT-decile baseline is empty "
            "(it needs prices) and every pick is ungradeable until a forward price "
            "is given at score time. Pass --price/--prices to pre-register a real book."
        )

    # --- baseline #3: the GBDT, honestly gated ---------------------------------
    gbdt_note = _gbdt_note(candidates, top=top)
    batch.strategies.append(StrategyBatch(Strategy.GBDT, n_picks=0, n_priced=0, note=gbdt_note))
    batch.notes.append(gbdt_note)

    # --- the agent (optional; needs a model) -----------------------------------
    if research_client is not None:
        requested = _top_ranked(by_id, top)
        agent_picks = _agent_picks(
            config,
            by_id,
            as_of=as_of,
            prices=prices,
            top=top,
            run_id=run_id,
            client=research_client,
            skeptic_client=skeptic_client,
            effort=effort,
        )
        all_picks.extend(agent_picks)
        batch.strategies.append(
            _strategy_batch(Strategy.AGENT, [p for p in agent_picks if p.weight > 0], None)
        )
        # An empty agent book after research ran on real candidates almost always
        # means the pipeline failed (bad key, model unreachable) rather than that
        # every name was vetoed -- research_entities swallows per-entity errors.
        # Say so, so a broken run is not mistaken for a clean "all vetoed" result.
        if requested and not agent_picks:
            batch.notes.append(
                f"--research ran on {len(requested)} candidate(s) but the pipeline returned "
                "no reports. The model may be unreachable or misconfigured; the agent book is "
                "empty for that reason, not because every name was vetoed."
            )

    batch.total_written = ledger.append(all_picks)
    return batch


def _picks_from_selections(
    strategy: Strategy,
    selections: list[Selection],
    by_id: dict[str, ScoredCandidate],
    as_of: date,
    prices: dict[str, float],
    run_id: str,
) -> list[PaperPick]:
    """Turn a strategy's selections into equal-weighted paper picks.

    Weight is 1/N over the whole book (weight * gradeable-set is renormalized at
    score time), so a name later found unpriceable does not silently reweight the
    rest here, and the book's intended composition is preserved as recorded.
    """
    if not selections:
        return []
    weight = 1.0 / len(selections)
    picks: list[PaperPick] = []
    for sel in selections:
        cand = by_id.get(sel.entity_id)
        picks.append(
            PaperPick(
                as_of=as_of,
                strategy=strategy,
                entity_id=sel.entity_id,
                name=cand.name if cand else None,
                cohort=cand.cohort.label() if cand else "",
                reference_price=prices.get(sel.entity_id),
                currency=None,  # trading currency is unknown without a feed; the
                # forward return is a same-instrument ratio, so it is unaffected.
                weight=weight,
                rank=sel.rank,
                score=sel.score,
                vetoed=None,
                features=finite_features(cand.metric_values) if cand else {},
                run_id=run_id,
                note=sel.note,
            )
        )
    return picks


def _agent_picks(
    config: Config,
    by_id: dict[str, ScoredCandidate],
    *,
    as_of: date,
    prices: dict[str, float],
    top: int,
    run_id: str,
    client: LLMClient,
    skeptic_client: LLMClient | None,
    effort: Effort | None,
) -> list[PaperPick]:
    """Research the top-N candidates and record the agent's book.

    The book is the names research did NOT veto, equal-weighted. Vetoed names are
    still recorded -- with weight 0 and the veto reason -- so the ledger captures
    what the agent rejected and a later analysis can ask whether vetoing helped.
    """
    # Local import: async lives at the edge here, matching the CLI's pattern of
    # running the research pipeline's event loop only where it is invoked.
    import asyncio

    entity_ids = [c.entity_id for c in _top_ranked(by_id, top)]
    if not entity_ids:
        return []

    reports = asyncio.run(
        research_entities(
            config,
            entity_ids,
            client=client,
            skeptic_client=skeptic_client,
            effort=effort,
        )
    )
    survivors = [r for r in reports if not r.vetoed]
    weight = 1.0 / len(survivors) if survivors else 0.0

    picks: list[PaperPick] = []
    for report in reports:
        cand = by_id.get(report.entity_id)
        vetoed = report.vetoed
        picks.append(
            PaperPick(
                as_of=as_of,
                strategy=Strategy.AGENT,
                entity_id=report.entity_id,
                name=report.name or (cand.name if cand else None),
                cohort=cand.cohort.label() if cand else "",
                reference_price=prices.get(report.entity_id),
                currency=None,
                weight=0.0 if vetoed else weight,
                rank=None,
                score=None,
                vetoed=vetoed,
                features=finite_features(cand.metric_values) if cand else {},
                run_id=run_id,
                note=("; ".join(report.memo.veto_reasons) or "vetoed") if vetoed else None,
            )
        )
    return picks


def _top_ranked(by_id: dict[str, ScoredCandidate], top: int) -> list[ScoredCandidate]:
    ranked = [c for c in by_id.values() if c.composite is not None]
    ranked.sort(key=lambda c: -(c.composite or 0.0))
    return ranked[:top]


def _gbdt_note(candidates: list[ScoredCandidate], *, top: int) -> str:
    """Try the GBDT baseline with the (currently empty) labeled history and turn
    its honest refusal into a note. The history comes from scored picks, which do
    not exist on a fresh ledger, so this all but always reports INSUFFICIENT --
    exactly the honest state until the forward archive fills in."""
    tree = baselines.GradientBoostedTree()
    history: list[LabeledObservation] = []  # populated from scored picks in a later phase
    try:
        tree.select(candidates, history=history, top=top)
    except InsufficientHistory as exc:
        return f"GBDT baseline: {exc}"
    return "GBDT baseline: trained and recorded."  # pragma: no cover - needs a full ledger


def _strategy_batch(
    strategy: Strategy, picks: list[PaperPick], note: str | None
) -> StrategyBatch:
    priced = sum(1 for p in picks if p.reference_price is not None)
    return StrategyBatch(strategy=strategy, n_picks=len(picks), n_priced=priced, note=note)
