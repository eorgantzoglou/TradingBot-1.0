"""Types for the paper-trade ledger and its forward scoring.

A *pick* is a pre-registration: at a point in time we recorded that a strategy
selected an entity, at a reference price, with the context that justified it.
Recording the pick before the outcome is known is the whole point -- it is what
makes forward paper trading credible evidence rather than a story told after the
fact (PLAN.md 6). Each strategy (the full agent, and the three dumb baselines it
must beat) writes its own picks over the same universe at the same instant, so
they can later be graded on the identical forward window.

Two hard rules this module encodes:

  - **Missing data is never a guessed value.** A pick with no reference price is
    recorded but ungradeable -- `reference_price=None`, not zero. A metric that
    was `inf`/`nan` in the screen is dropped from `features`, never serialized as
    a number, because `Infinity` is not valid JSON and a fake finite value is a
    lie. This mirrors the metrics layer's `ok=False` discipline.
  - **The report is a distribution, not a hit rate.** Bessembinder (2018): stock
    returns are so skewed that a strategy can be right 55% of the time and still
    lose money. So `StrategyScore` carries the full quantile spread and states
    the median next to the mean; the hit rate is present but never stands alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any


class Strategy(StrEnum):
    """Who selected the pick. Everything but AGENT is a benchmark the agent must
    beat before its cost is justified (PLAN.md 6)."""

    AGENT = "agent"
    """Survived the screen AND the LLM research pipeline without a veto."""

    SCREEN = "screen"
    """The deterministic composite rank, top-N. The product with no LLM at all."""

    UNIVERSE_EW = "universe_ew"
    """Baseline #1: the equal-weighted screened universe -- hold everything that
    passed the hard excludes. The bar 'did picking add anything over not picking'."""

    EV_EBIT_DECILE = "ev_ebit_decile"
    """Baseline #2: the cheapest EV/EBIT names within each cohort, no LLM. THE
    benchmark -- if the agent does not beat this, the LLM layer is cost, not
    signal, and the tool should say so out loud."""

    GBDT = "gbdt"
    """Baseline #3: a gradient-boosted tree on the same tabular features. Levy
    (2026) found a GBDT beats the best commercial LLM with no look-ahead. Needs
    labeled forward-return history to train; honestly INSUFFICIENT until enough
    picks have been scored (see baselines.GradientBoostedTree)."""


def finite_features(values: dict[str, float]) -> dict[str, float]:
    """Keep only the finite metric values -- the ones safe to store and rank on.

    The screen's `metric_values` carries raw `inf` for self-funding firms'
    cash-runway and can carry `nan` from a degenerate ratio. Neither is valid
    JSON, and neither is a real feature value, so both are dropped here rather
    than serialized as a number that later code would trust.
    """
    return {name: value for name, value in values.items() if math.isfinite(value)}


@dataclass(frozen=True, slots=True)
class PaperPick:
    """One pre-registered paper position, one line of the ledger JSONL."""

    as_of: date
    """The pre-registration date -- when the pick was recorded, before the
    outcome was known. The forward-evidence clock starts here."""

    strategy: Strategy
    entity_id: str
    name: str | None
    cohort: str
    """The (country / accounting-standard / sector) cohort label, for context."""

    reference_price: float | None
    """Entry price at `as_of`. Manual for now (no price feed -- PLAN.md 1.3
    deferral); None when none was supplied, which makes the pick ungradeable
    rather than wrong."""

    currency: str | None
    weight: float
    """Portfolio weight within this strategy's book, so the strategy return is a
    real weighted portfolio return. Equal-weight => 1/N. A pick the strategy
    recorded but excluded from its book (e.g. an agent veto) carries weight 0."""

    rank: int | None
    """Position in the strategy's ordered selection (1 = most preferred), or None
    when the strategy does not rank (equal-weight universe)."""

    score: float | None
    """The strategy's own score for this name (composite, -EV/EBIT, model prob).
    None when the strategy does not score."""

    vetoed: bool | None
    """Agent strategy only: whether the research pipeline vetoed the name. None
    for the deterministic strategies, which cannot veto."""

    features: dict[str, float] = field(default_factory=dict)
    """The finite ranking metrics at pick time -- provenance, and the GBDT
    baseline's training features. Always pre-filtered through `finite_features`."""

    run_id: str = ""
    """Ties every pick written by one `scout pick` invocation together."""

    note: str | None = None

    @property
    def pick_id(self) -> str:
        """Stable identity: one strategy picks one entity at most once per day."""
        return f"{self.as_of.isoformat()}:{self.strategy.value}:{self.entity_id}"

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready dict. `date` -> ISO string, enum -> its value, and
        `features` is already finite so it serializes cleanly."""
        return {
            "as_of": self.as_of.isoformat(),
            "strategy": self.strategy.value,
            "entity_id": self.entity_id,
            "name": self.name,
            "cohort": self.cohort,
            "reference_price": self.reference_price,
            "currency": self.currency,
            "weight": self.weight,
            "rank": self.rank,
            "score": self.score,
            "vetoed": self.vetoed,
            "features": self.features,
            "run_id": self.run_id,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaperPick:
        return cls(
            as_of=date.fromisoformat(data["as_of"]),
            strategy=Strategy(data["strategy"]),
            entity_id=data["entity_id"],
            name=data.get("name"),
            cohort=data.get("cohort", ""),
            reference_price=data.get("reference_price"),
            currency=data.get("currency"),
            weight=data.get("weight", 0.0),
            rank=data.get("rank"),
            score=data.get("score"),
            vetoed=data.get("vetoed"),
            features=data.get("features") or {},
            run_id=data.get("run_id", ""),
            note=data.get("note"),
        )


@dataclass(frozen=True, slots=True)
class ScoredPick:
    """One graded pick: the pick plus what price did over the forward window."""

    pick: PaperPick
    exit_price: float
    gross_return: float
    """exit / reference - 1, before costs."""

    net_return: float
    """gross_return - round-trip cost. The number the strategy actually earns."""


@dataclass(frozen=True, slots=True)
class Distribution:
    """The full spread of a set of returns. Reported instead of a lone mean,
    because a skewed distribution's mean and median tell different stories and
    the median is the more honest summary of a typical outcome (PLAN.md 6)."""

    n: int
    mean: float
    median: float
    stdev: float
    minimum: float
    p10: float
    p25: float
    p75: float
    p90: float
    maximum: float
    hit_rate: float
    """Fraction of picks with a positive net return. Present for continuity with
    the old score.js, but deliberately never the headline number."""


@dataclass(frozen=True, slots=True)
class StrategyScore:
    """How one strategy's book did over the forward window."""

    strategy: Strategy
    n_picks: int
    """Every pick the strategy recorded, including ungradeable and weight-0 ones."""

    n_scored: int
    """Picks that actually contributed to the book: had both prices and weight>0."""

    portfolio_return: float | None
    """Weight-weighted net return across the scored book -- the strategy's number.
    None when nothing in the book could be graded."""

    distribution: Distribution | None
    """The equal-weighted spread of the individual scored picks. None when
    nothing could be graded."""

    ungradeable: int = 0
    """Picks dropped for want of a reference or forward price -- a blind spot,
    reported rather than hidden."""

    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Comparison:
    """The agent measured against one baseline. The verdict the whole project
    turns on: does the LLM layer earn its cost?"""

    baseline: Strategy
    agent_return: float | None
    baseline_return: float | None
    delta: float | None
    """agent_return - baseline_return, in the same units (fraction). Positive =
    the agent beat the baseline. None when either side had nothing to grade."""

    verdict: str
    """Plain-language read, including the honest 'insufficient evidence' when the
    sample is too small to mean anything."""


@dataclass(frozen=True, slots=True)
class Evaluation:
    """The full forward-scoring result across every strategy in the ledger."""

    as_of_exit: date
    cost_bps: float
    """Round-trip cost applied to every gross return, in basis points."""

    scores: dict[Strategy, StrategyScore]
    comparisons: list[Comparison]
    total_scored: int
    warnings: list[str] = field(default_factory=list)
    """Cross-cutting caveats -- above all the small-sample warning carried over
    verbatim in spirit from the old score.js: a high hit rate on a handful of
    picks is noise, not evidence."""
