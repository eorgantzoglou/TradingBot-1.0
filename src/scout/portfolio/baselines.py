"""The dumb baselines the agent has to beat before it earns its cost.

PLAN.md 6 names three, in ascending order of how much they threaten the LLM's
value:

  1. **Equal-weighted screened universe** -- hold everything that passed the hard
     excludes. Answers 'did selecting add anything over not selecting?'.
  2. **EV/EBIT within-cohort decile** -- the cheapest names per cohort, no LLM.
     THE benchmark. If the agent cannot beat a one-line value screen, the whole
     research pipeline is cost rather than signal.
  3. **A gradient-boosted tree on the same tabular features** -- Levy (2026)
     found a GBDT beats the best commercial LLM by 2.7pp with no look-ahead. It
     needs labeled forward-return history to train, which does not exist until
     picks have been scored, so it reports INSUFFICIENT until then rather than
     fabricating a model from nothing.

A selector maps the screen's ranked candidates to an ordered `Selection` list.
It is pure and deterministic -- prices and any training history are passed in,
never fetched -- so a baseline is as testable as the metrics beneath it.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field

from scout.portfolio.models import finite_features
from scout.screen.models import ScoredCandidate

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Selection:
    """One name a baseline chose, with its place and score in that baseline."""

    entity_id: str
    rank: int | None
    score: float | None
    note: str | None = None


@dataclass(frozen=True, slots=True)
class LabeledObservation:
    """A past pick whose forward return is now known -- one GBDT training row.

    Assembled by `evaluate.py` from scored picks. `features` are the finite
    metrics recorded at pick time; `forward_return` is the realized net return.
    """

    features: dict[str, float]
    forward_return: float


class InsufficientHistory(RuntimeError):
    """The GBDT baseline cannot train yet: too few labeled observations. An
    honest 'not enough data', not an error to hide."""


def equal_weight_universe(candidates: list[ScoredCandidate]) -> list[Selection]:
    """Baseline #1: every survivor, unranked (equal weight is applied later).

    Ordered by composite for a stable, readable ledger -- names the screen could
    not score (too-small cohort) sort last but are still held, because 'hold the
    whole universe' means the whole universe.
    """
    ordered = sorted(
        candidates,
        key=lambda c: (c.composite is None, -(c.composite or 0.0)),
    )
    return [
        Selection(entity_id=c.entity_id, rank=None, score=c.composite)
        for c in ordered
    ]


def top_composite(candidates: list[ScoredCandidate], *, top: int) -> list[Selection]:
    """The deterministic product itself: the top-N by composite, no LLM.

    This is the 'screen' strategy -- ranking without research. Candidates with no
    composite (unrankable cohort) are excluded, because the screen makes no claim
    about them.
    """
    ranked = [c for c in candidates if c.composite is not None]
    ranked.sort(key=lambda c: -(c.composite or 0.0))
    return [
        Selection(entity_id=c.entity_id, rank=i, score=c.composite)
        for i, c in enumerate(ranked[:top], start=1)
    ]


def ev_ebit_decile(
    candidates: list[ScoredCandidate], *, decile_fraction: float = 0.1
) -> list[Selection]:
    """Baseline #2: the cheapest EV/EBIT names, within each cohort.

    Cheapness only compares within a cohort (PLAN.md 1.4 -- EV/EBIT under JGAAP
    is not the same claim as under IFRS), so the decile is taken per cohort and
    then pooled. In a small cohort the 'decile' rounds up to at least one name,
    so the baseline is never empty for a cohort that has any priced member.

    Needs `ev_ebit`, which needs a price. Without a price feed most names have
    none, so this baseline is legitimately empty until prices are supplied -- an
    honest gap (surfaced by the caller as a note), not a bug.
    """
    by_cohort: dict[str, list[ScoredCandidate]] = defaultdict(list)
    for cand in candidates:
        value = cand.metric_values.get("ev_ebit")
        # A negative EV/EBIT (loss-making) is not 'cheap', it is meaningless as a
        # multiple, so only finite positive values are eligible.
        if value is not None and math.isfinite(value) and value > 0:
            by_cohort[cand.cohort.label()].append(cand)

    chosen: list[tuple[float, ScoredCandidate]] = []
    for members in by_cohort.values():
        members.sort(key=lambda c: c.metric_values["ev_ebit"])  # cheapest first
        take = max(1, math.ceil(len(members) * decile_fraction))
        chosen.extend((m.metric_values["ev_ebit"], m) for m in members[:take])

    # Pool the per-cohort winners and rank globally by cheapness for the ledger.
    chosen.sort(key=lambda pair: pair[0])
    return [
        # score = -EV/EBIT so that, like every other strategy's score, higher is
        # more preferred.
        Selection(entity_id=c.entity_id, rank=i, score=-ev)
        for i, (ev, c) in enumerate(chosen, start=1)
    ]


@dataclass(slots=True)
class GradientBoostedTree:
    """Baseline #3, behind an honest gate.

    A GBDT that predicts forward return from the tabular features. It can only
    exist once there are enough (features -> realized return) rows to train on,
    and those accumulate only as picks are scored. Until then `select` raises
    `InsufficientHistory`, and the caller records that the baseline is not yet
    trainable rather than inventing picks.

    `lightgbm` is an optional dependency (the `gbdt` extra) precisely because
    this baseline does nothing useful for months; a fresh install should not pay
    for a compiled tree library it cannot use yet.
    """

    min_train_rows: int = 200
    """Below this, a tree learns noise. Deliberately conservative -- the point of
    a baseline is to be a fair bar, and a bar fit to 30 rows is not fair."""

    num_boost_round: int = 200
    feature_order: list[str] = field(default_factory=list)
    """Fixed at train time and reused at predict time, so a row missing a feature
    lines up by name, never by position."""

    def select(
        self,
        candidates: list[ScoredCandidate],
        *,
        history: list[LabeledObservation],
        top: int,
    ) -> list[Selection]:
        if len(history) < self.min_train_rows:
            raise InsufficientHistory(
                f"the GBDT baseline needs >= {self.min_train_rows} scored picks to "
                f"train; the ledger has {len(history)}. It becomes available as the "
                "forward archive accumulates -- until then it is honestly skipped."
            )

        model, order = self._train(history)
        scored: list[tuple[float, str]] = []
        for cand in candidates:
            row = [finite_features(cand.metric_values).get(name, math.nan) for name in order]
            predicted = float(model.predict([row])[0])
            scored.append((predicted, cand.entity_id))

        scored.sort(key=lambda pair: -pair[0])  # highest predicted return first
        return [
            Selection(entity_id=eid, rank=i, score=pred)
            for i, (pred, eid) in enumerate(scored[:top], start=1)
        ]

    def _train(self, history: list[LabeledObservation]):  # type: ignore[no-untyped-def]
        """Fit a LightGBM regressor on features -> forward return.

        Imported lazily so the dependency is only required when the baseline can
        actually run. The feature set is the union of every feature seen across
        the training rows, sorted for determinism.
        """
        try:
            # Imported lazily so the optional dependency is only required when
            # the baseline can actually run (see the class docstring).
            import lightgbm as lgb
        except ImportError as exc:  # pragma: no cover - exercised only with the extra absent
            raise InsufficientHistory(
                "the GBDT baseline needs the optional 'lightgbm' dependency "
                "(install the project's 'gbdt' extra: uv sync --extra gbdt)."
            ) from exc

        order = sorted({name for obs in history for name in obs.features})
        rows = [[obs.features.get(name, math.nan) for name in order] for obs in history]
        labels = [obs.forward_return for obs in history]

        dataset = lgb.Dataset(rows, label=labels, feature_name=order)
        # seed + deterministic so the baseline is reproducible run-to-run -- the
        # house rule is that every number here is deterministic and testable, and
        # a baseline that shifts on re-train would not be a fair, stable bar.
        params = {
            "objective": "regression",
            "verbosity": -1,
            "min_data_in_leaf": 5,
            "seed": 0,
            "deterministic": True,
            "force_row_wise": True,
        }
        model = lgb.train(params, dataset, num_boost_round=self.num_boost_round)
        return model, order
