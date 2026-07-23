"""The screen orchestrator: entities in, ranked watchlist out.

For every entity in the fundamentals store it gathers the pieces the other
modules produce -- the profile, the metric report -- then applies the hard
excludes, drops the failures, assigns each survivor to its peer cohort, and
ranks within that cohort. The output is a `ScreenResult` a human can eyeball:
the ranked names with their block scores, the excluded names with reasons, and
an honest tally of which excludes could not be evaluated for lack of data.

This is deliberately the last deterministic step. No LLM has run; the research
agents (phase 5) only ever see names that already survived here.
"""

from __future__ import annotations

from scout.config import Config
from scout.fundamentals.models import FundamentalsSnapshot
from scout.fundamentals.store import FundamentalsStore
from scout.metrics.base import MarketData
from scout.metrics.report import MetricReport, compute_metrics
from scout.screen.cohorts import standard_from_taxonomy
from scout.screen.excludes import ScreenInput, evaluate_excludes, exclusion_reasons, is_excluded
from scout.screen.models import (
    CohortKey,
    Decision,
    EntityProfile,
    ExcludedCandidate,
    ScoredCandidate,
    ScreenResult,
)
from scout.screen.profile import ProfileStore
from scout.screen.rank import RankInput, RankWeights, rank

# Metrics that feed the ranking blocks. Anything else in the report (flags like
# is_net_net) is informational and not ranked on.
_RANK_METRICS = frozenset(
    {
        "earnings_yield", "fcf_yield", "net_cash_to_mcap", "ev_ebit", "ev_sales", "ncav_to_mcap",
        "gp_to_assets", "roic", "piotroski_f", "accruals",
        "altman_z", "cash_runway_months", "beneish_m", "share_issuance",
    }
)


def _cohort_of(snapshot: FundamentalsSnapshot, profile: EntityProfile | None) -> CohortKey:
    country = (profile.country if profile else None) or snapshot.entity.country or "unknown"
    standard = standard_from_taxonomy(snapshot.taxonomy)
    sector = (profile.sector if profile else None) or "unknown"
    return CohortKey(country=country, accounting_standard=standard, sector=sector)


def _rank_metrics(report: MetricReport) -> dict[str, float]:
    """The computed, finite ranking metrics for one entity."""
    out: dict[str, float] = {}
    for name in _RANK_METRICS:
        metric = report.metrics.get(name)
        if metric is not None and metric.ok and metric.value is not None:
            out[name] = metric.value
    return out


def run_screen(
    config: Config,
    *,
    market_data: dict[str, MarketData] | None = None,
    min_cohort: int = 3,
    weights: RankWeights | None = None,
    include_excluded_sectors: bool = False,
) -> ScreenResult:
    """Screen every entity in the store into a ranked watchlist.

    `market_data` maps entity_id -> MarketData for the (deferred) price feed;
    without it the valuation block is simply absent and names rank on quality and
    safety, which is the honest state until prices are wired in.
    """
    market_data = market_data or {}
    result = ScreenResult(ranked=[], excluded=[])

    with FundamentalsStore(config.db_path, read_only=True) as store, _open_profiles(config) as profiles:
        entity_ids = store.all_entity_ids()
        rank_inputs: list[RankInput] = []
        cohort_counter: dict[str, int] = {}

        for entity_id in entity_ids:
            snapshots = store.snapshots_for_entity(entity_id)
            if not snapshots:
                continue
            latest = max(snapshots, key=lambda s: s.period_end)
            profile = profiles.get(entity_id) if profiles else None
            name = (profile.name if profile else None) or latest.entity.name

            # Financials and utilities are out of the ranked universe entirely
            # (their value multiples are not comparable), unless explicitly kept.
            if profile and profile.is_excluded_sector and not include_excluded_sectors:
                result.excluded.append(
                    ExcludedCandidate(entity_id, name, [f"excluded sector: {profile.sector}"])
                )
                continue

            report = compute_metrics(snapshots, market=market_data.get(entity_id))

            checks = evaluate_excludes(
                ScreenInput(entity_id=entity_id, profile=profile, snapshots=snapshots, report=report)
            )
            for check in checks:
                if check.decision == Decision.INSUFFICIENT:
                    result.insufficient_checks[check.rule] = (
                        result.insufficient_checks.get(check.rule, 0) + 1
                    )

            result.universe_size += 1

            if is_excluded(checks):
                result.excluded.append(
                    ExcludedCandidate(entity_id, name, exclusion_reasons(checks))
                )
                continue

            cohort = _cohort_of(latest, profile)
            cohort_counter[cohort.label()] = cohort_counter.get(cohort.label(), 0) + 1
            rank_inputs.append(
                RankInput(
                    entity_id=entity_id,
                    name=name,
                    cohort=cohort,
                    metrics=_rank_metrics(report) if report else {},
                    underfollowed=_is_underfollowed(profile),
                )
            )

    result.cohort_sizes = cohort_counter
    result.ranked = _rank(rank_inputs, min_cohort=min_cohort, weights=weights)
    result.notes = _notes(result, market_data)
    return result


def _rank(
    inputs: list[RankInput], *, min_cohort: int, weights: RankWeights | None
) -> list[ScoredCandidate]:
    if not inputs:
        return []
    if weights is None:
        return rank(inputs, min_cohort=min_cohort)
    return rank(inputs, weights=weights, min_cohort=min_cohort)


def _is_underfollowed(profile: EntityProfile | None) -> bool:
    """A conditioning flag, not a factor. Approximated for now by filer size:
    a smaller-reporting/non-accelerated filer is, by construction, less covered.
    A real coverage/institutional-ownership signal replaces this once 13F and
    analyst-count data are wired in."""
    if profile is None or not profile.filer_category:
        return False
    category = profile.filer_category.lower()
    return "non-accelerated" in category or "smaller reporting" in category


def _notes(result: ScreenResult, market_data: dict[str, MarketData]) -> list[str]:
    notes: list[str] = []
    if not market_data:
        notes.append(
            "No price data supplied: the valuation (cheap) block is absent, so names "
            "are ranked on quality and safety only. Supply prices to rank on cheapness."
        )
    if result.insufficient_checks:
        blind = ", ".join(sorted(result.insufficient_checks))
        notes.append(
            f"Excludes not evaluated for lack of data (blind spots): {blind}. "
            "These filings were NOT verified clean on those rules."
        )
    small = [c for c, n in result.cohort_sizes.items() if n < 3]
    if small:
        notes.append(
            f"{len(small)} cohort(s) too small to z-score -- their members are listed "
            "but unranked. Rank becomes meaningful as the universe grows."
        )
    return notes


class _NullProfiles:
    """Stand-in when no profile DB exists yet, so the screen still runs."""

    def get(self, entity_id: str) -> EntityProfile | None:
        return None

    def __enter__(self) -> _NullProfiles:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _open_profiles(config: Config):  # type: ignore[no-untyped-def]
    """Open the profile store read-only, or a null stand-in if enrichment has
    never run. The screen must not require `scout enrich` to have happened."""
    profiles_path = config.db_path
    if not profiles_path.exists():
        return _NullProfiles()
    try:
        store = ProfileStore(profiles_path, read_only=True)
        store.count()  # touches the table; raises if enrichment never created it
        return store
    except Exception:
        # A DB that exists but has no profiles table yet (fundamentals ingested,
        # never enriched) should not crash the screen.
        return _NullProfiles()
