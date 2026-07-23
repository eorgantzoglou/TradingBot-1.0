"""The deterministic screen: hard excludes, then within-cohort ranking.

No LLM runs here. The screen turns per-entity metrics into a ranked watchlist by
first dropping names that fail a hard exclude (dilution, reverse splits,
going-concern with no runway, shells, delinquency), then ranking survivors on a
cheap x quality x safety composite -- but only against peers in the same
(country x accounting-standard x sector) cohort. It is meant to be usable and
eyeballed on its own before any research agent runs on top of it (PLAN.md phase
4): if the screen is junk, no LLM will fix it.
"""

from scout.screen.models import (
    CohortKey,
    Decision,
    EntityProfile,
    ExcludeCheck,
    ExcludedCandidate,
    ScoredCandidate,
    ScreenResult,
)

__all__ = [
    "CohortKey",
    "Decision",
    "EntityProfile",
    "ExcludeCheck",
    "ExcludedCandidate",
    "ScoredCandidate",
    "ScreenResult",
]
