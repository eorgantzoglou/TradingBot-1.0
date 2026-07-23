"""Sector classification and cohort assignment.

Two jobs, both pure and testable:

  1. Map a SIC code to a coarse sector and decide whether it is one the screen
     excludes wholesale (financials and utilities -- EV/EBIT is meaningless for a
     bank's balance sheet and a regulated utility's returns are set by rate case,
     not the market).
  2. Assign an entity to a (country x accounting-standard x sector) cohort, so
     ranking only ever compares genuine peers (PLAN.md section 1.4).

SIC ranges are the US Standard Industrial Classification major divisions. Non-US
filers rarely carry SIC; they fall to sector "unknown", which forms its own
cohort rather than being force-fit against US peers.
"""

from __future__ import annotations

# US SIC major-group ranges -> coarse sector label. Ordered, first match wins.
_SIC_SECTORS: list[tuple[int, int, str]] = [
    (100, 999, "agriculture"),
    (1000, 1499, "mining"),
    (1500, 1799, "construction"),
    (2000, 3999, "manufacturing"),
    (4000, 4899, "transport_communications"),
    (4900, 4999, "utilities"),
    (5000, 5199, "wholesale"),
    (5200, 5999, "retail"),
    (6000, 6799, "finance"),
    (7000, 8999, "services"),
    (9100, 9999, "public_admin"),
]

# Sectors excluded from the ranked universe: their economics make the value
# multiples the screen ranks on non-comparable or meaningless.
_EXCLUDED_SECTORS = frozenset({"finance", "utilities"})


def sic_to_sector(sic: str | None) -> str:
    """Coarse sector for a SIC code, or "unknown" when absent/unparseable."""
    if not sic:
        return "unknown"
    try:
        code = int(sic)
    except (TypeError, ValueError):
        return "unknown"
    for low, high, sector in _SIC_SECTORS:
        if low <= code <= high:
            return sector
    return "unknown"


def is_excluded_sector(sector: str) -> bool:
    return sector in _EXCLUDED_SECTORS


def standard_from_taxonomy(taxonomy: str) -> str:
    """Normalize a filing taxonomy to an accounting-standard label for cohorting.

    us-gaap and ifrs-full are the two we parse today; a national extension
    taxonomy (e.g. a JGAAP or a Ukrainian extension) keeps its own label so it is
    never pooled with clean IFRS filers, whose numbers are more comparable.
    """
    if taxonomy == "us-gaap":
        return "US-GAAP"
    if taxonomy == "ifrs-full":
        return "IFRS"
    return taxonomy or "unknown"
