"""Hard-exclude engine for the screen -- where the expected value actually is.

PLAN.md section 3.2 is blunt about this: the hard excludes are "cheap to run, and
most of the value". More money is lost in microcaps by *owning* a dilution
machine, a shell or a going-concern than is made by ranking the survivors one
notch better. So this module is the part of the screen we most want to be right,
and the ranking (rank.py) is downstream of it -- a name that fails a hard exclude
is never ranked, never shown to the LLM, never bought.

The single biggest killer is dilution. Pontiff & Woodgate (2008, JF) found share
issuance predicts the cross-section more strongly than size, book-to-market or
momentum; a microcap that grew its share count 30% in a year is a wealth-transfer
machine pointed at its existing holders, and no valuation multiple survives that.
`check_dilution` is therefore the rule this file exists for.

The care point -- and the reason `check_shell` is deliberately conservative -- is
the pre-revenue-vs-shell distinction. A blank-check shell and an early clinical
biotech both report ~zero revenue; only one is worthless. False-excluding a real
pre-revenue microcap is a *miss* (we lose a candidate we wanted), so the shell
test fires only when there is positive evidence of non-operation (a non-operating
entity classification), never on absence of revenue alone.

Honesty about blind spots is a first-class output here (PLAN.md's "no silent
caps"). A rule whose input data is absent returns INSUFFICIENT, which is recorded
and surfaced but does NOT exclude -- "we could not verify this" and "this is fine"
are different claims, and letting the former masquerade as the latter is exactly
how a dilution machine slips through a screen that looks clean. Three rules
(reverse split, toxic convertibles, paid promotion) need filing-text or external
data that is not wired in yet; rather than pretend they passed, they declare
themselves INSUFFICIENT so the screen reports honestly what it did not check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scout.fundamentals.concepts import Concept
from scout.fundamentals.models import FundamentalsSnapshot
from scout.metrics.report import MetricReport
from scout.screen.models import Decision, EntityProfile, ExcludeCheck

# Thresholds are named here rather than buried in the rule bodies so the policy is
# readable in one place and a change is a one-line edit with a paper trail.
_DILUTION_YOY_LIMIT = 0.20
"""Share-count growth above this over a year is the dilution fingerprint."""

_MIN_RUNWAY_MONTHS = 12.0
"""Below one year of cash at the current burn is near-term insolvency risk."""

_NAME_CHANGE_LOOKBACK_MONTHS = 24
"""A name/ticker change inside this window is the classic shell-hijack pattern."""

_REVENUE_ZERO_EPS = 1.0
"""Revenue whose absolute value is under one reporting unit is treated as zero --
a filer reporting exactly 0 and one reporting a rounding-dust figure are the same
"no real sales" fact for the shell test."""


@dataclass
class ScreenInput:
    """Everything one entity's exclude rules need, gathered once.

    Each field can be absent (None / empty) because microcap coverage is ragged;
    the rules degrade to INSUFFICIENT rather than crashing on a missing piece.
    """

    entity_id: str
    profile: EntityProfile | None
    snapshots: list[FundamentalsSnapshot]
    """The entity's snapshot history, newest-not-required. May be length 1."""

    report: MetricReport | None
    """Computed metrics for the entity. None when nothing could be computed."""


# --------------------------------------------------------------------------- #
# Small helpers -- keep the rule bodies to their decision logic
# --------------------------------------------------------------------------- #


def _check(rule: str, decision: Decision, reason: str) -> ExcludeCheck:
    return ExcludeCheck(rule=rule, decision=decision, reason=reason)


def _latest_snapshot(inp: ScreenInput) -> FundamentalsSnapshot | None:
    """The most recent snapshot by period end, or None if there are none."""
    if not inp.snapshots:
        return None
    return max(inp.snapshots, key=lambda s: s.period_end)


# --------------------------------------------------------------------------- #
# Checkable-now rules -- these can actually EXCLUDE
# --------------------------------------------------------------------------- #


def check_dilution(inp: ScreenInput) -> ExcludeCheck:
    """Exclude names whose share count grew past the dilution limit YoY.

    The single most important microcap red flag (Pontiff & Woodgate). Reads the
    precomputed `share_issuance` metric, a YoY fraction; the metric itself is
    INSUFFICIENT (needs two annual filings) when the pair is unavailable, and we
    surface that reason verbatim rather than inventing our own.
    """
    rule = "dilution"
    metric = _metric(inp, "share_issuance")
    if metric is None:
        return _check(rule, Decision.INSUFFICIENT, "no metrics computed for entity")
    if not metric.ok:
        return _check(rule, Decision.INSUFFICIENT, metric.reason or "share issuance not computable")

    value = metric.value
    if value is not None and value > _DILUTION_YOY_LIMIT:
        return _check(rule, Decision.EXCLUDE, f"shares grew {value:.0%} YoY — dilution")
    return _check(rule, Decision.PASS, f"share growth {value:.0%} YoY within limit")


def check_cash_runway(inp: ScreenInput) -> ExcludeCheck:
    """Exclude names with under a year of cash at the current burn.

    PLAN.md pairs a going-concern OPINION with a sub-12-month runway; we have only
    the runway half wired in, so we exclude on runway alone but say explicitly in
    the reason that the opinion itself was not verified -- again, no pretending we
    checked more than we did. `inf` runway means the firm is self-funding (no
    burn) and is never a concern.
    """
    rule = "cash_runway"
    metric = _metric(inp, "cash_runway_months")
    if metric is None:
        return _check(rule, Decision.INSUFFICIENT, "no metrics computed for entity")
    if not metric.ok:
        return _check(rule, Decision.INSUFFICIENT, metric.reason or "cash runway not computable")

    value = metric.value
    if value is not None and not math.isinf(value) and value < _MIN_RUNWAY_MONTHS:
        return _check(
            rule,
            Decision.EXCLUDE,
            f"cash runway {value:.1f} months — near-term insolvency risk; "
            "note the going-concern OPINION itself was not verified",
        )
    return _check(rule, Decision.PASS, "cash runway of a year or more")


def check_delinquent_filings(inp: ScreenInput) -> ExcludeCheck:
    """Exclude filers with a recent NT 10-K / NT 10-Q (late-filing notice).

    The flag is a tri-state on the profile: True (delinquent), False (on time), or
    None (could not be determined) -- and None must not read as "on time".
    """
    rule = "delinquent_filings"
    if inp.profile is None or inp.profile.has_recent_late_filing is None:
        return _check(rule, Decision.INSUFFICIENT, "filing-history flags unavailable")
    if inp.profile.has_recent_late_filing:
        return _check(rule, Decision.EXCLUDE, "recent NT 10-K/10-Q late filing")
    return _check(rule, Decision.PASS, "no recent late-filing notice")


def check_name_change(inp: ScreenInput) -> ExcludeCheck:
    """Exclude names/tickers changed inside the shell-hijack lookback window.

    The profile carries months-since-most-recent-change when a change is known.
    When that is None we must distinguish "no change happened" from "we have no
    data": an existing profile with no (recent) former name is a genuine PASS,
    while no profile at all is INSUFFICIENT.
    """
    rule = "name_change"
    profile = inp.profile
    if profile is None:
        return _check(rule, Decision.INSUFFICIENT, "no profile to check name/ticker history")

    months = profile.name_changed_within_months
    if months is not None:
        if months <= _NAME_CHANGE_LOOKBACK_MONTHS:
            return _check(
                rule,
                Decision.EXCLUDE,
                f"name/ticker changed {months} months ago — shell-hijack pattern",
            )
        return _check(rule, Decision.PASS, f"last name change {months} months ago, outside window")

    # months is None but we DO have a profile: absence of a recent change is real
    # information here, not missing data.
    if not profile.former_names:
        return _check(rule, Decision.PASS, "no name changes on record")
    return _check(rule, Decision.PASS, "former name(s) on record but none recent")


def check_shell(inp: ScreenInput) -> ExcludeCheck:
    """Exclude shells / blank-checks: no operations, not merely pre-revenue.

    Deliberately conservative (see module docstring). Revenue at ~zero is a
    necessary but NOT sufficient condition -- an early biotech clears it too. We
    additionally require positive evidence of non-operation: an entity_type that
    is present and something other than "operating". Absent or "operating"
    classification with zero revenue reads as pre-revenue, which we PASS on
    purpose so we do not throw away a real microcap.
    """
    rule = "shell"
    latest = _latest_snapshot(inp)
    if latest is None:
        return _check(rule, Decision.INSUFFICIENT, "no snapshot to assess operations")

    revenue = latest.get(Concept.REVENUE)
    has_revenue = revenue is not None and abs(revenue) >= _REVENUE_ZERO_EPS
    if has_revenue:
        return _check(rule, Decision.PASS, "reports revenue -- operating")

    entity_type = inp.profile.entity_type if inp.profile is not None else None
    # entity_type None = unknown classification -> we withhold the shell judgement
    # rather than exclude on absence of revenue alone.
    if entity_type is None or entity_type == "operating":
        return _check(rule, Decision.PASS, "no revenue but not classified non-operating (pre-revenue, not a shell)")
    return _check(
        rule,
        Decision.EXCLUDE,
        f"no revenue and entity_type {entity_type!r} — shell / blank-check",
    )


# --------------------------------------------------------------------------- #
# Not-yet-wired rules -- honestly INSUFFICIENT, never a silent pass
# --------------------------------------------------------------------------- #


def check_reverse_split(inp: ScreenInput) -> ExcludeCheck:
    """Reverse split within 24 months -- data not wired in yet."""
    return _check(
        "reverse_split",
        Decision.INSUFFICIENT,
        "needs 8-K / full-text scan for reverse splits -- not yet wired in",
    )


def check_toxic_convertibles(inp: ScreenInput) -> ExcludeCheck:
    """Floating/discount (death-spiral) convertibles -- data not wired in yet."""
    return _check(
        "toxic_convertibles",
        Decision.INSUFFICIENT,
        "needs filing-text scan for floating/discount convertibles -- not yet wired in",
    )


def check_paid_promotion(inp: ScreenInput) -> ExcludeCheck:
    """Detectable paid stock promotion -- data not wired in yet."""
    return _check(
        "paid_promotion",
        Decision.INSUFFICIENT,
        "needs external promotion-database cross-reference -- not yet wired in",
    )


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


def _metric(inp: ScreenInput, name: str):
    """The named MetricValue from the report, or None if no report / not present.

    Both "no report" and "report without this metric" collapse to None here; the
    caller turns that into INSUFFICIENT, so a rule never has to re-derive the
    two-step lookup.
    """
    if inp.report is None:
        return None
    return inp.report.metrics.get(name)


# One line per rule. Adding a rule is adding it here; the engine and the tests
# both iterate this tuple, so nothing else needs to change.
ALL_RULES = (
    check_dilution,
    check_cash_runway,
    check_delinquent_filings,
    check_name_change,
    check_shell,
    check_reverse_split,
    check_toxic_convertibles,
    check_paid_promotion,
)


def evaluate_excludes(inp: ScreenInput) -> list[ExcludeCheck]:
    """Run every rule and return all checks -- passes, excludes and insufficients.

    We return the full set (not just the excludes) on purpose: the INSUFFICIENT
    checks are the screen's reported blind spots and the PASS checks are its audit
    trail, and both are wanted downstream.
    """
    return [rule(inp) for rule in ALL_RULES]


def is_excluded(checks: list[ExcludeCheck]) -> bool:
    """True iff any check decided EXCLUDE. INSUFFICIENT never excludes."""
    return any(check.decision is Decision.EXCLUDE for check in checks)


def exclusion_reasons(checks: list[ExcludeCheck]) -> list[str]:
    """The reasons of the EXCLUDE checks, in rule order -- why a name was dropped."""
    return [check.reason for check in checks if check.decision is Decision.EXCLUDE]
