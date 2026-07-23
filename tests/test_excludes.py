"""Tests for scout.screen.excludes -- the hard-exclude engine.

Every checkable rule is exercised on all three of its paths: the EXCLUDE path
(the red flag is present), the PASS path (it is absent), and the INSUFFICIENT
path (the data to judge it is missing). The last is not a formality -- the whole
point of the engine is that a rule it could not evaluate says so instead of
silently passing, so those cases are asserted as carefully as the excludes.

Fixtures are hand-built from the exact models the engine reads (EntityProfile,
FundamentalsSnapshot, MetricReport) so a passing test cannot be an artifact of a
convenience factory hiding a wrong default.
"""

from __future__ import annotations

import math
from datetime import date

from scout.fundamentals.concepts import Concept
from scout.fundamentals.models import CanonicalFact, EntityRef, FundamentalsSnapshot
from scout.metrics.base import MetricValue
from scout.metrics.report import MetricReport
from scout.screen.excludes import (
    ALL_RULES,
    ScreenInput,
    check_cash_runway,
    check_delinquent_filings,
    check_dilution,
    check_name_change,
    check_paid_promotion,
    check_reverse_split,
    check_shell,
    check_toxic_convertibles,
    evaluate_excludes,
    exclusion_reasons,
    is_excluded,
)
from scout.screen.models import Decision, EntityProfile, FormerName

# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #

_ENTITY = EntityRef(source="sec", entity_id="1", identifier_scheme="cik", name="Test Co")


def _snap(
    values: dict[Concept, float],
    *,
    period_end: date = date(2024, 12, 31),
    fiscal_period: str | None = "FY",
) -> FundamentalsSnapshot:
    """A snapshot carrying exactly the given concept values (only `value` matters)."""
    facts = {
        concept: CanonicalFact(
            entity_id="1",
            concept=concept,
            value=value,
            currency="USD",
            period_end=period_end,
            period_start=None,
            fiscal_year=period_end.year,
            fiscal_period=fiscal_period,
            accession="unit-acc",
            source_concept=f"test:{concept.value}",
            taxonomy="us-gaap",
        )
        for concept, value in values.items()
    }
    return FundamentalsSnapshot(
        entity=_ENTITY,
        period_end=period_end,
        fiscal_year=period_end.year,
        fiscal_period=fiscal_period,
        currency="USD",
        taxonomy="us-gaap",
        accession="unit-acc",
        filing_date=None,
        facts=facts,
    )


def _report(**metrics: MetricValue) -> MetricReport:
    """A MetricReport holding exactly the given named metric values."""
    return MetricReport(
        entity_id="1",
        period_end="2024-12-31",
        fiscal_period="FY",
        currency="USD",
        has_market_data=False,
        has_annual_pair=True,
        metrics=dict(metrics),
    )


def _pct_metric(name: str, value: float) -> MetricValue:
    return MetricValue.of(name, value, "pct", "cross-period")


def _count_metric(name: str, value: float) -> MetricValue:
    return MetricValue.of(name, value, "count", "point-in-time")


def _missing_metric(name: str, reason: str) -> MetricValue:
    return MetricValue.missing(name, "pct", "cross-period", reason)


def _profile(**kwargs) -> EntityProfile:
    return EntityProfile(entity_id="1", source="sec", **kwargs)


def _input(
    *,
    profile: EntityProfile | None = None,
    snapshots: list[FundamentalsSnapshot] | None = None,
    report: MetricReport | None = None,
) -> ScreenInput:
    return ScreenInput(
        entity_id="1",
        profile=profile,
        snapshots=snapshots or [],
        report=report,
    )


# --------------------------------------------------------------------------- #
# check_dilution
# --------------------------------------------------------------------------- #


def test_dilution_over_limit_excludes():
    inp = _input(report=_report(share_issuance=_pct_metric("share_issuance", 0.35)))
    check = check_dilution(inp)
    assert check.decision is Decision.EXCLUDE
    assert "35%" in check.reason


def test_dilution_within_limit_passes():
    inp = _input(report=_report(share_issuance=_pct_metric("share_issuance", 0.10)))
    assert check_dilution(inp).decision is Decision.PASS


def test_dilution_missing_metric_is_insufficient():
    metric = _missing_metric("share_issuance", "needs two annual filings")
    inp = _input(report=_report(share_issuance=metric))
    check = check_dilution(inp)
    assert check.decision is Decision.INSUFFICIENT
    assert check.reason == "needs two annual filings"


def test_dilution_no_report_is_insufficient():
    assert check_dilution(_input()).decision is Decision.INSUFFICIENT


# --------------------------------------------------------------------------- #
# check_cash_runway
# --------------------------------------------------------------------------- #


def test_runway_below_year_excludes():
    inp = _input(report=_report(cash_runway_months=_count_metric("cash_runway_months", 6.0)))
    check = check_cash_runway(inp)
    assert check.decision is Decision.EXCLUDE
    assert "6.0 months" in check.reason
    # We must be honest that the going-concern opinion itself was not verified.
    assert "OPINION" in check.reason


def test_runway_above_year_passes():
    inp = _input(report=_report(cash_runway_months=_count_metric("cash_runway_months", 24.0)))
    assert check_cash_runway(inp).decision is Decision.PASS


def test_runway_infinite_passes():
    inp = _input(report=_report(cash_runway_months=_count_metric("cash_runway_months", math.inf)))
    assert check_cash_runway(inp).decision is Decision.PASS


def test_runway_missing_is_insufficient():
    metric = MetricValue.missing("cash_runway_months", "count", "point-in-time", "no cash figure")
    inp = _input(report=_report(cash_runway_months=metric))
    assert check_cash_runway(inp).decision is Decision.INSUFFICIENT


# --------------------------------------------------------------------------- #
# check_delinquent_filings
# --------------------------------------------------------------------------- #


def test_late_filing_true_excludes():
    inp = _input(profile=_profile(has_recent_late_filing=True))
    assert check_delinquent_filings(inp).decision is Decision.EXCLUDE


def test_late_filing_false_passes():
    inp = _input(profile=_profile(has_recent_late_filing=False))
    assert check_delinquent_filings(inp).decision is Decision.PASS


def test_late_filing_none_is_insufficient():
    inp = _input(profile=_profile(has_recent_late_filing=None))
    assert check_delinquent_filings(inp).decision is Decision.INSUFFICIENT


def test_late_filing_no_profile_is_insufficient():
    assert check_delinquent_filings(_input()).decision is Decision.INSUFFICIENT


# --------------------------------------------------------------------------- #
# check_name_change
# --------------------------------------------------------------------------- #


def test_name_change_recent_excludes():
    inp = _input(profile=_profile(name_changed_within_months=12))
    check = check_name_change(inp)
    assert check.decision is Decision.EXCLUDE
    assert "12 months" in check.reason


def test_name_change_old_passes():
    inp = _input(profile=_profile(name_changed_within_months=40))
    assert check_name_change(inp).decision is Decision.PASS


def test_name_change_empty_former_names_passes():
    # Profile present, no change on record -> real PASS, not INSUFFICIENT.
    inp = _input(profile=_profile(former_names=()))
    assert check_name_change(inp).decision is Decision.PASS


def test_name_change_old_former_name_passes():
    former = FormerName(name="Old Shell Inc", from_date=None, to_date=date(2015, 1, 1))
    inp = _input(profile=_profile(former_names=(former,)))
    assert check_name_change(inp).decision is Decision.PASS


def test_name_change_no_profile_is_insufficient():
    assert check_name_change(_input()).decision is Decision.INSUFFICIENT


# --------------------------------------------------------------------------- #
# check_shell
# --------------------------------------------------------------------------- #


def test_shell_zero_revenue_non_operating_excludes():
    inp = _input(
        profile=_profile(entity_type="shell"),
        snapshots=[_snap({Concept.REVENUE: 0.0, Concept.TOTAL_ASSETS: 5_000.0})],
    )
    assert check_shell(inp).decision is Decision.EXCLUDE


def test_shell_zero_revenue_empty_entity_type_excludes():
    # An empty-string classification is present (not None) and != "operating".
    inp = _input(
        profile=_profile(entity_type=""),
        snapshots=[_snap({Concept.REVENUE: 0.0})],
    )
    assert check_shell(inp).decision is Decision.EXCLUDE


def test_shell_pre_revenue_biotech_passes():
    # Zero revenue but classified operating: a real pre-revenue firm, NOT a shell.
    inp = _input(
        profile=_profile(entity_type="operating"),
        snapshots=[_snap({Concept.REVENUE: 0.0, Concept.TOTAL_ASSETS: 80_000_000.0})],
    )
    assert check_shell(inp).decision is Decision.PASS


def test_shell_unknown_entity_type_passes():
    # No classification -> withhold judgement rather than exclude on revenue alone.
    inp = _input(snapshots=[_snap({Concept.REVENUE: 0.0})])
    assert check_shell(inp).decision is Decision.PASS


def test_shell_with_revenue_passes():
    inp = _input(
        profile=_profile(entity_type="shell"),
        snapshots=[_snap({Concept.REVENUE: 5_000_000.0})],
    )
    assert check_shell(inp).decision is Decision.PASS


def test_shell_no_snapshot_is_insufficient():
    inp = _input(profile=_profile(entity_type="shell"))
    assert check_shell(inp).decision is Decision.INSUFFICIENT


# --------------------------------------------------------------------------- #
# Not-yet-wired rules -- always INSUFFICIENT
# --------------------------------------------------------------------------- #


def test_not_wired_rules_always_insufficient():
    inp = _input(
        profile=_profile(entity_type="operating"),
        snapshots=[_snap({Concept.REVENUE: 1_000_000.0})],
        report=_report(),
    )
    for rule in (check_reverse_split, check_toxic_convertibles, check_paid_promotion):
        check = rule(inp)
        assert check.decision is Decision.INSUFFICIENT
        assert "not yet wired in" in check.reason


# --------------------------------------------------------------------------- #
# Engine: evaluate_excludes / is_excluded / exclusion_reasons
# --------------------------------------------------------------------------- #


def test_evaluate_returns_one_check_per_rule():
    checks = evaluate_excludes(_input())
    assert len(checks) == len(ALL_RULES)
    # Rule names are unique -- no rule silently shadows another.
    assert len({c.rule for c in checks}) == len(ALL_RULES)


def test_is_excluded_true_when_any_rule_excludes():
    inp = _input(report=_report(share_issuance=_pct_metric("share_issuance", 0.90)))
    checks = evaluate_excludes(inp)
    assert is_excluded(checks) is True
    assert any("dilution" in r for r in exclusion_reasons(checks))


def test_clean_healthy_entity_is_not_excluded():
    profile = _profile(
        entity_type="operating",
        has_recent_late_filing=False,
        former_names=(),
    )
    report = _report(
        share_issuance=_pct_metric("share_issuance", 0.02),
        cash_runway_months=_count_metric("cash_runway_months", 48.0),
    )
    inp = _input(
        profile=profile,
        snapshots=[_snap({Concept.REVENUE: 20_000_000.0})],
        report=report,
    )
    checks = evaluate_excludes(inp)
    assert is_excluded(checks) is False
    assert exclusion_reasons(checks) == []


def test_insufficient_never_counts_as_excluded():
    # Nothing wired in at all: every checkable rule is INSUFFICIENT, and the three
    # unwired ones too -- but the entity is NOT excluded, because "unverified" is
    # not "bad".
    checks = evaluate_excludes(_input())
    assert is_excluded(checks) is False
    assert all(c.decision is Decision.INSUFFICIENT for c in checks)
