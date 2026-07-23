"""Tests for scout.fundamentals.normalize -- the concept-mapping normalizer.

Two layers, both load-bearing:

  * GOLDEN tests against the REAL archived filings on disk (3M's 10-Q under
    us-gaap, a Ukrainian ESEF issuer under ifrs-full). Every value asserted here
    was verified by hand against the filing. These are the whole point: the SEC
    bug this project shipped once came from a fabricated fixture that agreed with
    buggy code, so nothing here is mocked -- the bytes are the ones the publisher
    disseminated. Guarded by skipif so a checkout without the archive still runs,
    but on a machine that has it they MUST pass.

  * UNIT tests that construct small `RawFact` lists by hand to pin each tricky
    behavior (period coherence, dimension exclusion, fallback warnings, ambiguity
    detection, FY inference, the shares cover-date relaxation) in isolation, so a
    regression is pinpointed to one rule rather than only surfacing as a golden
    mismatch.

The session-scoped `parsed_3m` fixture parses the 13MB 3M submission once.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from scout.fundamentals.concepts import Concept, PeriodType
from scout.fundamentals.models import EntityRef, RawFact
from scout.fundamentals.normalize import normalize_filing
from scout.fundamentals.parse.base import ParsedFiling
from scout.fundamentals.parse.esef import EsefJsonParser
from scout.fundamentals.parse.sec import SecXbrlParser

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SEC_PATH = (
    _REPO_ROOT
    / "data" / "archive" / "sec" / "2026" / "07" / "21"
    / "0000066740-26-000246" / "v1" / "0000066740-26-000246.txt"
)
_SEC_ACCESSION = "0000066740-26-000246"

_ESEF_PATH = (
    _REPO_ROOT
    / "data" / "archive" / "esef" / "2026" / "07" / "21"
    / "25825" / "v1" / "25825.json"
)
_ESEF_ACCESSION = "25825"

_needs_sec = pytest.mark.skipif(
    not _SEC_PATH.exists(), reason=f"real SEC archive not present at {_SEC_PATH}"
)
_needs_esef = pytest.mark.skipif(
    not _ESEF_PATH.exists(), reason=f"real ESEF archive not present at {_ESEF_PATH}"
)


def _span_days(fact, period_end: date) -> int | None:
    """The span a canonical fact covers, using the snapshot's period_end (which is
    what a duration CanonicalFact stores) and the fact's own period_start."""
    if fact.period_start is None:
        return None
    return (period_end - fact.period_start).days


# ---------------------------------------------------------------------------
# GOLDEN: 3M 10-Q, us-gaap
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def parsed_3m() -> ParsedFiling:
    """Parse the real 3M 10-Q exactly once for the whole session (it's 13MB)."""
    payload = _SEC_PATH.read_bytes()
    return SecXbrlParser().parse(payload, accession=_SEC_ACCESSION, entity_hint={"cik": "66740"})


@pytest.fixture(scope="session")
def snapshot_3m(parsed_3m: ParsedFiling):
    entity = EntityRef(
        source="sec", entity_id="66740", identifier_scheme="cik", name=parsed_3m.entity_name
    )
    snap = normalize_filing(parsed_3m, entity)
    assert snap is not None
    return snap


@_needs_sec
class TestGolden3M:
    def test_snapshot_identity(self, snapshot_3m) -> None:
        assert snapshot_3m.entity.entity_id == "66740"
        assert snapshot_3m.taxonomy == "us-gaap"
        assert snapshot_3m.period_end == date(2026, 6, 30)
        assert snapshot_3m.fiscal_period == "YTD6"
        assert snapshot_3m.fiscal_year == 2026
        assert snapshot_3m.currency == "USD"

    def test_number_of_concepts_resolved(self, snapshot_3m) -> None:
        assert len(snapshot_3m.facts) == 29

    def test_revenue(self, snapshot_3m) -> None:
        rev = snapshot_3m.facts[Concept.REVENUE]
        assert rev.value == 12_530_000_000
        assert rev.source_concept == "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
        assert rev.period_start == date(2026, 1, 1)
        assert _span_days(rev, snapshot_3m.period_end) == 180

    def test_total_assets_is_instant(self, snapshot_3m) -> None:
        assets = snapshot_3m.facts[Concept.TOTAL_ASSETS]
        assert assets.value == 34_924_000_000
        assert assets.period_start is None  # instant
        assert assets.source_concept == "us-gaap:Assets"

    def test_net_income(self, snapshot_3m) -> None:
        ni = snapshot_3m.facts[Concept.NET_INCOME]
        assert ni.value == 1_586_000_000
        assert _span_days(ni, snapshot_3m.period_end) == 180

    def test_operating_income(self, snapshot_3m) -> None:
        oi = snapshot_3m.facts[Concept.OPERATING_INCOME]
        assert oi.value == 2_381_000_000
        assert _span_days(oi, snapshot_3m.period_end) == 180

    def test_total_equity_is_instant(self, snapshot_3m) -> None:
        eq = snapshot_3m.facts[Concept.TOTAL_EQUITY]
        assert eq.value == 2_952_000_000
        assert eq.period_start is None  # instant

    def test_cash_resolves_via_fallback_tag(self, snapshot_3m) -> None:
        # 3M reports cash under the restricted-cash tag, not the plain one -- the
        # exact heterogeneity the ordered candidate list exists to handle.
        cash = snapshot_3m.facts[Concept.CASH_AND_EQUIVALENTS]
        assert cash.value == 2_955_000_000
        assert cash.source_concept == (
            "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"
        )

    def test_shares_outstanding_from_dei_cover_tag(self, snapshot_3m) -> None:
        shares = snapshot_3m.facts[Concept.SHARES_OUTSTANDING]
        assert shares.value == 515_722_417
        assert shares.source_concept == "dei:EntityCommonStockSharesOutstanding"

    def test_period_coherence_all_durations_share_one_span(self, snapshot_3m) -> None:
        # The whole point: a snapshot is one coherent period. Every income and
        # cash-flow figure must cover the SAME span (180 days here) -- never a mix
        # of the discrete 90-day quarter and the 180-day year-to-date.
        duration_spans = {
            concept.value: _span_days(fact, snapshot_3m.period_end)
            for concept, fact in snapshot_3m.facts.items()
            if concept.meta.period_type is PeriodType.DURATION
        }
        assert duration_spans, "expected at least one duration concept"
        assert set(duration_spans.values()) == {180}, duration_spans

    def test_income_and_cashflow_concepts_are_all_180_days(self, snapshot_3m) -> None:
        # Spell out the cross-statement coherence explicitly: income items and
        # cash-flow items (which 3M reports only year-to-date) line up.
        for concept in (
            Concept.REVENUE,
            Concept.COST_OF_REVENUE,
            Concept.NET_INCOME,
            Concept.OPERATING_INCOME,
            Concept.CASH_FROM_OPERATIONS,
            Concept.CAPEX,
        ):
            fact = snapshot_3m.facts[concept]
            assert _span_days(fact, snapshot_3m.period_end) == 180, concept.value

    def test_warnings_flag_fallback_tags(self, snapshot_3m) -> None:
        warnings = snapshot_3m.warnings
        assert len(warnings) >= 3
        for concept_name in ("cost_of_revenue", "interest_expense", "cash_and_equivalents"):
            assert any(
                concept_name in w and "fallback" in w for w in warnings
            ), f"expected a fallback-tag warning for {concept_name}: {warnings}"

    def test_no_false_ambiguity_warning(self, snapshot_3m) -> None:
        # The fixed bug: a 90-day fact and a 180-day fact for the same tag are
        # different spans, NOT differing duplicates, and must never be flagged as
        # ambiguous "differing non-dimensioned values".
        assert not any(
            "differing non-dimensioned values" in w for w in snapshot_3m.warnings
        ), snapshot_3m.warnings


# ---------------------------------------------------------------------------
# GOLDEN: Ukrainian ESEF issuer, ifrs-full
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def parsed_esef() -> ParsedFiling:
    payload = _ESEF_PATH.read_bytes()
    return EsefJsonParser().parse(
        payload, accession=_ESEF_ACCESSION, entity_hint={"lei": "44356194", "country": "UA"}
    )


@pytest.fixture(scope="session")
def snapshot_esef(parsed_esef: ParsedFiling):
    entity = EntityRef(
        source="esef",
        entity_id=parsed_esef.entity_id,
        identifier_scheme="national_id",
        country="UA",
        name=parsed_esef.entity_name,
    )
    snap = normalize_filing(parsed_esef, entity)
    assert snap is not None
    return snap


@_needs_esef
class TestGoldenEsef:
    def test_snapshot_identity(self, snapshot_esef) -> None:
        assert snapshot_esef.taxonomy == "ifrs-full"
        assert snapshot_esef.entity.entity_id == "44356194"
        assert snapshot_esef.period_end == date(2022, 1, 1)
        assert snapshot_esef.currency == "UAH"

    def test_fiscal_period_is_inferred_fy(self, snapshot_esef) -> None:
        # The IFRS facts carry no fiscal metadata; the normalizer infers FY from
        # the ~365-day annual duration ending on the reporting date.
        assert snapshot_esef.fiscal_period == "FY"

    def test_number_of_concepts_resolved(self, snapshot_esef) -> None:
        assert len(snapshot_esef.facts) == 23

    def test_revenue(self, snapshot_esef) -> None:
        rev = snapshot_esef.facts[Concept.REVENUE]
        assert rev.value == 19_459_000
        assert rev.source_concept == "ifrs-full:Revenue"

    def test_total_assets_is_instant(self, snapshot_esef) -> None:
        assets = snapshot_esef.facts[Concept.TOTAL_ASSETS]
        assert assets.value == 152_914_000
        assert assets.period_start is None  # instant

    def test_gross_profit_reported_directly(self, snapshot_esef) -> None:
        # Unlike 3M (which does not tag gross profit), this IFRS filer reports it.
        gp = snapshot_esef.facts[Concept.GROSS_PROFIT]
        assert gp.value == 9_453_000

    def test_every_duration_spans_about_a_year(self, snapshot_esef) -> None:
        for concept, fact in snapshot_esef.facts.items():
            if concept.meta.period_type is PeriodType.DURATION:
                span = _span_days(fact, snapshot_esef.period_end)
                assert span is not None and 350 <= span <= 380, (concept.value, span)


# ---------------------------------------------------------------------------
# UNIT: hand-built RawFact sets exercising one rule each (no archive needed)
# ---------------------------------------------------------------------------

_ENTITY = EntityRef(source="sec", entity_id="1", identifier_scheme="cik", name="Test Co")

# Canonical primary tags used across the unit tests.
_REVENUE_TAG = "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
_ASSETS_TAG = "us-gaap:Assets"
_SHARES_TAG = "dei:EntityCommonStockSharesOutstanding"


def _raw(
    tag: str,
    value: float,
    *,
    period_type: PeriodType = PeriodType.DURATION,
    period_start: date | None = None,
    period_end: date,
    is_dimensioned: bool = False,
    fiscal_period: str | None = None,
    fiscal_year: int | None = None,
    unit: str | None = "USD",
) -> RawFact:
    taxonomy, local_name = tag.split(":", 1)
    return RawFact(
        accession="unit-acc",
        taxonomy=taxonomy,
        local_name=local_name,
        value=value,
        unit=unit,
        period_type=period_type,
        period_start=period_start,
        period_end=period_end,
        is_dimensioned=is_dimensioned,
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
    )


def _filing(facts: list[RawFact], *, period_of_report: date | None) -> ParsedFiling:
    return ParsedFiling(
        accession="unit-acc",
        source="sec",
        entity_id="1",
        entity_name="Test Co",
        taxonomy="us-gaap",
        period_of_report=period_of_report,
        filing_date=None,
        facts=facts,
    )


_END = date(2026, 6, 30)
_START_90 = date(2026, 4, 1)  # 90 days before _END
_START_180 = date(2026, 1, 1)  # 180 days before _END


class TestPeriodCoherence:
    """Both a 90-day and a 180-day duration end on the same date; the fiscal
    period decides which one belongs to the snapshot."""

    def test_ytd6_picks_the_180_day_duration(self) -> None:
        facts = [
            _raw(_REVENUE_TAG, 90.0, period_start=_START_90, period_end=_END, fiscal_period="YTD6"),
            _raw(_REVENUE_TAG, 180.0, period_start=_START_180, period_end=_END, fiscal_period="YTD6"),
        ]
        snap = normalize_filing(_filing(facts, period_of_report=_END), _ENTITY)
        assert snap is not None
        rev = snap.facts[Concept.REVENUE]
        assert rev.value == 180.0
        assert rev.period_start == _START_180

    def test_ytd3_picks_the_90_day_duration(self) -> None:
        facts = [
            _raw(_REVENUE_TAG, 90.0, period_start=_START_90, period_end=_END, fiscal_period="YTD3"),
            _raw(_REVENUE_TAG, 180.0, period_start=_START_180, period_end=_END, fiscal_period="YTD3"),
        ]
        snap = normalize_filing(_filing(facts, period_of_report=_END), _ENTITY)
        assert snap is not None
        rev = snap.facts[Concept.REVENUE]
        assert rev.value == 90.0
        assert rev.period_start == _START_90


class TestDimensionExclusion:
    def test_non_dimensioned_total_is_chosen_over_dimensioned(self) -> None:
        facts = [
            _raw(_ASSETS_TAG, 40.0, period_type=PeriodType.INSTANT, period_end=_END, is_dimensioned=True),
            _raw(_ASSETS_TAG, 100.0, period_type=PeriodType.INSTANT, period_end=_END, is_dimensioned=False),
        ]
        snap = normalize_filing(_filing(facts, period_of_report=_END), _ENTITY)
        assert snap is not None
        assert snap.facts[Concept.TOTAL_ASSETS].value == 100.0

    def test_only_dimensioned_means_concept_is_absent(self) -> None:
        facts = [
            _raw(_ASSETS_TAG, 40.0, period_type=PeriodType.INSTANT, period_end=_END, is_dimensioned=True),
        ]
        snap = normalize_filing(_filing(facts, period_of_report=_END), _ENTITY)
        assert snap is not None
        assert Concept.TOTAL_ASSETS not in snap.facts


class TestFallbackWarning:
    def test_second_priority_tag_resolves_and_warns(self) -> None:
        # COST_OF_REVENUE's primary tag is CostOfGoodsAndServicesSold; supply only
        # the 2nd candidate, CostOfRevenue.
        facts = [
            _raw("us-gaap:CostOfRevenue", 50.0, period_start=_START_90, period_end=_END),
        ]
        snap = normalize_filing(_filing(facts, period_of_report=_END), _ENTITY)
        assert snap is not None
        fact = snap.facts[Concept.COST_OF_REVENUE]
        assert fact.source_concept == "us-gaap:CostOfRevenue"
        assert any(
            "cost_of_revenue" in w and "us-gaap:CostOfRevenue" in w and "fallback" in w
            for w in snap.warnings
        ), snap.warnings


class TestAmbiguity:
    def test_same_span_differing_values_warns(self) -> None:
        # Two non-dimensioned instants for the same tag and period, different
        # values -> the filing is internally inconsistent, so warn.
        facts = [
            _raw(_ASSETS_TAG, 100.0, period_type=PeriodType.INSTANT, period_end=_END),
            _raw(_ASSETS_TAG, 111.0, period_type=PeriodType.INSTANT, period_end=_END),
        ]
        snap = normalize_filing(_filing(facts, period_of_report=_END), _ENTITY)
        assert snap is not None
        assert any("differing non-dimensioned values" in w for w in snap.warnings), snap.warnings

    def test_different_spans_do_not_warn(self) -> None:
        # A 90-day and a 180-day duration for the same tag are expected companions,
        # NOT an ambiguous duplicate -- the false-ambiguity bug that was fixed.
        facts = [
            _raw(_REVENUE_TAG, 90.0, period_start=_START_90, period_end=_END, fiscal_period="YTD6"),
            _raw(_REVENUE_TAG, 180.0, period_start=_START_180, period_end=_END, fiscal_period="YTD6"),
        ]
        snap = normalize_filing(_filing(facts, period_of_report=_END), _ENTITY)
        assert snap is not None
        assert not any("differing non-dimensioned values" in w for w in snap.warnings), snap.warnings


class TestFiscalPeriodInference:
    def test_annual_duration_without_fiscal_metadata_infers_fy(self) -> None:
        end = date(2022, 1, 1)
        facts = [
            _raw(_REVENUE_TAG, 500.0, period_start=date(2021, 1, 1), period_end=end),
        ]
        snap = normalize_filing(_filing(facts, period_of_report=end), _ENTITY)
        assert snap is not None
        assert snap.fiscal_period == "FY"

    def test_quarter_length_duration_without_fiscal_metadata_is_not_fy(self) -> None:
        facts = [
            _raw(_REVENUE_TAG, 100.0, period_start=_START_90, period_end=_END),
        ]
        snap = normalize_filing(_filing(facts, period_of_report=_END), _ENTITY)
        assert snap is not None
        assert snap.fiscal_period != "FY"


class TestSharesCoverDateRelaxation:
    def test_shares_off_date_is_taken_with_a_warning(self) -> None:
        # Shares outstanding is stated 'as of' the cover date, which can fall a few
        # days after period_end; with none exactly on period_end it is still taken.
        cover_date = date(2026, 7, 5)  # 5 days after period_end
        facts = [
            _raw(
                _SHARES_TAG, 1000.0, period_type=PeriodType.INSTANT,
                period_end=cover_date, unit="shares",
            ),
        ]
        snap = normalize_filing(_filing(facts, period_of_report=_END), _ENTITY)
        assert snap is not None
        assert snap.facts[Concept.SHARES_OUTSTANDING].value == 1000.0
        assert any("cover-date" in w for w in snap.warnings), snap.warnings

    def test_other_instant_concepts_do_not_get_the_relaxation(self) -> None:
        # A balance-sheet instant that misses period_end is dropped, not relaxed --
        # only shares outstanding gets the cover-date treatment.
        off_date = date(2026, 7, 5)
        facts = [
            _raw(_ASSETS_TAG, 100.0, period_type=PeriodType.INSTANT, period_end=off_date),
        ]
        snap = normalize_filing(_filing(facts, period_of_report=_END), _ENTITY)
        assert snap is not None
        assert Concept.TOTAL_ASSETS not in snap.facts


class TestReturnsNone:
    def test_no_facts_returns_none(self) -> None:
        assert normalize_filing(_filing([], period_of_report=_END), _ENTITY) is None

    def test_no_period_of_report_returns_none(self) -> None:
        facts = [_raw(_REVENUE_TAG, 100.0, period_start=_START_180, period_end=_END)]
        assert normalize_filing(_filing(facts, period_of_report=None), _ENTITY) is None
