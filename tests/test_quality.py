"""Tests for scout.metrics.quality.

Two layers, mirroring test_normalize.py:

  * HAND-BUILT snapshots that pin every rule in isolation. The Piotroski block is
    a golden worked example: current + prior FY snapshots are engineered so every
    one of the nine signals is verifiable by eye, and a perfect 9, a 0, a mixed
    case, a missing-input degrade and the non-FY rejection are each asserted.

  * GOLDEN tests against the REAL archived filings on disk (3M's us-gaap 10-Q,
    the Ukrainian ifrs-full ESEF report), built exactly the way test_normalize.py
    builds them so the numbers are the ones the publishers disseminated, not a
    fixture that could agree with buggy code. Guarded by skipif.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from scout.fundamentals.concepts import Concept
from scout.fundamentals.models import CanonicalFact, EntityRef, FundamentalsSnapshot
from scout.fundamentals.normalize import normalize_filing
from scout.fundamentals.parse.esef import EsefJsonParser
from scout.fundamentals.parse.sec import SecXbrlParser
from scout.metrics.quality import (
    accruals,
    gross_profit_to_assets,
    piotroski_f_score,
    piotroski_from_history,
    roic,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Hand-built snapshot helpers
# ---------------------------------------------------------------------------

_ENTITY = EntityRef(source="test", entity_id="1", identifier_scheme="cik", name="Test Co")


def _snap(
    values: dict[Concept, float],
    *,
    fiscal_period: str = "FY",
    period_end: date = date(2025, 12, 31),
) -> FundamentalsSnapshot:
    """Build a snapshot whose get() returns exactly the given concept values."""
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
            accession="acc",
            source_concept="test:tag",
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
        accession="acc",
        filing_date=None,
        facts=facts,
    )


# ---------------------------------------------------------------------------
# GP/A, accruals, ROIC on hand-built snapshots
# ---------------------------------------------------------------------------


class TestGrossProfitToAssets:
    def test_reported_gross_profit_is_not_flagged_derived(self) -> None:
        snap = _snap({Concept.GROSS_PROFIT: 300.0, Concept.TOTAL_ASSETS: 1000.0})
        mv = gross_profit_to_assets(snap)
        assert mv.ok
        assert mv.value == pytest.approx(0.30)
        assert mv.basis == "annual"
        assert not any("derived" in w for w in mv.warnings)

    def test_derived_gross_profit_warns(self) -> None:
        snap = _snap(
            {Concept.REVENUE: 1000.0, Concept.COST_OF_REVENUE: 700.0, Concept.TOTAL_ASSETS: 1000.0}
        )
        mv = gross_profit_to_assets(snap)
        assert mv.ok
        assert mv.value == pytest.approx(0.30)
        assert any("derived" in w for w in mv.warnings)

    def test_interim_basis(self) -> None:
        snap = _snap(
            {Concept.GROSS_PROFIT: 100.0, Concept.TOTAL_ASSETS: 1000.0}, fiscal_period="YTD6"
        )
        assert gross_profit_to_assets(snap).basis == "interim"

    def test_missing_input_is_not_ok(self) -> None:
        snap = _snap({Concept.GROSS_PROFIT: 300.0})  # no assets
        mv = gross_profit_to_assets(snap)
        assert not mv.ok
        assert mv.value is None
        assert mv.reason


class TestAccruals:
    def test_basic(self) -> None:
        snap = _snap(
            {
                Concept.NET_INCOME: 200.0,
                Concept.CASH_FROM_OPERATIONS: 50.0,
                Concept.TOTAL_ASSETS: 1000.0,
            }
        )
        mv = accruals(snap)
        assert mv.ok
        assert mv.value == pytest.approx((200.0 - 50.0) / 1000.0)  # 0.15, a red flag

    def test_missing_cfo_is_not_ok(self) -> None:
        snap = _snap({Concept.NET_INCOME: 200.0, Concept.TOTAL_ASSETS: 1000.0})
        mv = accruals(snap)
        assert not mv.ok
        assert mv.reason


class TestRoic:
    def test_uses_effective_tax_rate(self) -> None:
        # tax 30 / pretax 150 -> 20% effective; NOPAT = 200 * 0.8 = 160.
        # invested = equity 500 + debt (100+300) - cash 100 = 800. ROIC = 0.20.
        snap = _snap(
            {
                Concept.OPERATING_INCOME: 200.0,
                Concept.INCOME_TAX_EXPENSE: 30.0,
                Concept.INCOME_BEFORE_TAX: 150.0,
                Concept.TOTAL_EQUITY: 500.0,
                Concept.SHORT_TERM_DEBT: 100.0,
                Concept.LONG_TERM_DEBT: 300.0,
                Concept.CASH_AND_EQUIVALENTS: 100.0,
            }
        )
        mv = roic(snap)
        assert mv.ok
        assert mv.inputs["effective_tax_rate"] == pytest.approx(0.20)
        assert mv.value == pytest.approx(160.0 / 800.0)

    def test_missing_tax_falls_back_to_21_percent_and_warns(self) -> None:
        snap = _snap(
            {
                Concept.OPERATING_INCOME: 100.0,
                Concept.TOTAL_EQUITY: 500.0,
                Concept.LONG_TERM_DEBT: 300.0,
                Concept.CASH_AND_EQUIVALENTS: 100.0,
            }
        )
        mv = roic(snap)
        assert mv.ok
        assert mv.inputs["effective_tax_rate"] == pytest.approx(0.21)
        assert any("21%" in w for w in mv.warnings)

    def test_nonsensical_tax_rate_is_clamped_and_warns(self) -> None:
        # pretax near zero -> raw rate 900/1 = 900, must clamp to 0.5 not explode.
        snap = _snap(
            {
                Concept.OPERATING_INCOME: 100.0,
                Concept.INCOME_TAX_EXPENSE: 900.0,
                Concept.INCOME_BEFORE_TAX: 1.0,
                Concept.TOTAL_EQUITY: 500.0,
                Concept.LONG_TERM_DEBT: 300.0,
                Concept.CASH_AND_EQUIVALENTS: 100.0,
            }
        )
        mv = roic(snap)
        assert mv.ok
        assert mv.inputs["effective_tax_rate"] == pytest.approx(0.50)
        assert any("clamped" in w for w in mv.warnings)
        # NOPAT = 100 * (1 - 0.5) = 50, not a negative blow-up.
        assert mv.value == pytest.approx(50.0 / 700.0)

    def test_non_positive_invested_capital_is_missing(self) -> None:
        # A net-cash firm: cash exceeds equity + debt -> invested capital <= 0.
        snap = _snap(
            {
                Concept.OPERATING_INCOME: 100.0,
                Concept.TOTAL_EQUITY: 200.0,
                Concept.LONG_TERM_DEBT: 50.0,
                Concept.CASH_AND_EQUIVALENTS: 400.0,
            }
        )
        mv = roic(snap)
        assert not mv.ok
        assert "invested capital" in mv.reason


# ---------------------------------------------------------------------------
# GOLDEN Piotroski worked example -- every signal verifiable by eye
# ---------------------------------------------------------------------------

# Prior-year (t-1) baseline shared by the perfect-9, missing-input examples.
#   ROA_p          = 100 / 1000            = 0.10
#   leverage_p     = LTD 300 / 1000        = 0.30
#   current_ratio  = 400 / 200             = 2.00
#   gross_margin_p = 240 / 800             = 0.30
#   asset_turn_p   = 800 / 1000            = 0.80
_PRIOR_9 = {
    Concept.NET_INCOME: 100.0,
    Concept.TOTAL_ASSETS: 1000.0,
    Concept.CASH_FROM_OPERATIONS: 120.0,
    Concept.LONG_TERM_DEBT: 300.0,
    Concept.CURRENT_ASSETS: 400.0,
    Concept.CURRENT_LIABILITIES: 200.0,
    Concept.SHARES_OUTSTANDING: 1000.0,
    Concept.REVENUE: 800.0,
    Concept.GROSS_PROFIT: 240.0,
}

# Current-year (t) engineered so ALL nine signals pass:
#   S1 ROA_t 0.20 > 0                    pass
#   S2 CFO_t 300 > 0                     pass
#   S3 ROA_t 0.20 > ROA_p 0.10          pass
#   S4 CFO/TA 0.30 > ROA_t 0.20         pass
#   S5 leverage 0.20 < 0.30             pass
#   S6 current ratio 3.00 > 2.00        pass
#   S7 shares 1000 <= 1000              pass
#   S8 gross margin 0.50 > 0.30         pass
#   S9 asset turnover 1.20 > 0.80       pass
_CURRENT_9 = {
    Concept.NET_INCOME: 200.0,
    Concept.TOTAL_ASSETS: 1000.0,
    Concept.CASH_FROM_OPERATIONS: 300.0,
    Concept.LONG_TERM_DEBT: 200.0,
    Concept.CURRENT_ASSETS: 600.0,
    Concept.CURRENT_LIABILITIES: 200.0,
    Concept.SHARES_OUTSTANDING: 1000.0,
    Concept.REVENUE: 1200.0,
    Concept.GROSS_PROFIT: 600.0,
}


class TestPiotroskiWorkedExample:
    def test_perfect_nine(self) -> None:
        mv = piotroski_f_score(_snap(_CURRENT_9), _snap(_PRIOR_9))
        assert mv.ok
        assert mv.value == 9.0
        assert mv.kind == "score"
        assert mv.basis == "cross-period"
        assert not mv.warnings
        for key in ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"):
            assert mv.inputs[key] == 1.0, key
        assert mv.inputs["score"] == 9.0

    def test_zero(self) -> None:
        # Every signal engineered to FAIL while staying evaluable (no warnings).
        prior = {
            Concept.NET_INCOME: 100.0,
            Concept.TOTAL_ASSETS: 1000.0,
            Concept.CASH_FROM_OPERATIONS: 120.0,
            Concept.LONG_TERM_DEBT: 300.0,
            Concept.CURRENT_ASSETS: 400.0,
            Concept.CURRENT_LIABILITIES: 200.0,
            Concept.SHARES_OUTSTANDING: 1000.0,
            Concept.REVENUE: 800.0,
            Concept.GROSS_PROFIT: 240.0,  # margin 0.30
        }
        current = {
            Concept.NET_INCOME: -50.0,  # ROA_t -0.05: S1 fail, S3 fail (< 0.10)
            Concept.TOTAL_ASSETS: 1000.0,
            Concept.CASH_FROM_OPERATIONS: -60.0,  # S2 fail; CFO/TA -0.06 <= ROA -0.05: S4 fail
            Concept.LONG_TERM_DEBT: 400.0,  # leverage 0.40 not < 0.30: S5 fail
            Concept.CURRENT_ASSETS: 200.0,
            Concept.CURRENT_LIABILITIES: 200.0,  # current ratio 1.0 not > 2.0: S6 fail
            Concept.SHARES_OUTSTANDING: 1200.0,  # dilution: S7 fail
            Concept.REVENUE: 600.0,  # asset turn 0.60 not > 0.80: S9 fail
            Concept.GROSS_PROFIT: 60.0,  # margin 0.10 not > 0.30: S8 fail
        }
        mv = piotroski_f_score(_snap(current), _snap(prior))
        assert mv.ok
        assert mv.value == 0.0
        assert not mv.warnings
        assert all(mv.inputs[f"S{i}"] == 0.0 for i in range(1, 10))

    def test_mixed_case_scores_four(self) -> None:
        # S1-S4 pass, S5-S9 fail -> 4. Hand-verified against the values below.
        current = {
            Concept.NET_INCOME: 200.0,  # ROA_t 0.20
            Concept.TOTAL_ASSETS: 1000.0,
            Concept.CASH_FROM_OPERATIONS: 300.0,  # CFO/TA 0.30 > ROA 0.20
            Concept.LONG_TERM_DEBT: 350.0,  # leverage 0.35 not < 0.30: S5 fail
            Concept.CURRENT_ASSETS: 300.0,
            Concept.CURRENT_LIABILITIES: 200.0,  # ratio 1.5 not > 2.0: S6 fail
            Concept.SHARES_OUTSTANDING: 1100.0,  # dilution: S7 fail
            Concept.REVENUE: 700.0,  # turnover 0.70 not > 0.80: S9 fail
            Concept.GROSS_PROFIT: 140.0,  # margin 0.20 not > 0.30: S8 fail
        }
        mv = piotroski_f_score(_snap(current), _snap(_PRIOR_9))
        assert mv.ok
        assert mv.value == 4.0
        assert [mv.inputs[f"S{i}"] for i in range(1, 10)] == [
            1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]

    def test_missing_input_zeroes_its_signal_and_warns(self) -> None:
        # Drop LONG_TERM_DEBT from the perfect-9 current -> S5 unevaluable.
        current = dict(_CURRENT_9)
        del current[Concept.LONG_TERM_DEBT]
        mv = piotroski_f_score(_snap(current), _snap(_PRIOR_9))
        assert mv.ok
        assert mv.value == 8.0  # the other eight still pass
        assert mv.inputs["S5"] == 0.0
        assert any("S5" in w for w in mv.warnings)
        assert len(mv.warnings) == 1

    def test_non_fy_current_is_rejected(self) -> None:
        mv = piotroski_f_score(_snap(_CURRENT_9, fiscal_period="YTD6"), _snap(_PRIOR_9))
        assert not mv.ok
        assert mv.value is None
        assert "FY" in mv.reason or "annual" in mv.reason

    def test_non_fy_prior_is_rejected(self) -> None:
        mv = piotroski_f_score(_snap(_CURRENT_9), _snap(_PRIOR_9, fiscal_period="Q4"))
        assert not mv.ok
        assert mv.reason


class TestPiotroskiFromHistory:
    def test_selects_recent_annual_pair(self) -> None:
        current = _snap(_CURRENT_9, period_end=date(2025, 12, 31))
        prior = _snap(_PRIOR_9, period_end=date(2024, 12, 31))
        # Order deliberately shuffled; select_annual_pair must re-sort.
        mv = piotroski_from_history([prior, current])
        assert mv.ok
        assert mv.value == 9.0

    def test_no_pair_is_missing(self) -> None:
        only_one = _snap(_CURRENT_9, period_end=date(2025, 12, 31))
        mv = piotroski_from_history([only_one])
        assert not mv.ok
        assert mv.reason


# ---------------------------------------------------------------------------
# GOLDEN: real 3M 10-Q (us-gaap) and Ukrainian ESEF report (ifrs-full)
# ---------------------------------------------------------------------------

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


@pytest.fixture(scope="session")
def snapshot_3m() -> FundamentalsSnapshot:
    payload = _SEC_PATH.read_bytes()
    parsed = SecXbrlParser().parse(payload, accession=_SEC_ACCESSION, entity_hint={"cik": "66740"})
    entity = EntityRef(
        source="sec", entity_id="66740", identifier_scheme="cik", name=parsed.entity_name
    )
    snap = normalize_filing(parsed, entity)
    assert snap is not None
    return snap


@pytest.fixture(scope="session")
def snapshot_esef() -> FundamentalsSnapshot:
    payload = _ESEF_PATH.read_bytes()
    parsed = EsefJsonParser().parse(payload, accession=_ESEF_ACCESSION, entity_hint={})
    entity = EntityRef("esef", parsed.entity_id, "national_id", country="UA")
    snap = normalize_filing(parsed, entity)
    assert snap is not None
    return snap


@_needs_sec
class TestGolden3M:
    def test_gross_profit_to_assets_is_derived(self, snapshot_3m) -> None:
        # 3M does not tag gross profit -> derives to revenue - COGS =
        # 12_530 - 7_391 = 5_139 (millions); / total assets 34_924 ~= 0.1472.
        mv = gross_profit_to_assets(snapshot_3m)
        assert mv.ok
        assert mv.value == pytest.approx(5_139_000_000 / 34_924_000_000, rel=1e-6)
        assert mv.value == pytest.approx(0.1472, abs=1e-4)
        assert any("derived" in w for w in mv.warnings)
        assert mv.basis == "interim"  # 3M snapshot is YTD6, not a full year

    def test_accruals(self, snapshot_3m) -> None:
        # (net income 1_586 - CFO 1_560) / assets 34_924 ~= 0.000745.
        mv = accruals(snapshot_3m)
        assert mv.ok
        assert mv.value == pytest.approx((1_586_000_000 - 1_560_000_000) / 34_924_000_000, rel=1e-6)
        assert mv.value == pytest.approx(0.000745, abs=1e-5)


@_needs_esef
class TestGoldenEsef:
    def test_gross_profit_to_assets_reported(self, snapshot_esef) -> None:
        # This IFRS filer reports gross profit directly (9_453) -> NOT derived.
        # / total assets 152_914 ~= 0.0618.
        mv = gross_profit_to_assets(snapshot_esef)
        assert mv.ok
        assert mv.value == pytest.approx(9_453_000 / 152_914_000, rel=1e-6)
        assert mv.value == pytest.approx(0.0618, abs=1e-4)
        assert not any("derived" in w for w in mv.warnings)
        assert mv.basis == "annual"

    def test_accruals_is_high(self, snapshot_esef) -> None:
        # net income 2_896 vs CFO -11_166: earnings far exceed cash -> a real
        # quality flag. (2_896 - (-11_166)) / 152_914 ~= 0.0919.
        mv = accruals(snapshot_esef)
        assert mv.ok
        assert mv.value == pytest.approx((2_896_000 - (-11_166_000)) / 152_914_000, rel=1e-6)
        assert mv.value == pytest.approx(0.0919, abs=1e-4)
