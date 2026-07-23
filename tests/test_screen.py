"""Integration test for the screen orchestrator, plus the name-change regression.

The individual pieces (profile, excludes, rank, cohorts) are unit-tested in
their own files. This drives run_screen end to end over a hand-built store so the
wiring is exercised: excludes drop the right names, sector exclusion works,
survivors land in the right cohort, and ranking orders them. No network.
"""

from __future__ import annotations

from datetime import date

from scout.config import Config
from scout.fundamentals.concepts import Concept
from scout.fundamentals.models import CanonicalFact, EntityRef, FundamentalsSnapshot
from scout.fundamentals.store import FundamentalsStore
from scout.screen.profile import ProfileStore, sec_profile_from_submissions
from scout.screen.screen import run_screen


def _fact(entity_id: str, concept: Concept, value: float, period_end: date) -> CanonicalFact:
    return CanonicalFact(
        entity_id=entity_id,
        concept=concept,
        value=value,
        currency="USD",
        period_end=period_end,
        period_start=None if concept.meta.period_type.value == "instant" else date(period_end.year, 1, 1),
        fiscal_year=period_end.year,
        fiscal_period="FY",
        accession=f"acc-{entity_id}",
        source_concept=f"us-gaap:{concept.value}",
        taxonomy="us-gaap",
    )


def _snapshot(entity_id: str, name: str, values: dict[Concept, float]) -> FundamentalsSnapshot:
    pe = date(2025, 12, 31)
    snap = FundamentalsSnapshot(
        entity=EntityRef("sec", entity_id, "cik", name=name, country="US"),
        period_end=pe,
        fiscal_year=2025,
        fiscal_period="FY",
        currency="USD",
        taxonomy="us-gaap",
        accession=f"acc-{entity_id}",
        filing_date=pe,
    )
    snap.facts = {c: _fact(entity_id, c, v, pe) for c, v in values.items()}
    return snap


def _healthy(entity_id: str, name: str, *, gp: float, cash: float) -> FundamentalsSnapshot:
    # Enough to compute GP/A, Altman, and a positive cash runway (CFO > 0).
    return _snapshot(
        entity_id, name,
        {
            Concept.REVENUE: 1000.0,
            Concept.COST_OF_REVENUE: 1000.0 - gp,
            Concept.TOTAL_ASSETS: 5000.0,
            Concept.CURRENT_ASSETS: 3000.0,
            Concept.CURRENT_LIABILITIES: 1000.0,
            Concept.RETAINED_EARNINGS: 2000.0,
            Concept.OPERATING_INCOME: 400.0,
            Concept.TOTAL_EQUITY: 3000.0,
            Concept.TOTAL_LIABILITIES: 2000.0,
            Concept.CASH_AND_EQUIVALENTS: cash,
            Concept.CASH_FROM_OPERATIONS: 300.0,  # positive -> runway inf, passes
            Concept.NET_INCOME: 350.0,
        },
    )


def _make_config(tmp_path) -> Config:  # type: ignore[no-untyped-def]
    return Config(user_agent="scout/0.1 test@example.com", data_dir=tmp_path)


def test_screen_ranks_healthy_excludes_distressed(tmp_path):
    config = _make_config(tmp_path)
    config.data_dir.mkdir(parents=True, exist_ok=True)

    with FundamentalsStore(config.db_path) as store:
        store.initialize()
        # Three healthy manufacturers -> a rankable cohort of 3.
        store.put_snapshot(_healthy("111", "Alpha Mfg", gp=600.0, cash=2000.0))
        store.put_snapshot(_healthy("222", "Beta Mfg", gp=300.0, cash=2000.0))
        store.put_snapshot(_healthy("333", "Gamma Mfg", gp=100.0, cash=2000.0))
        # A distressed name: burning cash, tiny runway -> hard exclude.
        distressed = _snapshot(
            "444", "Delta Distressed",
            {
                Concept.REVENUE: 100.0,
                Concept.TOTAL_ASSETS: 500.0,
                Concept.CASH_AND_EQUIVALENTS: 10.0,
                Concept.CASH_FROM_OPERATIONS: -240.0,  # burns 20/mo, ~0.5mo runway
                Concept.TOTAL_EQUITY: 100.0,
                Concept.TOTAL_LIABILITIES: 400.0,
            },
        )
        store.put_snapshot(distressed)

    # Profiles: three manufacturers, one financial (sector-excluded).
    with ProfileStore(config.db_path) as profiles:
        profiles.initialize()
        for eid, name in [("111", "Alpha Mfg"), ("222", "Beta Mfg"), ("333", "Gamma Mfg")]:
            profiles.put(_mfg_profile(eid, name))
        profiles.put(_finance_profile("444", "Delta Distressed"))

    result = run_screen(config, min_cohort=3)

    ranked_ids = [c.entity_id for c in result.ranked]
    excluded_ids = {e.entity_id for e in result.excluded}

    # Delta is a financial-sector name -> excluded before the cash test even runs.
    assert "444" in excluded_ids
    # The three manufacturers survive and are ranked as a cohort of 3.
    assert set(ranked_ids) == {"111", "222", "333"}
    # Highest gross profit -> highest quality -> ranked first (no price, so
    # ranking is quality+safety only; safety is identical, so GP/A decides).
    assert result.ranked[0].entity_id == "111"
    assert result.ranked[0].quality is not None
    # Blind spots are surfaced honestly.
    assert result.notes
    assert any("valuation" in n or "price" in n for n in result.notes)


def test_screen_runs_without_profiles(tmp_path):
    """The screen must work before `scout enrich` has ever run."""
    config = _make_config(tmp_path)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    with FundamentalsStore(config.db_path) as store:
        store.initialize()
        store.put_snapshot(_healthy("111", "Solo Co", gp=500.0, cash=1000.0))

    result = run_screen(config, min_cohort=1)
    # No profile DB -> everything falls to the 'unknown' sector cohort, still runs.
    assert result.universe_size == 1
    assert len(result.ranked) == 1


def test_current_name_in_former_names_is_not_a_change():
    """Regression: SEC lists the CURRENT name in formerNames with a rolling `to`
    date (Equifax). That must not read as a same-day name change and exclude the
    company."""
    doc = {
        "name": "EQUIFAX INC",
        "sic": "7320",
        "formerNames": [
            {"name": "EQUIFAX INC", "from": "1994-02-14T05:00:00.000Z",
             "to": "2026-07-22T04:00:00.000Z"},
        ],
        "filings": {"recent": {"form": ["10-K"], "filingDate": ["2026-07-22"]}},
    }
    profile = sec_profile_from_submissions(doc, "33185")
    assert profile.name_changed_within_months is None


def test_genuine_name_change_is_detected():
    doc = {
        "name": "NEWCO HOLDINGS",
        "sic": "3841",
        "formerNames": [
            {"name": "OLD SHELL CORP", "from": "2010-01-01T00:00:00.000Z",
             "to": "2026-05-01T00:00:00.000Z"},
        ],
        "filings": {"recent": {"form": ["8-K"], "filingDate": ["2026-07-01"]}},
    }
    profile = sec_profile_from_submissions(doc, "999")
    assert profile.name_changed_within_months is not None
    assert profile.name_changed_within_months <= 3


def _mfg_profile(entity_id: str, name: str):  # type: ignore[no-untyped-def]
    doc = {
        "name": name, "sic": "3841", "entityType": "operating",
        "tickers": [name[:3].upper()], "exchanges": ["NYSE"],
        "filings": {"recent": {"form": ["10-K"], "filingDate": ["2026-03-01"]}},
    }
    return sec_profile_from_submissions(doc, entity_id)


def _finance_profile(entity_id: str, name: str):  # type: ignore[no-untyped-def]
    doc = {
        "name": name, "sic": "6199", "entityType": "operating",
        "filings": {"recent": {"form": ["10-K"], "filingDate": ["2026-03-01"]}},
    }
    return sec_profile_from_submissions(doc, entity_id)
