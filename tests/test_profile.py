"""Tests for scout.screen.profile.

Three concerns:

  * sec_profile_from_submissions maps a real-shaped submissions document into an
    EntityProfile correctly (sector, exclusion, former-name dates, the NT
    late-filing flag) and degrades to None on malformed input rather than
    raising.
  * ProfileStore round-trips every field -- including None and empty tuples --
    and upserts rather than duplicates.
  * enrich orchestrates the two paths (SEC fetch vs. minimal) and, critically,
    coexists in one DuckDB file with the fundamentals store without a lock
    error, because it reads the entity list read-only and closes that handle
    before opening the read-write ProfileStore.

The submissions fixtures are modelled on the real 3M shape (CIK 66740). A single
opt-in live test (SCOUT_LIVE_TESTS=1) confirms the fixtures still match reality.
"""

from __future__ import annotations

import os
from datetime import date

import httpx
import pytest
import respx

from scout.config import Config
from scout.data.http import HttpClient
from scout.fundamentals.models import EntityRef, FundamentalsSnapshot
from scout.fundamentals.store import FundamentalsStore
from scout.screen import profile
from scout.screen.models import EntityProfile, FormerName

# ---------------------------------------------------------------------------
# Submissions fixtures (3M-shaped: parallel filings.recent arrays, formerNames)
# ---------------------------------------------------------------------------

# Clean manufacturer: no late filings, one historical name change. sic 3841 is
# "Surgical & Medical Instruments" -> manufacturing, not excluded.
_MANUFACTURING = {
    "name": "3M CO",
    "sic": "3841",
    "sicDescription": "Surgical & Medical Instruments & Apparatus",
    "tickers": ["MMM"],
    "exchanges": ["NYSE"],
    "category": "Large accelerated filer",
    "entityType": "operating",
    "fiscalYearEnd": "1231",
    "stateOfIncorporation": "DE",
    "formerNames": [
        {
            "name": "MINNESOTA MINING & MANUFACTURING CO",
            "from": "1994-01-01T05:00:00.000Z",
            "to": "2002-06-30T05:00:00.000Z",
        }
    ],
    "filings": {
        "recent": {
            "form": ["10-Q", "8-K", "10-K", "4"],
            "filingDate": ["2024-07-25", "2024-05-01", "2024-02-07", "2024-01-15"],
            "reportDate": ["2024-06-30", "", "2023-12-31", ""],
            "items": ["", "2.02", "", ""],
        }
    },
}

# Delinquent filer: an NT 10-K (late-filing notice) precedes the eventual 10-K.
_LATE_FILER = {
    "name": "LATE WIDGETS INC",
    "sic": "3559",
    "sicDescription": "Special Industry Machinery",
    "tickers": ["LATE"],
    "exchanges": ["Nasdaq"],
    "category": "Smaller reporting company",
    "entityType": "operating",
    "fiscalYearEnd": "1231",
    "formerNames": [],
    "filings": {
        "recent": {
            "form": ["10-K", "NT 10-K", "8-K"],
            "filingDate": ["2024-03-29", "2024-03-15", "2023-11-01"],
            "reportDate": ["2023-12-31", "2023-12-31", ""],
            "items": ["", "", "2.02"],
        }
    },
}

# Finance company: sic 6199 -> finance -> excluded from the ranked universe.
_FINANCE = {
    "name": "SOME CAPITAL CORP",
    "sic": "6199",
    "sicDescription": "Finance Services",
    "tickers": ["SCC"],
    "exchanges": ["NYSE"],
    "category": "Accelerated filer",
    "entityType": "operating",
    "fiscalYearEnd": "1231",
    "formerNames": [],
    "filings": {
        "recent": {
            "form": ["10-K", "8-K"],
            "filingDate": ["2024-02-20", "2023-12-01"],
            "reportDate": ["2023-12-31", ""],
            "items": ["", ""],
        }
    },
}


# ---------------------------------------------------------------------------
# sec_profile_from_submissions
# ---------------------------------------------------------------------------


def test_clean_manufacturer_profile():
    p = profile.sec_profile_from_submissions(_MANUFACTURING, "0000066740")

    assert p.entity_id == "0000066740"
    assert p.source == "sec"
    assert p.country == "US"
    assert p.name == "3M CO"
    assert p.sic == "3841"
    assert p.sic_description == "Surgical & Medical Instruments & Apparatus"
    assert p.sector == "manufacturing"
    assert p.is_excluded_sector is False
    assert p.tickers == ("MMM",)
    assert p.exchanges == ("NYSE",)
    assert p.filer_category == "Large accelerated filer"
    assert p.entity_type == "operating"
    assert p.fiscal_year_end == "1231"
    # Newest-first arrays: index 0 is the most recent filing.
    assert p.most_recent_form == "10-Q"
    assert p.most_recent_filing_date == date(2024, 7, 25)
    # History present, no NT form -> a definite False, not None.
    assert p.has_recent_late_filing is False


def test_former_name_dates_are_parsed():
    p = profile.sec_profile_from_submissions(_MANUFACTURING, "0000066740")

    assert p.former_names == (
        FormerName(
            name="MINNESOTA MINING & MANUFACTURING CO",
            from_date=date(1994, 1, 1),
            to_date=date(2002, 6, 30),
        ),
    )
    # 2002-06-30 -> 2024-07-25 is 265 raw months, minus one because the day of
    # the "now" date (25) precedes the change day (30).
    assert p.name_changed_within_months == 264


def test_no_former_names_yields_none_name_change():
    p = profile.sec_profile_from_submissions(_FINANCE, "0000000001")
    assert p.former_names == ()
    assert p.name_changed_within_months is None


def test_nt_form_sets_recent_late_filing():
    p = profile.sec_profile_from_submissions(_LATE_FILER, "0000000002")
    assert p.has_recent_late_filing is True
    # sic 3559 is machinery -> manufacturing, still not excluded.
    assert p.sector == "manufacturing"
    assert p.is_excluded_sector is False


def test_finance_sic_is_excluded():
    p = profile.sec_profile_from_submissions(_FINANCE, "0000000001")
    assert p.sector == "finance"
    assert p.is_excluded_sector is True
    assert p.has_recent_late_filing is False


def test_empty_filings_history_yields_none_late_flag():
    doc = {"name": "NO FILINGS INC", "sic": "3841", "filings": {"recent": {}}}
    p = profile.sec_profile_from_submissions(doc, "0000000003")
    # "we could not check" is not the same as "there are none".
    assert p.has_recent_late_filing is None
    assert p.most_recent_form is None
    assert p.most_recent_filing_date is None


def test_stale_nt_outside_window_is_not_flagged():
    # An NT 10-K from 3+ years before the newest filing is history, not a
    # current delinquency signal.
    doc = {
        "name": "RECOVERED INC",
        "sic": "3841",
        "filings": {
            "recent": {
                "form": ["10-K", "NT 10-K"],
                "filingDate": ["2024-03-01", "2020-01-15"],
            }
        },
    }
    p = profile.sec_profile_from_submissions(doc, "0000000004")
    assert p.has_recent_late_filing is False


def test_malformed_document_degrades_without_raising():
    # Every field the wrong type; must not raise, must produce a valid profile.
    doc = {
        "name": "BROKEN CO",
        "sic": "not-a-number",
        "tickers": "MMM",  # should be a list
        "exchanges": None,
        "formerNames": "garbage",  # should be a list
        "filings": "nope",  # should be a dict
    }
    p = profile.sec_profile_from_submissions(doc, "0000000005")

    assert p.name == "BROKEN CO"
    assert p.sector == "unknown"
    assert p.is_excluded_sector is False
    assert p.tickers == ()
    assert p.exchanges == ()
    assert p.former_names == ()
    assert p.has_recent_late_filing is None
    assert p.name_changed_within_months is None
    assert p.most_recent_form is None


def test_non_dict_document_returns_minimal_sec_profile():
    p = profile.sec_profile_from_submissions("totally not a dict", "0000000006")
    assert p.entity_id == "0000000006"
    assert p.source == "sec"
    assert p.country == "US"
    assert p.name is None
    assert p.sector is None  # nothing to derive from -- a truly minimal profile


def test_minimal_profile_carries_identity_only():
    p = profile.minimal_profile("LEI-UA-1", "esef", "UA", "Ukr Agri Co")
    assert p.entity_id == "LEI-UA-1"
    assert p.source == "esef"
    assert p.country == "UA"
    assert p.name == "Ukr Agri Co"
    assert p.sic is None
    assert p.sector is None
    assert p.is_excluded_sector is False
    assert p.has_recent_late_filing is None


def test_submissions_url_zero_pads_cik():
    assert profile.submissions_url("66740") == "https://data.sec.gov/submissions/CIK0000066740.json"
    assert profile.submissions_url("0000066740") == (
        "https://data.sec.gov/submissions/CIK0000066740.json"
    )
    assert profile.submissions_url(320193) == "https://data.sec.gov/submissions/CIK0000320193.json"


# ---------------------------------------------------------------------------
# ProfileStore round-trip
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    with profile.ProfileStore(tmp_path / "scout.duckdb") as s:
        s.initialize()
        yield s


def _full_profile(**overrides) -> EntityProfile:
    args = dict(
        entity_id="0000066740",
        source="sec",
        name="3M CO",
        country="US",
        sic="3841",
        sic_description="Surgical & Medical Instruments & Apparatus",
        sector="manufacturing",
        is_excluded_sector=False,
        tickers=("MMM",),
        exchanges=("NYSE",),
        filer_category="Large accelerated filer",
        entity_type="operating",
        former_names=(
            FormerName(
                name="MINNESOTA MINING & MANUFACTURING CO",
                from_date=date(1994, 1, 1),
                to_date=date(2002, 6, 30),
            ),
        ),
        fiscal_year_end="1231",
        has_recent_late_filing=False,
        name_changed_within_months=264,
        most_recent_form="10-Q",
        most_recent_filing_date=date(2024, 7, 25),
    )
    args.update(overrides)
    return EntityProfile(**args)


def test_initialize_is_idempotent(tmp_path):
    with profile.ProfileStore(tmp_path / "scout.duckdb") as s:
        s.initialize()
        s.initialize()  # must not raise


def test_put_and_get_roundtrip(store):
    p = _full_profile()
    store.put(p)
    got = store.get("0000066740")
    assert got == p


def test_none_and_empty_fields_survive(store):
    # A minimal non-SEC profile: empty tuples, all-None derived fields.
    p = profile.minimal_profile("LEI-UA-1", "esef", "UA", "Ukr Agri Co")
    store.put(p)
    got = store.get("LEI-UA-1")

    assert got == p
    assert got.tickers == ()
    assert got.exchanges == ()
    assert got.former_names == ()
    assert got.sic is None
    assert got.has_recent_late_filing is None
    assert got.name_changed_within_months is None
    assert got.most_recent_filing_date is None


def test_get_missing_returns_none(store):
    assert store.get("nope") is None


def test_has_and_count(store):
    assert store.has("0000066740") is False
    assert store.count() == 0

    store.put(_full_profile())

    assert store.has("0000066740") is True
    assert store.count() == 1


def test_put_upserts_not_duplicates(store):
    store.put(_full_profile(name="OLD NAME", most_recent_form="10-K"))
    store.put(_full_profile(name="3M CO", most_recent_form="10-Q"))

    assert store.count() == 1
    got = store.get("0000066740")
    assert got.name == "3M CO"
    assert got.most_recent_form == "10-Q"


def test_all_returns_every_profile_sorted(store):
    store.put(_full_profile(entity_id="0000000002"))
    store.put(_full_profile(entity_id="0000000001"))

    ids = [p.entity_id for p in store.all()]
    assert ids == ["0000000001", "0000000002"]


def test_has_recent_late_filing_true_roundtrips(store):
    # The tri-state flag: make sure True (not just None/False) survives.
    p = _full_profile(entity_id="0000000009", has_recent_late_filing=True)
    store.put(p)
    assert store.get("0000000009").has_recent_late_filing is True


# ---------------------------------------------------------------------------
# enrich orchestration (+ DuckDB coexistence with the fundamentals store)
# ---------------------------------------------------------------------------


def _seed_fundamentals(db_path, entities):
    """entities: iterable of (entity_id, source, scheme, country, name)."""
    with FundamentalsStore(db_path) as s:
        s.initialize()
        for entity_id, source, scheme, country, name in entities:
            s.put_snapshot(
                FundamentalsSnapshot(
                    entity=EntityRef(
                        source=source,
                        entity_id=entity_id,
                        identifier_scheme=scheme,
                        name=name,
                        country=country,
                    ),
                    period_end=date(2023, 12, 31),
                    fiscal_year=2023,
                    fiscal_period="FY",
                    currency="USD",
                    taxonomy="us-gaap",
                    accession=f"{entity_id}-acc",
                    filing_date=date(2024, 2, 1),
                    facts={},
                    warnings=[],
                )
            )


def _config(tmp_path) -> Config:
    return Config(user_agent="scout-test/0.1 test@example.com", data_dir=tmp_path)


async def test_enrich_stores_sec_and_minimal(tmp_path):
    config = _config(tmp_path)
    _seed_fundamentals(
        config.db_path,
        [
            ("0000066740", "sec", "cik", "US", "3M CO"),
            ("LEI-UA-1", "esef", "edrpou", "UA", "Ukr Agri Co"),
        ],
    )

    async with respx.mock:
        respx.get(profile.submissions_url("0000066740")).mock(
            return_value=httpx.Response(200, json=_MANUFACTURING)
        )
        result = await profile.enrich(config)

    assert result.enriched == 1
    assert result.minimal == 1
    assert result.failed == 0
    assert result.errors == []

    # Read back through a read-only store -- proves both stores coexist in the
    # one file and the writer released its lock cleanly.
    with profile.ProfileStore(config.db_path, read_only=True) as store:
        sec_p = store.get("0000066740")
        ua_p = store.get("LEI-UA-1")

    assert sec_p.sector == "manufacturing"
    assert sec_p.name == "3M CO"
    assert ua_p.source == "esef"
    assert ua_p.country == "UA"
    assert ua_p.name == "Ukr Agri Co"
    assert ua_p.sic is None


async def test_enrich_and_fundamentals_coexist_in_one_file(tmp_path):
    # The explicit coexistence check: after enrich has written profiles, the
    # fundamentals data must still be readable from the same file.
    config = _config(tmp_path)
    _seed_fundamentals(config.db_path, [("LEI-UA-1", "esef", "edrpou", "UA", "Ukr Agri Co")])

    result = await profile.enrich(config)  # no HTTP: the only entity is non-SEC
    assert result.minimal == 1

    with FundamentalsStore(config.db_path, read_only=True) as funds:
        assert funds.entity_count() == 1
    with profile.ProfileStore(config.db_path, read_only=True) as profiles:
        assert profiles.count() == 1


async def test_enrich_skips_existing_unless_reenrich(tmp_path):
    config = _config(tmp_path)
    _seed_fundamentals(config.db_path, [("0000066740", "sec", "cik", "US", "3M CO")])

    async with respx.mock:
        respx.get(profile.submissions_url("0000066740")).mock(
            return_value=httpx.Response(200, json=_MANUFACTURING)
        )

        first = await profile.enrich(config)
        assert first.enriched == 1

        second = await profile.enrich(config)
        assert second.enriched == 0
        assert second.skipped_existing == 1

        third = await profile.enrich(config, reenrich=True)
        assert third.enriched == 1
        assert third.skipped_existing == 0


async def test_enrich_respects_limit(tmp_path):
    config = _config(tmp_path)
    _seed_fundamentals(
        config.db_path,
        [
            ("0000000001", "sec", "cik", "US", "Alpha"),
            ("0000000002", "sec", "cik", "US", "Beta"),
        ],
    )

    async with respx.mock:
        # Only the first entity (ordered by id) should be fetched.
        respx.get(profile.submissions_url("0000000001")).mock(
            return_value=httpx.Response(200, json=_MANUFACTURING)
        )
        result = await profile.enrich(config, limit=1)

    assert result.enriched == 1
    with profile.ProfileStore(config.db_path, read_only=True) as store:
        assert store.count() == 1
        assert store.has("0000000001") is True
        assert store.has("0000000002") is False


async def test_enrich_collects_per_entity_errors(tmp_path):
    config = _config(tmp_path)
    _seed_fundamentals(
        config.db_path,
        [
            ("0000000001", "sec", "cik", "US", "Good Co"),
            ("0000000404", "sec", "cik", "US", "Gone Co"),
        ],
    )

    async with respx.mock:
        respx.get(profile.submissions_url("0000000001")).mock(
            return_value=httpx.Response(200, json=_MANUFACTURING)
        )
        respx.get(profile.submissions_url("0000000404")).mock(
            return_value=httpx.Response(404)
        )
        result = await profile.enrich(config)

    # One 404 does not abort the run; the good filer still gets a profile.
    assert result.enriched == 1
    assert result.failed == 1
    assert len(result.errors) == 1
    assert result.errors[0][0] == "0000000404"

    with profile.ProfileStore(config.db_path, read_only=True) as store:
        assert store.has("0000000001") is True
        assert store.has("0000000404") is False


async def test_enrich_empty_store_is_a_noop(tmp_path):
    config = _config(tmp_path)  # no fundamentals db at all
    result = await profile.enrich(config)
    assert result == profile.EnrichResult()


# ---------------------------------------------------------------------------
# Live SEC test -- opt-in via SCOUT_LIVE_TESTS to keep CI off the network.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("SCOUT_LIVE_TESTS"),
    reason="set SCOUT_LIVE_TESTS=1 to hit the real SEC submissions API",
)
async def test_live_3m_submissions():
    async with HttpClient(user_agent="scout-live-test/0.1 azantzos@gmail.com") as http:
        doc = await profile.fetch_submissions(http, "0000066740")
    p = profile.sec_profile_from_submissions(doc, "0000066740")

    assert "3M" in (p.name or "")
    assert p.sic == "3841"
    assert p.sector == "manufacturing"
    assert p.is_excluded_sector is False
    assert "NYSE" in p.exchanges
    assert p.former_names  # 3M was Minnesota Mining & Manufacturing
