"""Tests for the DuckDB fundamentals store.

The load-bearing property here is idempotency: re-ingesting the same filing
(raw facts) or re-running normalization (a snapshot) must leave the DB
looking exactly as if it had only ever happened once. That's exercised
explicitly, not just implied by the round-trip tests.
"""

from __future__ import annotations

from datetime import date

import pytest

from scout.fundamentals.concepts import Concept, PeriodType
from scout.fundamentals.models import CanonicalFact, EntityRef, FundamentalsSnapshot, RawFact
from scout.fundamentals.store import FundamentalsStore


@pytest.fixture
def store(tmp_path):
    with FundamentalsStore(tmp_path / "scout.duckdb") as s:
        s.initialize()
        yield s


def make_raw_fact(**overrides) -> RawFact:
    args = dict(
        accession="0001-acc",
        taxonomy="us-gaap",
        local_name="Revenues",
        value=1000.0,
        unit="USD",
        period_type=PeriodType.DURATION,
        period_start=date(2023, 1, 1),
        period_end=date(2023, 12, 31),
        is_dimensioned=False,
        decimals=-3,
        fiscal_year=2023,
        fiscal_period="FY",
    )
    args.update(overrides)
    return RawFact(**args)


def make_entity(**overrides) -> EntityRef:
    args = dict(source="sec", entity_id="0000320193", identifier_scheme="cik", name="Acme Inc", country="US")
    args.update(overrides)
    return EntityRef(**args)


def make_canonical_fact(**overrides) -> CanonicalFact:
    args = dict(
        entity_id="0000320193",
        concept=Concept.REVENUE,
        value=1000.0,
        currency="USD",
        period_end=date(2023, 12, 31),
        period_start=date(2023, 1, 1),
        fiscal_year=2023,
        fiscal_period="FY",
        accession="0001-acc",
        source_concept="us-gaap:Revenues",
        taxonomy="us-gaap",
    )
    args.update(overrides)
    return CanonicalFact(**args)


def make_snapshot(**overrides) -> FundamentalsSnapshot:
    entity = overrides.get("entity", make_entity())
    facts = overrides.pop("facts", None)
    if facts is None:
        cf = make_canonical_fact(entity_id=entity.entity_id)
        facts = {cf.concept: cf}
    args = dict(
        entity=entity,
        period_end=date(2023, 12, 31),
        fiscal_year=2023,
        fiscal_period="FY",
        currency="USD",
        taxonomy="us-gaap",
        accession="0001-acc",
        filing_date=date(2024, 2, 1),
        facts=facts,
        warnings=["fell back to Revenues from SalesRevenueNet"],
    )
    args.update(overrides)
    return FundamentalsSnapshot(**args)


# --------------------------------------------------------------------------
# initialize
# --------------------------------------------------------------------------


def test_initialize_is_idempotent(tmp_path):
    with FundamentalsStore(tmp_path / "scout.duckdb") as s:
        s.initialize()
        s.initialize()  # must not raise


# --------------------------------------------------------------------------
# raw facts
# --------------------------------------------------------------------------


def test_put_and_get_raw_facts_roundtrip(store):
    instant = make_raw_fact(
        local_name="CashAndCashEquivalentsAtCarryingValue",
        unit="USD",
        period_type=PeriodType.INSTANT,
        period_start=None,
        is_dimensioned=True,
        decimals=None,
    )
    duration = make_raw_fact()

    count = store.put_raw_facts([instant, duration])

    assert count == 2
    facts = store.raw_facts_for("0001-acc")
    assert len(facts) == 2

    by_name = {f.local_name: f for f in facts}
    got_instant = by_name["CashAndCashEquivalentsAtCarryingValue"]
    assert got_instant.period_start is None
    assert got_instant.is_dimensioned is True
    assert got_instant.decimals is None
    assert got_instant.period_type is PeriodType.INSTANT

    got_duration = by_name["Revenues"]
    assert got_duration == duration


def test_put_raw_facts_empty_list_is_a_noop(store):
    assert store.put_raw_facts([]) == 0
    assert store.raw_facts_for("nothing") == []


def test_has_accession(store):
    assert store.has_accession("0001-acc") is False
    store.put_raw_facts([make_raw_fact()])
    assert store.has_accession("0001-acc") is True


def test_reingesting_accession_replaces_not_duplicates(store):
    store.put_raw_facts([make_raw_fact(value=1000.0), make_raw_fact(local_name="CostOfRevenue", value=600.0)])

    # Re-ingest with a different fact set for the same accession, as would
    # happen if the parser were re-run over the same filing.
    store.put_raw_facts([make_raw_fact(value=1234.0)])

    facts = store.raw_facts_for("0001-acc")
    assert len(facts) == 1
    assert facts[0].value == 1234.0
    assert store.has_accession("0001-acc") is True


def test_reingesting_accession_does_not_touch_other_accessions(store):
    store.put_raw_facts([make_raw_fact(accession="acc-a")])
    store.put_raw_facts([make_raw_fact(accession="acc-b")])

    store.put_raw_facts([make_raw_fact(accession="acc-a", value=999.0)])

    assert len(store.raw_facts_for("acc-a")) == 1
    assert len(store.raw_facts_for("acc-b")) == 1


# --------------------------------------------------------------------------
# snapshots / canonical facts
# --------------------------------------------------------------------------


def test_put_and_get_snapshot_roundtrip(store):
    snapshot = make_snapshot()

    store.put_snapshot(snapshot)
    got = store.get_snapshot("0000320193", date(2023, 12, 31), "FY")

    assert got is not None
    assert got.entity == snapshot.entity
    assert got.period_end == snapshot.period_end
    assert got.fiscal_year == snapshot.fiscal_year
    assert got.fiscal_period == snapshot.fiscal_period
    assert got.currency == snapshot.currency
    assert got.taxonomy == snapshot.taxonomy
    assert got.accession == snapshot.accession
    assert got.filing_date == snapshot.filing_date
    assert got.warnings == snapshot.warnings
    assert got.facts == snapshot.facts


def test_snapshot_with_none_fiscal_period_roundtrips(store):
    snapshot = make_snapshot(fiscal_period=None, facts={})

    store.put_snapshot(snapshot)
    got = store.get_snapshot("0000320193", date(2023, 12, 31), None)

    assert got is not None
    assert got.fiscal_period is None
    assert got.facts == {}


def test_get_snapshot_missing_returns_none(store):
    assert store.get_snapshot("nope", date(2023, 12, 31), "FY") is None


def test_concept_and_none_currency_roundtrip(store):
    cf = make_canonical_fact(concept=Concept.SHARES_OUTSTANDING, currency=None, value=42.0)
    snapshot = make_snapshot(facts={cf.concept: cf})

    store.put_snapshot(snapshot)
    got = store.get_snapshot("0000320193", date(2023, 12, 31), "FY")

    fact = got.facts[Concept.SHARES_OUTSTANDING]
    assert fact.concept is Concept.SHARES_OUTSTANDING
    assert fact.currency is None
    assert fact.value == 42.0


def test_snapshots_for_entity_newest_period_first(store):
    store.put_snapshot(make_snapshot(period_end=date(2022, 12, 31), fiscal_year=2022))
    store.put_snapshot(make_snapshot(period_end=date(2023, 12, 31), fiscal_year=2023))
    store.put_snapshot(make_snapshot(period_end=date(2021, 12, 31), fiscal_year=2021))

    snapshots = store.snapshots_for_entity("0000320193")

    assert [s.period_end for s in snapshots] == [
        date(2023, 12, 31),
        date(2022, 12, 31),
        date(2021, 12, 31),
    ]


def test_latest_snapshot_returns_newest(store):
    store.put_snapshot(make_snapshot(period_end=date(2022, 12, 31), fiscal_year=2022))
    store.put_snapshot(make_snapshot(period_end=date(2023, 12, 31), fiscal_year=2023))

    latest = store.latest_snapshot("0000320193")

    assert latest.period_end == date(2023, 12, 31)


def test_latest_snapshot_none_when_entity_absent(store):
    assert store.latest_snapshot("nope") is None


def test_put_snapshot_upserts_same_key(store):
    cf = make_canonical_fact(value=1000.0)
    store.put_snapshot(make_snapshot(facts={cf.concept: cf}, currency="USD"))

    updated_cf = make_canonical_fact(value=2000.0)
    store.put_snapshot(make_snapshot(facts={updated_cf.concept: updated_cf}, currency="EUR"))

    snapshots = store.snapshots_for_entity("0000320193")
    assert len(snapshots) == 1
    assert snapshots[0].currency == "EUR"
    assert snapshots[0].facts[Concept.REVENUE].value == 2000.0


def test_put_snapshot_upsert_drops_concepts_no_longer_present(store):
    revenue = make_canonical_fact(concept=Concept.REVENUE)
    net_income = make_canonical_fact(concept=Concept.NET_INCOME, value=500.0)
    store.put_snapshot(make_snapshot(facts={revenue.concept: revenue, net_income.concept: net_income}))

    # Re-run normalization producing only revenue this time.
    store.put_snapshot(make_snapshot(facts={revenue.concept: revenue}))

    got = store.get_snapshot("0000320193", date(2023, 12, 31), "FY")
    assert set(got.facts) == {Concept.REVENUE}


# --------------------------------------------------------------------------
# coverage / counts
# --------------------------------------------------------------------------


def test_entity_and_snapshot_counts(store):
    assert store.entity_count() == 0
    assert store.snapshot_count() == 0

    store.put_snapshot(make_snapshot(entity=make_entity(entity_id="A"), period_end=date(2023, 12, 31)))
    store.put_snapshot(make_snapshot(entity=make_entity(entity_id="A"), period_end=date(2022, 12, 31)))
    store.put_snapshot(make_snapshot(entity=make_entity(entity_id="B"), period_end=date(2023, 12, 31)))

    assert store.entity_count() == 2
    assert store.snapshot_count() == 3


def test_coverage_reports_per_taxonomy_rollup(store):
    revenue = make_canonical_fact(entity_id="A", concept=Concept.REVENUE, taxonomy="us-gaap")
    net_income = make_canonical_fact(entity_id="A", concept=Concept.NET_INCOME, taxonomy="us-gaap")
    store.put_snapshot(
        make_snapshot(
            entity=make_entity(entity_id="A"),
            taxonomy="us-gaap",
            facts={revenue.concept: revenue, net_income.concept: net_income},
        )
    )

    ifrs_fact = make_canonical_fact(
        entity_id="B", concept=Concept.REVENUE, taxonomy="ifrs-full", source_concept="ifrs-full:Revenue"
    )
    store.put_snapshot(
        make_snapshot(
            entity=make_entity(entity_id="B"),
            taxonomy="ifrs-full",
            facts={ifrs_fact.concept: ifrs_fact},
        )
    )

    coverage = {row["taxonomy"]: row for row in store.coverage()}

    assert coverage["us-gaap"]["entities"] == 1
    assert coverage["us-gaap"]["snapshots"] == 1
    assert coverage["us-gaap"]["concepts"] == 2
    assert coverage["ifrs-full"]["entities"] == 1
    assert coverage["ifrs-full"]["concepts"] == 1


# --------------------------------------------------------------------------
# read-only
# --------------------------------------------------------------------------


def test_read_only_store_can_read_after_writer_closes(tmp_path):
    db_path = tmp_path / "scout.duckdb"
    with FundamentalsStore(db_path) as writer:
        writer.initialize()
        writer.put_snapshot(make_snapshot())

    with FundamentalsStore(db_path, read_only=True) as reader:
        got = reader.latest_snapshot("0000320193")

    assert got is not None
    assert got.facts[Concept.REVENUE].value == 1000.0
