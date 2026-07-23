"""`run_pick` end to end over a hand-built store.

Drives the deterministic path (no LLM, like a plain `scout pick`): screen the
universe, build each strategy's book, and append it to the ledger. Asserts the
wiring -- the right strategies are written, prices flow through to gradeable
picks, unpriced names are recorded ungradeable, and the GBDT is honestly skipped.
"""

from __future__ import annotations

from datetime import date

from scout.config import Config
from scout.fundamentals.concepts import Concept
from scout.fundamentals.models import CanonicalFact, EntityRef, FundamentalsSnapshot
from scout.fundamentals.store import FundamentalsStore
from scout.portfolio.ledger import Ledger
from scout.portfolio.models import Strategy
from scout.portfolio.pick import run_pick


def _fact(entity_id: str, concept: Concept, value: float, pe: date) -> CanonicalFact:
    return CanonicalFact(
        entity_id=entity_id,
        concept=concept,
        value=value,
        currency="USD",
        period_end=pe,
        period_start=None if concept.meta.period_type.value == "instant" else date(pe.year, 1, 1),
        fiscal_year=pe.year,
        fiscal_period="FY",
        accession=f"acc-{entity_id}",
        source_concept=f"us-gaap:{concept.value}",
        taxonomy="us-gaap",
    )


def _manufacturer(entity_id: str, name: str, *, gp: float) -> FundamentalsSnapshot:
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
    values = {
        Concept.REVENUE: 1000.0,
        Concept.COST_OF_REVENUE: 1000.0 - gp,
        Concept.TOTAL_ASSETS: 5000.0,
        Concept.CURRENT_ASSETS: 3000.0,
        Concept.CURRENT_LIABILITIES: 1000.0,
        Concept.RETAINED_EARNINGS: 2000.0,
        Concept.OPERATING_INCOME: 400.0,
        Concept.TOTAL_EQUITY: 3000.0,
        Concept.TOTAL_LIABILITIES: 2000.0,
        Concept.CASH_AND_EQUIVALENTS: 500.0,
        Concept.CASH_FROM_OPERATIONS: 300.0,
        Concept.NET_INCOME: 350.0,
        Concept.SHARES_OUTSTANDING: 100.0,  # lets market cap / EV/EBIT compute
        Concept.LONG_TERM_DEBT: 200.0,
    }
    snap.facts = {c: _fact(entity_id, c, v, pe) for c, v in values.items()}
    return snap


def _config(tmp_path) -> Config:  # type: ignore[no-untyped-def]
    return Config(user_agent="scout/0.1 test@example.com", data_dir=tmp_path)


def _seed(config: Config) -> None:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    with FundamentalsStore(config.db_path) as store:
        store.initialize()
        store.put_snapshot(_manufacturer("111", "Alpha Mfg", gp=600.0))
        store.put_snapshot(_manufacturer("222", "Beta Mfg", gp=300.0))
        store.put_snapshot(_manufacturer("333", "Gamma Mfg", gp=100.0))


def test_run_pick_writes_the_deterministic_books(tmp_path):
    config = _config(tmp_path)
    _seed(config)
    ledger = Ledger(tmp_path / "ledger.jsonl")

    batch = run_pick(
        config,
        as_of=date(2026, 7, 23),
        prices={"111": 50.0, "222": 50.0},  # 333 deliberately unpriced
        top=2,
        run_id="run-fixed",
        ledger=ledger,
    )

    assert batch.run_id == "run-fixed"
    assert batch.universe_size == 3

    picks = ledger.read()
    strategies = {p.strategy for p in picks}
    assert Strategy.UNIVERSE_EW in strategies
    assert Strategy.SCREEN in strategies
    # GBDT writes no picks (insufficient history) but is reported as a batch note.
    gbdt_batch = next(b for b in batch.strategies if b.strategy == Strategy.GBDT)
    assert "INSUFFICIENT" in gbdt_batch.note.upper() or "needs" in gbdt_batch.note

    # Universe = all three survivors, equal weight, and the unpriced name is
    # recorded ungradeable rather than dropped.
    universe = [p for p in picks if p.strategy == Strategy.UNIVERSE_EW]
    assert {p.entity_id for p in universe} == {"111", "222", "333"}
    assert all(p.weight == 1 / 3 for p in universe)
    unpriced = next(p for p in universe if p.entity_id == "333")
    assert unpriced.reference_price is None


def test_run_pick_ev_ebit_decile_populates_with_prices(tmp_path):
    config = _config(tmp_path)
    _seed(config)
    ledger = Ledger(tmp_path / "ledger.jsonl")

    run_pick(
        config,
        as_of=date(2026, 7, 23),
        prices={"111": 50.0, "222": 50.0, "333": 50.0},
        run_id="run-fixed",
        ledger=ledger,
    )
    ev = ledger.read_strategy(Strategy.EV_EBIT_DECILE)
    # With prices and one manufacturing cohort, the decile rounds up to the single
    # cheapest EV/EBIT name -- non-empty, and every chosen name is priced.
    assert ev
    assert all(p.reference_price is not None for p in ev)
    # Its features are finite (the inf cash-runway must have been filtered out).
    assert all("cash_runway_months" not in p.features for p in ev)


def test_run_pick_without_prices_notes_the_gap(tmp_path):
    config = _config(tmp_path)
    _seed(config)
    ledger = Ledger(tmp_path / "ledger.jsonl")

    batch = run_pick(config, as_of=date(2026, 7, 23), run_id="run-fixed", ledger=ledger)
    assert ledger.read_strategy(Strategy.EV_EBIT_DECILE) == []
    assert any("EV/EBIT" in n for n in batch.notes)
    # Every recorded pick is ungradeable without prices, but still recorded.
    assert all(p.reference_price is None for p in ledger.read())
