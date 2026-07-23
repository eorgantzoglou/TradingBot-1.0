"""The append-only paper-trade ledger.

Checks the properties a pre-registration record must have: appends never rewrite
history, a round trip is lossless, non-finite values cannot reach the file, and a
corrupt line fails loud rather than silently truncating the history a later score
would trust.
"""

from __future__ import annotations

from datetime import date

import pytest

from scout.portfolio.ledger import Ledger, LedgerError
from scout.portfolio.models import PaperPick, Strategy


def _pick(entity_id: str, *, strategy: Strategy = Strategy.SCREEN, **kw) -> PaperPick:
    base = dict(
        as_of=date(2026, 7, 23),
        strategy=strategy,
        entity_id=entity_id,
        name=f"Co {entity_id}",
        cohort="US / US-GAAP / manufacturing",
        reference_price=100.0,
        currency=None,
        weight=0.5,
        rank=1,
        score=0.3,
        vetoed=None,
        features={"roic": 0.15},
        run_id="run-1",
    )
    base.update(kw)
    return PaperPick(**base)  # type: ignore[arg-type]


def test_append_then_read_round_trips(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    picks = [_pick("111"), _pick("222", reference_price=None)]
    assert ledger.append(picks) == 2

    read = ledger.read()
    assert [p.entity_id for p in read] == ["111", "222"]
    assert read[0].reference_price == 100.0
    assert read[1].reference_price is None  # None survives, not coerced to 0
    assert read[0].features == {"roic": 0.15}
    assert read[0].strategy == Strategy.SCREEN


def test_append_is_additive_never_overwrites(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append([_pick("111")])
    ledger.append([_pick("222")])
    # The second append must not have rewritten the first -- pre-registration
    # history is immutable.
    assert [p.entity_id for p in ledger.read()] == ["111", "222"]


def test_missing_ledger_reads_empty(tmp_path):
    assert Ledger(tmp_path / "nope.jsonl").read() == []
    assert not Ledger(tmp_path / "nope.jsonl").exists()


def test_non_finite_feature_fails_loud_and_writes_nothing(tmp_path):
    """A stray inf/nan must never be written as the non-standard JSON tokens
    Python emits by default -- the whole batch is rejected instead."""
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(path)
    bad = _pick("111", features={"cash_runway_months": float("inf")})
    with pytest.raises(LedgerError):
        ledger.append([_pick("000"), bad])
    # Atomic: nothing written when any pick in the batch is unserializable.
    assert not path.exists()


def test_corrupt_middle_line_raises_with_line_number(tmp_path):
    """A malformed line that is NOT the last is a real integrity problem -- the
    read must fail loudly rather than silently truncate the history."""
    path = tmp_path / "ledger.jsonl"
    Ledger(path).append([_pick("111")])
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not valid json}\n")
    Ledger(path).append([_pick("222")])  # a valid line AFTER the corrupt one
    with pytest.raises(LedgerError, match="line 2"):
        Ledger(path).read()


def test_torn_final_line_is_tolerated(tmp_path):
    """A torn LAST line (an append interrupted by a crash) loses one pick, not the
    whole file -- the recoverable case the JSONL format was chosen for."""
    path = tmp_path / "ledger.jsonl"
    Ledger(path).append([_pick("111"), _pick("222")])
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"as_of": "2026-07-23", "strateg')  # torn mid-write, no newline
    read = Ledger(path).read()
    assert [p.entity_id for p in read] == ["111", "222"]  # the two good picks survive


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "ledger.jsonl"
    Ledger(path).append([_pick("111")])
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n   \n")
    assert len(Ledger(path).read()) == 1


def test_read_strategy_and_run_ids(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append([_pick("111", strategy=Strategy.SCREEN, run_id="run-1")])
    ledger.append([_pick("222", strategy=Strategy.UNIVERSE_EW, run_id="run-2")])
    assert [p.entity_id for p in ledger.read_strategy(Strategy.UNIVERSE_EW)] == ["222"]
    assert ledger.run_ids() == ["run-1", "run-2"]


def test_pick_id_is_stable_and_unique_per_day_strategy_entity():
    pick = _pick("111", strategy=Strategy.AGENT)
    assert pick.pick_id == "2026-07-23:agent:111"
