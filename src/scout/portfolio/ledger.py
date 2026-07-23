"""The paper-trade ledger: append-only picks as JSONL.

One pick per line, JSON. This is the old repo's `logger.js` pattern carried over
deliberately -- a JSONL append log is trivially parseable, survives a crash
mid-run (a torn last line loses one pick, not the file), and never rewrites
history, which is exactly the property a pre-registration record must have. If we
could edit past picks the whole point -- recording a bet before its outcome is
known -- would evaporate.

Two departures from the JS original, both from this project's rules:

  - **A failed write is NOT swallowed.** `logger.js` logged and continued because
    losing one signal never mattered. Here the ledger *is* the evidence, so a
    write that fails is raised, not warned-and-dropped -- silently losing a
    pre-registered pick would corrupt the very measurement this exists for.
  - **Non-finite floats never reach the file.** `Infinity`/`NaN` are not valid
    JSON; the models already strip them from `features`, and the writer asserts
    strict JSON so a stray one fails loud instead of writing a file that only
    Python's lenient reader can parse back.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

from scout.portfolio.models import PaperPick, Strategy

logger = logging.getLogger(__name__)


class LedgerError(RuntimeError):
    """A ledger line could not be read or written. Raised, never swallowed."""


class Ledger:
    """Append-only JSONL store of paper picks at `path`."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.exists()

    def append(self, picks: Iterable[PaperPick]) -> int:
        """Append picks, one JSON line each. Returns the number written.

        The whole batch is serialized in memory first, so a value that will not
        encode (a non-finite float that slipped past the model) raises before a
        single partial line is written -- an append either adds every pick or
        none, never half.
        """
        picks = list(picks)
        if not picks:
            return 0

        try:
            # allow_nan=False makes Infinity/NaN a hard error rather than emitting
            # the non-standard tokens Python's json writes by default.
            lines = [json.dumps(p.to_dict(), allow_nan=False) for p in picks]
        except ValueError as exc:
            raise LedgerError(
                f"a pick would not serialize to strict JSON ({exc}); this is a bug "
                "-- features should already be finite-filtered. Not writing the batch."
            ) from exc

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        except OSError as exc:
            raise LedgerError(f"could not write to ledger {self._path}: {exc}") from exc

        logger.info("appended %d pick(s) to %s", len(picks), self._path)
        return len(picks)

    def read(self) -> list[PaperPick]:
        """Every pick in the ledger, in file order.

        A malformed line is a data-integrity problem, not something to skip
        quietly -- the whole read fails with the offending line number so it can
        be fixed, rather than silently returning a truncated history that a score
        would then treat as complete.
        """
        if not self._path.exists():
            return []

        picks: list[PaperPick] = []
        try:
            with self._path.open(encoding="utf-8") as fh:
                for lineno, raw in enumerate(fh, start=1):
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        picks.append(PaperPick.from_dict(json.loads(line)))
                    except (ValueError, KeyError) as exc:
                        raise LedgerError(
                            f"{self._path} line {lineno} is not a valid pick: {exc}"
                        ) from exc
        except OSError as exc:
            raise LedgerError(f"could not read ledger {self._path}: {exc}") from exc
        return picks

    def read_strategy(self, strategy: Strategy) -> list[PaperPick]:
        """Just one strategy's picks -- convenience over `read`."""
        return [p for p in self.read() if p.strategy == strategy]

    def run_ids(self) -> list[str]:
        """Distinct `run_id`s present, oldest first -- each is one `scout pick`."""
        seen: dict[str, None] = {}
        for pick in self.read():
            if pick.run_id:
                seen.setdefault(pick.run_id, None)
        return list(seen)
