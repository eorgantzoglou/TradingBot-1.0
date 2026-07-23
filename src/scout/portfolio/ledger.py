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
import os
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
            # the non-standard tokens Python's json writes by default. A non-finite
            # value here means one slipped past validation -- features are
            # finite-filtered and prices are validated at the CLI, so name the pick
            # rather than assuming which field is at fault.
            lines = [json.dumps(p.to_dict(), allow_nan=False) for p in picks]
        except ValueError as exc:
            culprit = next((p.pick_id for p in picks if not _serializable(p)), "unknown")
            raise LedgerError(
                f"pick {culprit!r} has a non-finite value and will not serialize to strict "
                f"JSON ({exc}). Not writing the batch -- check its reference_price, score, "
                "weight and features."
            ) from exc

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
                # Flush and fsync so an append is durable on return: the ledger is
                # a pre-registration record, and a pick that "succeeded" but sat in
                # an OS buffer through a crash would be a silently lost bet.
                fh.flush()
                os.fsync(fh.fileno())
        except OSError as exc:
            raise LedgerError(f"could not write to ledger {self._path}: {exc}") from exc

        logger.info("appended %d pick(s) to %s", len(picks), self._path)
        return len(picks)

    def read(self) -> list[PaperPick]:
        """Every pick in the ledger, in file order.

        A malformed line in the MIDDLE is a data-integrity problem -- the read
        fails with the offending line number rather than silently returning a
        truncated history a score would treat as complete. But a torn FINAL line
        (the tail of an append interrupted by a crash or power loss) is expected
        and recoverable: it is skipped with a warning, so one lost pick never
        makes the whole ledger unreadable. That is the crash-tolerance the JSONL
        format was chosen for.
        """
        if not self._path.exists():
            return []

        try:
            with self._path.open(encoding="utf-8") as fh:
                content = [(n, raw.strip()) for n, raw in enumerate(fh, start=1) if raw.strip()]
        except OSError as exc:
            raise LedgerError(f"could not read ledger {self._path}: {exc}") from exc

        picks: list[PaperPick] = []
        for index, (lineno, line) in enumerate(content):
            try:
                picks.append(PaperPick.from_dict(json.loads(line)))
            except (ValueError, KeyError) as exc:
                is_last = index == len(content) - 1
                if is_last:
                    logger.warning(
                        "%s line %d looks like a torn final line (interrupted append); "
                        "skipping it: %s",
                        self._path, lineno, exc,
                    )
                    break
                raise LedgerError(
                    f"{self._path} line {lineno} is not a valid pick: {exc}"
                ) from exc
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


def _serializable(pick: PaperPick) -> bool:
    """Whether one pick encodes to strict JSON -- used only to name the culprit
    when a batch write fails, so the error can point at the offending pick."""
    try:
        json.dumps(pick.to_dict(), allow_nan=False)
        return True
    except ValueError:
        return False
