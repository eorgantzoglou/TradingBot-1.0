"""Tests for scout.fundamentals.ingest -- archive -> DuckDB orchestration.

These run the REAL ingester over REAL archived filings (one SEC 10-Q, one ESEF
xBRL-JSON) copied into a tmp archive, writing to a tmp DuckDB. Nothing is mocked:
the point of this layer is exactly the archive->parse->normalize->store round
trip, and a fabricated fixture is what shipped the SEC bug this project learned
from. Guarded by skipif so a checkout without the archive still runs, but on a
machine that has it they MUST pass.

The 13MB SEC submission is slow to parse, so the shared `ingested` fixture builds
the archive and runs one ingest once per module; tests that must run ingest a
second time (idempotency, reingest) build their own small archive so they never
disturb the shared DB.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest

from scout.config import Config
from scout.fundamentals.concepts import Concept
from scout.fundamentals.ingest import ingest
from scout.fundamentals.store import FundamentalsStore

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REAL_ARCHIVE = _REPO_ROOT / "data" / "archive"
_MANIFEST = _REAL_ARCHIVE / "manifest.jsonl"

# One SEC filing (3M 10-Q, us-gaap) and one ESEF filing (Ukrainian issuer,
# ifrs-full). Both are known to parse and normalize to a snapshot.
_SEC_DOC = "0000066740-26-000246"
_ESEF_DOC = "25825"
_SEC_ENTITY = "66740"

_needs_archive = pytest.mark.skipif(
    not (_REAL_ARCHIVE / "sec" / "2026" / "07" / "21" / _SEC_DOC / "v1" / f"{_SEC_DOC}.txt").exists()
    or not (_REAL_ARCHIVE / "esef" / "2026" / "07" / "21" / _ESEF_DOC / "v1" / f"{_ESEF_DOC}.json").exists()
    or not _MANIFEST.exists(),
    reason="real SEC+ESEF archive not present",
)


def _build_archive(data_dir: Path, doc_ids: set[str]) -> Config:
    """Copy the given archived documents (payload + manifest lines) into
    `data_dir/archive`, preserving their relative paths, and return a Config that
    reads that archive and writes a DuckDB under the same tmp `data_dir`.

    Copying the real filings rather than pointing at the shared archive keeps each
    test's DB isolated (db_path derives from data_dir) while still exercising the
    genuine bytes.
    """
    dest_archive = data_dir / "archive"
    dest_archive.mkdir(parents=True, exist_ok=True)

    kept_lines: list[str] = []
    for line in _MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if record["doc_id"] not in doc_ids:
            continue
        kept_lines.append(line)
        src = _REAL_ARCHIVE / record["path"]
        dst = dest_archive / record["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    (dest_archive / "manifest.jsonl").write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
    return Config(data_dir=data_dir, concurrency=2)


@pytest.fixture(scope="module")
def ingested(tmp_path_factory):
    """Build a two-document archive and run one fresh ingest, shared across the
    read-only tests. Sync so the 13MB SEC parse happens exactly once for the
    module, independent of any per-test event loop."""
    data_dir = tmp_path_factory.mktemp("ingest_shared")
    config = _build_archive(data_dir, {_SEC_DOC, _ESEF_DOC})
    result = asyncio.run(ingest(config))
    return config, result


@_needs_archive
class TestFreshIngest:
    def test_produces_snapshots_with_no_failures(self, ingested) -> None:
        _config, result = ingested
        assert result.considered == 2
        assert result.parsed == result.considered
        assert result.failed == 0
        assert result.snapshots > 0
        assert result.errors == []

    def test_snapshot_count_matches_result(self, ingested) -> None:
        config, result = ingested
        with FundamentalsStore(config.db_path, read_only=True) as store:
            assert store.snapshot_count() == result.snapshots

    def test_store_reconstructs_a_snapshot_with_its_facts(self, ingested) -> None:
        # Regression guard for the fiscal-key join bug: the snapshot header's
        # fiscal_period must match its canonical_facts' fiscal_period, or the
        # reconstructed snapshot comes back with an empty facts dict.
        config, _result = ingested
        with FundamentalsStore(config.db_path, read_only=True) as store:
            snapshot = store.latest_snapshot(_SEC_ENTITY)
            assert snapshot is not None
            assert snapshot.facts, "reconstructed snapshot has no facts -- fiscal-key join broke"
            # The join must round-trip a real value, not just non-empty rows.
            assert snapshot.get(Concept.REVENUE) == 12_530_000_000


@_needs_archive
class TestIdempotency:
    async def test_second_ingest_skips_everything_and_writes_nothing(self, tmp_path) -> None:
        config = _build_archive(tmp_path, {_SEC_DOC, _ESEF_DOC})

        first = await ingest(config)
        with FundamentalsStore(config.db_path, read_only=True) as store:
            count_after_first = store.snapshot_count()

        second = await ingest(config)
        with FundamentalsStore(config.db_path, read_only=True) as store:
            count_after_second = store.snapshot_count()

        assert second.skipped_existing == first.considered
        assert second.snapshots == 0
        assert second.parsed == 0
        assert count_after_second == count_after_first


@_needs_archive
class TestReingest:
    async def test_reingest_reparses_and_replaces_not_duplicates(self, tmp_path) -> None:
        config = _build_archive(tmp_path, {_SEC_DOC, _ESEF_DOC})

        first = await ingest(config)
        with FundamentalsStore(config.db_path, read_only=True) as store:
            count_after_first = store.snapshot_count()

        again = await ingest(config, reingest=True)
        with FundamentalsStore(config.db_path, read_only=True) as store:
            count_after_reingest = store.snapshot_count()

        assert again.skipped_existing == 0
        assert again.parsed == first.considered
        # Replace, not duplicate: the snapshot count is unchanged.
        assert count_after_reingest == count_after_first


@_needs_archive
class TestLimit:
    async def test_limit_caps_documents_ingested(self, tmp_path) -> None:
        config = _build_archive(tmp_path, {_SEC_DOC, _ESEF_DOC})

        result = await ingest(config, limit=1)

        assert result.considered == 1
        assert result.parsed == 1
        with FundamentalsStore(config.db_path, read_only=True) as store:
            assert store.snapshot_count() == result.snapshots
