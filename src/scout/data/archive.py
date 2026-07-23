"""Append-only archive of raw primary filings.

This is the most time-critical component in the project and the reason phase 0
comes before everything else. No retail vendor sells point-in-time fundamentals
outside the US, so the only way to get one is to build it forward -- and the
free sources purge:

    UK Companies House daily accounts   deleted after 60 days
    Japan TDnet                         rolling ~30-day window
    Nasdaq symbol directory             snapshot only, no archive published

None of that can be reconstructed retroactively. Every day this collector is not
running is a day of history permanently lost.

Design rules, all of which exist to keep the archive trustworthy as evidence:

  * Payloads are stored verbatim. No parsing, no normalization, no cleanup on
    the way in. A parser fix two years from now must be replayable over the
    whole history, which is only possible if we kept what was actually
    published.
  * Nothing is ever overwritten. A re-filed document becomes a new version;
    both are kept. Issuers revise filings, and "what did this look like on the
    day" is the entire question the archive exists to answer.
  * Ingest date and filing date are recorded separately. Conflating them is the
    classic look-ahead bug.

Layout:

    <root>/manifest.jsonl                       append-only index, one line per blob
    <root>/<source>/<YYYY>/<MM>/<DD>/<doc>/v<N>/<filename>
    <root>/<source>/<YYYY>/<MM>/<DD>/<doc>/v<N>/meta.json
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from scout.data.sources.base import RawDocument

MANIFEST_NAME = "manifest.jsonl"

# doc_ids come from foreign systems and land in paths. Accession numbers contain
# dashes, OAM filing ids contain slashes and colons.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe(component: str) -> str:
    cleaned = _UNSAFE.sub("_", component).strip("._")
    if not cleaned:
        cleaned = "unnamed"
    # Windows path components cap at 255; leave room for the version suffix.
    return cleaned[:120]


@dataclass(frozen=True, slots=True)
class StoredDocument:
    source: str
    doc_id: str
    version: int
    content_sha256: str
    size_bytes: int
    path: str
    """Relative to the archive root, POSIX separators, so the manifest is
    portable between machines."""

    filename: str
    ingested_at: str
    harvest_day: str
    filing_date: str | None
    form_type: str | None
    title: str | None
    url: str | None
    content_type: str | None
    entity: dict[str, str]
    meta: dict[str, Any]

    @property
    def is_revision(self) -> bool:
        return self.version > 1


class Archive:
    """Append-only store plus its manifest index.

    Cheap to construct: the manifest is read once into an in-memory index of
    (source, doc_id) -> {content hashes}, which is what `has` and `put` need to
    decide whether a document is new, unchanged, or a revision.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / MANIFEST_NAME
        self._index: dict[tuple[str, str], dict[str, int]] = {}
        self._load_index()

    # ---------------------------------------------------------------- index

    def _load_index(self) -> None:
        if not self._manifest_path.exists():
            return
        with self._manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    key = (record["source"], record["doc_id"])
                    self._index.setdefault(key, {})[record["content_sha256"]] = record["version"]
                except (json.JSONDecodeError, KeyError) as exc:
                    # A truncated final line is expected if a previous run was
                    # killed mid-write. Anything else is worth surfacing, but
                    # never worth refusing to start over.
                    raise ValueError(
                        f"{self._manifest_path}:{line_number} is not a valid manifest record "
                        f"({exc}). Repair or truncate the line; the payloads on disk are intact."
                    ) from exc

    # ---------------------------------------------------------------- query

    def has_seen(self, source: str, doc_id: str) -> bool:
        """True if any version of this document is already stored.

        Used to skip fetching during listing. Cheap and deliberately loose: a
        source that revises documents in place should call `has_content`
        instead.
        """
        return (source, doc_id) in self._index

    def has_content(self, source: str, doc_id: str, content_sha256: str) -> bool:
        return content_sha256 in self._index.get((source, doc_id), {})

    def versions(self, source: str, doc_id: str) -> int:
        return len(self._index.get((source, doc_id), {}))

    def __len__(self) -> int:
        return sum(len(hashes) for hashes in self._index.values())

    # ---------------------------------------------------------------- write

    def put(self, document: RawDocument, *, harvest_day: date | None = None) -> StoredDocument | None:
        """Store a document. Returns None if this exact content is already held.

        Same doc_id with different bytes is stored as a new version rather than
        replacing the old one -- that is a revision, and both sides of it are
        evidence.
        """
        ref = document.ref
        digest = hashlib.sha256(document.payload).hexdigest()

        if self.has_content(ref.source, ref.doc_id, digest):
            return None

        day = harvest_day or datetime.now(UTC).date()
        version = self.versions(ref.source, ref.doc_id) + 1

        directory = (
            self.root
            / _safe(ref.source)
            / f"{day.year:04d}"
            / f"{day.month:02d}"
            / f"{day.day:02d}"
            / _safe(ref.doc_id)
            / f"v{version}"
        )
        directory.mkdir(parents=True, exist_ok=True)

        filename = _safe(document.filename) or "payload.bin"
        (directory / filename).write_bytes(document.payload)

        stored = StoredDocument(
            source=ref.source,
            doc_id=ref.doc_id,
            version=version,
            content_sha256=digest,
            size_bytes=len(document.payload),
            path=(directory / filename).relative_to(self.root).as_posix(),
            filename=filename,
            ingested_at=datetime.now(UTC).isoformat(),
            harvest_day=day.isoformat(),
            filing_date=ref.filing_date.isoformat() if ref.filing_date else None,
            form_type=ref.form_type,
            title=ref.title,
            url=ref.url,
            content_type=document.content_type,
            entity=dict(ref.entity),
            meta=dict(ref.meta),
        )

        # meta.json sits beside the payload so a directory is self-describing
        # even if the manifest is ever lost or rebuilt.
        (directory / "meta.json").write_text(
            json.dumps(asdict(stored), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._append_manifest(stored)
        self._index.setdefault((ref.source, ref.doc_id), {})[digest] = version
        return stored

    def _append_manifest(self, stored: StoredDocument) -> None:
        line = json.dumps(asdict(stored), ensure_ascii=False, separators=(",", ":"))
        with self._manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()

    # ---------------------------------------------------------------- read

    def iter_manifest(
        self,
        *,
        source: str | None = None,
        since: date | None = None,
        until: date | None = None,
    ) -> Iterator[StoredDocument]:
        """Replay the manifest, optionally filtered by source and harvest day."""
        if not self._manifest_path.exists():
            return
        with self._manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if source and record["source"] != source:
                    continue
                day = date.fromisoformat(record["harvest_day"])
                if since and day < since:
                    continue
                if until and day > until:
                    continue
                yield StoredDocument(**record)

    def read_payload(self, stored: StoredDocument) -> bytes:
        return (self.root / stored.path).read_bytes()
