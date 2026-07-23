"""DuckDB storage for the fundamentals pipeline.

Two layers, matching the two layers in `models.py`:

  raw_facts        Append-only provenance: one row per RawFact, exactly as
                    filed. Never edited in place -- re-ingesting an accession
                    deletes and re-inserts its rows so the table always
                    reflects the current parse of that filing, but nothing
                    from *other* filings is ever touched.
  canonical_facts / snapshots
                    The queryable output of normalization: one row per
                    (entity, period, concept) plus one snapshot header row per
                    (entity, period). Upserted, because normalization is
                    expected to be re-run as the tag-mapping tables improve
                    and the latest run should simply replace the last one.

Keeping raw and canonical in separate tables is the point: if a mapping bug
is found, canonical_facts can be dropped and rebuilt entirely from raw_facts
without re-parsing or re-fetching a single filing.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import duckdb

from scout.fundamentals.concepts import Concept, PeriodType
from scout.fundamentals.models import CanonicalFact, EntityRef, FundamentalsSnapshot, RawFact

_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_facts (
    accession       VARCHAR NOT NULL,
    taxonomy        VARCHAR NOT NULL,
    local_name      VARCHAR NOT NULL,
    value           DOUBLE NOT NULL,
    unit            VARCHAR,
    period_type     VARCHAR NOT NULL,
    period_start    DATE,
    period_end      DATE NOT NULL,
    is_dimensioned  BOOLEAN NOT NULL,
    decimals        INTEGER,
    fiscal_year     INTEGER,
    fiscal_period   VARCHAR
);

-- Not a primary key: a single accession can legitimately repeat the same
-- (taxonomy, local_name, period_end) with different dimensions, and
-- is_dimensioned alone doesn't disambiguate multiple dimensioned facts. The
-- accession index is what makes put_raw_facts's delete-then-insert cheap.
CREATE INDEX IF NOT EXISTS idx_raw_facts_accession ON raw_facts (accession);

CREATE TABLE IF NOT EXISTS snapshots (
    entity_id           VARCHAR NOT NULL,
    period_end          DATE NOT NULL,
    fiscal_period       VARCHAR,
    entity_source       VARCHAR NOT NULL,
    entity_scheme       VARCHAR NOT NULL,
    entity_name         VARCHAR,
    entity_country      VARCHAR,
    fiscal_year         INTEGER,
    currency            VARCHAR,
    taxonomy            VARCHAR NOT NULL,
    accession           VARCHAR NOT NULL,
    filing_date         DATE,
    warnings            VARCHAR NOT NULL  -- JSON array of strings
);

CREATE INDEX IF NOT EXISTS idx_snapshots_key ON snapshots (entity_id, period_end, fiscal_period);

CREATE TABLE IF NOT EXISTS canonical_facts (
    entity_id       VARCHAR NOT NULL,
    period_end      DATE NOT NULL,
    fiscal_period   VARCHAR,
    concept         VARCHAR NOT NULL,
    value           DOUBLE NOT NULL,
    currency        VARCHAR,
    period_start    DATE,
    fiscal_year     INTEGER,
    accession       VARCHAR NOT NULL,
    source_concept  VARCHAR NOT NULL,
    taxonomy        VARCHAR NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_canonical_facts_key
    ON canonical_facts (entity_id, period_end, fiscal_period);
"""
# No PRIMARY KEY / UNIQUE constraint on either table: DuckDB enforces PRIMARY
# KEY columns as implicitly NOT NULL (unlike Postgres, where a PK column is
# NOT NULL but a multi-column UNIQUE constraint tolerates NULLs), and
# fiscal_period is legitimately NULL for filings that don't report one. A
# NOT NULL fiscal_period would reject exactly the rows the model says are
# valid. So the natural key (entity_id, period_end, fiscal_period[, concept])
# is enforced procedurally instead: every write is a DELETE of the existing
# key's rows followed by an INSERT, done as one statement pair per call, which
# gives the same "upsert" behavior without asking the column to be non-null.


class FundamentalsStore:
    """One DuckDB connection, opened against `Config.db_path`.

    A single writer connection is intentional: DuckDB allows one read-write
    connection to a file at a time, and the ingester's write phase is
    sequential after a concurrent parse fan-out, so there is never a reason
    to share this across threads.
    """

    def __init__(self, db_path: Path | str, *, read_only: bool = False) -> None:
        self._path = Path(db_path)
        if not read_only:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self._path), read_only=read_only)
        self._read_only = read_only

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> FundamentalsStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def initialize(self) -> None:
        """Create tables/indexes if absent. Safe to call on every open."""
        # CREATE TABLE/INDEX IF NOT EXISTS is naturally idempotent; DuckDB
        # runs each statement in the script in order.
        self._conn.execute(_SCHEMA)

    # ------------------------------------------------------------------
    # Raw facts
    # ------------------------------------------------------------------

    def put_raw_facts(self, facts: Sequence[RawFact]) -> int:
        """Replace all raw facts for the accession(s) present in `facts`.

        Delete-then-insert per accession, not a global truncate: re-ingesting
        one filing must never touch another filing's rows. Facts for more
        than one accession in a single call are supported (grouped so each
        accession's old rows are cleared exactly once).
        """
        if not facts:
            return 0

        accessions = {f.accession for f in facts}
        self._conn.executemany(
            "DELETE FROM raw_facts WHERE accession = ?", [(a,) for a in accessions]
        )
        self._conn.executemany(
            """
            INSERT INTO raw_facts (
                accession, taxonomy, local_name, value, unit, period_type,
                period_start, period_end, is_dimensioned, decimals,
                fiscal_year, fiscal_period
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    f.accession,
                    f.taxonomy,
                    f.local_name,
                    f.value,
                    f.unit,
                    f.period_type.value,
                    f.period_start,
                    f.period_end,
                    f.is_dimensioned,
                    f.decimals,
                    f.fiscal_year,
                    f.fiscal_period,
                )
                for f in facts
            ],
        )
        return len(facts)

    def has_accession(self, accession: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM raw_facts WHERE accession = ? LIMIT 1", [accession]
        ).fetchone()
        return row is not None

    def raw_facts_for(self, accession: str) -> list[RawFact]:
        rows = self._conn.execute(
            """
            SELECT accession, taxonomy, local_name, value, unit, period_type,
                   period_start, period_end, is_dimensioned, decimals,
                   fiscal_year, fiscal_period
            FROM raw_facts
            WHERE accession = ?
            """,
            [accession],
        ).fetchall()
        return [
            RawFact(
                accession=r[0],
                taxonomy=r[1],
                local_name=r[2],
                value=r[3],
                unit=r[4],
                period_type=PeriodType(r[5]),
                period_start=r[6],
                period_end=r[7],
                is_dimensioned=r[8],
                decimals=r[9],
                fiscal_year=r[10],
                fiscal_period=r[11],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Snapshots / canonical facts
    # ------------------------------------------------------------------

    def put_snapshot(self, snapshot: FundamentalsSnapshot) -> None:
        """Upsert the snapshot header and every canonical fact it carries.

        Both the header and its facts are replaced by key (delete then
        insert, mirroring `put_raw_facts`) rather than an ON CONFLICT upsert
        -- `fiscal_period` can be NULL and DuckDB requires PRIMARY KEY/UNIQUE
        columns to be NOT NULL, so there is no constraint to conflict on.
        Facts for concepts no longer produced by a re-run are removed because
        the whole fact set for the key is cleared before inserting.
        """
        entity = snapshot.entity
        key = (snapshot.entity.entity_id, snapshot.period_end, snapshot.fiscal_period)

        self._conn.execute(
            "DELETE FROM snapshots WHERE entity_id = ? AND period_end = ? "
            "AND fiscal_period IS NOT DISTINCT FROM ?",
            key,
        )
        self._conn.execute(
            """
            INSERT INTO snapshots (
                entity_id, period_end, fiscal_period, entity_source,
                entity_scheme, entity_name, entity_country, fiscal_year,
                currency, taxonomy, accession, filing_date, warnings
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                entity.entity_id,
                snapshot.period_end,
                snapshot.fiscal_period,
                entity.source,
                entity.identifier_scheme,
                entity.name,
                entity.country,
                snapshot.fiscal_year,
                snapshot.currency,
                snapshot.taxonomy,
                snapshot.accession,
                snapshot.filing_date,
                json.dumps(list(snapshot.warnings)),
            ],
        )

        # Clear this snapshot's old facts, then insert the current set --
        # simpler and just as correct as a per-concept upsert, and it
        # naturally drops concepts a re-run no longer produces.
        self._conn.execute(
            "DELETE FROM canonical_facts WHERE entity_id = ? AND period_end = ? "
            "AND fiscal_period IS NOT DISTINCT FROM ?",
            key,
        )
        if not snapshot.facts:
            return

        self._conn.executemany(
            """
            INSERT INTO canonical_facts (
                entity_id, period_end, fiscal_period, concept, value,
                currency, period_start, fiscal_year, accession,
                source_concept, taxonomy
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    fact.entity_id,
                    fact.period_end,
                    fact.fiscal_period,
                    fact.concept.value,
                    fact.value,
                    fact.currency,
                    fact.period_start,
                    fact.fiscal_year,
                    fact.accession,
                    fact.source_concept,
                    fact.taxonomy,
                )
                for fact in snapshot.facts.values()
            ],
        )

    def get_snapshot(
        self, entity_id: str, period_end: date, fiscal_period: str | None
    ) -> FundamentalsSnapshot | None:
        header = self._conn.execute(
            """
            SELECT entity_id, period_end, fiscal_period, entity_source,
                   entity_scheme, entity_name, entity_country, fiscal_year,
                   currency, taxonomy, accession, filing_date, warnings
            FROM snapshots
            WHERE entity_id = ? AND period_end = ? AND fiscal_period IS NOT DISTINCT FROM ?
            """,
            [entity_id, period_end, fiscal_period],
        ).fetchone()
        if header is None:
            return None
        return self._build_snapshot(header)

    def snapshots_for_entity(self, entity_id: str) -> list[FundamentalsSnapshot]:
        headers = self._conn.execute(
            """
            SELECT entity_id, period_end, fiscal_period, entity_source,
                   entity_scheme, entity_name, entity_country, fiscal_year,
                   currency, taxonomy, accession, filing_date, warnings
            FROM snapshots
            WHERE entity_id = ?
            ORDER BY period_end DESC
            """,
            [entity_id],
        ).fetchall()
        return [self._build_snapshot(h) for h in headers]

    def latest_snapshot(self, entity_id: str) -> FundamentalsSnapshot | None:
        snapshots = self.snapshots_for_entity(entity_id)
        return snapshots[0] if snapshots else None

    def _build_snapshot(self, header: tuple) -> FundamentalsSnapshot:
        (
            entity_id,
            period_end,
            fiscal_period,
            entity_source,
            entity_scheme,
            entity_name,
            entity_country,
            fiscal_year,
            currency,
            taxonomy,
            accession,
            filing_date,
            warnings_json,
        ) = header

        fact_rows = self._conn.execute(
            """
            SELECT concept, value, currency, period_start, fiscal_year,
                   accession, source_concept, taxonomy
            FROM canonical_facts
            WHERE entity_id = ? AND period_end = ? AND fiscal_period IS NOT DISTINCT FROM ?
            """,
            [entity_id, period_end, fiscal_period],
        ).fetchall()

        facts: dict[Concept, CanonicalFact] = {}
        for row in fact_rows:
            concept = Concept(row[0])
            facts[concept] = CanonicalFact(
                entity_id=entity_id,
                concept=concept,
                value=row[1],
                currency=row[2],
                period_end=period_end,
                period_start=row[3],
                fiscal_year=row[4],
                fiscal_period=fiscal_period,
                accession=row[5],
                source_concept=row[6],
                taxonomy=row[7],
            )

        return FundamentalsSnapshot(
            entity=EntityRef(
                source=entity_source,
                entity_id=entity_id,
                identifier_scheme=entity_scheme,
                name=entity_name,
                country=entity_country,
            ),
            period_end=period_end,
            fiscal_year=fiscal_year,
            fiscal_period=fiscal_period,
            currency=currency,
            taxonomy=taxonomy,
            accession=accession,
            filing_date=filing_date,
            facts=facts,
            warnings=json.loads(warnings_json),
        )

    # ------------------------------------------------------------------
    # Coverage / introspection
    # ------------------------------------------------------------------

    def entity_count(self) -> int:
        return self._conn.execute("SELECT COUNT(DISTINCT entity_id) FROM snapshots").fetchone()[0]

    def all_entity_ids(self) -> list[str]:
        """Every distinct entity in the store. The screen iterates this to build
        one candidate per entity."""
        rows = self._conn.execute(
            "SELECT DISTINCT entity_id FROM snapshots ORDER BY entity_id"
        ).fetchall()
        return [r[0] for r in rows]

    def snapshot_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]

    def coverage(self) -> list[dict]:
        """Per-taxonomy rollup for the CLI `status` output.

        Two separate aggregates joined on taxonomy, not one query joining
        snapshots to canonical_facts directly -- a direct join fans out one
        row per fact, which would inflate COUNT(*) on the snapshots side by
        however many concepts each snapshot has.
        """
        rows = self._conn.execute(
            """
            WITH snap_stats AS (
                SELECT taxonomy, COUNT(DISTINCT entity_id) AS entities, COUNT(*) AS snapshots
                FROM snapshots
                GROUP BY taxonomy
            ),
            concept_stats AS (
                SELECT s.taxonomy, COUNT(DISTINCT cf.concept) AS concepts
                FROM canonical_facts cf
                JOIN snapshots s
                    ON cf.entity_id = s.entity_id
                    AND cf.period_end = s.period_end
                    AND cf.fiscal_period IS NOT DISTINCT FROM s.fiscal_period
                GROUP BY s.taxonomy
            )
            SELECT snap_stats.taxonomy, entities, snapshots, COALESCE(concepts, 0) AS concepts
            FROM snap_stats
            LEFT JOIN concept_stats ON snap_stats.taxonomy = concept_stats.taxonomy
            ORDER BY snap_stats.taxonomy
            """
        ).fetchall()
        return [
            {"taxonomy": r[0], "entities": r[1], "snapshots": r[2], "concepts": r[3]}
            for r in rows
        ]
