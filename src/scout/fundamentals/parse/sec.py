"""Parse archived SEC submissions (.txt SGML) into `RawFact`s, fully offline.

WHY this shape: the harvest layer stores the exact bytes SEC disseminates -- one
`.txt` "full submission" per accession, SGML-wrapped, with the XBRL instance and
its linkbases embedded. edgartools can reconstruct the complete fact set from
those bytes with no network, which is the whole reason the archive is the source
of truth (see parse/base.py). This parser is a thin translator: it hands the
bytes to edgartools, then copies each numeric fact into a `RawFact` verbatim --
preserving taxonomy prefix, period, dimensions and unit. It does NO mapping to
canonical concepts; that is the normalizer's job, kept separate so a mapping fix
never forces a re-parse.

Two deliberate choices worth calling out:
  - Non-numeric facts (text blocks, policy narratives, dates-as-text) are
    dropped. They are not fundamentals and would only be noise to the metrics
    layer; we count them and note the count so the drop is visible, not silent.
  - Dimensioned facts (segment/product/geography breakdowns) are KEPT. The
    normalizer needs both the consolidated total and its breakdown and only it
    knows which concept wants which, so dropping here would destroy information.

A filing that carries no XBRL is not an error: a plain-text 8-K or an SGML doc
edgartools cannot reconstruct yields an empty `ParsedFiling` with a warning, not
an exception. Only genuine parse/data failures are swallowed that way -- a bug in
our own mapping (a `TypeError` we wrote) is allowed to surface, because hiding it
behind a warning would turn a code defect into silently missing fundamentals.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import pandas as pd

from scout.fundamentals.concepts import PeriodType
from scout.fundamentals.models import RawFact
from scout.fundamentals.parse.base import ParsedFiling

log = logging.getLogger(__name__)

# Form types that carry financial-statement XBRL worth parsing for fundamentals.
# 8-K is excluded on purpose: it rarely carries statements, and paying to parse
# every 8-K would swamp the ingester for almost no facts. Amendments ("10-K/A")
# are accepted by stripping the "/A" suffix before the membership test.
_FUNDAMENTAL_FORMS = frozenset({"10-K", "10-Q", "20-F", "40-F"})

# Only these columns are read from the facts dataframe. Selecting them up front
# keeps the per-row loop cheap (the full frame has 60+ dimension columns) and,
# because none of these names contain a hyphen or colon, lets `itertuples`
# expose them as plain attributes.
_FACT_COLUMNS = [
    "concept",
    "numeric_value",
    "unit_ref",
    "currency",
    "decimals",
    "period_type",
    "period_start",
    "period_end",
    "period_instant",
    "fiscal_period",
    "fiscal_year",
    "is_dimensioned",
]


class SecXbrlParser:
    """Reconstructs US-GAAP XBRL facts from an archived SEC full-submission.

    Stateless: one instance can parse any number of filings. `source` is the tag
    every produced fact and `ParsedFiling` carries, so downstream code can tell a
    SEC fact from an ESEF one without inspecting its taxonomy.
    """

    source = "sec"

    def can_parse(
        self, *, form_type: str | None, content_type: str | None, filename: str
    ) -> bool:
        # form_type is the reliable signal; content_type/filename are ignored
        # because SEC serves every form as text/plain .txt regardless of content.
        if form_type is None:
            # Unknown form: let parse() decide. A missing type is cheap to attempt
            # and returns empty-with-warning if there's no XBRL, so we don't skip.
            return True

        base_form = form_type.strip().upper().removesuffix("/A")
        return base_form in _FUNDAMENTAL_FORMS

    def parse(
        self, payload: bytes, *, accession: str, entity_hint: dict[str, str]
    ) -> ParsedFiling:
        # Reconstructing the XBRL is the only step that can fail on non-XBRL or
        # malformed submissions, so it is the only step wrapped: a failure here is
        # a data/parse problem and becomes an empty result plus a warning. Keeping
        # the wrap this tight means a bug in the mapping below still surfaces.
        reconstruction = self._reconstruct(payload)
        if reconstruction is None:
            return self._empty_filing(
                accession,
                entity_hint,
                "no XBRL instance found: edgartools could not reconstruct XBRL "
                "from the submission (not an XBRL filing, or malformed).",
            )

        sgml, xbrl, df = reconstruction

        entity_id = _entity_id(getattr(sgml, "cik", None), entity_hint)
        entity_name = getattr(xbrl, "entity_name", None)
        period_of_report = _to_date(getattr(xbrl, "period_of_report", None))
        filing_date = _to_date(getattr(sgml, "filing_date", None))

        warnings: list[str] = []

        if df is None or df.empty:
            warnings.append("no XBRL instance found: filing reconstructed but carried zero facts.")
            return ParsedFiling(
                accession=accession,
                source=self.source,
                entity_id=entity_id,
                entity_name=entity_name,
                taxonomy="us-gaap",
                period_of_report=period_of_report,
                filing_date=filing_date,
                facts=[],
                warnings=warnings,
            )

        facts, skipped_non_numeric, skipped_no_period = self._facts_from_dataframe(df, accession)

        if skipped_non_numeric:
            # Expected and normal (3M's 10-Q has ~100), but counted so the drop is
            # auditable rather than a silent gap between "facts filed" and "facts kept".
            warnings.append(
                f"skipped {skipped_non_numeric} non-numeric fact(s) "
                "(text blocks, dates, narratives) -- not fundamentals."
            )
        if skipped_no_period:
            warnings.append(
                f"skipped {skipped_no_period} fact(s) with no usable period end date."
            )

        return ParsedFiling(
            accession=accession,
            source=self.source,
            entity_id=entity_id,
            entity_name=entity_name,
            taxonomy="us-gaap",
            period_of_report=period_of_report,
            filing_date=filing_date,
            facts=facts,
            warnings=warnings,
        )

    # -- internals -----------------------------------------------------------

    def _reconstruct(self, payload: bytes) -> tuple[Any, Any, pd.DataFrame] | None:
        """Turn raw bytes into (sgml, xbrl, facts-dataframe), or None on failure.

        Returns None -- rather than raising -- for any edgartools parse/data
        failure, so a filing without XBRL yields an empty result upstream. The
        broad `except` is intentional and scoped to exactly these three calls:
        edgartools raises a variety of low-level errors (AttributeError,
        ValueError, KeyError) on non-XBRL input, and to the caller they all mean
        the same thing -- "no reconstructable XBRL here".
        """
        # Local imports: edgartools is heavyweight, and can_parse callers that
        # never reach parse() shouldn't pay its import cost.
        from edgar.sgml import FilingSGML
        from edgar.xbrl import XBRL

        try:
            # latin-1 decodes any byte 1:1 with no UnicodeDecodeError; SEC SGML is
            # effectively latin-1/ASCII and the XBRL inside declares its own encoding.
            sgml = FilingSGML.from_text(payload.decode("latin-1"))
            xbrl = XBRL.from_filing(sgml)
            df = xbrl.facts.to_dataframe()
        except Exception as exc:
            log.warning("SEC XBRL reconstruction failed: %s: %s", type(exc).__name__, exc)
            return None

        return sgml, xbrl, df

    def _facts_from_dataframe(
        self, df: pd.DataFrame, accession: str
    ) -> tuple[list[RawFact], int, int]:
        """Map each dataframe row to a RawFact, skipping non-fundamentals.

        Runs OUTSIDE the reconstruction try/except on purpose: a failure in here
        would be our own bug, and we want it to surface, not hide as a warning.
        """
        # Some filings may not surface every optional column; guard so a missing
        # one becomes None-for-all rather than a KeyError.
        available = [c for c in _FACT_COLUMNS if c in df.columns]
        subset = df[available]

        facts: list[RawFact] = []
        skipped_non_numeric = 0
        skipped_no_period = 0

        for row in subset.itertuples(index=False):
            raw = row._asdict()

            numeric_value = raw.get("numeric_value")
            if pd.isna(numeric_value):
                # Non-numeric fact (text block, date, boolean narrative). Not a
                # fundamental; drop it but count it for the warning.
                skipped_non_numeric += 1
                continue

            period_type = _period_type(raw.get("period_type"))
            period_start, period_end = _period_bounds(raw, period_type)
            if period_end is None:
                # A fact with no resolvable end date can't be placed in time, and
                # RawFact.period_end is non-optional -- skip rather than fabricate.
                skipped_no_period += 1
                continue

            taxonomy, local_name = _split_concept(raw.get("concept"))

            facts.append(
                RawFact(
                    accession=accession,
                    taxonomy=taxonomy,
                    local_name=local_name,
                    value=float(numeric_value),
                    unit=_unit(raw.get("currency"), raw.get("unit_ref")),
                    period_type=period_type,
                    period_start=period_start,
                    period_end=period_end,
                    is_dimensioned=bool(raw.get("is_dimensioned")),
                    decimals=_to_int(raw.get("decimals")),
                    fiscal_year=_to_int(raw.get("fiscal_year")),
                    fiscal_period=_to_str(raw.get("fiscal_period")),
                )
            )

        return facts, skipped_non_numeric, skipped_no_period

    def _empty_filing(
        self, accession: str, entity_hint: dict[str, str], warning: str
    ) -> ParsedFiling:
        """A no-XBRL result: identity from the harvest hint, zero facts, a note."""
        return ParsedFiling(
            accession=accession,
            source=self.source,
            entity_id=_entity_id(None, entity_hint),
            entity_name=None,
            taxonomy="us-gaap",
            period_of_report=None,
            filing_date=None,
            facts=[],
            warnings=[warning],
        )


# -- pure helpers (module-level so they're trivially unit-testable) ----------


def _split_concept(concept: Any) -> tuple[str, str]:
    """"us-gaap:Revenues" -> ("us-gaap", "Revenues").

    A concept with no prefix (rare, e.g. a bare extension name) has no taxonomy,
    so the whole string becomes the local name and taxonomy is empty.
    """
    text = "" if concept is None else str(concept)
    prefix, sep, local = text.partition(":")
    if not sep:
        return "", text
    return prefix, local


def _unit(currency: Any, unit_ref: Any) -> str | None:
    """Currency for monetary facts (e.g. "USD"); otherwise the unit_ref ("shares",
    "pure"). Currency wins when populated because it is the human-meaningful unit;
    unit_ref is the fallback for non-monetary facts that have no currency."""
    currency_str = _to_str(currency)
    if currency_str:
        return currency_str
    return _to_str(unit_ref)


def _period_type(value: Any) -> PeriodType:
    """"instant"/"duration" text -> PeriodType. Anything else defaults to DURATION
    (the safe fallback: an unknown period shape is more likely a span than a
    point, and the normalizer re-checks period shape against the concept anyway)."""
    if _to_str(value) == "instant":
        return PeriodType.INSTANT
    return PeriodType.DURATION


def _period_bounds(row: dict[str, Any], period_type: PeriodType) -> tuple[date | None, date]:
    """Resolve (start, end) dates for a fact.

    Instant facts are a point in time: no start, and the balance date lives in
    `period_instant`. Duration facts span `period_start`..`period_end`. Returns
    end as `date | None` (the caller skips the row if it's None); typed as `date`
    in the non-None path so the common case reads cleanly.
    """
    if period_type is PeriodType.INSTANT:
        instant = _to_date(row.get("period_instant")) or _to_date(row.get("period_end"))
        return None, instant  # type: ignore[return-value]

    start = _to_date(row.get("period_start"))
    end = _to_date(row.get("period_end"))
    return start, end  # type: ignore[return-value]


def _to_date(value: Any) -> date | None:
    """Coerce a cell to `datetime.date`, or None.

    The facts dataframe delivers dates as ISO strings ("2026-06-30") and
    `period_of_report`/`filing_date` likewise, but edgartools versions have
    shipped these as pandas Timestamps too -- so we handle Timestamp, datetime,
    date and string, and treat NaN/empty as absent."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return None
    # Tolerate a trailing time component ("2026-06-30T00:00:00") by taking the date.
    return date.fromisoformat(text[:10])


def _to_int(value: Any) -> int | None:
    """Coerce to int, or None. `decimals` arrives as text including "INF"
    (infinite precision / exact), which has no integer form -- that and any
    unparseable/absent value become None rather than raising."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value

    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text)) if "." in text else int(text)
    except ValueError:
        # "INF"/"-INF" (and any other non-numeric decimals token) -> unknown precision.
        return None


def _to_str(value: Any) -> str | None:
    """Coerce to a non-empty string, or None (NaN/empty become None)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return text


def _entity_id(cik: Any, entity_hint: dict[str, str]) -> str:
    """The SEC entity key is the CIK, zero-stripped ("0000066740" -> "66740").

    Prefer the CIK the filing itself states; fall back to the harvest manifest's
    hint when the filing doesn't (a non-XBRL submission often has no CIK on the
    reconstructed object). Empty string if neither is available -- an unknown
    entity is still a valid, if unjoinable, result."""
    raw = cik if cik not in (None, "") else entity_hint.get("cik")
    if raw in (None, ""):
        return ""

    text = str(raw).strip()
    # Strip leading zeros without turning "0" into "" -- lstrip then restore.
    stripped = text.lstrip("0")
    return stripped or "0"
