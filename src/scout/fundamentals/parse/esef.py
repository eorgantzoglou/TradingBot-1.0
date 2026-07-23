"""Parses xBRL-JSON (OIM) ESEF filings into `RawFact`s, fully offline.

Why plain JSON, not Arelle: filings.xbrl.org archives a pre-normalized
xBRL-JSON sibling of every ESEF report -- the OIM (Open Information Model)
serialization where every fact is already resolved to a concept, entity,
period and (optional) unit. There is no taxonomy DTS to load, no linkbases to
resolve, no inline-XBRL HTML to strip; it is a JSON object with a `facts` dict.
Parsing it is stdlib `json`, which is simpler and keeps a heavy, slow
dependency (Arelle) out of the harvest-time hot path. The packaged ESEF ZIP
(inline XBRL + taxonomy package) is a materially different, harder problem --
`can_parse` refuses `.zip` on purpose so a future Arelle-based parser can claim
that form explicitly without this one silently mis-handling it.

Every fact is parsed independently and defensively: one malformed fact (a
missing period, an unprefixed concept) is skipped and counted rather than
aborting the whole document, because a single stray disclosure fact should
never cost the rest of a filing's numbers.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime

from scout.fundamentals.concepts import PeriodType
from scout.fundamentals.models import RawFact
from scout.fundamentals.parse.base import ParsedFiling

# The four dimensions the OIM format defines as "core" for a fact. Anything
# else in a fact's `dimensions` dict is a typed/explicit XBRL dimension
# (segment, product, geography, ...) -- see `is_dimensioned` on RawFact.
_CORE_DIMENSION_KEYS = frozenset({"concept", "entity", "period", "unit"})

# The concept whose value is the filer's name, when present. Text, so it never
# survives the numeric-only fact filter -- looked up separately.
_ENTITY_NAME_CONCEPT = "NameOfReportingEntity"

# ESEF filers report under IFRS; this is the dominant standard for every
# filing this parser handles, regardless of which extension taxonomies
# individual facts use.
_TAXONOMY = "ifrs-full"


class EsefJsonParser:
    """Parses archived xBRL-JSON documents from filings.xbrl.org."""

    source = "esef"

    def can_parse(self, *, form_type: str | None, content_type: str | None, filename: str) -> bool:
        # Packaged ESEF ZIPs (inline XBRL + taxonomy package) need Arelle to
        # unpack -- not handled here. Only the pre-normalized xBRL-JSON
        # sibling qualifies; a future Arelle-based parser can claim ".zip".
        if filename.endswith(".zip"):
            return False
        if content_type and "json" in content_type.lower():
            return True
        return filename.endswith(".json")

    def parse(self, payload: bytes, *, accession: str, entity_hint: dict[str, str]) -> ParsedFiling:
        try:
            doc = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return self._empty(accession, entity_hint, f"payload is not valid JSON: {exc}")

        if not isinstance(doc, dict):
            return self._empty(accession, entity_hint, "payload is not a JSON object")

        raw_facts = doc.get("facts")
        if not isinstance(raw_facts, dict):
            return self._empty(accession, entity_hint, "no 'facts' object -- not an xBRL-JSON (OIM) document")

        facts: list[RawFact] = []
        entity_ids: Counter[str] = Counter()
        entity_name: str | None = None
        skipped_non_numeric = 0
        skipped_malformed = 0

        for fact in raw_facts.values():
            outcome, entity_id, name = self._parse_one_fact(fact, accession)
            if entity_id is not None:
                entity_ids[entity_id] += 1
            if name is not None and entity_name is None:
                entity_name = name

            if outcome == "malformed":
                skipped_malformed += 1
                continue
            if outcome == "non_numeric":
                skipped_non_numeric += 1
                continue
            facts.append(outcome)  # a RawFact

        # The authoritative entity id is whatever the facts themselves carry
        # (see module docstring in the caller's task brief: for Ukrainian
        # filers this is an 8-digit EDRPOU code, not the LEI the harvest
        # metadata may have labelled it as). Fall back to the manifest hint
        # only when no fact carried an entity dimension at all.
        if entity_ids:
            entity_id = entity_ids.most_common(1)[0][0]
        else:
            entity_id = entity_hint.get("lei") or entity_hint.get("cik") or ""

        warnings: list[str] = []
        if skipped_non_numeric:
            warnings.append(f"skipped {skipped_non_numeric} non-numeric fact(s) (text/date values)")
        if skipped_malformed:
            warnings.append(f"skipped {skipped_malformed} malformed fact(s) (missing/invalid concept/period/entity)")
        if not facts:
            warnings.append("no numeric facts parsed from xBRL-JSON document")

        return ParsedFiling(
            accession=accession,
            source=self.source,
            entity_id=entity_id,
            entity_name=entity_name,
            taxonomy=_TAXONOMY,
            period_of_report=_period_of_report(facts),
            filing_date=None,  # not present in the JSON; the harvest manifest has it
            facts=facts,
            warnings=warnings,
        )

    @staticmethod
    def _parse_one_fact(
        fact: object, accession: str
    ) -> tuple[RawFact | str, str | None, str | None]:
        """Parse one raw fact dict.

        Returns (outcome, entity_id, entity_name) where outcome is a RawFact
        on success, or the literal "malformed"/"non_numeric" on skip.
        entity_id/entity_name are reported separately because they matter even
        for facts whose value we drop (a text NameOfReportingEntity fact still
        tells us who filed; a non-numeric fact's entity still counts towards
        the dominant entity id).
        """
        if not isinstance(fact, dict):
            return "malformed", None, None

        dims = fact.get("dimensions")
        if not isinstance(dims, dict):
            return "malformed", None, None

        concept = dims.get("concept")
        split = _split_concept(concept) if isinstance(concept, str) else None
        if split is None:
            return "malformed", None, None
        taxonomy, local_name = split

        entity_dim = dims.get("entity")
        entity_id = _extract_entity_id(entity_dim) if isinstance(entity_dim, str) and entity_dim else None

        entity_name = None
        if local_name == _ENTITY_NAME_CONCEPT:
            value = fact.get("value")
            if isinstance(value, str) and value.strip():
                entity_name = value

        numeric_value = _to_numeric(fact.get("value"))
        if numeric_value is None:
            return "non_numeric", entity_id, entity_name

        period_str = dims.get("period")
        if not isinstance(period_str, str) or not period_str:
            return "malformed", entity_id, entity_name
        try:
            period_type, period_start, period_end = _parse_period(period_str)
        except ValueError:
            return "malformed", entity_id, entity_name

        raw_fact = RawFact(
            accession=accession,
            taxonomy=taxonomy,
            local_name=local_name,
            value=numeric_value,
            unit=_extract_unit(dims.get("unit")),
            period_type=period_type,
            period_start=period_start,
            period_end=period_end,
            is_dimensioned=bool(set(dims) - _CORE_DIMENSION_KEYS),
            decimals=_extract_decimals(fact.get("decimals")),
        )
        return raw_fact, entity_id, entity_name

    @staticmethod
    def _empty(accession: str, entity_hint: dict[str, str], warning: str) -> ParsedFiling:
        return ParsedFiling(
            accession=accession,
            source="esef",
            entity_id=entity_hint.get("lei") or entity_hint.get("cik") or "",
            entity_name=None,
            taxonomy=_TAXONOMY,
            period_of_report=None,
            filing_date=None,
            warnings=[warning],
        )


def _split_concept(concept: str) -> tuple[str, str] | None:
    """"taxonomy:LocalName" -> (taxonomy, LocalName), or None if unprefixed."""
    if ":" not in concept:
        return None
    taxonomy, local_name = concept.split(":", 1)
    if not taxonomy or not local_name:
        return None
    return taxonomy, local_name


def _extract_entity_id(entity: str) -> str:
    """"scheme:44356194" -> "44356194". Whatever precedes the last colon is
    the identifier scheme (a URI for national registries, "lei" elsewhere);
    only the trailing identifier is stored, matching what other RawFacts key
    entities by."""
    return entity.rsplit(":", 1)[-1]


def _extract_unit(unit: object) -> str | None:
    """"iso4217:USD" -> "USD"; "xbrli:pure"/"xbrli:shares" -> "pure"/"shares";
    absent or unrecognized -> None."""
    if not isinstance(unit, str) or not unit:
        return None
    if unit == "xbrli:pure":
        return "pure"
    if unit == "xbrli:shares":
        return "shares"
    if ":" in unit:
        return unit.split(":", 1)[1]
    return unit


def _to_numeric(value: object) -> float | None:
    """A monetary/numeric fact value is a string or number that parses as
    float. Booleans (a bool is technically an int in Python) and ISO date
    strings are deliberately NOT numeric -- checked first / rejected by
    float() naturally, respectively."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _extract_decimals(decimals: object) -> int | None:
    if isinstance(decimals, bool) or not isinstance(decimals, int):
        return None
    return decimals


def _parse_period(period_str: str) -> tuple[PeriodType, date | None, date]:
    """OIM period dimension: an INSTANT is a single ISO datetime; a DURATION
    is two ISO datetimes joined by "/", and the END of the span (the second
    one) is what gets stored as `period_end` regardless of shape."""
    if "/" in period_str:
        start_str, end_str = period_str.split("/", 1)
        return PeriodType.DURATION, _parse_datetime(start_str), _parse_datetime(end_str)
    return PeriodType.INSTANT, None, _parse_datetime(period_str)


def _parse_datetime(value: str) -> date:
    return datetime.fromisoformat(value).date()


def _period_of_report(facts: list[RawFact]) -> date | None:
    """The most common period_end among duration facts (the fiscal year the
    income statement covers), or failing that the latest instant (the
    balance-sheet date)."""
    duration_ends = [f.period_end for f in facts if f.period_type is PeriodType.DURATION]
    if duration_ends:
        return Counter(duration_ends).most_common(1)[0][0]
    instant_ends = [f.period_end for f in facts if f.period_type is PeriodType.INSTANT]
    return max(instant_ends) if instant_ends else None
