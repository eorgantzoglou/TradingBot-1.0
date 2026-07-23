"""Entity-profile enrichment: identity, classification and filing-history facts.

Why this exists, field by field, is the whole point of the module:

  * SIC -> sector is what lets the screen drop financials and utilities wholesale
    (EV/EBIT is meaningless for a bank's balance sheet and a regulated utility's
    returns are set by rate case, not the market) and cohort everyone else
    against genuine peers. See cohorts.py.
  * former names are the shell-hijack tell: a microcap that changed its name a
    few months ago, took on a new "operating" story and started raising capital
    is the classic reverse-merger dilution pattern. name_changed_within_months
    surfaces exactly that recency.
  * NT 10-K / NT 10-Q ("notification of late filing") forms are the delinquency
    signal -- a company that keeps missing its own filing deadlines is a company
    whose numbers you cannot trust are current.

Two data paths. SEC filers get a full profile from the submissions API; every
other source (ESEF/EDINET/...) gets a minimal identity-only profile until its
own registry is wired in, because the screen degrades on missing fields rather
than failing. A malformed SEC payload degrades the same way: a bad fetch must
never crash a run over thousands of filers.

Storage shares the fundamentals DuckDB file. DuckDB permits only one read-write
handle per file, so `enrich` reads the entity list through a read-only
connection, closes it, and only then opens the read-write ProfileStore.
"""

from __future__ import annotations

import calendar
import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import duckdb

from scout.config import Config
from scout.data.http import HttpClient
from scout.screen.cohorts import is_excluded_sector, sic_to_sector
from scout.screen.models import EntityProfile, FormerName

logger = logging.getLogger(__name__)

# CIK is zero-padded to 10 digits in this endpoint's path -- "CIK0000066740.json".
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"

# "NT " prefixes the late-filing notice forms: NT 10-K, NT 10-Q, and their /A
# amendments. Matching the prefix rather than an exact set catches all of them.
_LATE_FORM_PREFIX = "NT "

# How recent an NT filing has to be, relative to the newest filing, to count as
# a *current* delinquency signal rather than ancient history.
_RECENT_LATE_WINDOW_MONTHS = 24


def submissions_url(cik: str | int) -> str:
    """The SEC submissions endpoint for a CIK, zero-padded to 10 digits."""
    return _SUBMISSIONS_URL.format(cik=str(cik).strip())


# ----------------------------------------------------------------------------
# Building a profile from a submissions document
# ----------------------------------------------------------------------------


def sec_profile_from_submissions(doc: object, entity_id: str) -> EntityProfile:
    """Build an EntityProfile from a parsed SEC submissions JSON document.

    Defensive by contract: any missing or malformed field degrades to
    None/empty rather than raising, and a payload that isn't even a dict still
    returns a valid, minimal SEC profile. The screen must survive a bad fetch.
    """
    if not isinstance(doc, dict):
        return EntityProfile(entity_id=entity_id, source="sec", country="US")

    sic = _as_str(doc.get("sic"))
    sector = sic_to_sector(sic)

    recent = _recent_filings(doc)
    forms = _as_str_list(recent.get("form"))
    filing_dates = [_parse_iso_date(v) for v in _as_list(recent.get("filingDate"))]
    # The arrays are newest-first, so index 0 is the most recent filing and the
    # natural "now" for every recency calculation below -- deterministic, unlike
    # date.today(), which is what makes these fields testable.
    reference = filing_dates[0] if filing_dates else None

    former_names = _former_names(doc)

    return EntityProfile(
        entity_id=entity_id,
        source="sec",
        name=_as_str(doc.get("name")),
        country="US",  # SEC EDGAR filers are, by definition, SEC registrants.
        sic=sic,
        sic_description=_as_str(doc.get("sicDescription")),
        sector=sector,
        is_excluded_sector=is_excluded_sector(sector),
        tickers=_as_str_tuple(doc.get("tickers")),
        exchanges=_as_str_tuple(doc.get("exchanges")),
        filer_category=_as_str(doc.get("category")),
        entity_type=_as_str(doc.get("entityType")),
        former_names=former_names,
        fiscal_year_end=_as_str(doc.get("fiscalYearEnd")),
        has_recent_late_filing=_recent_late_filing(forms, filing_dates, reference),
        name_changed_within_months=_name_changed_within_months(
            former_names, reference, _as_str(doc.get("name"))
        ),
        most_recent_form=forms[0] if forms else None,
        most_recent_filing_date=reference,
    )


def minimal_profile(
    entity_id: str, source: str, country: str | None, name: str | None
) -> EntityProfile:
    """Identity-only profile for non-SEC filers (ESEF/EDINET/...).

    Carries just enough for the screen to cohort the entity by country;
    everything the submissions API would supply is left None. Wired-in national
    registries can enrich these later.
    """
    return EntityProfile(entity_id=entity_id, source=source, country=country, name=name)


def _recent_filings(doc: dict) -> dict:
    """`filings.recent`, or an empty dict if the structure is absent/malformed."""
    filings = doc.get("filings")
    recent = filings.get("recent") if isinstance(filings, dict) else None
    return recent if isinstance(recent, dict) else {}


def _recent_late_filing(
    forms: list[str], filing_dates: list[date | None], reference: date | None
) -> bool | None:
    """Whether a recent NT (late-filing notice) form appears in the history.

    Returns None when there is no filing history at all -- "we could not check"
    is a different claim from "there are none", and the screen treats them
    differently. An NT form is only counted when we can confirm it is within the
    recency window; an undated one is skipped rather than assumed recent.
    """
    if not forms:
        return None

    now = reference or date.today()
    cutoff = _add_months(now, -_RECENT_LATE_WINDOW_MONTHS)
    for i, form in enumerate(forms):
        if not form.startswith(_LATE_FORM_PREFIX):
            continue
        filed = filing_dates[i] if i < len(filing_dates) else None
        if filed is not None and filed >= cutoff:
            return True
    return False


def _name_changed_within_months(
    former_names: tuple[FormerName, ...], reference: date | None, current_name: str | None
) -> int | None:
    """Months from the most recent GENUINE name change to the newest filing.

    None when there are no dated former names. A small number here is the
    shell-hijack recency signal the excludes screen keys off.

    SEC lists the CURRENT name inside formerNames with a `to` date that rolls
    forward to the latest filing (Equifax's only "former" name is "EQUIFAX INC"
    with to=today). Counting that as a same-day name change would false-exclude
    every such filer, so an entry whose name equals the current name is not a
    change and is ignored.
    """
    current = (current_name or "").strip().casefold()
    end_dates = [
        fn.to_date
        for fn in former_names
        if fn.to_date is not None and fn.name.strip().casefold() != current
    ]
    if not end_dates:
        return None
    now = reference or date.today()
    return _months_between(max(end_dates), now)


def _former_names(doc: dict) -> tuple[FormerName, ...]:
    raw = doc.get("formerNames")
    if not isinstance(raw, list):
        return ()
    names: list[FormerName] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = _as_str(entry.get("name"))
        if name is None:
            continue
        names.append(
            FormerName(
                name=name,
                from_date=_parse_iso_date(entry.get("from")),
                to_date=_parse_iso_date(entry.get("to")),
            )
        )
    return tuple(names)


# ----------------------------------------------------------------------------
# Small, total parsing helpers -- none of these ever raise on bad input.
# ----------------------------------------------------------------------------


def _parse_iso_date(value: object) -> date | None:
    """A calendar date from a SEC date field.

    SEC ships both plain dates ("2024-02-07") and full ISO datetimes
    ("2002-02-06T05:00:00.000Z"); we only need the date, and both start with
    the same 10-character prefix.
    """
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _as_str(value: object) -> str | None:
    if value is None:
        return None
    text = value.strip() if isinstance(value, str) else str(value)
    return text or None


def _as_list(value: object) -> list:
    return value if isinstance(value, list) else []


def _as_str_list(value: object) -> list[str]:
    return [v for v in _as_list(value) if isinstance(v, str)]


def _as_str_tuple(value: object) -> tuple[str, ...]:
    return tuple(_as_str_list(value))


def _days_in_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _add_months(d: date, months: int) -> date:
    """Shift a date by a (possibly negative) number of months, clamping the day
    to the target month's length so 31 Jan - 1 month is 28/29 Feb, not an error.
    """
    total = d.month - 1 + months
    year = d.year + total // 12
    month = total % 12 + 1
    return date(year, month, min(d.day, _days_in_month(year, month)))


def _months_between(earlier: date, later: date) -> int:
    """Whole calendar months from `earlier` to `later` (negative if reversed)."""
    months = (later.year - earlier.year) * 12 + (later.month - earlier.month)
    if later.day < earlier.day:
        months -= 1
    return months


# ----------------------------------------------------------------------------
# ProfileStore -- DuckDB persistence in the fundamentals file
# ----------------------------------------------------------------------------

# tickers/exchanges are JSON arrays; former_names a JSON array of {name,from,to}
# with dates as ISO strings. is_excluded_sector is always known (defaults False),
# so it is NOT NULL; the two derived flags are genuinely tri-state (True / False
# / not-determinable) and stay nullable so None round-trips faithfully.
_PROFILE_SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    entity_id                   VARCHAR PRIMARY KEY,
    source                      VARCHAR NOT NULL,
    name                        VARCHAR,
    country                     VARCHAR,
    sic                         VARCHAR,
    sic_description             VARCHAR,
    sector                      VARCHAR,
    is_excluded_sector          BOOLEAN NOT NULL,
    tickers                     VARCHAR NOT NULL,
    exchanges                   VARCHAR NOT NULL,
    filer_category              VARCHAR,
    entity_type                 VARCHAR,
    former_names                VARCHAR NOT NULL,
    fiscal_year_end             VARCHAR,
    has_recent_late_filing      BOOLEAN,
    name_changed_within_months  INTEGER,
    most_recent_form            VARCHAR,
    most_recent_filing_date     DATE
);
"""

# One source of truth for column order, so INSERT and every SELECT stay in sync.
_COLUMNS = (
    "entity_id, source, name, country, sic, sic_description, sector, "
    "is_excluded_sector, tickers, exchanges, filer_category, entity_type, "
    "former_names, fiscal_year_end, has_recent_late_filing, "
    "name_changed_within_months, most_recent_form, most_recent_filing_date"
)


class ProfileStore:
    """DuckDB storage for EntityProfiles, in the same file as FundamentalsStore.

    One read-write connection per file at a time (a DuckDB constraint), so the
    screen opens this and the fundamentals store BOTH read-only, and the writer
    -- `enrich` -- is the only read-write user.
    """

    def __init__(self, db_path: Path | str, *, read_only: bool = False) -> None:
        self._path = Path(db_path)
        if not read_only:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self._path), read_only=read_only)
        self._read_only = read_only

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ProfileStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def initialize(self) -> None:
        """Create the profiles table if absent. Safe to call on every open."""
        self._conn.execute(_PROFILE_SCHEMA)

    def put(self, profile: EntityProfile) -> None:
        """Upsert one profile, keyed on entity_id.

        Delete-then-insert, mirroring the fundamentals store: re-enriching an
        entity replaces its row rather than duplicating it.
        """
        self._conn.execute("DELETE FROM profiles WHERE entity_id = ?", [profile.entity_id])
        self._conn.execute(
            f"INSERT INTO profiles ({_COLUMNS}) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                profile.entity_id,
                profile.source,
                profile.name,
                profile.country,
                profile.sic,
                profile.sic_description,
                profile.sector,
                profile.is_excluded_sector,
                json.dumps(list(profile.tickers)),
                json.dumps(list(profile.exchanges)),
                profile.filer_category,
                profile.entity_type,
                _dump_former_names(profile.former_names),
                profile.fiscal_year_end,
                profile.has_recent_late_filing,
                profile.name_changed_within_months,
                profile.most_recent_form,
                profile.most_recent_filing_date,
            ],
        )

    def get(self, entity_id: str) -> EntityProfile | None:
        row = self._conn.execute(
            f"SELECT {_COLUMNS} FROM profiles WHERE entity_id = ?", [entity_id]
        ).fetchone()
        return _row_to_profile(row) if row is not None else None

    def all(self) -> list[EntityProfile]:
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM profiles ORDER BY entity_id"
        ).fetchall()
        return [_row_to_profile(r) for r in rows]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]

    def has(self, entity_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM profiles WHERE entity_id = ? LIMIT 1", [entity_id]
        ).fetchone()
        return row is not None


def _dump_former_names(former_names: tuple[FormerName, ...]) -> str:
    return json.dumps(
        [
            {
                "name": fn.name,
                "from": fn.from_date.isoformat() if fn.from_date else None,
                "to": fn.to_date.isoformat() if fn.to_date else None,
            }
            for fn in former_names
        ]
    )


def _load_former_names(raw: str) -> tuple[FormerName, ...]:
    return tuple(
        FormerName(
            name=entry["name"],
            from_date=_date_or_none(entry.get("from")),
            to_date=_date_or_none(entry.get("to")),
        )
        for entry in json.loads(raw)
    )


def _date_or_none(value: object) -> date | None:
    return date.fromisoformat(value) if isinstance(value, str) else None


def _row_to_profile(row: tuple) -> EntityProfile:
    (
        entity_id,
        source,
        name,
        country,
        sic,
        sic_description,
        sector,
        is_excluded,
        tickers_json,
        exchanges_json,
        filer_category,
        entity_type,
        former_names_json,
        fiscal_year_end,
        has_recent_late_filing,
        name_changed_within_months,
        most_recent_form,
        most_recent_filing_date,
    ) = row
    return EntityProfile(
        entity_id=entity_id,
        source=source,
        name=name,
        country=country,
        sic=sic,
        sic_description=sic_description,
        sector=sector,
        is_excluded_sector=bool(is_excluded),
        tickers=tuple(json.loads(tickers_json)),
        exchanges=tuple(json.loads(exchanges_json)),
        filer_category=filer_category,
        entity_type=entity_type,
        former_names=_load_former_names(former_names_json),
        fiscal_year_end=fiscal_year_end,
        has_recent_late_filing=has_recent_late_filing,
        name_changed_within_months=name_changed_within_months,
        most_recent_form=most_recent_form,
        most_recent_filing_date=most_recent_filing_date,
    )


# ----------------------------------------------------------------------------
# enrich -- the batch orchestration
# ----------------------------------------------------------------------------


@dataclass(slots=True)
class EnrichResult:
    enriched: int = 0
    """SEC filers fetched and profiled."""

    skipped_existing: int = 0
    """Already had a profile and reenrich was off."""

    minimal: int = 0
    """Non-SEC filers stored with an identity-only profile."""

    failed: int = 0
    """SEC fetches that errored (404, network) -- collected, never fatal."""

    errors: list[tuple[str, str]] = field(default_factory=list)
    """(entity_id, message) for each failure, for the CLI to surface."""


@dataclass(frozen=True, slots=True)
class _EntityRow:
    entity_id: str
    source: str
    country: str | None
    name: str | None


async def enrich(config: Config, *, reenrich: bool = False, limit: int | None = None) -> EnrichResult:
    """Build and store a profile for every entity in the fundamentals store.

    Sequencing matters: the entity list is read through a read-only connection
    that is CLOSED before the read-write ProfileStore opens, because DuckDB
    allows only one read-write handle to the shared file. A single CIK's 404 or
    network error is collected and the run continues.
    """
    result = EnrichResult()

    entities = _read_entity_rows(config.db_path)
    if limit is not None:
        entities = entities[:limit]
    if not entities:
        return result

    async with HttpClient(user_agent=config.user_agent) as http:
        with ProfileStore(config.db_path) as store:
            store.initialize()
            for entity in entities:
                if not reenrich and store.has(entity.entity_id):
                    result.skipped_existing += 1
                    continue
                await _enrich_one(http, store, entity, result)
    return result


async def _enrich_one(
    http: HttpClient, store: ProfileStore, entity: _EntityRow, result: EnrichResult
) -> None:
    if entity.source != "sec":
        # No submissions API for non-SEC sources yet: store identity + country
        # so the screen can still cohort them.
        store.put(minimal_profile(entity.entity_id, entity.source, entity.country, entity.name))
        result.minimal += 1
        return

    try:
        doc = await fetch_submissions(http, entity.entity_id)
    except Exception as exc:
        # One filer's HTTP/network failure must not abort a run over thousands.
        # Logged AND collected -- never silently swallowed.
        logger.warning("enrich: submissions fetch failed for %s: %s", entity.entity_id, exc)
        result.failed += 1
        result.errors.append((entity.entity_id, str(exc)))
        return

    store.put(sec_profile_from_submissions(doc, entity.entity_id))
    result.enriched += 1


async def fetch_submissions(http: HttpClient, cik: str | int) -> object:
    """Fetch and parse the SEC submissions document for a CIK."""
    return await http.get_json(submissions_url(cik))


def _read_entity_rows(db_path: Path) -> list[_EntityRow]:
    """Distinct entities to enrich, read through a short-lived read-only handle.

    Reads straight from the snapshots table rather than adding a method to
    FundamentalsStore: this is a pure read, and keeping it self-contained is
    what lets the connection close before the read-write ProfileStore opens.
    """
    if not db_path.exists():
        return []
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = conn.execute(
            """
            SELECT entity_id,
                   any_value(entity_source),
                   any_value(entity_country),
                   any_value(entity_name)
            FROM snapshots
            GROUP BY entity_id
            ORDER BY entity_id
            """
        ).fetchall()
    except duckdb.Error:
        # No snapshots table yet (nothing harvested) -- nothing to enrich.
        return []
    finally:
        conn.close()
    return [_EntityRow(entity_id=r[0], source=r[1], country=r[2], name=r[3]) for r in rows]
