"""SEC EDGAR (US).

Data here is US federal government work product: public domain, no copyright,
no redistribution restriction. That is not true of any other source in this
project (see `esef.py`'s licence note, and PLAN.md section 2) -- worth keeping
in mind if this archive is ever shared rather than only queried.

Listing is day-indexed off the EDGAR daily index
(`daily-index/{year}/QTR{q}/form.{YYYYMMDD}.idx`), which is the fixed-width,
form-sorted variant. There is also a `index.json` for the same day; the .idx
file is preferred here because its column layout is stable and it is what
every existing third-party EDGAR parser is built against, so deviations are
easy to notice.

Weekends, federal holidays and the pre-EDGAR era all have no index file for a
given day -- SEC returns 404, which is a normal empty day, not an error (see
`Source.list_documents`'s docstring on why that distinction matters: an empty
day and a broken request must not look the same, but a *missing* index file
and an *empty* one are the same thing here).
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import date

from scout.data.http import HttpClient
from scout.data.sources.base import DocumentRef, RawDocument

logger = logging.getLogger(__name__)

_ARCHIVES = "https://www.sec.gov/Archives"
_SEC_WWW = "https://www.sec.gov"

# The forms this project actually reads. EDGAR carries several hundred form
# types (most of it fund/proxy noise for our purposes); this is the subset
# that matters for equity research -- annuals/quarterlies, foreign-private and
# Canadian-MJDS equivalents, capital-raise and registration statements,
# insider transactions, and the delinquency/deregistration forms that feed the
# hard-excludes screen in PLAN.md section 5. Configurable because a future
# stage may want more (e.g. 13F for institutional-ownership context).
DEFAULT_FORM_TYPES: frozenset[str] = frozenset(
    {
        "10-K",
        "10-Q",
        "8-K",
        "20-F",
        "40-F",
        "S-1",
        "S-3",
        "424B3",
        "424B5",
        "3",
        "4",
        "5",
        "NT 10-K",
        "NT 10-Q",
        "25",
        "15-12B",
        "15-12G",
    }
)

# The idx format is fixed-width in principle, but exact column widths have
# drifted across eras, and header-position slicing breaks the moment a
# company name runs one character long. A field-separator regex (2+ spaces
# between columns) is robust to that and to the header itself changing, as
# long as the columns are still ordered form/company/cik/date/file. Lines that
# don't match -- blank separators, truncated rows, anything SEC's generator
# has ever emitted that we haven't seen -- are skipped and counted rather than
# raising, per the Source contract.
#
# The Date Filed column is undashed YYYYMMDD in the live files
# ("20260721"). Dashes are accepted too, because the historical archives are
# not uniform and it costs one character to tolerate both.
_ROW_RE = re.compile(
    r"^(?P<form>\S(?:.*\S)?)\s{2,}"
    r"(?P<company>\S(?:.*\S)?)\s{2,}"
    r"(?P<cik>\d{1,10})\s{2,}"
    r"(?P<date>\d{4}-?\d{2}-?\d{2})\s{2,}"
    r"(?P<file>\S+)\s*$"
)

_HEADER_SENTINEL = "Form Type"

# EDGAR accession numbers show up dashed (index files, the UI) and undashed
# (some JSON APIs, directory names) depending on where they came from.
# Normalizing once here means `doc_id` is stable regardless of source.
_ACCESSION_RE = re.compile(r"^(\d{10})-?(\d{2})-?(\d{6})$")


def normalize_accession(raw: str) -> str:
    """Canonical dashed accession number, e.g. "0001234567-26-000012"."""
    match = _ACCESSION_RE.match(raw.strip())
    if not match:
        raise ValueError(f"not a recognizable SEC accession number: {raw!r}")
    return "-".join(match.groups())


def _accession_from_file_name(file_name: str) -> str:
    """The idx "File Name" column is a path like
    "edgar/data/1234567/0001234567-26-000012.txt"; the accession is the
    basename without extension."""
    base = file_name.rsplit("/", 1)[-1]
    if base.endswith(".txt"):
        base = base[: -len(".txt")]
    return base


@dataclass(frozen=True, slots=True)
class _IdxRow:
    form_type: str
    company: str
    cik: str
    filed: date
    file_name: str


def _parse_idx(text: str, *, day: date) -> tuple[list[_IdxRow], int]:
    """Parse the body of a form.*.idx file.

    Returns the successfully parsed rows and a count of lines skipped as
    malformed. A day with zero index rows because the header itself is
    unrecognizable is treated differently from a day with no filings at all:
    the former means SEC changed the format underneath us and is raised
    loudly, the latter (caught earlier, via the 404) is a normal empty day.
    """
    lines = text.splitlines()

    header_index = next((i for i, line in enumerate(lines) if line.startswith(_HEADER_SENTINEL)), None)
    if header_index is None:
        raise ValueError(
            f"SEC daily index for {day} has no recognizable header "
            f"(expected a line starting with {_HEADER_SENTINEL!r}); the .idx format may have changed."
        )
    # The header WRAPS: "Form Type  Company Name  CIK" on one line and
    # "Date Filed  File Name" on the next, with the dashed rule after both. So
    # scan forward for the separator rather than assuming it is adjacent --
    # assuming adjacency rejects every real file SEC publishes.
    separator_index = next(
        (
            i
            for i in range(header_index + 1, min(header_index + 6, len(lines)))
            if lines[i].lstrip().startswith("---")
        ),
        None,
    )
    if separator_index is None:
        raise ValueError(
            f"SEC daily index for {day}: found the header but no dashed separator line "
            "within the following 5 lines; the .idx format may have changed."
        )

    rows: list[_IdxRow] = []
    skipped = 0
    for line in lines[separator_index + 1 :]:
        if not line.strip():
            continue
        match = _ROW_RE.match(line)
        if not match:
            skipped += 1
            continue
        try:
            # date.fromisoformat accepts both "20260721" and "2026-07-21" on
            # Python 3.11+, which is why the regex tolerates either.
            filed = date.fromisoformat(match.group("date"))
        except ValueError:
            skipped += 1
            continue
        rows.append(
            _IdxRow(
                form_type=match.group("form").strip(),
                company=match.group("company").strip(),
                cik=match.group("cik"),
                filed=filed,
                file_name=match.group("file"),
            )
        )
    return rows, skipped


class SecSource:
    """`Source` implementation for SEC EDGAR."""

    name = "sec"

    def __init__(self, http: HttpClient, *, form_types: frozenset[str] | None = None) -> None:
        self._http = http
        self._form_types = form_types if form_types is not None else DEFAULT_FORM_TYPES

    def available(self) -> bool:
        # No API key: the only credential EDGAR requires is the descriptive
        # User-Agent, which HttpClient refuses to construct without (see
        # http.py). If we got this far, we're good to go.
        return True

    async def list_documents(self, day: date) -> AsyncIterator[DocumentRef]:
        quarter = (day.month - 1) // 3 + 1
        # Note the "edgar" path segment: the daily index lives under
        # /Archives/edgar/daily-index/, while the File Name column inside the
        # index is already relative to /Archives/ and carries its own "edgar/"
        # prefix. Getting this wrong yields a 404 on every weekday, which looks
        # exactly like a quiet weekend -- hence the warning below.
        url = f"{_ARCHIVES}/edgar/daily-index/{day.year}/QTR{quarter}/form.{day:%Y%m%d}.idx"

        response = await self._http.get(url)
        if response.status_code == 404:
            # Weekends, federal holidays and pre-EDGAR dates genuinely have no
            # index. On a business day a 404 means our URL or SEC's layout is
            # wrong, and silently reporting an empty day would lose history we
            # cannot backfill.
            if day.weekday() < 5:
                logger.warning(
                    "sec: no daily index for %s (a weekday) at %s -- a federal holiday, "
                    "or EDGAR's layout changed. Verify the URL before trusting an empty day.",
                    day,
                    url,
                )
            return
        response.raise_for_status()

        # Older idx files carry Latin-1 punctuation in company names; decoding
        # permissively beats raising the whole day over a handful of bytes.
        text = response.content.decode("latin-1")
        rows, skipped = _parse_idx(text, day=day)
        if skipped:
            logger.warning("sec: skipped %d malformed daily-index line(s) for %s", skipped, day)

        for row in rows:
            if row.form_type not in self._form_types:
                continue
            try:
                doc_id = normalize_accession(_accession_from_file_name(row.file_name))
            except ValueError:
                logger.warning(
                    "sec: could not derive an accession number from file name %r on %s",
                    row.file_name,
                    day,
                )
                continue
            yield DocumentRef(
                source=self.name,
                doc_id=doc_id,
                filing_date=row.filed,
                form_type=row.form_type,
                title=row.company,
                url=f"{_ARCHIVES}/{row.file_name}",
                entity={"cik": str(int(row.cik))},
            )

    async def fetch(self, ref: DocumentRef) -> RawDocument:
        # Refs from our own list_documents always carry a url pointing at the
        # full submission text file; the fallback below exists for refs built
        # by hand (tests, or a future backfill from a different SEC listing).
        url = ref.url or self._submission_url(ref)
        payload = await self._http.get_bytes(url)
        return RawDocument(
            ref=ref,
            payload=payload,
            filename=f"{ref.doc_id}.txt",
            content_type="text/plain",
        )

    def _submission_url(self, ref: DocumentRef) -> str:
        cik = ref.entity.get("cik")
        if not cik:
            raise ValueError(f"cannot fetch {ref.doc_id}: ref has no url and no cik in entity")
        accession_nodash = ref.doc_id.replace("-", "")
        return f"{_ARCHIVES}/edgar/data/{int(cik)}/{accession_nodash}/{ref.doc_id}.txt"


async def fetch_company_tickers(http: HttpClient) -> list[dict]:
    """CIK <-> ticker <-> company-name mapping, refreshed a few times a day by
    SEC. The universe/entity-resolution layer needs this to join EDGAR filers
    against GLEIF/OpenFIGI without guessing tickers from company names.

    Surprisingly, the payload is not a JSON array: it's an object keyed by
    stringified indices ("0", "1", "2", ...) for no documented reason. This
    flattens that into a plain list of {cik, ticker, title}.
    """
    payload = await http.get_json(f"{_SEC_WWW}/files/company_tickers.json")
    if not isinstance(payload, dict):
        raise ValueError("unexpected company_tickers.json shape: expected a JSON object")
    return [
        {
            "cik": str(row.get("cik_str", "")),
            "ticker": row.get("ticker"),
            "title": row.get("title"),
        }
        for row in payload.values()
        if isinstance(row, dict)
    ]
