"""UK -- Companies House.

Two unrelated capabilities live in this one module because they share a
publisher, nothing else:

1. **Bulk daily accounts** (`CompaniesHouseSource.list_documents`/`fetch`).
   No API key needed. This is the time-critical one -- see PLAN.md 1.3 and 2.

       ################################################################
       # THESE FILES ARE PURGED 60 DAYS AFTER PUBLICATION.            #
       # Every day this collector doesn't run is a day of history     #
       # permanently and unrecoverably lost. Nothing about this can   #
       # be backfilled later -- there is no archive to backfill from. #
       ################################################################

   Files are published Tuesday-Saturday; Tuesday's file covers accounts filed
   Saturday, Sunday and Monday. So a missing file on a Sunday or Monday (and
   sometimes bank holidays) is completely normal and must not raise -- only a
   genuine fetch failure (5xx, timeout) should.

   We store the whole ZIP as one document per day (`accounts-bulk-YYYY-MM-DD`)
   rather than exploding it into per-company entries. The archive keeps
   payloads verbatim by design (see archive.py); a parser can open the zip
   later, and re-exploding it costs us nothing we can't redo, whereas the raw
   bytes existing at all is the thing that can't be redone once purged.

2. **The REST API** (`fetch_company_profile`, `fetch_filing_history`). Needs
   `SourceCredentials.companies_house_key`, sent as HTTP Basic auth username
   with an empty password (no bearer tokens here). Rate limit is 600 requests
   per 5 minutes and Companies House bans repeat offenders -- `HttpClient`
   already enforces this host's limit, so these helpers just need to go
   through it, never around it.

-----------------------------------------------------------------------------
THE TRAP (PLAN.md 1.4): the bulk accounts in (1) are FRS 102/105 **statutory
entity** accounts -- the UK subsidiary or parent company in isolation. A
listed group's annual report, by contrast, is UK-adopted IFRS at the
**consolidated group** level. Different accounting basis, different
consolidation scope, different numbers for what looks like the same line
item. These must never be merged into one field downstream; keep them as
distinct facts with distinct provenance.
-----------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date
from typing import Any

from scout.data.http import HttpClient
from scout.data.sources.base import DocumentRef, RawDocument

BULK_BASE = "http://download.companieshouse.gov.uk"
REST_BASE = "https://api.company-information.service.gov.uk"


class CompaniesHouseSource:
    """Harvests the daily bulk accounts ZIP. No credentials required.

    The REST helpers (`fetch_company_profile`, `fetch_filing_history`) live
    as module-level functions below, not on this class, because they need a
    key this class doesn't hold -- `available()` here is about the bulk path
    only.
    """

    name = "companies_house"

    def __init__(self, http: HttpClient) -> None:
        self._http = http

    def available(self) -> bool:
        # The bulk daily-accounts path needs no key at all, so this source is
        # always usable. The REST helpers below additionally need
        # SourceCredentials.companies_house_key -- that's the caller's concern
        # when it chooses to call them, not something `available()` gates.
        return True

    async def list_documents(self, day: date) -> AsyncIterator[DocumentRef]:
        """The single bulk ZIP published for `day`, if one exists.

        Files publish Tuesday-Saturday covering the previous day(s); a 404 on
        Sunday/Monday/holidays is the expected, normal case, not an error. We
        HEAD first rather than pulling the whole (large) ZIP just to find out
        whether today's file exists.
        """
        filename = f"Accounts_Bulk_Data-{day.isoformat()}.zip"
        url = f"{BULK_BASE}/{filename}"

        response = await self._http.request("HEAD", url)
        if response.status_code == 404:
            return  # no file for this day -- normal on Sun/Mon/holidays
        response.raise_for_status()

        yield DocumentRef(
            source=self.name,
            doc_id=f"accounts-bulk-{day.isoformat()}",
            filing_date=day,
            form_type="accounts_bulk",
            title=f"Companies House daily accounts bulk data for {day.isoformat()}",
            url=url,
            meta={"filename": filename},
        )

    async def fetch(self, ref: DocumentRef) -> RawDocument:
        payload = await self._http.get_bytes(ref.url)
        filename = ref.meta.get("filename") or f"{ref.doc_id}.zip"
        return RawDocument(ref=ref, payload=payload, filename=filename, content_type="application/zip")


async def fetch_company_profile(http: HttpClient, key: str, company_number: str) -> Any:
    """GET /company/{company_number}. HTTP Basic auth: key as username, empty password."""
    url = f"{REST_BASE}/company/{company_number}"
    return await http.get_json(url, auth=(key, ""))


async def fetch_filing_history(http: HttpClient, key: str, company_number: str) -> Any:
    """GET /company/{company_number}/filing-history. Same auth as above."""
    url = f"{REST_BASE}/company/{company_number}/filing-history"
    return await http.get_json(url, auth=(key, ""))
