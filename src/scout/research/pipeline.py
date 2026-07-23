"""The research pipeline: one screened candidate -> a cited memo.

Chains the stages in the only order that keeps the trust guarantee intact:

    evidence  ->  extract  ->  VERIFY citations  ->  debate  ->  memo

Verification sits BEFORE the debate on purpose. The bull, bear and skeptic all
argue from `verified_findings` only, so an unquotable claim the extractor
invented is gone before it can influence the debate or the memo. And the verdict
is decided in code (`decide_verdict`), not by the memo writer -- the model
phrases the case, the gate stays deterministic (design rule 1).

Candidates are researched concurrently with a bounded `asyncio.gather`, the same
shape the harness uses everywhere: one slow or failing filing never blocks the
rest of the watchlist.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from scout.config import Config
from scout.data.archive import Archive
from scout.fundamentals.store import FundamentalsStore
from scout.harness.protocol import Effort, LLMClient
from scout.metrics.base import MarketData
from scout.metrics.report import compute_metrics
from scout.research.analysts import run_debate
from scout.research.evidence import build_evidence_pack
from scout.research.extract import extract_findings
from scout.research.memo import write_memo
from scout.research.models import AnalystView, Finding, ResearchMemo, SkepticVerdict, Verdict
from scout.research.verify import verify_findings
from scout.screen.profile import ProfileStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ResearchReport:
    entity_id: str
    name: str | None
    memo: ResearchMemo
    verified_findings: list[Finding]
    dropped_citations: list[tuple[Finding, str]]
    bull: AnalystView
    bear: AnalystView
    skeptic: SkepticVerdict
    warnings: list[str] = field(default_factory=list)

    @property
    def vetoed(self) -> bool:
        return self.memo.verdict == Verdict.VETO

    @property
    def fabrication_rate(self) -> float:
        total = len(self.verified_findings) + len(self.dropped_citations)
        return len(self.dropped_citations) / total if total else 0.0


async def research_candidate(
    pack,  # EvidencePack
    *,
    client: LLMClient,
    skeptic_client: LLMClient | None = None,
    name: str | None = None,
    effort: Effort | None = None,
) -> ResearchReport:
    """Run the full pipeline for one candidate whose evidence pack is built.

    `skeptic_client` defaults to `client`, but the plan recommends a different
    model family there for independence -- the signature keeps that option open.
    """
    skeptic_client = skeptic_client or client

    extraction = await extract_findings(client, pack, effort=effort)

    # The linchpin: drop any finding whose quote is not really in the filing,
    # BEFORE it can reach the debate or the memo.
    verification = verify_findings(extraction.findings, pack)
    if verification.dropped:
        logger.info(
            "%s: dropped %d/%d findings with unverifiable citations",
            pack.entity_id,
            verification.dropped_count,
            verification.verified_count + verification.dropped_count,
        )

    bull, bear, skeptic = await run_debate(
        client, client, skeptic_client, pack, verification.verified, effort=effort
    )

    memo = await write_memo(
        client, pack, verification.verified, bull, bear, skeptic, effort=effort
    )

    warnings = list(pack.warnings)
    if verification.fabrication_rate > 0.3:
        warnings.append(
            f"high citation-failure rate ({verification.fabrication_rate:.0%}): the extractor "
            "invented quotes for this candidate -- treat its findings with extra caution"
        )

    return ResearchReport(
        entity_id=pack.entity_id,
        name=name,
        memo=memo,
        verified_findings=verification.verified,
        dropped_citations=verification.dropped,
        bull=bull,
        bear=bear,
        skeptic=skeptic,
        warnings=warnings,
    )


def gather_filings_for_entity(archive: Archive, entity_id: str, *, max_filings: int = 3) -> list[tuple[str, bytes]]:
    """The archived SEC filings for one entity, most recent first.

    Matches on the CIK the harvest recorded. Capped, because the most recent
    periodic report plus a recent 8-K or two is enough evidence and each filing
    costs tokens to read.
    """
    matches = [
        stored
        for stored in archive.iter_manifest(source="sec")
        if stored.entity.get("cik") == entity_id
    ]
    # Most recent first by filing date, falling back to harvest day.
    matches.sort(key=lambda s: (s.filing_date or "", s.harvest_day), reverse=True)

    filings: list[tuple[str, bytes]] = []
    for stored in matches[:max_filings]:
        try:
            filings.append((stored.doc_id, archive.read_payload(stored)))
        except OSError as exc:
            logger.warning("could not read %s: %s", stored.path, exc)
    return filings


async def research_entities(
    config: Config,
    entity_ids: list[str],
    *,
    client: LLMClient,
    skeptic_client: LLMClient | None = None,
    market_data: dict[str, MarketData] | None = None,
    concurrency: int | None = None,
    effort: Effort | None = None,
) -> list[ResearchReport]:
    """Research a list of entities, concurrently and fault-tolerantly."""
    market_data = market_data or {}
    archive = Archive(config.archive_dir)
    semaphore = asyncio.Semaphore(concurrency or config.concurrency)

    with FundamentalsStore(config.db_path, read_only=True) as store:
        profiles = _open_profiles(config)
        packs: list[tuple[str, str | None, object]] = []
        for entity_id in entity_ids:
            snapshots = store.snapshots_for_entity(entity_id)
            if not snapshots:
                continue
            profile = profiles.get(entity_id) if profiles else None
            name = (profile.name if profile else None) or snapshots[0].entity.name
            report = compute_metrics(snapshots, market=market_data.get(entity_id))
            filings = gather_filings_for_entity(archive, entity_id)
            pack = build_evidence_pack(
                entity_id,
                filings,
                report=report,
                company_name=name,
                tickers=tuple(profile.tickers) if profile else (),
            )
            packs.append((entity_id, name, pack))
        if profiles and hasattr(profiles, "close"):
            profiles.close()

    async def run_one(entity_id: str, name: str | None, pack) -> ResearchReport | None:  # type: ignore[no-untyped-def]
        async with semaphore:
            try:
                return await research_candidate(
                    pack, client=client, skeptic_client=skeptic_client, name=name, effort=effort
                )
            except Exception as exc:
                logger.warning("research failed for %s: %s", entity_id, exc)
                return None

    reports = await asyncio.gather(*(run_one(eid, name, pack) for eid, name, pack in packs))
    return [r for r in reports if r is not None]


def _open_profiles(config: Config):  # type: ignore[no-untyped-def]
    if not config.db_path.exists():
        return None
    try:
        store = ProfileStore(config.db_path, read_only=True)
        store.count()
        return store
    except Exception:
        return None
