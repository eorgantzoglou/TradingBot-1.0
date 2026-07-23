"""Build an evidence pack for one candidate from its archived filings. No LLM.

Filings are large and mostly boilerplate, so we do not hand a whole 13 MB
submission to a model. We extract the primary document's narrative text and pull
the passages most likely to carry a red flag -- going-concern language, toxic
convertible terms, related-party dealings, dilution, reverse splits -- plus the
computed metrics, into a compact pack. This is retrieval, and it is where the
token budget is won or lost.

Two deliberate choices:

  * ANONYMISATION. The company name and tickers are redacted to placeholders
    before the pack reaches the judgment stage. Glasserman & Lin (2023) found
    anonymised inputs actually outperform, removing both look-ahead and a
    "distraction" effect where the model's stored knowledge of a company
    interferes. Dates are kept -- a dilution finding needs them, and name/ticker
    are the strongest memorisation levers anyway.
  * VERIFIABILITY. The full anonymised text per accession is retained in the
    pack so verify.py can confirm every quoted span the model returns is a real
    substring of what it was shown. A quote that isn't there is a fabrication.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from edgar.sgml import FilingSGML

from scout.metrics.report import MetricReport

# Red-flag categories -> the patterns that flag a passage worth extracting. These
# are a RETRIEVAL heuristic, not a verdict: they decide what the LLM reads, and
# the LLM (then verify.py) decides what is real. Kept deliberately broad -- a
# false positive costs a few tokens, a false negative hides a fraud.
_RED_FLAG_PATTERNS: dict[str, str] = {
    "going_concern": r"going concern|substantial doubt|ability to continue",
    "toxic_convertible": r"convertible|conversion price|variable conversion|"
    r"lowest.{0,40}(trading|closing) price|look-?back|discount to market",
    "reverse_split": r"reverse (stock )?split|share consolidation",
    "related_party": r"related part(y|ies)|affiliate transaction|officer loan|due to (officer|director)",
    "dilution": r"at-the-market|equity line|ATM (offering|program)|dilut|"
    r"shares? (issued|to be issued)|registered direct|standby equity",
    "default": r"in default|event of default|forbearance|cross-default",
    "auditor": r"change in (registrant'?s )?certifying accountant|dismissed.{0,40}auditor|"
    r"resignation of.{0,20}auditor|material weakness",
    "litigation": r"material (legal )?proceeding|SEC (investigation|subpoena)|class action",
}

_MAX_EXCERPT_CHARS = 900
_CONTEXT_BEFORE = 250
_CONTEXT_AFTER = 500
_MAX_EXCERPTS_PER_CATEGORY = 4


@dataclass(frozen=True, slots=True)
class Excerpt:
    source_accession: str
    category: str
    text: str
    char_start: int
    """Offset into the (anonymised) full text for this accession -- a stable
    locator so a human, or a re-run, can find the passage again."""


@dataclass(slots=True)
class EvidencePack:
    entity_id: str
    excerpts: list[Excerpt] = field(default_factory=list)
    texts_by_accession: dict[str, str] = field(default_factory=dict)
    """The full anonymised narrative per accession. verify.py checks quoted spans
    against these; the extractor is shown the excerpts (or, for short filings,
    the whole thing)."""

    metrics_block: str = ""
    """Pre-formatted, code-computed numbers to inject into prompts. The model
    never recomputes these."""

    anonymized: bool = True
    redactions: dict[str, str] = field(default_factory=dict)
    """placeholder -> what it replaced, so the memo writer can restore identity."""

    warnings: list[str] = field(default_factory=list)

    @property
    def has_narrative(self) -> bool:
        return bool(self.texts_by_accession)


def extract_primary_text(filing_bytes: bytes) -> str:
    """The primary document's narrative as clean text, or "" if not extractable.

    Offline, from archived bytes. The primary document is the attachment whose
    type is the form itself (the 10-K/10-Q HTML), not an exhibit or the XBRL.
    """
    try:
        sgml = FilingSGML.from_text(filing_bytes.decode("latin-1"))
    except Exception:
        return ""

    form = sgml.form or ""
    primary = None
    for attachment in sgml.attachments:
        doc_type = getattr(attachment, "document_type", "") or ""
        name = str(getattr(attachment, "document", "") or "")
        if doc_type == form and name.lower().endswith((".htm", ".html")):
            primary = attachment
            break
    if primary is None:
        # Fall back to the first HTML attachment -- some filers mis-tag the type.
        for attachment in sgml.attachments:
            name = str(getattr(attachment, "document", "") or "")
            if name.lower().endswith((".htm", ".html")):
                primary = attachment
                break
    if primary is None:
        return ""

    content = getattr(primary, "content", None)
    if not isinstance(content, str) or len(content) < 100:
        return ""
    return _html_to_text(content)


def _html_to_text(html: str) -> str:
    """Strip tags and normalise whitespace. Deliberately simple: we want
    readable prose for an LLM and a stable string for span verification, not a
    faithful DOM."""
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&#160;", " ")
    text = text.replace("&amp;", "&").replace("&#8217;", "'").replace("&#8216;", "'")
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _anonymize(text: str, redactions: dict[str, str]) -> str:
    """Replace each redaction target with its placeholder, longest first so a
    ticker inside a name is not half-redacted.

    Word boundaries are mandatory: a short ticker like "CONC" would otherwise
    redact the "conc" inside "concern" -> "going [TICKER]ern", which both
    corrupts the text and, worse, breaks span verification of legitimate quotes.
    """
    for placeholder, target in sorted(redactions.items(), key=lambda kv: -len(kv[1])):
        if target:
            text = re.sub(rf"\b{re.escape(target)}\b", placeholder, text, flags=re.IGNORECASE)
    return text


def _select_excerpts(accession: str, text: str) -> list[Excerpt]:
    """Windows around red-flag matches, deduplicated by overlap."""
    excerpts: list[Excerpt] = []
    for category, pattern in _RED_FLAG_PATTERNS.items():
        seen_starts: list[int] = []
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start = max(0, match.start() - _CONTEXT_BEFORE)
            # Skip a hit that falls inside a window we already took for this
            # category -- the same going-concern note matches five times.
            if any(abs(start - s) < _MAX_EXCERPT_CHARS for s in seen_starts):
                continue
            seen_starts.append(start)
            end = min(len(text), match.end() + _CONTEXT_AFTER)
            excerpts.append(
                Excerpt(
                    source_accession=accession,
                    category=category,
                    text=text[start:end].strip(),
                    char_start=start,
                )
            )
            if len(seen_starts) >= _MAX_EXCERPTS_PER_CATEGORY:
                break
    return excerpts


def build_evidence_pack(
    entity_id: str,
    filings: list[tuple[str, bytes]],
    *,
    report: MetricReport | None = None,
    company_name: str | None = None,
    tickers: tuple[str, ...] = (),
    anonymize: bool = True,
) -> EvidencePack:
    """Assemble the pack for one candidate.

    `filings` is [(accession, raw_bytes), ...] -- typically the candidate's most
    recent periodic report plus any recent 8-K/S-1. Non-SEC filings that yield no
    narrative simply contribute nothing, with a warning.
    """
    redactions: dict[str, str] = {}
    if anonymize:
        if company_name:
            redactions["[COMPANY]"] = company_name
        for i, ticker in enumerate(tickers):
            redactions[f"[TICKER{i or ''}]"] = ticker

    pack = EvidencePack(entity_id=entity_id, anonymized=anonymize, redactions=redactions)

    for accession, raw in filings:
        text = extract_primary_text(raw)
        if not text:
            pack.warnings.append(f"{accession}: no narrative text extracted (no primary HTML document)")
            continue
        if anonymize:
            text = _anonymize(text, redactions)
        pack.texts_by_accession[accession] = text
        pack.excerpts.extend(_select_excerpts(accession, text))

    if not pack.has_narrative:
        pack.warnings.append(
            "no narrative evidence available -- research limited to computed metrics "
            "(common for non-SEC filings, whose text is not yet extracted)"
        )

    pack.metrics_block = _format_metrics(report)
    return pack


def _format_metrics(report: MetricReport | None) -> str:
    """A compact, code-computed metric block for prompt injection. The model is
    shown these numbers and told not to recompute or invent any."""
    if report is None:
        return "No computed metrics available."
    lines = [f"Period: {report.period_end} ({report.fiscal_period or '?'}), currency {report.currency or '?'}"]
    for name, metric in report.metrics.items():
        if metric.ok and metric.value is not None:
            lines.append(f"  {name} = {metric.value:.4g} [{metric.basis}]")
    if len(lines) == 1:
        lines.append("  (no metrics computed)")
    return "\n".join(lines)
