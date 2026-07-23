"""Citation verification: drop any finding whose quote isn't really in the filing.

This is the trust guarantee of the whole phase. The extractor is instructed to
quote verbatim, but instruction is not enforcement -- an LLM will occasionally
paraphrase, merge two sentences, or invent a plausible-sounding line. verify.py
checks each `quoted_span` against the exact anonymised text the model was shown
(from the evidence pack) and drops the finding if the quote cannot be found.

Matching is exact-substring first, then a whitespace-normalised retry (models
reflow spacing and line breaks), then a high-threshold token-overlap fallback for
minor punctuation differences. It is deliberately strict: a finding that only
"mostly" matches is dropped, because the point is that a human can click the
citation and read the exact words. A dropped finding is recorded, not silently
discarded -- an extractor that fabricates a lot is itself a signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from scout.research.evidence import EvidencePack
from scout.research.models import Finding

# A quote shorter than this is too generic to verify meaningfully ("in default")
# and too easy to match by accident; we still require it to appear, but flag it.
_MIN_QUOTE_CHARS = 12
_FUZZY_THRESHOLD = 0.90


@dataclass(slots=True)
class VerificationResult:
    verified: list[Finding] = field(default_factory=list)
    dropped: list[tuple[Finding, str]] = field(default_factory=list)
    """(finding, reason) for each finding whose citation could not be confirmed."""

    @property
    def verified_count(self) -> int:
        return len(self.verified)

    @property
    def dropped_count(self) -> int:
        return len(self.dropped)

    @property
    def fabrication_rate(self) -> float:
        total = self.verified_count + self.dropped_count
        return self.dropped_count / total if total else 0.0


def quote_in_text(quote: str, text: str) -> bool:
    """Whether `quote` appears in `text`: exact, then whitespace-normalised, then
    high token-overlap. The one matcher shared by finding verification here and
    the agent brief (`agent/brief.py`), so both enforce the same 'a human can
    click it and read the exact words' bar. A quote below the minimum length is
    too generic to verify and is rejected."""
    q = quote.strip()
    if len(q) < _MIN_QUOTE_CHARS:
        return False
    if q in text:
        return True
    if _normalize(q) in _normalize(text):
        return True
    return _token_overlap(q, _normalize(text)) >= _FUZZY_THRESHOLD


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _token_overlap(needle: str, haystack_norm: str) -> float:
    """Fraction of the quote's tokens that appear, in order-insensitive terms.
    A last-resort tolerance for punctuation/entity noise, not a license to
    paraphrase -- hence the 0.90 bar."""
    tokens = [t for t in re.findall(r"\w+", needle.casefold()) if t]
    if not tokens:
        return 0.0
    haystack_tokens = set(re.findall(r"\w+", haystack_norm))
    hits = sum(1 for t in tokens if t in haystack_tokens)
    return hits / len(tokens)


def verify_finding(finding: Finding, pack: EvidencePack) -> tuple[bool, str]:
    """Confirm one finding's quote appears in its cited filing.

    Returns (ok, reason). The accession must be one the pack actually holds, and
    the quote must be found in THAT filing's text -- a real quote attributed to
    the wrong filing is still wrong.
    """
    quote = finding.quoted_span.strip()
    if len(quote) < _MIN_QUOTE_CHARS:
        return False, f"quote too short to verify ({len(quote)} chars)"

    text = pack.texts_by_accession.get(finding.source_accession)
    if text is None:
        # The model cited an accession that wasn't in its evidence -- either a
        # hallucinated id or the wrong filing. Try the other filings before
        # failing, but record the mismatch.
        found_elsewhere = _find_in_any(quote, pack)
        if found_elsewhere:
            return False, f"quote found but attributed to wrong/absent accession {finding.source_accession!r}"
        return False, f"cited accession {finding.source_accession!r} not in evidence pack"

    if quote in text:
        return True, "exact match"
    if _normalize(quote) in _normalize(text):
        return True, "match after whitespace normalisation"
    if _token_overlap(quote, _normalize(text)) >= _FUZZY_THRESHOLD:
        return True, "high token-overlap match"
    return False, "quote not found in cited filing"


def _find_in_any(quote: str, pack: EvidencePack) -> bool:
    norm_quote = _normalize(quote)
    return any(norm_quote in _normalize(text) for text in pack.texts_by_accession.values())


def verify_findings(findings: list[Finding], pack: EvidencePack) -> VerificationResult:
    """Partition findings into verified and dropped. Order preserved."""
    result = VerificationResult()
    for finding in findings:
        ok, reason = verify_finding(finding, pack)
        if ok:
            result.verified.append(finding)
        else:
            result.dropped.append((finding, reason))
    return result
