"""Phase 5: cited LLM research over screened candidates.

The harness (phase 1) meets the screen (phase 4) here. For each candidate that
survived the screen, the pipeline builds an evidence pack from its filings,
extracts red-flag findings each anchored to a verbatim quote, runs a bull/bear
debate and an adversarial skeptic, verifies every citation against the source,
and writes a cited memo. The LLM can VETO a candidate but never promote or rank
one (design rule 1), and every number in the output is code-computed, not
generated.
"""

from scout.research.models import (
    AnalystView,
    ExtractionResult,
    Finding,
    FindingCategory,
    ResearchMemo,
    Severity,
    SkepticVerdict,
    Stance,
    Verdict,
)

__all__ = [
    "AnalystView",
    "ExtractionResult",
    "Finding",
    "FindingCategory",
    "ResearchMemo",
    "Severity",
    "SkepticVerdict",
    "Stance",
    "Verdict",
]
