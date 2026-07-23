"""Stage 1 of the research pipeline: pull cited red-flag findings from filings.

This is the FORENSIC-READER stage of design rule 1. It hands the model the
evidence pack for one candidate and asks for an `ExtractionResult` -- a list of
qualitative findings, each anchored to a verbatim quote from the filing text.
The model never emits a number (metrics/ owns those) and never ranks or
recommends; it can only surface facts that verify.py will later confirm against
the exact anonymised text the model was shown.

Everything hard about getting a validated object back -- the output-mode ladder,
the repair loop, schema normalisation -- lives in the harness. This module's job
is narrow: build a good prompt (stable system prefix, per-candidate evidence
last, for prompt-cache reuse) and delegate to `complete_structured`.
"""

from __future__ import annotations

from scout.harness.protocol import Effort, LLMClient, Message
from scout.harness.structured import complete_structured
from scout.research.evidence import EvidencePack
from scout.research.models import ExtractionResult
from scout.research.prompts.extract import SYSTEM_PROMPT, build_user_prompt

# The system message is index 0 and holds only fixed instructions, so it is the
# stable prefix the provider can cache across every candidate in a run.
_CACHE_PREFIX_UPTO = 0


async def extract_findings(
    client: LLMClient,
    pack: EvidencePack,
    *,
    effort: Effort | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> ExtractionResult:
    """Extract verbatim-anchored red-flag findings from one candidate's evidence.

    Returns the validated `ExtractionResult` (an empty findings list is a valid,
    expected result for a clean company). Raises whatever the harness raises when
    it cannot get a valid object after its repair budget -- we do not paper over a
    failed extraction, because a half-valid result would become a wrong claim in
    a memo.
    """
    messages = [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=build_user_prompt(pack)),
    ]

    result = await complete_structured(
        client,
        messages,
        ExtractionResult,
        effort=effort,
        temperature=temperature,
        max_tokens=max_tokens,
        cache_prefix_upto=_CACHE_PREFIX_UPTO,
    )
    return result.value
