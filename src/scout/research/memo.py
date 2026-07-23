"""Memo synthesis: the code owns the verdict, the model only writes the prose.

This is the last stage of the research pipeline and the one place the two halves
of design rule 1 are enforced together:

  * THE VERDICT IS CODE, NOT A PROMPT. `decide_verdict` is a pure function -- a
    confirmed CRITICAL finding, or a skeptic that flags the candidate
    disqualifying, is a VETO, full stop. It is unit-tested exhaustively because
    it is the actual gate that removes a fraud from the shortlist. An LLM does
    not get a vote here: it cannot be prompted, cajoled, or hallucinated into
    waving a toxic convertible through, and it cannot invent a buy.
  * THE MODEL WRITES PROSE, THEN CODE OVERRULES ITS DECISION FIELDS. `write_memo`
    lets the model phrase the headline and thesis, but AFTER the call it
    overwrites verdict, veto_reasons and entity_id with the authoritative
    values. The model's own guess at these is discarded on purpose -- even a
    well-behaved model must not be able to escape the deterministic gate, so we
    do not trust it to echo the decision correctly, we replace it.

Every number in the memo comes from the code-computed metrics block; the prompt
forbids the model from computing or inventing figures.
"""

from __future__ import annotations

from scout.harness.protocol import Effort, LLMClient
from scout.harness.structured import complete_structured
from scout.research.evidence import EvidencePack
from scout.research.models import (
    AnalystView,
    Finding,
    ResearchMemo,
    Severity,
    SkepticVerdict,
    Verdict,
)
from scout.research.prompts.memo import build_memo_messages


def decide_verdict(
    verified_findings: list[Finding], skeptic: SkepticVerdict
) -> tuple[Verdict, list[str]]:
    """Decide VETO / NO_VETO deterministically. Pure -- no LLM, no I/O.

    VETO if EITHER a verified finding is CRITICAL (a confirmed, disqualifying red
    flag) OR the skeptic flags the candidate disqualifying. The reasons are the
    critical findings' claims, plus the skeptic's reasoning when it is the one
    vetoing. Anything else is NO_VETO with no reasons -- which is not a buy, only
    an absence of a disqualifier (see Verdict docstring).
    """
    critical = [f for f in verified_findings if f.severity is Severity.CRITICAL]

    # No gate tripped: the candidate simply was not removed by research.
    if not critical and not skeptic.disqualifying:
        return Verdict.NO_VETO, []

    veto_reasons = [f.claim for f in critical]
    if skeptic.disqualifying:
        veto_reasons.append(skeptic.reasoning)
    return Verdict.VETO, veto_reasons


async def write_memo(
    client: LLMClient,
    pack: EvidencePack,
    verified_findings: list[Finding],
    bull: AnalystView,
    bear: AnalystView,
    skeptic: SkepticVerdict,
    *,
    effort: Effort | None = None,
    temperature: float | None = 0.2,
    max_tokens: int | None = 1500,
) -> ResearchMemo:
    """Write the final memo for one candidate.

    Flow: decide the verdict in code -> build the prompt around that decision ->
    let the model write the prose -> overwrite the decision fields with the
    authoritative values. The model phrases the case; it never owns the verdict.
    """
    # Decide first, so the prompt presents the verdict as a fixed fact and the
    # model has nothing to overturn.
    verdict, veto_reasons = decide_verdict(verified_findings, skeptic)

    messages = build_memo_messages(
        pack, verified_findings, bull, bear, skeptic, verdict, veto_reasons
    )
    result = await complete_structured(
        client,
        messages,
        ResearchMemo,
        effort=effort,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # The model wrote headline/thesis; its verdict, veto_reasons and entity_id
    # are discarded, not trusted. Code owns the gate and the identity -- a model
    # cannot escape a VETO or attach the memo to the wrong entity.
    return result.value.model_copy(
        update={
            "entity_id": pack.entity_id,
            "verdict": verdict,
            "veto_reasons": veto_reasons,
        }
    )
