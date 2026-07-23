"""The bull/bear/skeptic debate stage of the research pipeline.

Why a debate instead of one summary: a single pass over a filing tends to average
its signals into a bland paragraph, and the point that should VETO a candidate --
the confirmed toxic convertible, the going-concern note -- is exactly the detail
an averaging summary smooths away. Forcing an explicit BULL case and an explicit
BEAR case makes each side surface its best material, so nothing decisive hides in
the middle.

Why a separate skeptic, ideally on a different model family: the skeptic's entire
value is independence. A model reviewing its own bull case shares that model's
blind spots and will happily wave through a flaw it never thought to look for.
`run_debate` therefore takes the skeptic's client as a distinct parameter so the
caller can route it to a different family (PLAN.md); the three may be the same
object, but the seam exists on purpose.

Why caution by default: these are microcaps, and the loss function is asymmetric
-- one confirmed death-spiral convertible costs more than ten missed cheap names.
So the skeptic is told that an unresolved red flag is a reason to veto, not to
wave through.
"""

from __future__ import annotations

import asyncio

from scout.harness.protocol import Effort, LLMClient
from scout.harness.structured import complete_structured
from scout.research.evidence import EvidencePack
from scout.research.models import AnalystView, Finding, SkepticVerdict, Stance
from scout.research.prompts.analysts import (
    build_analyst_messages,
    build_skeptic_messages,
)

# The system message is index 0 and holds the stable, cacheable instructions; the
# varying evidence is the last message. Point the cache breakpoint at the system
# prompt so every candidate reuses the same prefix.
_CACHE_PREFIX_UPTO = 0


async def argue(
    client: LLMClient,
    stance: Stance,
    pack: EvidencePack,
    verified_findings: list[Finding],
    *,
    effort: Effort | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_repairs: int = 2,
) -> AnalystView:
    """Build the strongest case for `stance` (BULL or BEAR) from the evidence alone.

    Returns an AnalystView whose `stance` field matches the requested side, with
    every point grounded in a metric or a verified finding.
    """
    messages = build_analyst_messages(stance, pack, verified_findings)
    result = await complete_structured(
        client,
        messages,
        AnalystView,
        effort=effort,
        temperature=temperature,
        max_tokens=max_tokens,
        max_repairs=max_repairs,
        cache_prefix_upto=_CACHE_PREFIX_UPTO,
    )
    return result.value


async def skeptic_review(
    client: LLMClient,
    pack: EvidencePack,
    verified_findings: list[Finding],
    bull: AnalystView,
    *,
    effort: Effort | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_repairs: int = 2,
) -> SkepticVerdict:
    """Adversarially review the bull case.

    Refutes unsupported bull points and makes the one call the pipeline needs from
    it: is a confirmed red flag disqualifying? Defaults toward caution -- for a
    microcap, uncertainty vetoes.
    """
    messages = build_skeptic_messages(pack, verified_findings, bull)
    result = await complete_structured(
        client,
        messages,
        SkepticVerdict,
        effort=effort,
        temperature=temperature,
        max_tokens=max_tokens,
        max_repairs=max_repairs,
        cache_prefix_upto=_CACHE_PREFIX_UPTO,
    )
    return result.value


async def run_debate(
    bull_client: LLMClient,
    bear_client: LLMClient,
    skeptic_client: LLMClient,
    pack: EvidencePack,
    verified_findings: list[Finding],
    *,
    effort: Effort | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_repairs: int = 2,
) -> tuple[AnalystView, AnalystView, SkepticVerdict]:
    """Run the full debate: bull and bear concurrently, then the skeptic.

    The three clients are separate parameters so the skeptic can be a different
    model family for independence; passing the same object for all three is fine.
    Returns (bull, bear, skeptic).
    """
    # Bull and bear are independent, so argue them at the same time.
    bull, bear = await asyncio.gather(
        argue(
            bull_client,
            Stance.BULL,
            pack,
            verified_findings,
            effort=effort,
            temperature=temperature,
            max_tokens=max_tokens,
            max_repairs=max_repairs,
        ),
        argue(
            bear_client,
            Stance.BEAR,
            pack,
            verified_findings,
            effort=effort,
            temperature=temperature,
            max_tokens=max_tokens,
            max_repairs=max_repairs,
        ),
    )

    # The skeptic reviews the FINISHED bull case, so it must run after the gather,
    # not inside it.
    skeptic = await skeptic_review(
        skeptic_client,
        pack,
        verified_findings,
        bull,
        effort=effort,
        temperature=temperature,
        max_tokens=max_tokens,
        max_repairs=max_repairs,
    )
    return bull, bear, skeptic
