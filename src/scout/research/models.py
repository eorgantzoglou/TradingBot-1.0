"""Schemas for the research pipeline.

Every model here is written to the INTERSECTION of the OpenAI and Anthropic JSON
Schema subsets (PLAN.md section 4.3): no `minimum`/`maximum`/`minLength`/
`maxLength` as schema keywords, because Anthropic silently drops them. Bounds are
enforced by Pydantic validators instead -- the wire schema is a hint to the
decoder, `model_validate()` is the gate.

The types encode design rule 1: the LLM produces qualitative Findings, each
anchored to a verbatim quoted span from the filing, and a veto/no-veto verdict.
It never emits a number (those are injected from `metrics/`) and never emits a
buy or a rank -- it can only remove a candidate the deterministic screen already
chose, never promote one.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class FindingCategory(StrEnum):
    DILUTION = "dilution"
    TOXIC_CONVERTIBLE = "toxic_convertible"
    GOING_CONCERN = "going_concern"
    REVERSE_SPLIT = "reverse_split"
    RELATED_PARTY = "related_party"
    AUDITOR = "auditor"
    LITIGATION = "litigation"
    PROMOTION = "promotion"
    CUSTOMER_CONCENTRATION = "customer_concentration"
    OTHER = "other"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    """A CRITICAL finding, once its citation verifies, vetoes the candidate --
    a confirmed toxic convertible or going-concern-with-no-runway is
    disqualifying, not a discussion point."""


class Finding(BaseModel):
    """One qualitative claim, anchored to the filing text that supports it.

    `quoted_span` must be a VERBATIM substring of the evidence the model was
    shown. verify.py checks that literally; a finding whose span cannot be found
    is dropped before it reaches the memo, because an unanchored claim is exactly
    the hallucination this design exists to prevent.
    """

    category: FindingCategory
    claim: str = Field(description="One sentence stating the concern, in plain language.")
    quoted_span: str = Field(
        description="A verbatim quote from the provided filing text that supports the claim. "
        "Copy it exactly, including punctuation. Do not paraphrase."
    )
    source_accession: str = Field(
        description="The accession id of the filing the quote is from, exactly as labelled in the evidence."
    )
    severity: Severity

    @field_validator("claim", "quoted_span", "source_accession")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must be non-empty")
        return v.strip()


class ExtractionResult(BaseModel):
    """What the extractor pulls from one candidate's evidence pack."""

    findings: list[Finding] = Field(
        description="Every red-flag or noteworthy qualitative fact found, each with a verbatim quote. "
        "Empty list if the filings show nothing of concern."
    )
    notes: str = Field(
        default="",
        description="Brief free-text observations that are not tied to a specific quote.",
    )


class Stance(StrEnum):
    BULL = "bull"
    BEAR = "bear"


class AnalystView(BaseModel):
    """One side of the debate, argued only from the evidence provided."""

    stance: Stance
    points: list[str] = Field(
        description="The strongest points for this stance, each grounded in the evidence or metrics shown. "
        "3 to 6 points."
    )

    @field_validator("points")
    @classmethod
    def _some_points(cls, v: list[str]) -> list[str]:
        cleaned = [p.strip() for p in v if p and p.strip()]
        if not cleaned:
            raise ValueError("at least one point required")
        return cleaned[:8]


class SkepticVerdict(BaseModel):
    """The adversarial pass, ideally run on a different model family.

    Its job is to REFUTE -- to find the weakest link in the bull case and to
    decide whether any confirmed red flag is disqualifying. It defaults toward
    caution: uncertainty is a reason to veto a microcap, not to wave it through.
    """

    refuted_claims: list[str] = Field(
        default_factory=list,
        description="Claims from the bull case or the findings that the evidence does NOT actually support.",
    )
    disqualifying: bool = Field(
        description="True if a confirmed red flag makes this candidate un-investable regardless of cheapness."
    )
    reasoning: str = Field(description="Why disqualifying is true or false, in two or three sentences.")


class Verdict(StrEnum):
    NO_VETO = "no_veto"
    """No disqualifying red flag was confirmed. This is NOT a buy recommendation
    -- the deterministic screen's rank stands; research simply did not remove it."""

    VETO = "veto"
    """A confirmed, disqualifying red flag. The candidate is removed."""


class ResearchMemo(BaseModel):
    """The final human-readable output for one candidate.

    Prose is the LLM's; every number in it is injected from `metrics/` and never
    generated. The verdict can only be VETO or NO_VETO -- the model cannot rank
    or recommend buying.
    """

    entity_id: str
    headline: str = Field(description="One sentence: what this company is and the single most important fact.")
    thesis: str = Field(description="2-4 sentences on the investment case the screen implies, and its main risk.")
    verdict: Verdict
    veto_reasons: list[str] = Field(
        default_factory=list,
        description="If vetoed, the specific confirmed red flags. Empty if NO_VETO.",
    )

    @field_validator("veto_reasons")
    @classmethod
    def _clean(cls, v: list[str]) -> list[str]:
        return [r.strip() for r in v if r and r.strip()]
