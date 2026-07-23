"""Prompt for the memo-synthesis stage: prose only, never the decision.

The memo writer is the one LLM call in the pipeline that produces human-facing
prose, and it is deliberately fenced in. Two things it is NOT allowed to do, and
the prompt says so in as many words:

  * It does not decide the verdict. VETO / NO_VETO is settled by deterministic
    code (`memo.decide_verdict`) BEFORE this prompt is built; the decision and
    its reasons are handed to the model as fixed facts to write around, not a
    question to answer. Design rule 1: a model cannot wave a confirmed fraud
    through, nor invent a buy.
  * It does not produce numbers. Every figure lives in the code-computed metrics
    block; the model quotes those and is told not to compute or invent any.

Keeping this text out of `memo.py` keeps the cache-friendly system prompt in one
obvious place and lets the wording change without touching the call logic.
"""

from __future__ import annotations

from scout.harness.protocol import Message
from scout.research.evidence import EvidencePack
from scout.research.models import AnalystView, Finding, SkepticVerdict, Verdict

MEMO_SYSTEM = """\
You are the memo writer for a deep-research equity scout. You produce the final \
human-readable memo for one already-screened, already-researched candidate.

Hard rules, no exceptions:
- You do NOT decide the verdict. The verdict (VETO or NO_VETO) has already been \
decided by deterministic code and is given to you below as a fixed fact. Write \
the prose so it is consistent with that decision. Never argue for a different \
verdict, never recommend buying, and never rank the candidate.
- You do NOT produce numbers. Every figure you may cite is in the METRICS block, \
which is authoritative and code-computed. Do not compute, estimate, or invent \
any number that is not printed there.
- Write from the evidence given only: the findings, the bull case, the bear \
case, and the skeptic's verdict. Do not add outside knowledge about the company.

Produce a ResearchMemo:
- headline: one sentence -- what this company is and the single most important fact.
- thesis: 2-4 sentences on the investment case the screen implies and its main risk.
- verdict and veto_reasons: echo the decided verdict below; code will overwrite \
these regardless, so match them.\
"""


def _format_findings(findings: list[Finding]) -> str:
    if not findings:
        return "None verified."
    return "\n".join(
        f"- [{f.severity.value}] {f.category.value}: {f.claim}" for f in findings
    )


def _format_view(view: AnalystView) -> str:
    return "\n".join(f"- {point}" for point in view.points)


def _format_skeptic(skeptic: SkepticVerdict) -> str:
    lines = [
        f"Disqualifying: {skeptic.disqualifying}",
        f"Reasoning: {skeptic.reasoning}",
    ]
    if skeptic.refuted_claims:
        lines.append("Refuted claims:")
        lines.extend(f"- {claim}" for claim in skeptic.refuted_claims)
    return "\n".join(lines)


def _format_verdict(verdict: Verdict, veto_reasons: list[str]) -> str:
    lines = [f"DECIDED VERDICT (authoritative, do not change): {verdict.value}"]
    if veto_reasons:
        lines.append("Veto reasons:")
        lines.extend(f"- {reason}" for reason in veto_reasons)
    else:
        lines.append("Veto reasons: none")
    return "\n".join(lines)


def build_memo_messages(
    pack: EvidencePack,
    verified_findings: list[Finding],
    bull: AnalystView,
    bear: AnalystView,
    skeptic: SkepticVerdict,
    verdict: Verdict,
    veto_reasons: list[str],
) -> list[Message]:
    """Assemble the memo prompt. Pure: same inputs -> same messages.

    The decided verdict comes last and is labelled authoritative, so a model
    scanning for the answer finds the code's decision, not a question.
    """
    user = "\n\n".join(
        [
            f"METRICS (authoritative, code-computed -- quote these, invent nothing):\n{pack.metrics_block}",
            f"VERIFIED FINDINGS (each citation already confirmed against the filing):\n{_format_findings(verified_findings)}",
            f"BULL CASE:\n{_format_view(bull)}",
            f"BEAR CASE:\n{_format_view(bear)}",
            f"SKEPTIC:\n{_format_skeptic(skeptic)}",
            _format_verdict(verdict, veto_reasons),
        ]
    )
    return [
        Message(role="system", content=MEMO_SYSTEM),
        Message(role="user", content=user),
    ]
