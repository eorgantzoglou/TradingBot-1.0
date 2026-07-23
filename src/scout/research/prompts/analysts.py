"""The debate prompts: a BULL, a BEAR, and an adversarial SKEPTIC.

Design rule 1 again, in three voices. Each analyst argues ONLY from the evidence,
metrics and verified findings it is shown -- never from stored knowledge of the
(anonymised) company -- and every point must trace back to a metric or a finding.
Numbers are quoted from the authoritative COMPUTED METRICS block; the model never
invents or recomputes one.

Two rules the debate lives or dies by:

  * It is a DEBATE, not a summary. The bull prompt demands the strongest honest
    case FOR and the bear the strongest AGAINST -- a strawman on either side
    wastes the exercise, because the whole point is to surface the best material
    each side can muster rather than averaging them into mush.
  * The SKEPTIC only refutes. It cannot buy, rank, or promote; it finds the weak
    links in the bull case and makes one call -- is a confirmed red flag fatal?
    It is told, explicitly, to default toward caution: for a microcap an
    unresolved red flag is a reason to veto, not to wave through.

As in extract.py, the fixed SYSTEM prompts are the cache-friendly prefix and the
per-candidate evidence is built last, so the cached prefix does not drift.
"""

from __future__ import annotations

from scout.harness.protocol import Message
from scout.research.evidence import EvidencePack
from scout.research.models import AnalystView, Finding, Stance

# --------------------------------------------------------------------------
# Stable system prompts (the cached prefix)
# --------------------------------------------------------------------------

# Shared across bull and bear: the ground rules that make this a grounded debate
# rather than two models free-associating about a company they cannot name.
_ANALYST_GROUND_RULES = """\
You are an equity analyst in a structured debate about a single microcap company, \
one side of a bull-versus-bear argument that a separate skeptic will later review.

Rules you must follow without exception:
- Argue ONLY from the evidence, computed metrics, and verified findings provided \
below. You have NO outside knowledge of this company, and you must not invent any.
- Every point you make must be traceable to a specific metric or a specific \
finding. If the evidence does not support a point, do not make it.
- All numbers come from the COMPUTED METRICS block, exactly as written there. \
Never invent, estimate, restate, or recompute a figure.
- This is a DEBATE, not a summary. Make the STRONGEST honest case for your side; \
a weak or half-hearted case wastes the debate.
- The evidence is ANONYMISED: any [COMPANY] or [TICKER] token is a redaction \
standing in for a real name or ticker removed on purpose. Do not guess the \
identity; assess only what the text and numbers say.

Return 3 to 6 points, each a single grounded sentence."""

BULL_SYSTEM = (
    _ANALYST_GROUND_RULES
    + """

YOUR SIDE IS BULL. Build the strongest genuine case FOR this being an attractive \
investment: cheapness on the metrics, balance-sheet strength, cash relative to \
market value, the ABSENCE of confirmed red flags -- whatever the evidence actually \
supports. Do not strawman your own side and do not concede the bear's points for \
them. Set "stance" to "bull"."""
)

BEAR_SYSTEM = (
    _ANALYST_GROUND_RULES
    + """

YOUR SIDE IS BEAR. Build the strongest genuine case AGAINST: dilution, \
going-concern language, toxic or death-spiral financing, weak or deteriorating \
metrics, and every confirmed red flag in the findings. Do not strawman the risk \
and do not soften a real concern. Set "stance" to "bear"."""
)

# The skeptic is the adversarial gate. Its independence is the reason it should
# run on a different model family (see analysts.run_debate); its instructions are
# deliberately one-directional -- it can only take away, never add.
SKEPTIC_SYSTEM = """\
You are an independent skeptic reviewing a BULL CASE built for a single microcap \
company. The evidence is ANONYMISED: any [COMPANY] or [TICKER] token is a \
redaction for a real name or ticker removed on purpose -- do not speculate about \
the identity.

Your ONLY job is REFUTATION. You cannot recommend buying, ranking, or promoting \
this candidate. You can only find weaknesses and decide whether a confirmed red \
flag is fatal. You may use ONLY the evidence, metrics, verified findings, and the \
bull case provided; numbers come from the COMPUTED METRICS block and are never \
recomputed.

Do exactly two things:
1. In "refuted_claims", list every bull point (or finding) the evidence does NOT \
actually support -- claims that overreach, assume facts not in the metrics, or \
ignore a confirmed red flag. If the bull case is fully supported, return an empty \
list.
2. In "disqualifying", decide whether any CONFIRMED red flag makes this candidate \
un-investable no matter how cheap it looks. A confirmed live toxic / death-spiral \
convertible, a going-concern statement with no described funding path, or an \
active dilution death-spiral is disqualifying.

DEFAULT TOWARD CAUTION. This is a microcap: uncertainty is a reason to VETO, not \
to wave through. If a serious red flag is left unresolved by the evidence, treat \
that unresolved risk as a reason to set disqualifying = true, not false. A cheap \
valuation NEVER offsets a fatal red flag. When genuinely torn, veto.

Explain your call in "reasoning" in two or three sentences."""


# --------------------------------------------------------------------------
# Per-candidate evidence (built last, kept out of the cached prefix)
# --------------------------------------------------------------------------

_ANON_NOTE = (
    "This evidence is ANONYMISED: [COMPANY] and [TICKER] tokens are redactions "
    "standing in for the real company name and tickers, removed on purpose."
)


def _format_findings(findings: list[Finding]) -> str:
    """Render the citation-verified findings the analysts may rely on.

    These have already passed verify.py, so each quote is known to be real -- the
    analysts can lean on them without re-checking.
    """
    if not findings:
        return (
            "VERIFIED FINDINGS: none. The citation-checked extraction surfaced no "
            "red flags in the available filings."
        )

    lines = ["VERIFIED FINDINGS (each quote is confirmed present in the cited filing):"]
    for i, finding in enumerate(findings, 1):
        lines.append(
            f"  {i}. [{finding.category.value} / {finding.severity.value}] {finding.claim}\n"
            f'     quote ({finding.source_accession}): "{finding.quoted_span}"'
        )
    return "\n".join(lines)


def _format_excerpts(pack: EvidencePack) -> str:
    """The retrieved red-flag passages, so an analyst has qualitative context
    beyond the one-line findings."""
    if not pack.excerpts:
        return ""
    lines = ["EVIDENCE EXCERPTS (anonymised passages retrieved from the filings):"]
    lines.extend(
        f"  [{excerpt.source_accession} / {excerpt.category}] {excerpt.text}"
        for excerpt in pack.excerpts
    )
    return "\n".join(lines)


def _format_evidence(pack: EvidencePack, verified_findings: list[Finding]) -> str:
    """The shared evidence body: anonymisation note, metrics, findings, excerpts.

    Metrics come first (short, and they set the frame), findings next, raw
    excerpts last. This is the varying content, so it is always the final message
    -- the fixed system prompt stays the cached prefix.
    """
    sections = [
        _ANON_NOTE,
        "COMPUTED METRICS (authoritative — use these numbers exactly, do not recompute):\n"
        + pack.metrics_block,
        _format_findings(verified_findings),
    ]
    excerpts = _format_excerpts(pack)
    if excerpts:
        sections.append(excerpts)
    return "\n\n".join(sections)


def _format_bull_case(bull: AnalystView) -> str:
    """The bull's points, laid out for the skeptic to attack one by one."""
    lines = ["BULL CASE TO REVIEW (the points you must try to refute):"]
    lines.extend(f"  {i}. {point}" for i, point in enumerate(bull.points, 1))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Message builders (pure: evidence in, messages out)
# --------------------------------------------------------------------------


def build_analyst_messages(
    stance: Stance,
    pack: EvidencePack,
    verified_findings: list[Finding],
) -> list[Message]:
    """System prompt for `stance` plus the shared evidence body.

    System first, evidence last -- the ordering the cache prefix depends on.
    """
    system = BULL_SYSTEM if stance is Stance.BULL else BEAR_SYSTEM
    return [
        Message(role="system", content=system),
        Message(role="user", content=_format_evidence(pack, verified_findings)),
    ]


def build_skeptic_messages(
    pack: EvidencePack,
    verified_findings: list[Finding],
    bull: AnalystView,
) -> list[Message]:
    """The skeptic's prompt: the same evidence PLUS the bull case to refute.

    The bull case is appended to the evidence body so the skeptic can point at
    specific bull claims -- this is what makes the review adversarial rather than
    a fresh, independent read.
    """
    body = f"{_format_evidence(pack, verified_findings)}\n\n{_format_bull_case(bull)}"
    return [
        Message(role="system", content=SKEPTIC_SYSTEM),
        Message(role="user", content=body),
    ]
