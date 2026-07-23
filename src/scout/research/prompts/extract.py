"""The extraction prompt: a FORENSIC READER, not a decider or a calculator.

Design rule 1 lives here in prose. The model's only job is to surface
qualitative red flags that are literally written in a filing and anchor each one
to a VERBATIM quote. It does not compute numbers (metrics/ owns those and injects
them), it does not rank, and it does not recommend -- it can only flag facts a
downstream verifier can confirm against the exact text the model was shown.

The verbatim-quote requirement is the anti-hallucination anchor of the whole
phase: verify.py drops any finding whose `quoted_span` is not a real substring of
the evidence, so a paraphrased or invented quote costs the model the finding.
Telling it that up front is cheaper than repairing it after.

Ordering matters for cost: SYSTEM_PROMPT is fixed text and is sent as the cached
prefix, while build_user_prompt() carries the per-candidate metrics and filing
text -- the part that changes every call -- so it comes last.
"""

from __future__ import annotations

from scout.research.evidence import EvidencePack

# A filing whose full anonymised text is at or under this many characters is
# shown WHOLE -- context beats retrieval when it is affordable, and the excerpt
# windows can clip the sentence that carries the actual red flag. Longer filings
# fall back to the pre-selected excerpts to keep the prompt (and its cost)
# bounded.
_FULL_TEXT_MAX_CHARS = 12_000


# The stable, cache-friendly prefix. Everything a reader-not-decider needs to
# know that does NOT change between candidates belongs here; anything that varies
# per candidate belongs in build_user_prompt().
SYSTEM_PROMPT = """\
You are a forensic reader of SEC and other primary company filings for a \
microcap equity screen. Your ONLY job is to extract qualitative red flags that \
are explicitly written in the filing text you are shown.

You are a READER, not a decider and not a calculator:
- You do NOT recommend buying, selling, or ranking anything. You only flag \
concerns a human and an automated verifier can check.
- You do NOT compute, restate, or estimate any number. A separate, authoritative \
COMPUTED METRICS block is provided; treat those numbers as final and never \
recompute or contradict them. Findings are about qualitative facts, not figures.

EVERY finding you emit MUST include:
- `quoted_span`: a VERBATIM substring copied exactly from the provided filing \
text -- same words, same punctuation, same order. Do NOT paraphrase, do NOT \
summarise, and do NOT put an ellipsis (...) inside the quote. If you cannot copy \
an exact supporting sentence, do not make the finding. An unquotable claim is a \
hallucination and will be discarded.
- `source_accession`: the accession id of the filing the quote came from, copied \
exactly from the "FILING <accession>" label above that filing's text.

Look for, and categorise accordingly:
- going_concern: substantial-doubt / ability-to-continue language.
- toxic_convertible: convertibles with variable or look-back conversion pricing, \
discounts to market, floorless resets -- death-spiral terms.
- dilution: at-the-market (ATM) programs, equity lines, standby equity, \
registered directs, large share issuance mechanics.
- reverse_split: reverse stock splits or share consolidations.
- related_party: transactions with officers, directors, or affiliates.
- auditor: auditor changes, dismissals, resignations, or material weakness in \
internal controls.
- litigation: material legal proceedings, SEC investigations or subpoenas, class \
actions.
- customer_concentration: dependence on one or a few customers.
- promotion: stock-promotion or paid-promotion disclosures.
- other: anything materially concerning that fits no category above.

Rate each finding's severity as low, medium, high, or critical. Reserve \
CRITICAL for disqualifying items only -- a live toxic / death-spiral convertible, \
or an explicit going-concern statement with no described path to funding. Most \
real concerns are medium or high; do not inflate.

If the filings show nothing of concern, return an EMPTY findings list. A clean \
company is a valid and expected outcome -- do NOT invent concerns to fill the \
list. Precision matters more than recall here; a false flag wastes a human's time \
and an unverifiable one is dropped anyway.

Return only the structured object. Put any observation you cannot tie to a \
specific quote in `notes`, not in a finding.\
"""


def _redaction_note(pack: EvidencePack) -> str:
    """Explain the placeholders so the model does not treat them as real names.

    The evidence is anonymised (company name -> [COMPANY], tickers -> [TICKER*])
    on purpose; without this note a model tends to speculate about the redacted
    identity, which is exactly the stored-knowledge "distraction" the
    anonymisation exists to remove.
    """
    if not pack.anonymized:
        return ""
    return (
        "NOTE ON REDACTIONS: This evidence is anonymised. Any [COMPANY] or "
        "[TICKER] token is a redaction standing in for the real company name or "
        "ticker, which have been removed on purpose. Do not guess or speculate "
        "about the company's identity; assess only what the text says.\n\n"
    )


def _render_filing(accession: str, pack: EvidencePack) -> str:
    """One filing's evidence: the whole short text, else its excerpts.

    Short filings are cheap enough to show whole, which avoids an excerpt window
    clipping the sentence that actually carries the flag. Long filings fall back
    to the pre-selected excerpts so the prompt stays bounded.
    """
    header = f"=== FILING {accession} ===\n"

    full_text = pack.texts_by_accession.get(accession)
    if full_text and len(full_text) <= _FULL_TEXT_MAX_CHARS:
        return f"{header}{full_text}"

    excerpts = [e for e in pack.excerpts if e.source_accession == accession]
    if excerpts:
        blocks = [f"[excerpt: {e.category}]\n{e.text}" for e in excerpts]
        return header + "\n\n".join(blocks)

    # Long filing with no excerpts selected: still show the (truncated) text so
    # the model has something to read, and say plainly that it was clipped.
    if full_text:
        clipped = full_text[:_FULL_TEXT_MAX_CHARS]
        return f"{header}[text truncated for length]\n{clipped}"

    return f"{header}(no narrative text extracted for this filing)"


def _ordered_accessions(pack: EvidencePack) -> list[str]:
    """Filing order, texts first (dict preserves insertion order), then any
    accession that appears only in excerpts."""
    ordered = list(pack.texts_by_accession.keys())
    for excerpt in pack.excerpts:
        if excerpt.source_accession not in ordered:
            ordered.append(excerpt.source_accession)
    return ordered


def build_user_prompt(pack: EvidencePack) -> str:
    """Assemble the per-candidate message: metrics, redaction note, filings.

    Everything here varies per candidate, so it is sent AFTER the fixed
    SYSTEM_PROMPT to keep the cached prefix stable. The authoritative metrics
    come first (they are short and set the frame), then the filing text the model
    must quote from -- each block labelled with the accession the model must cite.
    """
    parts = [
        "COMPUTED METRICS (authoritative — do not recompute)\n" + pack.metrics_block,
        "",
        _redaction_note(pack).rstrip(),
        "FILING EVIDENCE\n"
        "Each filing is labelled with its accession id. When you cite a quote, "
        "put that exact id in `source_accession`.",
    ]

    accessions = _ordered_accessions(pack)
    if accessions:
        parts.extend(_render_filing(accession, pack) for accession in accessions)
    else:
        parts.append(
            "(No narrative filing text is available for this candidate. Base any "
            "finding only on text you can quote; if there is none, return an empty "
            "findings list.)"
        )

    # Drop the empty string the redaction note leaves when not anonymised.
    return "\n\n".join(part for part in parts if part)
