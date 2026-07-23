"""The deep-research agent: the LLM in the driver's seat, on a leash of code.

`scout research` runs a fixed pipeline where the LLM is a reader that can only
veto. This package is the other half of the original vision: an agent that
*drives* -- it searches filings and the web, pulls a company's numbers, and
decides what to look at next -- while the disciplined pieces it was built on
(code-computed metrics, citation verification, the code-owned veto) remain the
tools and checks it must pass through.

The one rule that relaxes here (deliberately): the agent may surface and
recommend a candidate, not only veto one. Everything else holds -- every number
comes from a metric tool, every filing/web claim must carry a verifiable quote,
and `decide_verdict` still owns the veto. Agency, on a provenance leash.
"""
