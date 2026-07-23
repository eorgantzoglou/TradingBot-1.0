"""Contracts for the agent loop, its tools, and its output.

Kept in one file so the loop, the toolbox and the brief composer all agree on the
shapes without importing each other. The Pydantic models are written to the same
intersection-of-subsets discipline as `research/models.py` (no schema-keyword
constraints; validators instead), because the agent runs on the same harness.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, field_validator

from scout.research.evidence import EvidencePack
from scout.research.models import Verdict

# The reserved tool name the model calls when it has gathered enough evidence and
# wants the loop to stop and compose the brief. A sentinel tool rather than a
# discriminated "final" action, because a single flat schema (always a tool call)
# is the most portable shape across the structured-output ladder.
FINISH_TOOL = "finish"


class AgentAction(BaseModel):
    """One step the model chooses: a thought plus the tool to call next."""

    thought: str = Field(
        description="One or two sentences: what you learned last step and what you need next."
    )
    tool: str = Field(
        description=f"The tool to call now, or '{FINISH_TOOL}' when you have enough "
        "evidence to write the brief."
    )
    tool_input: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the tool, as a JSON object. {} for the finish tool.",
    )

    @field_validator("thought", "tool")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must be non-empty")
        return v.strip()


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What a tool hands back: text the model reads, plus evidence it gathered.

    `output` is always a string, because it becomes an `Observation:` message in
    the transcript -- the model reads prose, not Python objects. A tool that fails
    returns `ok=False` with the reason AS the output, so a failure is something
    the agent reasons about (and can retry differently), never a crash.

    The evidence fields are how a tool feeds the guardrail: `read_filing` returns
    the FULL filing text in `filing_texts` (keyed by accession) even though its
    `output` is only an excerpt, so the brief's quotes can later be verified
    against the complete text the model actually had access to. Same for
    `web_pages` (url -> text) and the code-computed `metrics_block`.
    """

    output: str
    ok: bool = True
    filing_texts: dict[str, str] = field(default_factory=dict)
    web_pages: dict[str, str] = field(default_factory=dict)
    metrics_block: str | None = None

    @classmethod
    def error(cls, reason: str) -> ToolResult:
        return cls(output=f"ERROR: {reason}", ok=False)


@dataclass(frozen=True, slots=True)
class Tool:
    """A capability the agent can call. `run` is async and self-contained.

    `parameters` is a short, human-readable description of the arguments, shown to
    the model in the tool catalogue -- deliberately prose, not a JSON Schema, so
    the catalogue stays cheap and stable in the prompt (which is what makes prompt
    caching hit). Validation of the actual call is the tool's own job.
    """

    name: str
    summary: str
    parameters: str
    run: Callable[[dict[str, Any]], Awaitable[ToolResult]]


@dataclass(slots=True)
class AgentStep:
    """One executed turn, kept for the transcript and the audit trail."""

    thought: str
    tool: str
    tool_input: dict[str, Any]
    observation: str
    ok: bool


@dataclass(slots=True)
class AgentRun:
    """The full trace of one investigation, before the brief is composed.

    `evidence` and `web_sources` are what the tools accumulated -- the substrate
    the brief's claims are verified against. `finished` distinguishes an agent
    that decided it was done from one that hit the step cap (a meaningful
    difference: the latter's brief rests on whatever it had gathered so far)."""

    goal: str
    steps: list[AgentStep] = field(default_factory=list)
    evidence: EvidencePack | None = None
    web_sources: dict[str, str] = field(default_factory=dict)
    """url -> fetched page text, so a web-sourced claim can be span-verified too."""

    metrics_block: str = ""
    """The most recent code-computed metrics a tool produced -- injected into the
    brief so its numbers are the deterministic ones, never the model's."""

    finished: bool = False
    stop_reason: str = ""


class BriefFinding(BaseModel):
    """A claim in the brief, anchored to a source the loop actually fetched.

    Same anchoring discipline as research.Finding, but the source can be a filing
    accession OR a web URL -- so it carries a free `source` string that
    verification resolves against either the evidence pack or the fetched pages.
    """

    claim: str = Field(description="One sentence stating the point, in plain language.")
    quoted_span: str = Field(
        description="A verbatim quote from a filing or web page you fetched that supports "
        "the claim. Copy it exactly."
    )
    source: str = Field(
        description="Where the quote is from: a filing accession, or a web URL you fetched."
    )
    is_red_flag: bool = Field(
        description="True if this is a risk/red flag (dilution, going concern, promotion, etc.)."
    )

    @field_validator("claim", "quoted_span", "source")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must be non-empty")
        return v.strip()


class AgentBrief(BaseModel):
    """The model's composed output, BEFORE code applies the guardrails.

    The model proposes a recommendation and findings; code then verifies every
    finding's quote, drops the unverifiable ones, and decides the verdict. The
    `recommendation` is advisory prose -- it is allowed to surface a candidate
    (the relaxation of rule 1), but it can never override the code-owned verdict.
    """

    headline: str = Field(description="One sentence: what this company is and the key fact.")
    thesis: str = Field(description="2-4 sentences on the case and its main risk.")
    recommendation: str = Field(
        description="Your read: is this worth further work? Why? This is advisory -- "
        "you may surface it, but you cannot recommend buying, and a code veto overrides you."
    )
    findings: list[BriefFinding] = Field(
        default_factory=list,
        description="Every notable fact, each with a verbatim quote from a source you fetched.",
    )


@dataclass(slots=True)
class FinishedBrief:
    """The final, guardrail-checked artifact written to disk.

    Assembled by `brief.py`: the model's prose, plus the verified findings, the
    dropped (unverifiable) ones, and the code-decided verdict. This is the shape
    `output/report.py` renders.
    """

    entity_id: str | None
    subject: str
    """What was investigated -- an entity name, or the free-text thesis."""

    model: str
    headline: str
    thesis: str
    recommendation: str
    verdict: Verdict
    veto_reasons: list[str]
    verified_findings: list[BriefFinding] = field(default_factory=list)
    dropped_findings: list[tuple[BriefFinding, str]] = field(default_factory=list)
    metrics_block: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def vetoed(self) -> bool:
        return self.verdict == Verdict.VETO


# A finding the composer verified against real evidence, paired with how it was
# confirmed -- reused by brief.py so the mapping to research.Finding stays local.
VerifiedBriefFinding = tuple[BriefFinding, str]
