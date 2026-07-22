"""Reasoning normalization -- the four-convention mess, in one place.

Nothing above `adapters/` is allowed to know that "where did the model put its
thinking?" has four incompatible answers depending on which server you happen to
be talking to (PLAN.md section 4.2):

    message.reasoning_content    vLLM, LiteLLM proxy
    message.reasoning            OpenRouter, some Ollama paths
    message.thinking             Ollama native /api/chat
    <think>...</think> in content llama.cpp, LM Studio, unconfigured vLLM

The single most valuable behaviour in this file is the last one: **empty content
with populated reasoning is an error, not a success.** That is a hybrid-thinking
model (Qwen3.x, DeepSeek-R1, some Gemma builds) that spent its whole output
budget thinking, and returning "" for it means the failure surfaces three layers
later as a confusing JSON parse error instead of here, where we can name the fix.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from scout.harness.errors import EmptyContentError
from scout.harness.protocol import Effort

# Checked in this order. A proxy chain (LiteLLM in front of Ollama, say) can
# populate more than one, and we keep all of them -- see _merge. The order only
# decides which text comes first in the merged string.
_REASONING_FIELDS = ("reasoning_content", "reasoning", "thinking")

_CLOSED_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
# Generation cut short by max_tokens leaves an opening tag with no closer, and
# everything after it is thinking that never got to become an answer.
_UNTERMINATED_THINK = re.compile(r"<think>(.*)\Z", re.DOTALL | re.IGNORECASE)

_EMPTY_WITH_REASONING = (
    "The model returned reasoning but no answer: `content` was empty while the "
    "reasoning field held {n} characters. This is a hybrid-thinking model "
    "(Qwen3.x, DeepSeek-R1, some Gemma builds) that spent its entire output "
    "budget thinking and never emitted the answer.\n"
    "Fix: set REASONING_EFFORT=none in .env to turn thinking off for this model. "
    "If you need the thinking, raise MAX_TOKENS instead so the answer fits after it."
)

_EMPTY_ENTIRELY = (
    "The model returned an empty response: no content, no reasoning, no tool "
    "call. The request was accepted but produced nothing. Check that the server "
    "actually has a model loaded, that MAX_TOKENS is not near zero, and that no "
    "stop sequence matches the very first token."
)


def normalize_message(message: Mapping[str, Any]) -> tuple[str, str | None]:
    """Resolve one chat-completion message into `(content, reasoning)`.

    Raises EmptyContentError rather than returning empty content -- see the
    module docstring for why that is the point of this function.
    """
    content = _as_text(message.get("content"))
    parts = [_as_text(message.get(name)) for name in _REASONING_FIELDS]

    content, inline = strip_think_tags(content)
    parts.extend(inline)
    reasoning = _merge(parts)

    if content.strip():
        return content.strip(), reasoning

    if reasoning:
        raise EmptyContentError(_EMPTY_WITH_REASONING.format(n=len(reasoning)), reasoning=reasoning)
    raise EmptyContentError(_EMPTY_ENTIRELY)


def strip_think_tags(content: str) -> tuple[str, list[str]]:
    """Pull inline `<think>` blocks out of `content`.

    Returns the content with the blocks removed and the list of block bodies.
    Handles three shapes seen in the wild:

      1. `<think>x</think>answer`  -- the normal llama.cpp / LM Studio case.
      2. `<think>x`                -- unterminated, because max_tokens cut the
         generation off mid-thought. Everything after the tag is reasoning and
         there is no answer at all.
      3. `x</think>answer`         -- an orphan closer, which happens when the
         chat template opens the think block in the *prompt* so the model only
         ever emits the closing tag.
    """
    if not content:
        return "", []

    reasoning: list[str] = []
    lowered = content.lower()
    open_at = lowered.find("<think>")
    close_at = lowered.find("</think>")

    # Shape 3: a closer with no opener before it. Handle it first, otherwise the
    # paired-block regex below would see nothing and the leading thinking would
    # be silently served as the answer.
    if close_at != -1 and (open_at == -1 or close_at < open_at):
        reasoning.append(content[:close_at])
        content = content[close_at + len("</think>") :]

    # Shape 1: properly closed pairs.
    content = _CLOSED_THINK.sub(lambda m: _capture(m.group(1), reasoning), content)

    # Shape 2: whatever opening tag is left has no closer.
    unterminated = _UNTERMINATED_THINK.search(content)
    if unterminated:
        reasoning.append(unterminated.group(1))
        content = content[: unterminated.start()]

    return content, reasoning


def openai_effort_params(effort: Effort | None) -> dict[str, Any]:
    """Translate the normalized effort enum for an OpenAI-shaped request.

    An unset effort sends nothing at all: many local servers and older models
    reject `reasoning_effort` outright, and the value of "unset" is precisely
    "do not touch this knob". `"none"` is a real value on OpenAI now and is the
    lever that turns a hybrid-thinking local model back into a fast extractor.
    """
    if effort is None:
        return {}
    return {"reasoning_effort": effort}


# Budgets in output tokens. Anthropic's floor is 1024, so "low" is the floor;
# "medium" is roughly one page of thinking, which is what a filing-extraction
# task needs; "high" is for the adversarial-review stage where the model is
# actually arguing with itself. Deliberately not scaled off max_tokens -- a
# budget that moves when you change max_tokens makes runs non-comparable.
_THINKING_BUDGETS: dict[str, int] = {"low": 1024, "medium": 4096, "high": 16384}
_MIN_THINKING_BUDGET = 1024  # Anthropic rejects anything smaller.
_ANSWER_HEADROOM = 512  # budget_tokens must be < max_tokens, and the answer needs room too.


def anthropic_thinking_params(effort: Effort | None, max_tokens: int) -> dict[str, Any]:
    """Translate the normalized effort enum for an Anthropic Messages request.

    `budget_tokens` comes out of `max_tokens`, not on top of it, so we always
    leave headroom for the answer itself -- a budget equal to max_tokens
    produces exactly the empty-content failure this module exists to catch.
    """
    if effort is None:
        return {}
    if effort == "none":
        return {"thinking": {"type": "disabled"}}

    budget = min(_THINKING_BUDGETS[effort], max_tokens - _ANSWER_HEADROOM)
    if budget < _MIN_THINKING_BUDGET:
        raise ValueError(
            f"max_tokens={max_tokens} is too small for thinking: Anthropic's minimum "
            f"budget is {_MIN_THINKING_BUDGET} tokens and the answer needs another "
            f"{_ANSWER_HEADROOM}. Raise MAX_TOKENS to at least "
            f"{_MIN_THINKING_BUDGET + _ANSWER_HEADROOM}, or set REASONING_EFFORT=none."
        )
    return {"thinking": {"type": "enabled", "budget_tokens": budget}}


def _capture(text: str, sink: list[str]) -> str:
    """Record a matched think-block body and replace it with nothing."""
    sink.append(text)
    return ""


def _as_text(value: Any) -> str:
    """Coerce a content field to a string.

    Usually a string, but some servers return the OpenAI "content parts" array
    (`[{"type": "text", "text": ...}]`) even on a plain chat completion.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        return "".join(part.get("text", "") for part in value if isinstance(part, Mapping))
    return str(value)


def _merge(parts: Sequence[str]) -> str | None:
    """Join every place reasoning turned up, dropping exact repeats.

    Proxies routinely echo the same text into two fields (LiteLLM copies
    `reasoning_content` to `reasoning`), and concatenating that would double the
    thinking in the trace and in the cost report.
    """
    seen: set[str] = set()
    merged: list[str] = []
    for part in parts:
        stripped = part.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        merged.append(stripped)
    return "\n\n".join(merged) or None
