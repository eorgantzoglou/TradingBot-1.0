"""Structured output: the capability ladder and the validate -> re-ask repair loop.

Two jobs, both of which every serious LLM pipeline ends up writing:

  1. **The ladder.** Ask for the most constrained output the provider can
     actually give us, and walk down one rung at a time when it says no:

         native json_schema  ->  strict tool call  ->  json_object  ->  free text

     Only `Capabilities` decides where we start; only `UnsupportedParameterError`
     makes us move down. We never retry the same rung on the same error, because
     a provider that rejected a parameter once will reject it again.

  2. **The repair loop.** Every rung ends in `model_validate()`. A model that
     produces the wrong shape gets told exactly what was wrong and asked again,
     at most `max_repairs` times, and then we fail loudly. Returning a
     half-valid object would push the failure into `metrics/`, where it becomes
     a wrong number in a memo instead of a stack trace.

Why hand-written rather than `instructor`: the loop *is* the interesting part
(PLAN.md section 4.3), and the failure mode we care about -- a local server that
accepts `response_format` and silently ignores it -- is exactly what a library
that hides the ladder cannot show us.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from scout.harness.errors import (
    NoJsonFoundError,
    SchemaValidationError,
    UnsupportedParameterError,
)
from scout.harness.protocol import (
    Capabilities,
    Effort,
    LLMClient,
    Message,
    ModelResponse,
    OutputMode,
)

T = TypeVar("T", bound=BaseModel)

# Most constrained first. TEXT is always reachable: it needs nothing from the
# provider beyond returning a string.
LADDER: tuple[OutputMode, ...] = (
    OutputMode.NATIVE_SCHEMA,
    OutputMode.STRICT_TOOL,
    OutputMode.JSON_OBJECT,
    OutputMode.TEXT,
)

# Keys under these are user-chosen field names, not JSON Schema keywords. A
# model with a field literally called "minimum" must not lose it to keyword
# stripping.
_NAMED_KEY_CONTAINERS = frozenset({"properties", "$defs", "definitions", "patternProperties"})


@dataclass(frozen=True)
class StructuredResult(Generic[T]):
    """A validated object plus the story of how we got it.

    `attempts` counts provider calls issued, including rungs the provider
    rejected (so a clean first-try is 1); `repairs` counts re-asks; `mode_used`
    is the ladder rung that finally worked. All
    three are worth logging -- a run where the frontier model needed repairs is
    a prompt bug, and a run that silently fell to TEXT is a config bug.
    """

    value: T
    response: ModelResponse
    attempts: int
    mode_used: OutputMode
    repairs: int


# --------------------------------------------------------------------------
# Schema generation
# --------------------------------------------------------------------------


def schema_for(
    model_type: type[BaseModel],
    *,
    exclude_keywords: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """JSON Schema for `model_type`, minus keywords the provider rejects.

    IMPORTANT, and the reason `research/models.py` must use validators rather
    than schema keywords: Anthropic's JSON Schema subset omits `minimum`,
    `maximum`, `minLength` and `maxLength`. Those keywords do not error -- they
    are *dropped*, here and/or by the provider. A `Field(ge=0.0, le=1.0)` on a
    confidence score therefore constrains nothing on the wire.

    What saves us is that Pydantic enforces the constraint locally anyway: the
    wire schema is a hint to the decoder, `model_validate()` is the gate. A
    model that returns confidence 1.7 will fail validation and enter the repair
    loop even though the schema never mentioned a maximum. Never rely on the
    emitted schema to enforce a range.
    """
    schema = model_type.model_json_schema()
    return _normalize(schema, exclude_keywords)


def _normalize(node: Any, exclude: frozenset[str]) -> Any:
    """Strip excluded keywords and close every object, recursively."""
    if isinstance(node, list):
        return [_normalize(item, exclude) for item in node]
    if not isinstance(node, dict):
        return node

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in exclude:
            continue
        if key in _NAMED_KEY_CONTAINERS and isinstance(value, dict):
            out[key] = {name: _normalize(sub, exclude) for name, sub in value.items()}
            continue
        out[key] = _normalize(value, exclude)

    # Both frontier strict modes require this, and it costs nothing on the
    # others -- an extra key is never something we want from an extraction.
    if "properties" in out and out.get("type", "object") == "object":
        out.setdefault("additionalProperties", False)
    return out


# --------------------------------------------------------------------------
# JSON extraction
# --------------------------------------------------------------------------

_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_CLOSE_THINK = re.compile(r"</think\s*>", re.IGNORECASE)
_OPEN_THINK = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
_FENCE = re.compile(r"```[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n(.*?)```", re.DOTALL)

_OPENERS = {"{": "}", "[": "]"}


def extract_json(text: str) -> Any:
    """Pull the first parseable JSON value out of whatever the model said.

    Handles, in order of how often they actually happen: raw JSON; a
    ```json fenced block; JSON buried in prose ("Sure! Here's the object: {...}
    Let me know if..."); and a leading `<think>` block from a hybrid-thinking
    model whose reasoning was never split out of `content`.

    Raises NoJsonFoundError when there is no object or array at all.
    """
    stripped = _strip_reasoning(text)

    candidates = [stripped, *_fenced_blocks(stripped)]
    for candidate in candidates:
        parsed = _try_loads(candidate.strip())
        if parsed is not _MISSING:
            return parsed

    scanned = _scan_for_json(stripped)
    if scanned is not _MISSING:
        return scanned

    raise NoJsonFoundError(
        "No JSON object or array found in the model response.",
        raw_text=text,
        errors="expected a JSON object or array; found none",
    )


class _Missing:
    """Sentinel: `None` is a legitimate parse result, so we cannot use it."""


_MISSING = _Missing()


def _try_loads(text: str) -> Any:
    if not text:
        return _MISSING
    try:
        return json.loads(text)
    except ValueError:
        return _MISSING


def _strip_reasoning(text: str) -> str:
    """Remove inline reasoning that adapters did not (or could not) split out."""
    cleaned = _THINK_BLOCK.sub("", text)

    # Some servers emit only the closing tag, so everything before the last one
    # is reasoning.
    closes = list(_CLOSE_THINK.finditer(cleaned))
    if closes:
        cleaned = cleaned[closes[-1].end() :]

    # An unterminated opening tag means the model never came back out of
    # thinking; anything after it is reasoning, not an answer.
    opening = _OPEN_THINK.search(cleaned)
    if opening:
        cleaned = cleaned[: opening.start()]

    return cleaned


def _fenced_blocks(text: str) -> list[str]:
    return [match.group(1) for match in _FENCE.finditer(text)]


def _scan_for_json(text: str) -> Any:
    """Try every `{`/`[` in the text as the start of a balanced JSON value."""
    for index, char in enumerate(text):
        if char not in _OPENERS:
            continue
        chunk = _balanced_slice(text, index)
        if chunk is None:
            continue
        parsed = _try_loads(chunk)
        if parsed is not _MISSING:
            return parsed
    return _MISSING


def _balanced_slice(text: str, start: int) -> str | None:
    """Slice from `start` to its matching bracket, ignoring bracket characters
    inside string literals.

    String-awareness is the whole point: `{"note": "closes at 12:00 } maybe"}`
    balances only if the `}` inside the string is skipped, and `"a \\" b"` only
    if the escaped quote does not end the string.
    """
    opener = text[start]
    closer = _OPENERS[opener]
    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


# --------------------------------------------------------------------------
# The ladder + repair loop
# --------------------------------------------------------------------------


def ladder_for(capabilities: Capabilities) -> list[OutputMode]:
    """The rungs this client actually supports, most constrained first."""
    modes: list[OutputMode] = []
    if capabilities.native_schema:
        modes.append(OutputMode.NATIVE_SCHEMA)
    if capabilities.strict_tools:
        modes.append(OutputMode.STRICT_TOOL)
    if capabilities.json_object:
        modes.append(OutputMode.JSON_OBJECT)
    modes.append(OutputMode.TEXT)
    return modes


@dataclass
class _Tally:
    attempts: int = 0
    repairs: int = 0


async def complete_structured(
    client: LLMClient,
    messages: list[Message],
    model_type: type[T],
    *,
    effort: Effort | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_repairs: int = 2,
    cache_prefix_upto: int | None = None,
) -> StructuredResult[T]:
    """Get a validated `model_type` out of `client`, or raise.

    Walks the ladder on UnsupportedParameterError and runs the repair loop on
    invalid output. Raises SchemaValidationError once the repair budget is
    spent -- deliberately, because the alternative is a plausible-looking
    memo built on a field the model made up.
    """
    schema = schema_for(
        model_type, exclude_keywords=client.capabilities.restricted_schema_keywords
    )
    tally = _Tally()
    last_unsupported: UnsupportedParameterError | None = None

    for mode in ladder_for(client.capabilities):
        try:
            return await _run_mode(
                client,
                messages,
                model_type,
                schema=schema,
                mode=mode,
                tally=tally,
                effort=effort,
                temperature=temperature,
                max_tokens=max_tokens,
                max_repairs=max_repairs,
                cache_prefix_upto=cache_prefix_upto,
            )
        except UnsupportedParameterError as exc:
            # Down one rung. Never up, and never this rung again.
            last_unsupported = exc

    raise last_unsupported or UnsupportedParameterError(
        f"{client.name} advertised no usable output mode for {model_type.__name__}."
    )


async def _run_mode(
    client: LLMClient,
    messages: list[Message],
    model_type: type[T],
    *,
    schema: dict[str, Any],
    mode: OutputMode,
    tally: _Tally,
    effort: Effort | None,
    temperature: float | None,
    max_tokens: int | None,
    max_repairs: int,
    cache_prefix_upto: int | None,
) -> StructuredResult[T]:
    """One rung: call, validate, re-ask on failure, give up loudly."""
    conversation = _messages_for_mode(messages, schema, mode, model_type)

    while True:
        # Counted before the call, so a rung the provider rejects still shows up
        # as the round trip it cost.
        tally.attempts += 1
        response = await client.complete(
            conversation,
            output_mode=mode,
            json_schema=schema,
            schema_name=model_type.__name__,
            effort=effort,
            temperature=temperature,
            max_tokens=max_tokens,
            cache_prefix_upto=cache_prefix_upto,
        )

        try:
            value = parse_and_validate(response.text, model_type)
        except SchemaValidationError as exc:
            if tally.repairs >= max_repairs:
                raise
            tally.repairs += 1
            conversation = [
                *conversation,
                Message(role="assistant", content=response.text),
                Message(role="user", content=_repair_prompt(exc, model_type)),
            ]
            continue

        return StructuredResult(
            value=value,
            response=response,
            attempts=tally.attempts,
            mode_used=mode,
            repairs=tally.repairs,
        )


def parse_and_validate(text: str, model_type: type[T]) -> T:
    """Extract JSON from `text` and validate it. Raises SchemaValidationError."""
    data = extract_json(text)
    try:
        return model_type.model_validate(data)
    except ValidationError as exc:
        raise SchemaValidationError(
            f"Response did not validate as {model_type.__name__}.",
            raw_text=text,
            errors=str(exc),
        ) from exc


def _messages_for_mode(
    messages: list[Message],
    schema: dict[str, Any],
    mode: OutputMode,
    model_type: type[BaseModel],
) -> list[Message]:
    """Append the schema to the prompt for rungs the decoder does not enforce.

    NATIVE_SCHEMA and STRICT_TOOL constrain decoding, so the schema is already
    on the wire and repeating it only burns tokens and destabilises the cache
    prefix. JSON_OBJECT and TEXT have nothing but the prompt.
    """
    if mode in (OutputMode.NATIVE_SCHEMA, OutputMode.STRICT_TOOL):
        return list(messages)

    instruction = (
        f"Respond with a single JSON object matching this JSON Schema for "
        f"{model_type.__name__}. Output only the JSON -- no prose, no markdown "
        f"fences, no commentary.\n\n{json.dumps(schema, indent=2)}"
    )
    return [*messages, Message(role="user", content=instruction)]


def _repair_prompt(exc: SchemaValidationError, model_type: type[BaseModel]) -> str:
    """Quote the failure back at the model.

    Verbatim error text, because a paraphrase loses the field path -- and the
    field path is the only part the model needs.
    """
    return (
        f"That response was not a valid {model_type.__name__}. The validator "
        f"reported:\n\n{exc.errors}\n\n"
        "Return the corrected JSON object only. Do not explain the fix, do not "
        "apologise, and do not wrap the JSON in markdown fences. Keep every "
        "field you already got right."
    )
