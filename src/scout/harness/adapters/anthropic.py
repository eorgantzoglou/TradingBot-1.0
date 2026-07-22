"""Anthropic adapter, native Messages API.

Not the OpenAI-compat shim: the native API is where the two things we actually
want live -- explicit `cache_control` breakpoints (reads bill at 0.1x, so the
break-even is two reads, and the 60-candidate fan-out in stage 3 shares one long
system prompt) and thinking blocks that arrive as structured content rather than
as `<think>` tags glued into the text.

The one constraint to absorb up front is the JSON Schema subset, which is
narrower than OpenAI's. See RESTRICTED_SCHEMA_KEYWORDS below.
"""

from __future__ import annotations

import json
from typing import Any, NoReturn

import anthropic

from scout.harness.errors import ProviderError, UnsupportedParameterError
from scout.harness.protocol import (
    Capabilities,
    Effort,
    Message,
    ModelResponse,
    OutputMode,
    Usage,
)
from scout.harness.reasoning import anthropic_thinking_params, normalize_message

ANTHROPIC_BASE_URL = "https://api.anthropic.com"

# Keywords Anthropic's JSON Schema subset does not accept. Sending them is a 400,
# so we strip them before the request instead of discovering it at runtime.
#
# The trap this creates is in research/models.py, not here: a Pydantic
# `Field(ge=0)` emits `minimum: 0`, which vanishes at this boundary. Numeric and
# length constraints must therefore be written as Pydantic *validators*, which
# run on our side after parsing, not as schema keywords, which do not survive.
#
# Deliberately NOT stripped: `$ref` / `$defs`. Removing those produces an
# invalid schema rather than a laxer one. Anthropic rejects *recursive* schemas,
# which is a modelling constraint to respect upstream, not something a rewriter
# can fix.
RESTRICTED_SCHEMA_KEYWORDS: frozenset[str] = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "maxItems",
        "uniqueItems",
        "minContains",
        "maxContains",
        "minProperties",
        "maxProperties",
    }
)

# Sub-objects of a schema whose keys are user-chosen names, not keywords. Without
# this, a field legitimately called "maximum" would be stripped out of
# `properties` as if it were a constraint.
_NAME_KEYED_CONTAINERS = frozenset({"properties", "$defs", "definitions", "patternProperties"})

ANTHROPIC_CAPABILITIES = Capabilities(
    native_schema=True,
    strict_tools=True,
    # No native "JSON mode" -- the JSON_OBJECT rung is served by prompting, which
    # is exactly what that rung means anyway.
    json_object=True,
    effort=True,
    prompt_cache=True,
    restricted_schema_keywords=RESTRICTED_SCHEMA_KEYWORDS,
)

_JSON_ONLY_INSTRUCTION = (
    "Respond with a single JSON object and nothing else. No prose before or "
    "after it, and no markdown code fence."
)


def strip_unsupported_schema_keywords(schema: Any) -> Any:
    """Rewrite a JSON Schema into Anthropic's subset.

    Recursive. Also forces `additionalProperties: false` on every object, which
    Anthropic requires rather than merely permits.
    """
    if isinstance(schema, list):
        return [strip_unsupported_schema_keywords(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key in RESTRICTED_SCHEMA_KEYWORDS:
            continue
        # minItems survives, but only as 0 or 1 -- "at least three" is not
        # expressible and has to become a validator.
        if key == "minItems" and value not in (0, 1):
            continue
        if key in _NAME_KEYED_CONTAINERS and isinstance(value, dict):
            cleaned[key] = {
                name: strip_unsupported_schema_keywords(sub) for name, sub in value.items()
            }
            continue
        cleaned[key] = strip_unsupported_schema_keywords(value)

    if cleaned.get("type") == "object" or "properties" in cleaned:
        cleaned["additionalProperties"] = False
    return cleaned


class AnthropicAdapter:
    """`LLMClient` over the Anthropic SDK. See `protocol.LLMClient` for the contract."""

    name = "anthropic"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 180.0,
        capabilities: Capabilities = ANTHROPIC_CAPABILITIES,
        default_max_tokens: int = 4096,
        client: Any | None = None,
    ):
        self.model = model
        self.capabilities = capabilities
        self.base_url = base_url or ANTHROPIC_BASE_URL
        # Anthropic requires max_tokens on every request; the protocol makes it
        # optional, so the adapter owns the default.
        self._default_max_tokens = default_max_tokens
        self._client = client or anthropic.AsyncAnthropic(
            api_key=api_key, base_url=base_url, timeout=timeout
        )

    async def complete(
        self,
        messages: list[Message],
        *,
        output_mode: OutputMode = OutputMode.TEXT,
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "response",
        effort: Effort | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache_prefix_upto: int | None = None,
    ) -> ModelResponse:
        budget = max_tokens or self._default_max_tokens
        system_blocks, turns = self._split_messages(messages, cache_prefix_upto)
        if not turns:
            raise ValueError("Anthropic needs at least one non-system message.")

        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": budget,
            "messages": turns,
        }

        thinking = anthropic_thinking_params(effort, budget)
        if (
            output_mode is OutputMode.STRICT_TOOL
            and thinking.get("thinking", {}).get("type") == "enabled"
        ):
            # The API rejects extended thinking combined with a forced
            # tool_choice. The forced tool is what makes this rung a rung, so
            # thinking is what gives way.
            thinking = {"thinking": {"type": "disabled"}}
        params.update(thinking)

        if temperature is not None and thinking.get("thinking", {}).get("type") != "enabled":
            # Anthropic rejects any temperature other than 1 while thinking is
            # enabled, so we drop ours rather than fail the call.
            params["temperature"] = temperature

        system_blocks = system_blocks + self._system_suffix(output_mode, json_schema)
        if system_blocks:
            # System prompts go top-level, never in `messages` -- a "system" role
            # inside messages is a 400 on this API.
            params["system"] = system_blocks

        params.update(self._output_params(output_mode, json_schema, schema_name))

        try:
            response = await self._client.messages.create(**params)
        except anthropic.APIError as exc:
            self._raise_mapped(exc)

        text, reasoning = normalize_message(_collect_blocks(response, output_mode))

        return ModelResponse(
            text=text,
            reasoning=reasoning,
            usage=_usage(getattr(response, "usage", None)),
            model=getattr(response, "model", self.model) or self.model,
            output_mode=output_mode,
            raw=response.model_dump(),
        )

    def _split_messages(
        self, messages: list[Message], cache_prefix_upto: int | None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Separate system messages out and place the cache breakpoint.

        `cache_prefix_upto` is the index of the last message that is stable
        across calls; its final content block carries the breakpoint, and
        everything before it is cached with it. Cached reads bill at 0.1x base
        input, so a breakpoint pays for itself on the *second* read -- worth it
        for a shared rubric across 60 candidates, not worth it for a one-off.
        """
        system_blocks: list[dict[str, Any]] = []
        turns: list[dict[str, Any]] = []
        blocks_by_index: dict[int, dict[str, Any]] = {}

        for index, message in enumerate(messages):
            block: dict[str, Any] = {"type": "text", "text": message.content}
            blocks_by_index[index] = block
            if message.role == "system":
                system_blocks.append(block)
            else:
                turns.append({"role": message.role, "content": [block]})

        if cache_prefix_upto is not None and cache_prefix_upto in blocks_by_index:
            blocks_by_index[cache_prefix_upto]["cache_control"] = {"type": "ephemeral"}

        return system_blocks, turns

    def _system_suffix(
        self, output_mode: OutputMode, json_schema: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """The JSON_OBJECT rung, which Anthropic has no native mode for."""
        if output_mode is not OutputMode.JSON_OBJECT:
            return []
        instruction = _JSON_ONLY_INSTRUCTION
        if json_schema is not None:
            instruction += "\n\nIt must conform to this JSON Schema:\n" + json.dumps(
                strip_unsupported_schema_keywords(json_schema), indent=2
            )
        # Appended after the caller's system text and never given a cache
        # breakpoint: it is stable, but it is also short enough not to matter.
        return [{"type": "text", "text": instruction}]

    def _output_params(
        self, output_mode: OutputMode, json_schema: dict[str, Any] | None, schema_name: str
    ) -> dict[str, Any]:
        if output_mode in (OutputMode.TEXT, OutputMode.JSON_OBJECT):
            return {}

        if json_schema is None:
            raise ValueError(f"output_mode={output_mode.value} requires a json_schema")

        schema = strip_unsupported_schema_keywords(json_schema)

        if output_mode is OutputMode.NATIVE_SCHEMA:
            # Structured outputs are GA -- no beta header any more.
            return {"output_config": {"format": {"type": "json_schema", "schema": schema}}}

        return {
            "tools": [
                {
                    "name": schema_name,
                    "description": f"Return the {schema_name} payload.",
                    "input_schema": schema,
                    "strict": True,
                }
            ],
            "tool_choice": {"type": "tool", "name": schema_name},
        }

    def _raise_mapped(self, exc: anthropic.APIError) -> NoReturn:
        endpoint = f"{self.base_url}/v1/messages"

        # APITimeoutError subclasses APIConnectionError; order matters.
        if isinstance(exc, anthropic.APITimeoutError):
            raise ProviderError(
                f"{self.base_url} accepted the connection and then went quiet "
                f"(timed out waiting for '{self.model}'). Raise REQUEST_TIMEOUT_S, "
                "or lower MAX_TOKENS / REASONING_EFFORT so the response arrives sooner.",
                endpoint=endpoint,
            ) from exc

        if isinstance(exc, anthropic.APIConnectionError):
            raise ProviderError(
                f"Could not connect to {self.base_url} -- nothing answered. Check "
                "network access and LLM_BASE_URL.",
                endpoint=endpoint,
            ) from exc

        status = getattr(exc, "status_code", None)
        detail = str(exc)

        if status in (400, 422):
            raise UnsupportedParameterError(
                f"Anthropic rejected a request parameter for '{self.model}' "
                f"(HTTP {status}): {detail}",
                status=status,
                endpoint=endpoint,
            ) from exc

        if status in (401, 403):
            raise ProviderError(
                f"Anthropic refused the credentials (HTTP {status}). Check "
                "ANTHROPIC_API_KEY / LLM_API_KEY.",
                status=status,
                endpoint=endpoint,
            ) from exc

        if status == 404:
            raise ProviderError(
                f"Model '{self.model}' not found (HTTP 404). Check MODEL_NAME against "
                f"the list at {self.base_url}/v1/models.",
                status=status,
                endpoint=endpoint,
            ) from exc

        raise ProviderError(
            f"Anthropic returned an error for '{self.model}'"
            + (f" (HTTP {status})" if status else "")
            + f": {detail}",
            status=status,
            endpoint=endpoint,
        ) from exc


def _collect_blocks(response: Any, output_mode: OutputMode) -> dict[str, Any]:
    """Flatten Anthropic content blocks into the shape normalize_message expects.

    Routed through the shared normalizer rather than parsed inline so that the
    empty-content rule -- thinking present, answer absent -- behaves identically
    on every provider.
    """
    text_parts: list[str] = []
    reasoning_parts: list[str] = []

    for block in getattr(response, "content", None) or []:
        kind = getattr(block, "type", None)
        if kind == "text":
            text_parts.append(block.text)
        elif kind == "thinking":
            reasoning_parts.append(getattr(block, "thinking", "") or "")
        elif kind == "redacted_thinking":
            # Encrypted by the API; unreadable by us, but its presence is worth
            # recording so a trace does not look like the model thought nothing.
            reasoning_parts.append("[redacted thinking block]")
        elif kind == "tool_use" and output_mode is OutputMode.STRICT_TOOL:
            text_parts.append(json.dumps(block.input))

    return {"content": "".join(text_parts), "thinking": "\n\n".join(reasoning_parts)}


def _usage(raw: Any) -> Usage:
    """Map Anthropic usage onto ours.

    `input_tokens` and `cached_input_tokens` are DISJOINT, as cost.py requires:
    cache *reads* (0.1x) become cached_input_tokens, cache *writes* (1.25x) are
    folded into plain input, and Anthropic's own `input_tokens` already excludes
    both. Folding writes in rather than tracking them separately slightly
    under-bills the first call of a cached run; that is the deliberate trade for
    keeping Usage to four fields.

    Thinking tokens are not reported separately -- Anthropic counts them inside
    `output_tokens` -- so reasoning_tokens stays 0 rather than being invented.
    """
    if raw is None:
        return Usage()
    base = int(getattr(raw, "input_tokens", 0) or 0)
    created = int(getattr(raw, "cache_creation_input_tokens", 0) or 0)
    read = int(getattr(raw, "cache_read_input_tokens", 0) or 0)
    return Usage(
        input_tokens=base + created,
        output_tokens=int(getattr(raw, "output_tokens", 0) or 0),
        cached_input_tokens=read,
        reasoning_tokens=0,
    )
