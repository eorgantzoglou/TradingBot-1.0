"""Adapter tests: request shaping per OutputMode, and error mapping per status.

No network. Both adapters take a `client=` seam; the fakes below record the
kwargs the adapter built and hand back a real SDK response model, so the
assertions are against the actual wire shape rather than against a mock's idea
of it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import anthropic as anthropic_sdk
import httpx
import openai as openai_sdk
import pytest
from anthropic.types import Message as AnthropicMessage
from openai.types.chat import ChatCompletion

from scout.harness.adapters.anthropic import (
    RESTRICTED_SCHEMA_KEYWORDS,
    AnthropicAdapter,
    strip_unsupported_schema_keywords,
)
from scout.harness.adapters.openai_compat import (
    OpenAICompatAdapter,
    capabilities_for,
)
from scout.harness.errors import ProviderError, UnsupportedParameterError
from scout.harness.protocol import Capabilities, LLMClient, Message, OutputMode

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}
PROMPT = [Message(role="system", content="be brief"), Message(role="user", content="hello")]


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class _FakeEndpoint:
    """Records the request kwargs; returns a canned response or raises."""

    def __init__(self, result: Any):
        self._result = result
        self.captured: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.captured = kwargs
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def _openai_client(result: Any) -> Any:
    endpoint = _FakeEndpoint(result)
    return SimpleNamespace(chat=SimpleNamespace(completions=endpoint), _endpoint=endpoint)


def _anthropic_client(result: Any) -> Any:
    endpoint = _FakeEndpoint(result)
    return SimpleNamespace(messages=endpoint, _endpoint=endpoint)


def _completion(message: dict[str, Any], usage: dict[str, Any] | None = None) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "cmpl-1",
            "created": 0,
            "model": "test-model",
            "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            "usage": usage,
        }
    )


def _anthropic_message(
    content: list[dict[str, Any]], usage: dict[str, Any] | None = None
) -> AnthropicMessage:
    return AnthropicMessage.model_validate(
        {
            "id": "msg-1",
            "model": "claude-test",
            "role": "assistant",
            "type": "message",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "content": content,
            "usage": usage or {"input_tokens": 0, "output_tokens": 0},
        }
    )


def _openai_adapter(result: Any, **kwargs: Any) -> OpenAICompatAdapter:
    return OpenAICompatAdapter(
        model="test-model",
        api_key="placeholder",
        base_url="http://localhost:11434/v1",
        capabilities=Capabilities(native_schema=True, strict_tools=True),
        client=_openai_client(result),
        **kwargs,
    )


def _anthropic_adapter(result: Any, **kwargs: Any) -> AnthropicAdapter:
    return AnthropicAdapter(
        model="claude-test", api_key="placeholder", client=_anthropic_client(result), **kwargs
    )


def _status_error(kind: type, status: int, url: str) -> Exception:
    request = httpx.Request("POST", url)
    return kind("boom", response=httpx.Response(status, request=request), body=None)


# --------------------------------------------------------------------------
# Both adapters satisfy the protocol
# --------------------------------------------------------------------------


def test_adapters_are_llm_clients():
    assert isinstance(_openai_adapter(None), LLMClient)
    assert isinstance(_anthropic_adapter(None), LLMClient)


# --------------------------------------------------------------------------
# OpenAI-compat: capability defaults
# --------------------------------------------------------------------------


def test_capabilities_default_to_openai_when_no_base_url():
    caps = capabilities_for(None, "gpt-5")
    assert caps.native_schema and caps.strict_tools and caps.effort and caps.prompt_cache


def test_capabilities_for_ollama_do_not_claim_effort_or_strict_tools():
    caps = capabilities_for("http://localhost:11434/v1", "qwen3:8b")
    assert caps.native_schema is True
    assert caps.strict_tools is False
    assert caps.effort is False


def test_capabilities_for_an_unknown_backend_are_conservative():
    caps = capabilities_for("http://my-box.lan:9999/v1", "whatever")
    assert caps.native_schema is False
    assert caps.strict_tools is False
    assert caps.json_object is True


def test_capabilities_are_not_sniffed_from_the_model_name():
    # Same server, wildly different models -> identical defaults. Only the
    # caller knows how the server was launched.
    assert capabilities_for("http://localhost:8000/v1", "qwen3-32b") == capabilities_for(
        "http://localhost:8000/v1", "llama-3.1-8b"
    )


# --------------------------------------------------------------------------
# OpenAI-compat: request shaping per OutputMode
# --------------------------------------------------------------------------


async def test_text_mode_sends_nothing_extra():
    adapter = _openai_adapter(_completion({"role": "assistant", "content": "hi"}))
    await adapter.complete(PROMPT, output_mode=OutputMode.TEXT)
    sent = adapter._client._endpoint.captured
    assert "response_format" not in sent
    assert "tools" not in sent
    assert sent["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]


async def test_json_object_mode():
    adapter = _openai_adapter(_completion({"role": "assistant", "content": "{}"}))
    await adapter.complete(PROMPT, output_mode=OutputMode.JSON_OBJECT)
    assert adapter._client._endpoint.captured["response_format"] == {"type": "json_object"}


async def test_native_schema_mode_is_strict():
    adapter = _openai_adapter(_completion({"role": "assistant", "content": "{}"}))
    await adapter.complete(PROMPT, output_mode=OutputMode.NATIVE_SCHEMA, json_schema=SCHEMA)
    response_format = adapter._client._endpoint.captured["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == SCHEMA


async def test_strict_tool_mode_forces_the_tool_and_reads_its_arguments():
    completion = _completion(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "response", "arguments": '{"answer": "42"}'},
                }
            ],
        }
    )
    adapter = _openai_adapter(completion)
    result = await adapter.complete(
        PROMPT, output_mode=OutputMode.STRICT_TOOL, json_schema=SCHEMA, schema_name="response"
    )

    sent = adapter._client._endpoint.captured
    assert sent["tools"][0]["function"]["strict"] is True
    assert sent["tools"][0]["function"]["parameters"] == SCHEMA
    assert sent["tool_choice"] == {"type": "function", "function": {"name": "response"}}
    assert json.loads(result.text) == {"answer": "42"}


async def test_schema_modes_refuse_to_send_without_a_schema():
    adapter = _openai_adapter(_completion({"role": "assistant", "content": "hi"}))
    with pytest.raises(ValueError):
        await adapter.complete(PROMPT, output_mode=OutputMode.NATIVE_SCHEMA)


async def test_sampling_and_effort_params():
    adapter = _openai_adapter(_completion({"role": "assistant", "content": "hi"}))
    await adapter.complete(PROMPT, effort="none", temperature=0.2, max_tokens=512)
    sent = adapter._client._endpoint.captured
    assert sent["reasoning_effort"] == "none"
    assert sent["temperature"] == 0.2
    assert sent["max_tokens"] == 512


async def test_unset_effort_and_temperature_are_omitted_entirely():
    adapter = _openai_adapter(_completion({"role": "assistant", "content": "hi"}))
    await adapter.complete(PROMPT)
    sent = adapter._client._endpoint.captured
    assert "reasoning_effort" not in sent
    assert "temperature" not in sent
    assert "max_tokens" not in sent


# --------------------------------------------------------------------------
# OpenAI-compat: response normalization
# --------------------------------------------------------------------------


async def test_reasoning_is_lifted_out_of_the_response():
    adapter = _openai_adapter(
        _completion({"role": "assistant", "content": "<think>a</think>42", "reasoning": "b"})
    )
    result = await adapter.complete(PROMPT)
    assert result.text == "42"
    assert result.reasoning == "b\n\na"
    assert result.output_mode is OutputMode.TEXT
    assert result.raw["id"] == "cmpl-1"


async def test_usage_includes_cached_and_reasoning_tokens():
    adapter = _openai_adapter(
        _completion(
            {"role": "assistant", "content": "42"},
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 30,
                "total_tokens": 130,
                "prompt_tokens_details": {"cached_tokens": 64},
                "completion_tokens_details": {"reasoning_tokens": 12},
            },
        )
    )
    result = await adapter.complete(PROMPT)
    # cost.py bills input and cached input at different rates and requires them
    # to be disjoint, so the cached tokens come back out of prompt_tokens.
    assert result.usage.input_tokens == 36
    assert result.usage.cached_input_tokens == 64
    assert result.usage.reasoning_tokens == 12


async def test_missing_usage_block_is_not_fatal():
    adapter = _openai_adapter(_completion({"role": "assistant", "content": "42"}))
    result = await adapter.complete(PROMPT)
    assert result.usage.input_tokens == 0


# --------------------------------------------------------------------------
# OpenAI-compat: error mapping
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "status"),
    [(openai_sdk.BadRequestError, 400), (openai_sdk.UnprocessableEntityError, 422)],
)
async def test_400_and_422_are_downgradeable(kind, status):
    adapter = _openai_adapter(_status_error(kind, status, "http://localhost:11434/v1"))
    with pytest.raises(UnsupportedParameterError) as excinfo:
        await adapter.complete(PROMPT)
    assert excinfo.value.status == status


async def test_401_points_at_the_api_key():
    adapter = _openai_adapter(
        _status_error(openai_sdk.AuthenticationError, 401, "http://localhost:11434/v1")
    )
    with pytest.raises(ProviderError) as excinfo:
        await adapter.complete(PROMPT)
    assert not isinstance(excinfo.value, UnsupportedParameterError)
    assert "LLM_API_KEY" in str(excinfo.value)


async def test_404_names_the_model_and_the_models_endpoint():
    adapter = _openai_adapter(
        _status_error(openai_sdk.NotFoundError, 404, "http://localhost:11434/v1")
    )
    with pytest.raises(ProviderError) as excinfo:
        await adapter.complete(PROMPT)
    message = str(excinfo.value)
    assert "test-model" in message
    assert "http://localhost:11434/v1/models" in message


async def test_connection_refused_says_the_server_is_not_running():
    adapter = _openai_adapter(
        openai_sdk.APIConnectionError(request=httpx.Request("POST", "http://localhost:11434/v1"))
    )
    with pytest.raises(ProviderError) as excinfo:
        await adapter.complete(PROMPT)
    assert "nothing is listening" in str(excinfo.value)


async def test_timeout_says_the_server_went_quiet():
    adapter = _openai_adapter(
        openai_sdk.APITimeoutError(request=httpx.Request("POST", "http://localhost:11434/v1"))
    )
    with pytest.raises(ProviderError) as excinfo:
        await adapter.complete(PROMPT)
    message = str(excinfo.value)
    assert "went quiet" in message
    assert "cold-loading" in message


async def test_other_statuses_are_plain_provider_errors():
    adapter = _openai_adapter(
        _status_error(openai_sdk.InternalServerError, 500, "http://localhost:11434/v1")
    )
    with pytest.raises(ProviderError) as excinfo:
        await adapter.complete(PROMPT)
    assert excinfo.value.status == 500
    assert not isinstance(excinfo.value, UnsupportedParameterError)


# --------------------------------------------------------------------------
# Anthropic: schema subset
# --------------------------------------------------------------------------


def test_restricted_keywords_cover_the_documented_subset():
    for keyword in ("minimum", "maximum", "minLength", "maxLength", "multipleOf", "maxItems"):
        assert keyword in RESTRICTED_SCHEMA_KEYWORDS


def test_strip_removes_unsupported_keywords_recursively():
    schema = {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "minimum": 0, "maximum": 100},
            "tags": {"type": "array", "items": {"type": "string", "maxLength": 20}},
        },
    }
    cleaned = strip_unsupported_schema_keywords(schema)
    assert cleaned["properties"]["score"] == {"type": "integer"}
    assert cleaned["properties"]["tags"]["items"] == {"type": "string"}


def test_strip_keeps_a_property_that_is_merely_named_like_a_keyword():
    schema = {"type": "object", "properties": {"maximum": {"type": "number", "minimum": 0}}}
    cleaned = strip_unsupported_schema_keywords(schema)
    assert "maximum" in cleaned["properties"]
    assert cleaned["properties"]["maximum"] == {"type": "number"}


def test_strip_keeps_minitems_only_when_it_is_zero_or_one():
    kept = strip_unsupported_schema_keywords({"type": "array", "minItems": 1})
    dropped = strip_unsupported_schema_keywords({"type": "array", "minItems": 3})
    assert kept["minItems"] == 1
    assert "minItems" not in dropped


def test_strip_forces_additional_properties_false():
    cleaned = strip_unsupported_schema_keywords(SCHEMA)
    assert cleaned["additionalProperties"] is False


# --------------------------------------------------------------------------
# Anthropic: request shaping
# --------------------------------------------------------------------------


async def test_system_messages_go_top_level_not_in_messages():
    adapter = _anthropic_adapter(_anthropic_message([{"type": "text", "text": "hi"}]))
    await adapter.complete(PROMPT)
    sent = adapter._client._endpoint.captured
    assert sent["system"] == [{"type": "text", "text": "be brief"}]
    assert [m["role"] for m in sent["messages"]] == ["user"]


async def test_native_schema_uses_output_config():
    adapter = _anthropic_adapter(_anthropic_message([{"type": "text", "text": "{}"}]))
    await adapter.complete(PROMPT, output_mode=OutputMode.NATIVE_SCHEMA, json_schema=SCHEMA)
    sent = adapter._client._endpoint.captured
    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert sent["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert "extra_headers" not in sent  # structured outputs are GA, no beta header


async def test_strict_tool_forces_the_tool_and_returns_its_input_as_text():
    response = _anthropic_message(
        [{"type": "tool_use", "id": "tu-1", "name": "response", "input": {"answer": "42"}}]
    )
    adapter = _anthropic_adapter(response)
    result = await adapter.complete(
        PROMPT, output_mode=OutputMode.STRICT_TOOL, json_schema=SCHEMA, effort="high"
    )
    sent = adapter._client._endpoint.captured
    assert sent["tools"][0]["strict"] is True
    assert sent["tool_choice"] == {"type": "tool", "name": "response"}
    # Forced tool use and extended thinking are mutually exclusive on this API.
    assert sent["thinking"] == {"type": "disabled"}
    assert json.loads(result.text) == {"answer": "42"}


async def test_json_object_mode_is_prompted_not_configured():
    adapter = _anthropic_adapter(_anthropic_message([{"type": "text", "text": "{}"}]))
    await adapter.complete(PROMPT, output_mode=OutputMode.JSON_OBJECT, json_schema=SCHEMA)
    sent = adapter._client._endpoint.captured
    assert "output_config" not in sent
    assert "tools" not in sent
    assert "single JSON object" in sent["system"][-1]["text"]


async def test_thinking_is_enabled_and_temperature_dropped():
    adapter = _anthropic_adapter(
        _anthropic_message([{"type": "text", "text": "42"}]), default_max_tokens=8000
    )
    await adapter.complete(PROMPT, effort="medium", temperature=0.2)
    sent = adapter._client._endpoint.captured
    assert sent["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    # Anthropic only accepts temperature=1 while thinking, so ours is dropped.
    assert "temperature" not in sent


async def test_effort_none_disables_thinking_and_keeps_temperature():
    adapter = _anthropic_adapter(_anthropic_message([{"type": "text", "text": "42"}]))
    await adapter.complete(PROMPT, effort="none", temperature=0.2)
    sent = adapter._client._endpoint.captured
    assert sent["thinking"] == {"type": "disabled"}
    assert sent["temperature"] == 0.2


async def test_cache_breakpoint_lands_on_the_named_message():
    adapter = _anthropic_adapter(_anthropic_message([{"type": "text", "text": "42"}]))
    await adapter.complete(PROMPT, cache_prefix_upto=0)
    sent = adapter._client._endpoint.captured
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in sent["messages"][0]["content"][0]


async def test_no_cache_breakpoint_by_default():
    adapter = _anthropic_adapter(_anthropic_message([{"type": "text", "text": "42"}]))
    await adapter.complete(PROMPT)
    sent = adapter._client._endpoint.captured
    assert "cache_control" not in sent["system"][0]


async def test_messages_without_a_user_turn_are_rejected():
    adapter = _anthropic_adapter(_anthropic_message([{"type": "text", "text": "42"}]))
    with pytest.raises(ValueError):
        await adapter.complete([Message(role="system", content="only a system prompt")])


# --------------------------------------------------------------------------
# Anthropic: response normalization and usage
# --------------------------------------------------------------------------


async def test_thinking_blocks_become_reasoning():
    response = _anthropic_message(
        [
            {"type": "thinking", "thinking": "let me check", "signature": "sig"},
            {"type": "text", "text": "42"},
        ]
    )
    adapter = _anthropic_adapter(response)
    result = await adapter.complete(PROMPT)
    assert result.text == "42"
    assert result.reasoning == "let me check"


async def test_thinking_with_no_text_block_is_an_empty_content_error():
    from scout.harness.errors import EmptyContentError

    response = _anthropic_message(
        [{"type": "thinking", "thinking": "ran out of budget", "signature": "sig"}]
    )
    adapter = _anthropic_adapter(response)
    with pytest.raises(EmptyContentError):
        await adapter.complete(PROMPT)


async def test_usage_keeps_cache_reads_disjoint_from_input():
    response = _anthropic_message(
        [{"type": "text", "text": "42"}],
        usage={
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 7,
            "cache_read_input_tokens": 3,
        },
    )
    adapter = _anthropic_adapter(response)
    result = await adapter.complete(PROMPT)
    # Cache writes fold into input (billed at 1.25x); cache reads stay separate
    # (billed at 0.1x). The two fields never overlap.
    assert result.usage.input_tokens == 17
    assert result.usage.cached_input_tokens == 3
    assert result.usage.output_tokens == 5


# --------------------------------------------------------------------------
# Anthropic: error mapping
# --------------------------------------------------------------------------


async def test_anthropic_400_is_downgradeable():
    adapter = _anthropic_adapter(
        _status_error(anthropic_sdk.BadRequestError, 400, "https://api.anthropic.com/v1/messages")
    )
    with pytest.raises(UnsupportedParameterError):
        await adapter.complete(PROMPT)


async def test_anthropic_401_points_at_the_key():
    adapter = _anthropic_adapter(
        _status_error(
            anthropic_sdk.AuthenticationError, 401, "https://api.anthropic.com/v1/messages"
        )
    )
    with pytest.raises(ProviderError) as excinfo:
        await adapter.complete(PROMPT)
    assert "ANTHROPIC_API_KEY" in str(excinfo.value)


async def test_anthropic_404_names_the_model():
    adapter = _anthropic_adapter(
        _status_error(anthropic_sdk.NotFoundError, 404, "https://api.anthropic.com/v1/messages")
    )
    with pytest.raises(ProviderError) as excinfo:
        await adapter.complete(PROMPT)
    assert "claude-test" in str(excinfo.value)


async def test_anthropic_timeout_is_distinguished_from_refusal():
    timeout = _anthropic_adapter(
        anthropic_sdk.APITimeoutError(
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        )
    )
    with pytest.raises(ProviderError) as excinfo:
        await timeout.complete(PROMPT)
    assert "went quiet" in str(excinfo.value)

    refused = _anthropic_adapter(
        anthropic_sdk.APIConnectionError(
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        )
    )
    with pytest.raises(ProviderError) as excinfo:
        await refused.complete(PROMPT)
    assert "Could not connect" in str(excinfo.value)
