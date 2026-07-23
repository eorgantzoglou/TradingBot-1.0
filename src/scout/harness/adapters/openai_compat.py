"""OpenAI-shaped adapter: OpenAI itself, Ollama, LM Studio, vLLM, OpenRouter.

One adapter for five backends, because the `openai` SDK with `base_url=` already
speaks to all of them. What differs between them is not the wire format but what
they *support*, which is why `Capabilities` is a constructor argument rather than
something this file guesses.

**Capabilities are never sniffed from the model name.** A local server's
capabilities depend on which model is loaded and how the server was launched --
vLLM only emits `reasoning_content` if it was started with `--reasoning-parser`,
and the same GGUF behaves differently under llama.cpp and LM Studio. Guessing
from the string "qwen3" would be wrong roughly half the time and wrong silently.
`capabilities_for()` offers defaults per backend; the caller overrides them.
"""

from __future__ import annotations

from typing import Any, NoReturn

import openai

from scout.harness.errors import ProviderError, UnsupportedParameterError
from scout.harness.protocol import (
    Capabilities,
    Effort,
    Message,
    ModelResponse,
    OutputMode,
    Usage,
)
from scout.harness.reasoning import (
    deepseek_effort_params,
    normalize_message,
    openai_effort_params,
)

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Conservative defaults per backend. Every one of these is overridable, and the
# caller is expected to override once it knows what it actually launched.
#
#   OpenAI      everything, including implicit prompt caching above ~1024 tokens
#               of stable prefix (no cache_control to set -- just keep the prefix
#               stable, which is why prompts are ordered stable-first).
#   vLLM        XGrammar is the default constrained-decoding backend, so native
#               schema is safe. `reasoning_effort` is not: it needs
#               --reasoning-parser at server launch, so default it off.
#   Ollama      supports response_format json_schema via its OpenAI-compat shim,
#               but not strict tools and not reasoning_effort (it takes `think`).
#   LM Studio   same story, and its strict-tool support has been inconsistent.
#   OpenRouter  a router, not a model: whether a request survives depends on
#               which upstream it picks today. Default everything off but
#               json_object and effort, both of which it normalizes itself.
_BACKEND_DEFAULTS: dict[str, Capabilities] = {
    "api.openai.com": Capabilities(
        native_schema=True, strict_tools=True, json_object=True, effort=True, prompt_cache=True
    ),
    "openrouter.ai": Capabilities(
        native_schema=False, strict_tools=False, json_object=True, effort=True, prompt_cache=False
    ),
    # DeepSeek's OpenAI-compatible endpoint. json_object is solid; native
    # json_schema is not documented as supported (and not in thinking mode), so
    # start the ladder at the JSON-object rung rather than eat a 400 on every
    # call. Context caching is automatic (like OpenAI), so prompt_cache is on.
    "api.deepseek.com": Capabilities(
        native_schema=False, strict_tools=False, json_object=True, effort=True, prompt_cache=True
    ),
    # Default ports, because that is how these are addressed in practice.
    ":11434": Capabilities(  # Ollama
        native_schema=True, strict_tools=False, json_object=True, effort=False, prompt_cache=False
    ),
    ":1234": Capabilities(  # LM Studio
        native_schema=True, strict_tools=False, json_object=True, effort=False, prompt_cache=False
    ),
    ":8000": Capabilities(  # vLLM
        native_schema=True, strict_tools=True, json_object=True, effort=False, prompt_cache=False
    ),
}

_UNKNOWN_BACKEND = Capabilities(
    native_schema=False, strict_tools=False, json_object=True, effort=False, prompt_cache=False
)


def capabilities_for(base_url: str | None, model: str) -> Capabilities:
    """Best-guess defaults for a backend, meant to be overridden.

    `model` is accepted so callers can key their own overrides off it, but it is
    deliberately not inspected here -- see the module docstring.
    """
    del model  # See module docstring: sniffing the model name is the bug, not the feature.
    if not base_url:
        return _BACKEND_DEFAULTS["api.openai.com"]

    haystack = base_url.lower()
    for marker, capabilities in _BACKEND_DEFAULTS.items():
        if marker in haystack:
            return capabilities
    return _UNKNOWN_BACKEND


def reasoning_style_for(base_url: str | None) -> str:
    """Which reasoning-request dialect this endpoint speaks.

    Keyed off the base URL (an explicit config value), the same way
    `capabilities_for` keys its defaults -- NOT off the model name, which the
    module docstring explains is the bug. Only DeepSeek needs a non-OpenAI
    dialect so far: its `reasoning_effort` rejects "none", so thinking is turned
    off through its own `thinking` toggle instead (see `deepseek_effort_params`).
    """
    if base_url and "deepseek" in base_url.lower():
        return "deepseek"
    return "openai"


class OpenAICompatAdapter:
    """`LLMClient` over the OpenAI SDK. See `protocol.LLMClient` for the contract."""

    name = "openai_compat"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        capabilities: Capabilities,
        base_url: str | None = None,
        timeout: float = 180.0,
        reasoning_style: str | None = None,
        client: Any | None = None,
    ):
        self.model = model
        self.capabilities = capabilities
        self.base_url = base_url or OPENAI_DEFAULT_BASE_URL
        # How this endpoint expects the reasoning knob to be set. Auto-detected
        # from the base URL, overridable for tests.
        self.reasoning_style = reasoning_style or reasoning_style_for(self.base_url)
        # `client` is the seam the tests inject through; production always builds
        # its own. Retries/backoff/429 handling are the SDK's job, not ours.
        self._client = client or openai.AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout
        )

    def _effort_params(self, effort: Effort | None) -> dict[str, Any]:
        """Effort translated for this endpoint's dialect."""
        if self.reasoning_style == "deepseek":
            return deepseek_effort_params(effort)
        return openai_effort_params(effort)

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
        # OpenAI-shaped backends cache implicitly above a stable ~1024-token
        # prefix; there is no breakpoint to place, so the hint is unused here.
        del cache_prefix_upto

        params: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            # `max_tokens` rather than `max_completion_tokens`: every local
            # server accepts the former and most reject the latter. OpenAI's
            # reasoning models want the newer name and return a 400 saying so,
            # which reaches the caller as UnsupportedParameterError carrying the
            # provider's own (actionable) text.
            params["max_tokens"] = max_tokens
        params.update(self._effort_params(effort))
        params.update(self._output_params(output_mode, json_schema, schema_name))

        try:
            completion = await self._client.chat.completions.create(**params)
        except openai.APIError as exc:
            self._raise_mapped(exc)

        choice = completion.choices[0]
        raw_message = choice.message.model_dump()

        if output_mode is OutputMode.STRICT_TOOL:
            # In strict-tool mode the answer is the tool's arguments and
            # `content` is empty by design -- feed the arguments in as content so
            # the empty-content rule still fires when the model produced neither.
            arguments = _tool_arguments(choice.message)
            if arguments:
                raw_message = {**raw_message, "content": arguments}

        text, reasoning = normalize_message(raw_message)

        return ModelResponse(
            text=text,
            reasoning=reasoning,
            usage=_usage(getattr(completion, "usage", None)),
            model=getattr(completion, "model", self.model) or self.model,
            output_mode=output_mode,
            raw=completion.model_dump(),
        )

    def _output_params(
        self, output_mode: OutputMode, json_schema: dict[str, Any] | None, schema_name: str
    ) -> dict[str, Any]:
        if output_mode is OutputMode.TEXT:
            return {}

        if output_mode is OutputMode.JSON_OBJECT:
            return {"response_format": {"type": "json_object"}}

        if json_schema is None:
            # Our bug, not the provider's: fail here rather than send a request
            # that can only come back as a confusing 400.
            raise ValueError(f"output_mode={output_mode.value} requires a json_schema")

        if output_mode is OutputMode.NATIVE_SCHEMA:
            return {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": json_schema,
                        "strict": True,
                    },
                }
            }

        return {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": schema_name,
                        "description": f"Return the {schema_name} payload.",
                        "parameters": json_schema,
                        "strict": True,
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": schema_name}},
        }

    def _raise_mapped(self, exc: openai.APIError) -> NoReturn:
        """Turn an SDK exception into the harness taxonomy.

        The distinction the ladder in `structured.py` depends on is 400/422 ->
        UnsupportedParameterError (retry with one fewer feature) versus
        everything else -> ProviderError (stop).
        """
        endpoint = f"{self.base_url}/chat/completions"

        # APITimeoutError subclasses APIConnectionError, so it must be checked
        # first. The two mean genuinely different things for a local box.
        if isinstance(exc, openai.APITimeoutError):
            raise ProviderError(
                f"{self.base_url} accepted the connection and then went quiet "
                f"(timed out waiting for '{self.model}'). Usually the server is "
                "cold-loading the model into VRAM, or the machine went to sleep "
                "mid-request. Retry once the model is resident, or raise "
                "REQUEST_TIMEOUT_S.",
                endpoint=endpoint,
            ) from exc

        if isinstance(exc, openai.APIConnectionError):
            raise ProviderError(
                f"Could not connect to {self.base_url} -- nothing is listening there. "
                "Start the server (`ollama serve`, `lms server start`, or your vLLM "
                "command) or fix LLM_BASE_URL.",
                endpoint=endpoint,
            ) from exc

        status = getattr(exc, "status_code", None)
        detail = str(exc)

        if status in (400, 422):
            raise UnsupportedParameterError(
                f"{self.base_url} rejected a request parameter for '{self.model}' "
                f"(HTTP {status}): {detail}",
                status=status,
                endpoint=endpoint,
            ) from exc

        if status in (401, 403):
            raise ProviderError(
                f"{self.base_url} refused the credentials (HTTP {status}). Check "
                "LLM_API_KEY -- note that local servers still require a non-empty "
                "placeholder value.",
                status=status,
                endpoint=endpoint,
            ) from exc

        if status == 404:
            raise ProviderError(
                f"Model '{self.model}' not found at {self.base_url} (HTTP 404). "
                f"List what this server actually serves with GET {self.base_url}/models, "
                "and check MODEL_NAME against it.",
                status=status,
                endpoint=endpoint,
            ) from exc

        raise ProviderError(
            f"{self.base_url} returned an error for '{self.model}'"
            + (f" (HTTP {status})" if status else "")
            + f": {detail}",
            status=status,
            endpoint=endpoint,
        ) from exc


def _tool_arguments(message: Any) -> str | None:
    """The JSON arguments of the first tool call, if the model made one."""
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return None
    function = getattr(tool_calls[0], "function", None)
    return getattr(function, "arguments", None)


def _usage(raw: Any) -> Usage:
    """Map OpenAI usage onto ours.

    `input_tokens` and `cached_input_tokens` must be DISJOINT -- cost.py bills
    them at different rates and would double-charge every cached call otherwise.
    OpenAI folds cached tokens into `prompt_tokens`, so we subtract them back
    out here. (Anthropic reports the two separately already.)
    """
    if raw is None:
        return Usage()
    prompt_details = getattr(raw, "prompt_tokens_details", None)
    completion_details = getattr(raw, "completion_tokens_details", None)

    prompt_tokens = int(getattr(raw, "prompt_tokens", 0) or 0)
    cached_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)
    return Usage(
        # max() guards a server that reports cached_tokens without folding them
        # into prompt_tokens; a negative token count corrupts the whole ledger.
        input_tokens=max(0, prompt_tokens - cached_tokens),
        output_tokens=int(getattr(raw, "completion_tokens", 0) or 0),
        cached_input_tokens=cached_tokens,
        reasoning_tokens=int(getattr(completion_details, "reasoning_tokens", 0) or 0),
    )
