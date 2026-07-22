"""Shared test fixtures.

No network, ever. `FakeClient` implements the `LLMClient` Protocol with
scriptable responses, so the ladder, the repair loop and the cache can all be
driven through failure paths that a real provider only produces by accident.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest

from scout.harness.errors import UnsupportedParameterError
from scout.harness.protocol import (
    Capabilities,
    Effort,
    Message,
    ModelResponse,
    OutputMode,
    Usage,
)

# What a script entry may be:
#   str        -> returned as the response text
#   Exception  -> raised
#   ModelResponse -> returned as-is
ScriptEntry = str | Exception | ModelResponse


class FakeClient:
    """In-memory LLMClient.

    `script` is consumed one entry per call. `unsupported_modes` makes the
    client reject a rung the way a real provider does (HTTP 400 ->
    UnsupportedParameterError), which is the only way to exercise the ladder
    walking down.
    """

    def __init__(
        self,
        script: Sequence[ScriptEntry] | None = None,
        *,
        capabilities: Capabilities | None = None,
        name: str = "fake",
        model: str = "fake-model-1",
        unsupported_modes: Sequence[OutputMode] = (),
        usage: Usage | None = None,
        repeat_last: bool = False,
    ) -> None:
        self.script: list[ScriptEntry] = list(script or [])
        self.capabilities = capabilities or Capabilities()
        self.name = name
        self.model = model
        self.unsupported_modes = set(unsupported_modes)
        self.usage = usage or Usage(input_tokens=100, output_tokens=20)
        self.repeat_last = repeat_last

        # Every call, in order, for assertions about what the ladder actually did.
        self.calls: list[dict[str, Any]] = []

    @property
    def modes_tried(self) -> list[OutputMode]:
        return [call["output_mode"] for call in self.calls]

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
        self.calls.append(
            {
                "messages": list(messages),
                "output_mode": output_mode,
                "json_schema": json_schema,
                "schema_name": schema_name,
                "effort": effort,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "cache_prefix_upto": cache_prefix_upto,
            }
        )

        if output_mode in self.unsupported_modes:
            raise UnsupportedParameterError(
                f"{self.name} does not support output mode {output_mode.value}.",
                status=400,
            )

        entry = self._next_entry()
        if isinstance(entry, Exception):
            raise entry
        if isinstance(entry, ModelResponse):
            return entry
        return ModelResponse(
            text=entry,
            reasoning=None,
            usage=self.usage,
            model=self.model,
            output_mode=output_mode,
            raw={"fake": True},
        )

    def _next_entry(self) -> ScriptEntry:
        if not self.script:
            raise AssertionError(f"{self.name} ran out of scripted responses.")
        if len(self.script) == 1 and self.repeat_last:
            return self.script[0]
        return self.script.pop(0)


@pytest.fixture
def make_client() -> Callable[..., FakeClient]:
    """Factory so each test builds the client it needs."""
    return FakeClient


@pytest.fixture
def all_modes() -> Capabilities:
    """A provider that advertises every rung of the ladder."""
    return Capabilities(native_schema=True, strict_tools=True, json_object=True, effort=True)
