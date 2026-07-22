"""Harness error taxonomy.

The distinction that matters is *retryable-with-fewer-features* vs *fatal*.
`UnsupportedParameterError` is the signal that lets the structured-output ladder
in `structured.py` downgrade one capability at a time instead of giving up.
"""

from __future__ import annotations


class HarnessError(Exception):
    """Base for every error this package raises."""


class ProviderError(HarnessError):
    """The provider returned an error we could not interpret as anything better."""

    def __init__(self, message: str, *, status: int | None = None, endpoint: str | None = None):
        super().__init__(message)
        self.status = status
        self.endpoint = endpoint


class UnsupportedParameterError(ProviderError):
    """The provider rejected a request parameter (HTTP 400/422).

    Raised for things like `reasoning_effort` on a model that does not take it,
    or `response_format` on a server without JSON mode. The caller is expected to
    retry with that capability dropped rather than to fail.
    """


class EmptyContentError(HarnessError):
    """The model produced no answer.

    Kept separate from ProviderError because the most common cause is specific
    and fixable: a hybrid-thinking model (Qwen3.x, DeepSeek-R1, some Gemma
    builds) left thinking enabled, so the answer went into a nonstandard
    reasoning field and `content` came back empty. `reasoning.py` detects that
    case and puts the fix in the message.
    """

    def __init__(self, message: str, *, reasoning: str | None = None):
        super().__init__(message)
        self.reasoning = reasoning


class SchemaValidationError(HarnessError):
    """Model output did not validate against the requested Pydantic model.

    Carries the raw text and the validation error so the repair loop can quote
    the failure back to the model.
    """

    def __init__(self, message: str, *, raw_text: str, errors: str):
        super().__init__(message)
        self.raw_text = raw_text
        self.errors = errors


class NoJsonFoundError(SchemaValidationError):
    """The response contained no parseable JSON object at all."""
