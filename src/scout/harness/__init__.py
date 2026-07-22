"""The LLM harness.

Hand-written rather than adopted, for the reasons in PLAN.md section 4: for a
static-shape pipeline (fan-out -> debate -> synthesize) the agent frameworks are
net-negative, and the genuinely hard parts of this problem -- reasoning-field
normalization across heterogeneous backends, JSON-schema subset intersection,
provenance, deterministic replay -- are precisely the parts none of them solve.
"""

from scout.harness.errors import (
    EmptyContentError,
    HarnessError,
    NoJsonFoundError,
    ProviderError,
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
    Usage,
    fingerprint,
)

__all__ = [
    "Capabilities",
    "Effort",
    "EmptyContentError",
    "HarnessError",
    "LLMClient",
    "Message",
    "ModelResponse",
    "NoJsonFoundError",
    "OutputMode",
    "ProviderError",
    "SchemaValidationError",
    "UnsupportedParameterError",
    "Usage",
    "fingerprint",
]
