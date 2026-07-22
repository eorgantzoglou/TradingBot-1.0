"""Assemble an `LLMClient` from configuration.

One place where the provider choice is made, so every caller above -- the CLI,
the research pipeline, the tests -- gets an identically-configured client and
nobody re-implements the wiring.
"""

from __future__ import annotations

from scout.config import Config
from scout.harness.adapters.anthropic import AnthropicAdapter
from scout.harness.adapters.openai_compat import OpenAICompatAdapter, capabilities_for
from scout.harness.cache import CachedClient, ReplayCache
from scout.harness.protocol import Capabilities, LLMClient


def build_client(
    config: Config,
    *,
    use_cache: bool | None = None,
    prompt_version: str | None = None,
    capabilities: Capabilities | None = None,
) -> LLMClient:
    """Build the configured client, wrapped in the replay cache unless disabled.

    `capabilities` overrides the backend defaults. That override matters for
    local servers: whether a vLLM instance can do grammar-constrained decoding
    depends on how it was launched and which model is loaded, which no amount
    of sniffing the model name will tell you.
    """
    llm = config.llm

    if llm.provider == "anthropic":
        inner: LLMClient = AnthropicAdapter(
            model=llm.model,
            api_key=llm.api_key,
            base_url=llm.base_url,
            timeout=llm.timeout,
            default_max_tokens=llm.max_tokens,
            **({"capabilities": capabilities} if capabilities else {}),
        )
    else:
        inner = OpenAICompatAdapter(
            model=llm.model,
            api_key=llm.api_key,
            base_url=llm.base_url,
            timeout=llm.timeout,
            capabilities=capabilities or capabilities_for(llm.base_url, llm.model),
        )

    enabled = config.enable_cache if use_cache is None else use_cache
    if not enabled:
        return inner
    return CachedClient(
        inner, ReplayCache(config.cache_dir), prompt_version=prompt_version
    )
