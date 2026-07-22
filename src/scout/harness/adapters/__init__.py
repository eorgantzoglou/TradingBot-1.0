"""Provider adapters.

Two, because the OpenAI SDK with `base_url=` already reaches Ollama, LM Studio,
vLLM and OpenRouter. Everything provider-specific -- parameter names, reasoning
field conventions, JSON-schema subsets -- stops here.
"""
