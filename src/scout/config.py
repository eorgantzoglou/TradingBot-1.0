"""Configuration, entirely from the environment.

Same principle as every other part of this project: nothing about which model or
which data source we use is hardcoded, so switching providers is a .env edit.

Deliberately plain dataclasses with explicit validation rather than a settings
framework -- this file should be readable start to finish by someone who has
never seen it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _str(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _opt(name: str) -> str | None:
    return _str(name) or None


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name) or default)
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(float(_str(name) or default))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    raw = _str(name).lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


@dataclass(frozen=True, slots=True)
class LLMConfig:
    provider: str = "openai_compat"
    """"anthropic" for the native Messages API, "openai_compat" for OpenAI
    itself and for every local server (Ollama, LM Studio, vLLM) and OpenRouter."""

    model: str = ""
    api_key: str = ""
    base_url: str | None = None
    temperature: float | None = 0.2
    max_tokens: int = 4096
    effort: str | None = None
    """Normalized reasoning budget: none | low | medium | high, or unset.

    Set "none" for hybrid-thinking local models (Qwen3.x, DeepSeek-R1). They
    reason before answering by default, which for routine structured extraction
    is pure latency -- measured at 193x on this project's predecessor. Leave
    unset for models that reject the parameter; adapters degrade on a 400."""

    timeout: float = 180.0

    def validate(self) -> list[str]:
        problems = []
        if not self.model:
            problems.append("MODEL_NAME is required (the model id your provider serves).")
        if not self.api_key:
            problems.append(
                "LLM_API_KEY is required (any placeholder works for local servers "
                "like Ollama or LM Studio)."
            )
        if self.provider not in {"anthropic", "openai_compat"}:
            problems.append(f'LLM_PROVIDER must be "anthropic" or "openai_compat", got "{self.provider}".')
        if self.effort and self.effort not in {"none", "low", "medium", "high"}:
            problems.append(f'REASONING_EFFORT must be none|low|medium|high, got "{self.effort}".')
        return problems


@dataclass(frozen=True, slots=True)
class SourceCredentials:
    """Per-source API keys. All optional: a source with no key reports itself
    unavailable and is skipped with a warning rather than failing the harvest."""

    edinet_key: str | None = None
    opendart_key: str | None = None
    companies_house_key: str | None = None
    openfigi_key: str | None = None


@dataclass(frozen=True, slots=True)
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    credentials: SourceCredentials = field(default_factory=SourceCredentials)

    user_agent: str = ""
    """Sent on every outbound request. MUST include a contact email -- SEC EDGAR
    returns a 403 HTML page without one, and blocks "unclassified bots"."""

    data_dir: Path = Path("data")
    cache_dir: Path = Path(".cache/llm")
    enable_cache: bool = True
    concurrency: int = 8

    @property
    def archive_dir(self) -> Path:
        return self.data_dir / "archive"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "scout.duckdb"

    @property
    def ledger_path(self) -> Path:
        """Append-only paper-trade ledger (phase 6). Lives beside the archive and
        db so the whole point-in-time record is one directory to back up."""
        return self.data_dir / "ledger.jsonl"

    @property
    def reports_dir(self) -> Path:
        """Where `scout research` saves the cited memos it produces, as Markdown
        (to read) and JSON (to diff/automate), partitioned by day."""
        return self.data_dir / "reports"

    def validate(self) -> list[str]:
        problems = list(self.llm.validate())
        if not self.user_agent:
            problems.append(
                "USER_AGENT is required, in the form 'scout/0.1 you@example.com'. "
                "SEC EDGAR rejects requests without a contact email."
            )
        elif "@" not in self.user_agent:
            problems.append(f'USER_AGENT must include a contact email, got "{self.user_agent}".')
        return problems


def load_config() -> Config:
    """Read and validate configuration. Raises with every problem at once."""
    config = Config(
        llm=LLMConfig(
            provider=_str("LLM_PROVIDER", "openai_compat"),
            model=_str("MODEL_NAME"),
            api_key=_str("LLM_API_KEY") or _str("OPENAI_API_KEY") or _str("ANTHROPIC_API_KEY"),
            base_url=_opt("LLM_BASE_URL") or _opt("OPENAI_BASE_URL"),
            temperature=_float("TEMPERATURE", 0.2),
            max_tokens=_int("MAX_TOKENS", 4096),
            effort=_opt("REASONING_EFFORT"),
            timeout=_float("REQUEST_TIMEOUT_S", 180.0),
        ),
        credentials=SourceCredentials(
            edinet_key=_opt("EDINET_API_KEY"),
            opendart_key=_opt("OPENDART_API_KEY"),
            companies_house_key=_opt("COMPANIES_HOUSE_API_KEY"),
            openfigi_key=_opt("OPENFIGI_API_KEY"),
        ),
        user_agent=_str("USER_AGENT"),
        data_dir=Path(_str("DATA_DIR", "data")),
        cache_dir=Path(_str("LLM_CACHE_DIR", ".cache/llm")),
        enable_cache=_bool("LLM_CACHE", True),
        concurrency=_int("CONCURRENCY", 8),
    )

    problems = config.validate()
    if problems:
        raise ValueError(
            "Invalid configuration:\n  - "
            + "\n  - ".join(problems)
            + "\nCopy env.example to .env and fill in the values."
        )
    return config


def load_config_for_harvest() -> Config:
    """Harvest needs a User-Agent and source keys but no LLM at all.

    Kept separate so `scout harvest` runs on a machine with no model configured.
    The archive is time-critical -- publishers purge on 30- and 60-day windows --
    so it must never be blocked on LLM setup.
    """
    config = Config(
        credentials=SourceCredentials(
            edinet_key=_opt("EDINET_API_KEY"),
            opendart_key=_opt("OPENDART_API_KEY"),
            companies_house_key=_opt("COMPANIES_HOUSE_API_KEY"),
            openfigi_key=_opt("OPENFIGI_API_KEY"),
        ),
        user_agent=_str("USER_AGENT"),
        data_dir=Path(_str("DATA_DIR", "data")),
        concurrency=_int("CONCURRENCY", 8),
    )
    problems = [p for p in config.validate() if "USER_AGENT" in p]
    if problems:
        raise ValueError(
            "Invalid configuration:\n  - "
            + "\n  - ".join(problems)
            + "\nCopy env.example to .env and fill in the values."
        )
    return config
