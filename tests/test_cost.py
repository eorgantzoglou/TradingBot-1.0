"""Tests for token and cost accounting."""

from __future__ import annotations

import pytest

from scout.harness.cost import (
    CACHE_READ_MULTIPLIER,
    PRICES,
    CostLedger,
    Price,
    cost_of,
    price_for,
)
from scout.harness.protocol import ModelResponse, OutputMode, Usage


def response(model: str, usage: Usage) -> ModelResponse:
    return ModelResponse(
        text="{}",
        reasoning=None,
        usage=usage,
        model=model,
        output_mode=OutputMode.NATIVE_SCHEMA,
    )


# --------------------------------------------------------------------------
# Price lookup
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "claude-opus-4-8",
        "claude-opus-4-8-20260101",  # date-suffixed ids must still resolve
        "anthropic/claude-opus-4-8",  # OpenRouter prefix
        "us.anthropic.claude-opus-4-8",  # Bedrock prefix
        "CLAUDE-OPUS-4-8",
    ],
)
def test_longest_prefix_match_resolves_id_variants(model):
    assert price_for(model) == PRICES["claude-opus-4-8"]


def test_more_specific_prefix_wins():
    assert price_for("gpt-5-mini-2026-01-30") == PRICES["gpt-5-mini"]
    assert price_for("gpt-5") == PRICES["gpt-5"]
    assert price_for("gpt-4o-mini") == PRICES["gpt-4o-mini"]


def test_local_models_are_a_known_zero():
    assert price_for("qwen3-30b-a3b").input == 0.0
    assert price_for("local/whatever") is not None


def test_unknown_model_has_no_price():
    assert price_for("some-model-nobody-has-heard-of") is None


def test_deepseek_v4_is_priced_not_counted_free():
    # A hosted DeepSeek run must be billed at its real rate, not fall through to
    # the local-model zero entry.
    flash = price_for("deepseek-v4-flash")
    assert flash is not None
    assert flash.input == 0.14
    assert flash.output == 0.28
    assert flash.cached_input == 0.0028
    assert price_for("deepseek-v4-pro").output == 0.87


def test_anthropic_cache_reads_are_a_tenth_of_base_input():
    opus = PRICES["claude-opus-4-8"]
    assert opus.cached_input == pytest.approx(opus.input * CACHE_READ_MULTIPLIER)


@pytest.mark.parametrize(("ttl", "multiplier"), [("5m", 1.25), ("1h", 2.0)])
def test_cache_write_multipliers(ttl, multiplier):
    opus = PRICES["claude-opus-4-8"]
    assert opus.cache_write(ttl) == pytest.approx(opus.input * multiplier)


def test_unknown_cache_ttl_is_an_error_not_a_guess():
    with pytest.raises(ValueError, match="Unknown cache TTL"):
        PRICES["claude-opus-4-8"].cache_write("7d")


# --------------------------------------------------------------------------
# Per-call cost
# --------------------------------------------------------------------------


def test_cached_input_is_priced_separately_from_fresh_input():
    price = Price(input=5.0, output=25.0, cached_input=0.5)
    usage = Usage(input_tokens=1_000_000, cached_input_tokens=1_000_000, output_tokens=1_000_000)

    assert cost_of(usage, price) == pytest.approx(5.0 + 0.5 + 25.0)


def test_reasoning_tokens_are_not_double_counted():
    price = Price(input=1.0, output=1.0)
    without = Usage(input_tokens=100, output_tokens=100)
    with_thinking = Usage(input_tokens=100, output_tokens=100, reasoning_tokens=1000)

    assert cost_of(without, price) == cost_of(with_thinking, price)


# --------------------------------------------------------------------------
# The ledger
# --------------------------------------------------------------------------


def test_rollup_by_stage_and_total():
    ledger = CostLedger()

    with ledger.stage("research"):
        ledger.record(response("claude-opus-4-8", Usage(input_tokens=1_000_000, output_tokens=0)))
        ledger.record(response("claude-opus-4-8", Usage(input_tokens=0, output_tokens=1_000_000)))
    with ledger.stage("review"):
        ledger.record(response("claude-haiku-4-5", Usage(input_tokens=1_000_000, output_tokens=0)))

    stages = ledger.snapshot()
    assert stages["research"].calls == 2
    assert stages["research"].cost_usd == pytest.approx(30.0)
    assert stages["review"].cost_usd == pytest.approx(1.0)

    total = ledger.totals()
    assert total.calls == 3
    assert total.usage.input_tokens == 2_000_000
    assert total.cost_usd == pytest.approx(31.0)


def test_cached_tokens_show_up_in_the_rollup():
    ledger = CostLedger()
    usage = Usage(input_tokens=100_000, cached_input_tokens=900_000, output_tokens=0)

    with ledger.stage("fanout"):
        ledger.record(response("claude-opus-4-8", usage))

    stage = ledger.stages["fanout"]
    assert stage.usage.cached_input_tokens == 900_000
    # 0.1M at $5 + 0.9M at $0.50 -- the prompt-cache saving is the whole point.
    assert stage.cost_usd == pytest.approx(0.5 + 0.45)
    assert "cached" in ledger.render()


def test_unknown_model_costs_zero_and_records_a_warning():
    ledger = CostLedger()

    with ledger.stage("research"):
        cost = ledger.record(response("mystery-model-9", Usage(input_tokens=1_000_000)))

    assert cost == 0.0
    assert ledger.totals().cost_usd == 0.0
    assert len(ledger.warnings) == 1
    assert "mystery-model-9" in ledger.warnings[0]
    assert "warning:" in ledger.render()


def test_unknown_model_warns_once_not_per_call():
    ledger = CostLedger()
    for _ in range(5):
        ledger.record(response("mystery-model-9", Usage(input_tokens=10)))

    assert len(ledger.warnings) == 1
    assert ledger.totals().calls == 5


def test_records_outside_a_stage_land_in_the_default_bucket():
    ledger = CostLedger()
    ledger.record(response("claude-haiku-4-5", Usage(input_tokens=1_000_000)))

    assert "unattributed" in ledger.stages


def test_explicit_stage_argument_wins_over_the_context_manager():
    """Concurrent fan-out shares one ledger, so the stage has to be passable."""
    ledger = CostLedger()
    with ledger.stage("research"):
        ledger.record(response("claude-haiku-4-5", Usage(input_tokens=10)), stage="review")

    assert ledger.stages["review"].calls == 1
    assert ledger.stages["research"].calls == 0


def test_nested_stages_restore_the_outer_one():
    ledger = CostLedger()
    with ledger.stage("outer"):
        with ledger.stage("inner"):
            assert ledger.current_stage == "inner"
        assert ledger.current_stage == "outer"
    assert ledger.current_stage == "unattributed"


def test_render_shows_every_stage_and_a_total():
    ledger = CostLedger()
    with ledger.stage("research"):
        ledger.record(response("claude-opus-4-8", Usage(input_tokens=1_000_000, output_tokens=1000)))

    table = ledger.render()
    assert "research" in table
    assert "TOTAL" in table
    assert "1,000,000" in table
