"""Tests for the four-convention reasoning normalizer.

The interesting cases are the failures, not the successes: a normalizer that
returns "" for a hybrid-thinking model is worse than no normalizer at all,
because the bug then surfaces as a JSON parse error three layers up.
"""

from __future__ import annotations

import pytest

from scout.harness.errors import EmptyContentError
from scout.harness.reasoning import (
    anthropic_thinking_params,
    normalize_message,
    openai_effort_params,
    strip_think_tags,
)

# --------------------------------------------------------------------------
# The four conventions
# --------------------------------------------------------------------------


def test_reasoning_content_field_vllm_litellm():
    text, reasoning = normalize_message({"content": "42", "reasoning_content": "counting"})
    assert text == "42"
    assert reasoning == "counting"


def test_reasoning_field_openrouter():
    text, reasoning = normalize_message({"content": "42", "reasoning": "counting"})
    assert text == "42"
    assert reasoning == "counting"


def test_thinking_field_ollama_native():
    text, reasoning = normalize_message({"content": "42", "thinking": "counting"})
    assert text == "42"
    assert reasoning == "counting"


def test_inline_think_tags_llama_cpp():
    text, reasoning = normalize_message({"content": "<think>counting</think>42"})
    assert text == "42"
    assert reasoning == "counting"


def test_no_reasoning_at_all():
    text, reasoning = normalize_message({"content": "42"})
    assert text == "42"
    assert reasoning is None


# --------------------------------------------------------------------------
# Resolution order and merging
# --------------------------------------------------------------------------


def test_resolution_order_puts_reasoning_content_first():
    text, reasoning = normalize_message(
        {"content": "42", "reasoning": "second", "reasoning_content": "first"}
    )
    assert text == "42"
    assert reasoning == "first\n\nsecond"


def test_reasoning_in_two_places_is_merged():
    _, reasoning = normalize_message(
        {"content": "42", "reasoning_content": "field", "thinking": "other"}
    )
    assert reasoning == "field\n\nother"


def test_identical_reasoning_echoed_into_two_fields_is_not_doubled():
    # LiteLLM copies reasoning_content into reasoning; concatenating would
    # double the thinking in traces and in the cost report.
    _, reasoning = normalize_message(
        {"content": "42", "reasoning_content": "same", "reasoning": "same"}
    )
    assert reasoning == "same"


def test_field_reasoning_and_inline_tags_both_survive():
    _, reasoning = normalize_message(
        {"content": "<think>inline</think>42", "reasoning_content": "field"}
    )
    assert reasoning == "field\n\ninline"


# --------------------------------------------------------------------------
# Think-tag shapes
# --------------------------------------------------------------------------


def test_closed_think_tag_with_surrounding_text():
    text, reasoning = normalize_message({"content": "before<think>mid</think>after"})
    assert text == "beforeafter"
    assert reasoning == "mid"


def test_multiple_closed_think_blocks():
    text, reasoning = normalize_message({"content": "<think>a</think>X<think>b</think>Y"})
    assert text == "XY"
    assert reasoning == "a\n\nb"


def test_unterminated_think_tag_with_an_answer_before_it():
    # max_tokens cut generation off mid-thought after a partial answer.
    text, reasoning = normalize_message({"content": "partial<think>ran out of"})
    assert text == "partial"
    assert reasoning == "ran out of"


def test_unterminated_think_tag_with_no_answer_is_an_error():
    with pytest.raises(EmptyContentError) as excinfo:
        normalize_message({"content": "<think>still thinking when the budget ran out"})
    assert "REASONING_EFFORT=none" in str(excinfo.value)


def test_orphan_closing_tag_from_a_prefilled_chat_template():
    # Some templates open <think> in the prompt, so the model emits only the closer.
    text, reasoning = normalize_message({"content": "hidden thoughts</think>the answer"})
    assert text == "the answer"
    assert reasoning == "hidden thoughts"


def test_think_tags_are_case_insensitive():
    text, reasoning = normalize_message({"content": "<THINK>a</THINK>b"})
    assert text == "b"
    assert reasoning == "a"


def test_strip_think_tags_leaves_untagged_content_alone():
    content, blocks = strip_think_tags("nothing to see")
    assert content == "nothing to see"
    assert blocks == []


# --------------------------------------------------------------------------
# The rule that matters most
# --------------------------------------------------------------------------


def test_empty_content_with_reasoning_names_the_fix():
    with pytest.raises(EmptyContentError) as excinfo:
        normalize_message({"content": "", "reasoning_content": "I should answer 42"})
    message = str(excinfo.value)
    assert "REASONING_EFFORT=none" in message
    assert "hybrid-thinking" in message
    # The thinking is attached so the caller can log what the model was doing.
    assert excinfo.value.reasoning == "I should answer 42"


def test_whitespace_only_content_counts_as_empty():
    with pytest.raises(EmptyContentError) as excinfo:
        normalize_message({"content": "   \n\t ", "thinking": "hmm"})
    assert "REASONING_EFFORT=none" in str(excinfo.value)


def test_missing_content_key_with_reasoning():
    with pytest.raises(EmptyContentError):
        normalize_message({"reasoning": "hmm"})


def test_both_empty_gets_a_different_plainer_message():
    with pytest.raises(EmptyContentError) as excinfo:
        normalize_message({"content": ""})
    message = str(excinfo.value)
    assert "REASONING_EFFORT=none" not in message
    assert "MAX_TOKENS" in message
    assert excinfo.value.reasoning is None


def test_content_parts_array_is_flattened():
    text, _ = normalize_message(
        {"content": [{"type": "text", "text": "4"}, {"type": "text", "text": "2"}]}
    )
    assert text == "42"


# --------------------------------------------------------------------------
# Request-side effort translation
# --------------------------------------------------------------------------


def test_openai_effort_unset_sends_nothing():
    assert openai_effort_params(None) == {}


@pytest.mark.parametrize("effort", ["none", "low", "medium", "high"])
def test_openai_effort_passes_through(effort):
    assert openai_effort_params(effort) == {"reasoning_effort": effort}


def test_anthropic_thinking_unset_sends_nothing():
    assert anthropic_thinking_params(None, 4096) == {}


def test_anthropic_thinking_none_disables():
    assert anthropic_thinking_params("none", 4096) == {"thinking": {"type": "disabled"}}


@pytest.mark.parametrize(
    ("effort", "expected"),
    [("low", 1024), ("medium", 4096), ("high", 16384)],
)
def test_anthropic_thinking_budgets(effort, expected):
    params = anthropic_thinking_params(effort, 64_000)
    assert params == {"thinking": {"type": "enabled", "budget_tokens": expected}}


def test_anthropic_thinking_budget_leaves_room_for_the_answer():
    params = anthropic_thinking_params("high", 4096)
    budget = params["thinking"]["budget_tokens"]
    assert budget < 4096
    assert budget == 4096 - 512


def test_anthropic_thinking_refuses_a_max_tokens_it_cannot_fit():
    with pytest.raises(ValueError) as excinfo:
        anthropic_thinking_params("low", 1200)
    assert "MAX_TOKENS" in str(excinfo.value)
