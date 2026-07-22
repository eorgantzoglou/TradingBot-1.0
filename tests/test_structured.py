"""Tests for the tier ladder, the repair loop and JSON extraction."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

from scout.harness.errors import NoJsonFoundError, SchemaValidationError
from scout.harness.protocol import Capabilities, Message, OutputMode
from scout.harness.structured import (
    complete_structured,
    extract_json,
    ladder_for,
    schema_for,
)


class Claim(BaseModel):
    text: str
    source_id: str
    confidence: float = Field(ge=0.0, le=1.0)


PROMPT = [Message(role="user", content="Extract the claim.")]
GOOD = json.dumps({"text": "Net cash exceeds market cap.", "source_id": "0001", "confidence": 0.8})


# --------------------------------------------------------------------------
# Ladder selection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("capabilities", "expected"),
    [
        (
            Capabilities(native_schema=True, strict_tools=True, json_object=True),
            [OutputMode.NATIVE_SCHEMA, OutputMode.STRICT_TOOL, OutputMode.JSON_OBJECT, OutputMode.TEXT],
        ),
        (
            Capabilities(strict_tools=True, json_object=True),
            [OutputMode.STRICT_TOOL, OutputMode.JSON_OBJECT, OutputMode.TEXT],
        ),
        (Capabilities(json_object=True), [OutputMode.JSON_OBJECT, OutputMode.TEXT]),
        (Capabilities(json_object=False), [OutputMode.TEXT]),
    ],
)
def test_ladder_reflects_capabilities(capabilities, expected):
    assert ladder_for(capabilities) == expected


async def test_starts_at_highest_advertised_rung(make_client, all_modes):
    client = make_client([GOOD], capabilities=all_modes)

    result = await complete_structured(client, PROMPT, Claim)

    assert result.mode_used is OutputMode.NATIVE_SCHEMA
    assert result.attempts == 1
    assert result.repairs == 0
    assert result.value.confidence == 0.8


async def test_local_server_without_json_mode_lands_on_text(make_client):
    client = make_client([GOOD], capabilities=Capabilities(json_object=False))

    result = await complete_structured(client, PROMPT, Claim)

    assert result.mode_used is OutputMode.TEXT
    # TEXT has no decoder constraint, so the schema has to be in the prompt.
    assert "JSON Schema" in client.calls[0]["messages"][-1].content


# --------------------------------------------------------------------------
# Walking down on UnsupportedParameterError
# --------------------------------------------------------------------------


async def test_downgrades_one_rung_per_rejection(make_client, all_modes):
    client = make_client(
        [GOOD],
        capabilities=all_modes,
        unsupported_modes=[OutputMode.NATIVE_SCHEMA, OutputMode.STRICT_TOOL],
    )

    result = await complete_structured(client, PROMPT, Claim)

    assert client.modes_tried == [
        OutputMode.NATIVE_SCHEMA,
        OutputMode.STRICT_TOOL,
        OutputMode.JSON_OBJECT,
    ]
    assert result.mode_used is OutputMode.JSON_OBJECT
    # Rejected calls still cost a round trip, so they count as attempts.
    assert result.attempts == 3


async def test_never_retries_the_same_rung(make_client, all_modes):
    client = make_client(
        [GOOD],
        capabilities=all_modes,
        unsupported_modes=[OutputMode.NATIVE_SCHEMA],
    )

    await complete_structured(client, PROMPT, Claim)

    assert client.modes_tried.count(OutputMode.NATIVE_SCHEMA) == 1


async def test_raises_when_every_rung_is_rejected(make_client, all_modes):
    client = make_client(
        [GOOD],
        capabilities=all_modes,
        unsupported_modes=list(OutputMode),
    )

    with pytest.raises(Exception) as excinfo:
        await complete_structured(client, PROMPT, Claim)

    assert "output mode" in str(excinfo.value)
    assert len(client.modes_tried) == 4


# --------------------------------------------------------------------------
# The repair loop
# --------------------------------------------------------------------------


async def test_repair_succeeds_on_second_attempt(make_client, all_modes):
    bad = json.dumps({"text": "x", "source_id": "0001", "confidence": "very high"})
    client = make_client([bad, GOOD], capabilities=all_modes)

    result = await complete_structured(client, PROMPT, Claim)

    assert result.repairs == 1
    assert result.attempts == 2
    assert result.value.confidence == 0.8

    # The re-ask quotes the validator verbatim and echoes the bad output back.
    repair_turn = client.calls[1]["messages"]
    assert repair_turn[-2].role == "assistant"
    assert repair_turn[-2].content == bad
    assert "confidence" in repair_turn[-1].content


async def test_repair_exhausts_and_fails_loud(make_client, all_modes):
    bad = json.dumps({"text": "x", "source_id": "0001"})  # confidence missing
    client = make_client([bad], capabilities=all_modes, repeat_last=True)

    with pytest.raises(SchemaValidationError) as excinfo:
        await complete_structured(client, PROMPT, Claim, max_repairs=2)

    # Initial call plus two repairs, then it gives up rather than half-return.
    assert len(client.calls) == 3
    assert excinfo.value.raw_text == bad
    assert "confidence" in excinfo.value.errors


async def test_no_repairs_allowed_fails_immediately(make_client, all_modes):
    client = make_client(["not json at all"], capabilities=all_modes, repeat_last=True)

    with pytest.raises(NoJsonFoundError):
        await complete_structured(client, PROMPT, Claim, max_repairs=0)

    assert len(client.calls) == 1


async def test_constraints_dropped_from_the_wire_schema_are_still_enforced(make_client):
    """The Anthropic-subset case: no maximum on the wire, still caught locally."""
    anthropic_like = Capabilities(
        native_schema=True,
        restricted_schema_keywords=frozenset({"minimum", "maximum", "minLength", "maxLength"}),
    )
    out_of_range = json.dumps({"text": "x", "source_id": "0001", "confidence": 1.7})
    client = make_client([out_of_range], capabilities=anthropic_like, repeat_last=True)

    with pytest.raises(SchemaValidationError):
        await complete_structured(client, PROMPT, Claim, max_repairs=1)

    sent_schema = json.dumps(client.calls[0]["json_schema"])
    assert "maximum" not in sent_schema  # gone from the schema...
    assert "less than or equal" in str(client.calls[1]["messages"][-1].content)  # ...caught anyway


# --------------------------------------------------------------------------
# schema_for
# --------------------------------------------------------------------------


def test_schema_for_strips_only_keywords_not_field_names():
    class Bounds(BaseModel):
        minimum: int  # a field that happens to share a keyword's name
        score: float = Field(ge=0.0)

    schema = schema_for(Bounds, exclude_keywords=frozenset({"minimum"}))

    assert "minimum" in schema["properties"]
    assert "minimum" not in schema["properties"]["score"]
    assert schema["additionalProperties"] is False


def test_schema_for_keeps_everything_when_nothing_is_excluded():
    schema = schema_for(Claim)
    assert schema["properties"]["confidence"]["maximum"] == 1.0


# --------------------------------------------------------------------------
# extract_json
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('  \n {"a": 1}\n ', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('```\n{"a": 1}\n```', {"a": 1}),
        ('Sure! Here you go:\n```json\n{"a": 1}\n```\nHope that helps.', {"a": 1}),
        ('Here is the object: {"a": 1} — let me know if you need more.', {"a": 1}),
        ("<think>I should answer with JSON.</think>\n{\"a\": 1}", {"a": 1}),
        ("<think>\nlong\nreasoning {not json}\n</think>```json\n{\"a\": 1}\n```", {"a": 1}),
        ("reasoning that was never opened</think>{\"a\": 1}", {"a": 1}),
        ('[{"a": 1}, {"a": 2}]', [{"a": 1}, {"a": 2}]),
        ('Results:\n[1, 2, 3]', [1, 2, 3]),
    ],
)
def test_extract_json_handles_messy_shapes(raw, expected):
    assert extract_json(raw) == expected


def test_extract_json_is_string_aware():
    """Braces and escaped quotes inside string values must not confuse the scan."""
    raw = 'Answer: {"note": "closes at 12:00 } and reopens {", "quote": "he said \\"buy\\""} done'
    assert extract_json(raw) == {
        "note": "closes at 12:00 } and reopens {",
        "quote": 'he said "buy"',
    }


def test_extract_json_skips_a_leading_brace_that_is_not_json():
    raw = 'The template {placeholder} is filled in as {"a": 1}.'
    assert extract_json(raw) == {"a": 1}


def test_extract_json_handles_nested_arrays_and_objects():
    raw = 'prose {"outer": {"inner": [1, {"deep": "]}"}]}} trailing'
    assert extract_json(raw) == {"outer": {"inner": [1, {"deep": "]}"}]}}


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "I could not find that information.",
        "<think>Only reasoning, no answer at all.",
        "{unbalanced: ",
    ],
)
def test_extract_json_raises_when_there_is_nothing_to_parse(raw):
    with pytest.raises(NoJsonFoundError):
        extract_json(raw)
