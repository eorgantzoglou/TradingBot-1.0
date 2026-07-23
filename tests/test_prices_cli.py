"""`_load_prices` input validation (H1).

A manual price is the only price source for now, so a malformed one must fail
loud at the CLI boundary rather than flow through as inf/nan and silently poison
the score. `_fail` raises typer.Exit, so a rejected input surfaces as a non-zero
exit, which is what these assert.
"""

from __future__ import annotations

import json

import pytest
import typer

from scout.cli import _load_prices


def test_valid_prices_parse_from_inline_and_file(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"111": 12.5, " 222 ": 3.0}), encoding="utf-8")
    prices = _load_prices(["333=7"], str(path))
    assert prices == {"111": 12.5, "222": 3.0, "333": 7.0}  # file key stripped too


@pytest.mark.parametrize("bad", ["x=inf", "x=nan", "x=-5", "x=0", "x=abc"])
def test_inline_non_finite_or_non_positive_is_rejected(bad):
    with pytest.raises(typer.Exit):
        _load_prices([bad], None)


def test_inline_without_equals_is_rejected():
    with pytest.raises(typer.Exit):
        _load_prices(["justanid"], None)


def test_file_infinity_token_is_rejected(tmp_path):
    # Python's json.loads parses the non-standard `Infinity` token by default;
    # _load_prices must still reject the resulting inf.
    path = tmp_path / "p.json"
    path.write_text('{"111": Infinity}', encoding="utf-8")
    with pytest.raises(typer.Exit):
        _load_prices(None, str(path))


def test_file_boolean_is_rejected(tmp_path):
    # JSON true would otherwise coerce to 1.0 and read as a real quote.
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"111": True}), encoding="utf-8")
    with pytest.raises(typer.Exit):
        _load_prices(None, str(path))


def test_missing_file_is_rejected():
    with pytest.raises(typer.Exit):
        _load_prices(None, "does-not-exist.json")
