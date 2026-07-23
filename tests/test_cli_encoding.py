"""The UTF-8 output shim.

Regression guard for a real crash: `scout research` produced a valid memo and
then died rendering the `✗` veto glyph to a Greek-locale (cp1253) Windows
console, because Python raises UnicodeEncodeError instead of degrading. The shim
forces UTF-8 with replacement before the Console is built.
"""

from __future__ import annotations

from scout.cli import _force_utf8_output


class _FakeStream:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[dict] = []
        self._fail = fail

    def reconfigure(self, **kwargs) -> None:
        self.calls.append(kwargs)
        if self._fail:
            raise ValueError("cannot reconfigure a detached stream")


def test_forces_utf8_with_replacement(monkeypatch):
    out, err = _FakeStream(), _FakeStream()
    monkeypatch.setattr("scout.cli.sys.stdout", out)
    monkeypatch.setattr("scout.cli.sys.stderr", err)

    _force_utf8_output()

    for stream in (out, err):
        assert stream.calls == [{"encoding": "utf-8", "errors": "replace"}]


def test_stream_without_reconfigure_is_skipped(monkeypatch):
    # An object with no `reconfigure` (a plain buffer, a redirect) must not crash.
    monkeypatch.setattr("scout.cli.sys.stdout", object())
    monkeypatch.setattr("scout.cli.sys.stderr", object())
    _force_utf8_output()  # no exception


def test_reconfigure_failure_is_swallowed(monkeypatch):
    # A stream that refuses reconfiguration must be left as-is, not fatal.
    monkeypatch.setattr("scout.cli.sys.stdout", _FakeStream(fail=True))
    monkeypatch.setattr("scout.cli.sys.stderr", _FakeStream(fail=True))
    _force_utf8_output()  # no exception
