"""Filing parsers: one per filing shape, all producing `RawFact`s offline."""

from scout.fundamentals.parse.base import FilingParser, ParsedFiling

__all__ = ["FilingParser", "ParsedFiling"]
