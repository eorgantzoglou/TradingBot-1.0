"""Fundamentals: archived filings -> canonical, comparable financial facts.

Pipeline: parse (bytes -> RawFact) -> normalize (RawFact -> CanonicalFact via
the concept vocabulary) -> store (DuckDB). Everything downstream reads
`FundamentalsSnapshot`s and never touches a raw XBRL tag.
"""

from scout.fundamentals.concepts import Concept, PeriodType, Statement
from scout.fundamentals.models import (
    CanonicalFact,
    EntityRef,
    FundamentalsSnapshot,
    RawFact,
)

__all__ = [
    "CanonicalFact",
    "Concept",
    "EntityRef",
    "FundamentalsSnapshot",
    "PeriodType",
    "RawFact",
    "Statement",
]
