"""Deterministic financial metrics over `FundamentalsSnapshot`s.

Every number the screen ranks on is computed here, in code, and unit-tested --
the LLM never computes a metric (PLAN.md design rule 1). Modules:

  valuation  cheapness: EV/EBIT, EV/Sales, P/B, net-cash/mcap, FCF yield, NCAV
  quality    Novy-Marx GP/A, ROIC, accruals, Piotroski F-Score
  risk       Beneish M, Altman Z'', share-issuance (dilution), cash runway
  liquidity  tradeability: ADV, dollar-volume floor, position capacity

Each returns `MetricValue`s that carry provenance and degrade to ok=False with a
reason rather than guessing when a microcap filing omits an input.
"""

from scout.metrics.base import (
    MarketData,
    MetricValue,
    enterprise_value,
    gross_profit,
    safe_div,
    select_annual_pair,
    total_debt,
    working_capital,
)

__all__ = [
    "MarketData",
    "MetricValue",
    "enterprise_value",
    "gross_profit",
    "safe_div",
    "select_annual_pair",
    "total_debt",
    "working_capital",
]
