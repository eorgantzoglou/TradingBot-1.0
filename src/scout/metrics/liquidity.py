"""Tradeability metrics.

Liquidity is not a footnote for microcaps -- it is the dominant real-world
constraint, and the reason most published microcap backtests are fiction
(PLAN.md section 1.5, 3.5). A 3-6% round-trip spread and depth that a single
retail order walks through several price levels routinely eat any edge the
fundamentals suggest. So the screen must be able to reject a name it cannot
actually trade, and size positions to what the market can absorb.

These take market inputs (price, average daily volume) the caller supplies;
fundamentals do not carry them. The math is deliberately simple and pessimistic
-- an optimistic liquidity model is worse than none, because it green-lights
positions that cannot be exited.
"""

from __future__ import annotations

from dataclasses import dataclass

from scout.metrics.base import MetricValue, safe_div


@dataclass(frozen=True, slots=True)
class LiquidityInputs:
    """What the (deferred, paid) market-data layer will supply per name."""

    price: float | None = None
    avg_daily_volume_shares: float | None = None
    """Average shares traded per day, ideally a 20-30 session mean."""

    bid: float | None = None
    ask: float | None = None

    @property
    def avg_daily_dollar_volume(self) -> float | None:
        return (
            self.price * self.avg_daily_volume_shares
            if self.price is not None and self.avg_daily_volume_shares is not None
            else None
        )


def avg_daily_dollar_volume(inputs: LiquidityInputs) -> MetricValue:
    """Average daily dollar volume -- the headline liquidity number."""
    addv = inputs.avg_daily_dollar_volume
    if addv is None:
        return MetricValue.missing(
            "adv", "currency", "point-in-time", "price or average daily volume missing"
        )
    return MetricValue.of("adv", addv, "currency", "point-in-time")


def quoted_spread_pct(inputs: LiquidityInputs) -> MetricValue:
    """Round-trip quoted spread as a percent of the mid price.

    The single most under-modelled cost in microcap investing. A $0.50 spread on
    a $10 stock is 5% round-trip, versus ~0.005% on a mega-cap.
    """
    if inputs.bid is None or inputs.ask is None or inputs.bid <= 0 or inputs.ask <= 0:
        return MetricValue.missing(
            "quoted_spread_pct", "pct", "point-in-time", "no two-sided quote"
        )
    mid = (inputs.bid + inputs.ask) / 2
    spread = safe_div((inputs.ask - inputs.bid), mid)
    if spread is None:
        return MetricValue.missing(
            "quoted_spread_pct", "pct", "point-in-time", "degenerate quote"
        )
    return MetricValue.of(
        "quoted_spread_pct", spread * 100, "pct", "point-in-time",
        inputs={"bid": inputs.bid, "ask": inputs.ask},
    )


def position_capacity(
    inputs: LiquidityInputs, *, participation: float = 0.10, days_to_exit: float = 5.0
) -> MetricValue:
    """Largest position (in currency) you could exit within `days_to_exit`,
    taking no more than `participation` of daily volume.

    Defaults: 10% of ADV per day over 5 days. Being a larger fraction of volume
    means moving your own exit price -- the market-impact term every microcap
    backtest omits and every microcap trader pays.
    """
    addv = inputs.avg_daily_dollar_volume
    if addv is None:
        return MetricValue.missing(
            "position_capacity", "currency", "point-in-time",
            "price or average daily volume missing",
        )
    capacity = addv * participation * days_to_exit
    return MetricValue.of(
        "position_capacity", capacity, "currency", "point-in-time",
        inputs={"adv": addv, "participation": participation, "days_to_exit": days_to_exit},
    )


def meets_liquidity_floor(
    inputs: LiquidityInputs, *, intended_position: float, min_multiple: float = 20.0
) -> MetricValue:
    """Flag: is average daily dollar volume at least `min_multiple` times the
    intended position size?

    The plan's hard rule is ADV >= 20x position size. Below that, entering and
    exiting moves the price against you enough to matter. Returned as a flag
    (1.0 pass / 0.0 fail) so the screen can use it as a hard exclude.
    """
    addv = inputs.avg_daily_dollar_volume
    if addv is None:
        return MetricValue.missing(
            "meets_liquidity_floor", "flag", "point-in-time",
            "price or average daily volume missing",
        )
    passes = addv >= min_multiple * intended_position
    return MetricValue.of(
        "meets_liquidity_floor", 1.0 if passes else 0.0, "flag", "point-in-time",
        inputs={"adv": addv, "intended_position": intended_position, "min_multiple": min_multiple},
    )
