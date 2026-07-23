"""Phase 6: the paper-trade ledger and forward scoring.

No retail point-in-time fundamentals exist outside the US (PLAN.md 1.3), so we
cannot honestly backtest this screen. Forward paper trading is therefore not just
the best evidence -- it is the *only* credible evidence. This package writes
timestamped, pre-registered picks (`ledger.py`) the day the screen works, and
grades them later against three dumb baselines (`evaluate.py`), reporting the
full return distribution rather than a hit rate that a coin flip would beat.
"""
