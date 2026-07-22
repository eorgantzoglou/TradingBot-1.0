"""Scout: a global deep-research equity scout.

Two co-equal goals, in dependency order:

  1. `scout.harness` -- a hand-written LLM harness: provider abstraction,
     reasoning normalization across four incompatible conventions, a
     structured-output ladder with a repair loop, a content-addressed replay
     cache, and per-stage cost accounting.
  2. `scout.data` -- an append-only archive of primary filings from SEC EDGAR,
     EDINET (Japan), OpenDART (Korea), Companies House (UK) and
     filings.xbrl.org (EU/UK), which over time becomes the point-in-time
     database that no vendor sells at retail.

See PLAN.md for the reasoning behind both, and for what the evidence does and
does not support about the strategy on top.
"""

__version__ = "0.1.0"
