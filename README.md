# scout

A global deep-research equity scout: harvest primary regulatory filings, run a
cited LLM research pipeline over them, and measure honestly whether the picks
are any good.

Two co-equal goals — this is a side/educational project, so the harness matters
as much as the application:

1. **The harness** (`scout.harness`) — a hand-written LLM pipeline: provider
   abstraction, reasoning normalization across four incompatible conventions,
   a structured-output ladder with a repair loop, a content-addressed replay
   cache, and per-stage cost accounting.
2. **The application** (`scout.data`, and later `scout.metrics` / `scout.screen`
   / `scout.research`) — find globally-listed small/microcap companies that are
   cheap *and* healthy *and* under-followed.

It does **not** place orders, and nothing it produces is investment advice.
[PLAN.md](PLAN.md) has the full design, the research it rests on, and a blunt
section on what the evidence does and does not support.

## Status

| Phase | State |
|---|---|
| 0. Archive harvester | **done** — SEC, ESEF, EDINET, OpenDART, Companies House |
| 1. LLM harness | **done** — adapters, reasoning, structured output, cache, cost |
| 2. Data layer (parse → DuckDB) | **done** — SEC + ESEF parsed and normalized offline; FX per IAS 21 |
| 3. Metrics | **done** — valuation, quality, risk, liquidity; deterministic, golden-tested |
| 4. Screen | **done** — hard excludes, SIC/sector enrichment, within-cohort ranking |
| 5. Research agents | **done** — cited extraction, citation verification, debate, veto memo |
| 6. Ledger + forward scoring | **done** — pre-registered picks, forward scoring vs three dumb baselines |

## Quickstart

```powershell
uv sync --extra dev
copy env.example .env     # then edit it
uv run scout doctor
```

The only required setting is `USER_AGENT`, which must include a real contact
email — SEC EDGAR returns a 403 without one and blocks "unclassified bots".

```powershell
uv run scout harvest --days 1      # collect yesterday's filings
uv run scout status                # what the archive holds
uv run scout ingest                # parse the archive into normalized fundamentals
uv run scout fundamentals          # coverage; --entity <CIK/LEI> for one snapshot
uv run scout metrics -e 66740 --price 150   # all metrics for one entity
uv run scout enrich                # fetch SIC/sector, exchange, filing history
uv run scout screen --show-excluded         # ranked candidate watchlist
uv run scout research --top 5      # cited LLM research over the top candidates
uv run scout pick --prices p.json  # pre-register today's paper picks per strategy
uv run scout score --prices q.json # grade past picks vs the three dumb baselines
uv run scout llm-check             # exercise the whole harness, no data needed
```

### ⚠️ Start harvesting before you write any more code

No retail vendor sells point-in-time fundamentals outside the US, so the only
way to get one is to build it forward. The free sources purge:

| Source | Retention |
|---|---|
| UK Companies House daily accounts | **60 days** |
| Japan TDnet | **~30 days** |
| Nasdaq symbol directory | snapshot only, no archive published |

None of that can be reconstructed retroactively. Every day the collector is not
running is a day of history permanently lost. Schedule `scout harvest` daily —
it is the single highest-value thing in this repo, and it costs nothing.

## Commands

| Command | What it does |
|---|---|
| `scout doctor` | Validate config; show which sources are usable and why |
| `scout harvest --days N` | Harvest the last N days into the archive |
| `scout harvest --from 2026-07-01 --to 2026-07-20` | Backfill a range |
| `scout harvest -s sec -s esef --limit 5` | Restrict sources; cap documents (smoke test) |
| `scout status` | Archive contents by source, size, day range |
| `scout ingest` | Parse archived filings into normalized fundamentals (DuckDB) |
| `scout fundamentals [--entity ID]` | Coverage by taxonomy, or one entity's latest snapshot |
| `scout metrics -e ID [--price P]` | All deterministic metrics for one entity |
| `scout enrich` | Fetch entity profiles (SIC/sector, exchange, filing history) |
| `scout screen [--show-excluded]` | Ranked candidate watchlist |
| `scout research [--top N \| -e ID]` | Cited LLM research + veto memo (needs a model) |
| `scout pick [--prices f] [--research]` | Pre-register today's paper picks per strategy into the ledger |
| `scout score --prices f` | Grade the ledger's picks against forward prices, vs three baselines |
| `scout llm-check` | Round-trip the harness on a synthetic filing |

Add `-v` before the subcommand for INFO logging: `scout -v harvest --days 1`.

## Data sources

Everything in phase 0 is free and needs no paid subscription.

| Source | Coverage | Credential |
|---|---|---|
| **SEC EDGAR** | US, all filers | none (User-Agent only) |
| **filings.xbrl.org** | EU + UK ESEF filings | none |
| **EDINET** | Japan | free `EDINET_API_KEY` |
| **OpenDART** | South Korea | free `OPENDART_API_KEY` |
| **Companies House** | UK bulk accounts | none for bulk |

Two coverage caveats worth knowing before you trust a screen built on this:

- **ESEF applies to regulated markets only.** AIM, Euronext Growth, Nasdaq First
  North and Scale are MTFs and are exempt — and that is where most European
  microcaps list. `filings.xbrl.org` also has no Germany or Ireland at all.
- **Companies House bulk filings are FRS 102/105 statutory *entity* accounts**,
  whereas the same listed company's annual report is UK-adopted IFRS at the
  *group* level. Different numbers, different scope; never mix them in one field.

SEC data is public domain with no redistribution restriction — the only source
here with that property. `filings.xbrl.org` currently states no usage
restrictions. Everything else is subject to its publisher's terms.

## Architecture

```
src/scout/
  cli.py                 typer entry point
  config.py              env-driven, validated, nothing hardcoded
  harness/               THE LLM HARNESS -- hand-written on purpose
    protocol.py          LLMClient Protocol, Message/Usage/ModelResponse, OutputMode
    reasoning.py         the four-convention normalizer
    adapters/            anthropic.py, openai_compat.py
    structured.py        tier ladder + validate->re-ask repair loop
    cache.py             content-addressed replay cache
    cost.py              per-stage token and dollar rollup
    build.py             assemble a client from config
  data/
    http.py              per-host rate limiting, retry, backoff
    archive.py           append-only raw filing store
    harvest.py           list -> fetch -> archive orchestration
    sources/             sec, esef, edinet, opendart, companies_house
```

### Fundamentals, briefly

`scout ingest` turns archived filings into comparable financial facts, entirely
offline from the stored bytes (verified: edgartools reconstructs a full US-GAAP
fact set from a stored submission with no network; ESEF xBRL-JSON is plain OIM
JSON). The hard part — the "sleeper task" the plan flagged — is the concept
mapping in `fundamentals/normalize.py`, which has to solve three problems at once
or silently corrupt every metric built on top:

- **Tag heterogeneity.** The same quantity has many tags. 3M reports cash as
  `CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents`, not the plain
  tag. Each concept has an ordered candidate list; the first that resolves wins,
  and using a fallback is recorded as a warning.
- **Dimensions.** Revenue appears once consolidated and 24 times split by
  segment. Only the non-dimensioned total is ever accepted as a concept's value.
- **Period selection.** A single Q2 10-Q contains, ending on the same date, both
  a 90-day (discrete quarter) and a 180-day (year-to-date) figure — and
  cash-flow items exist *only* as year-to-date. The normalizer targets the span
  matching the fiscal period so every line item in a snapshot covers the **same**
  period; each fact also stores its exact span, so nothing downstream has to
  assume. Getting this wrong pairs a 3-month income figure with 6-month cash
  flow — an incoherent snapshot that quietly breaks any cross-statement ratio.

Everything is code-computed and golden-tested against real filings in both
taxonomies (3M / us-gaap, a Ukrainian issuer / ifrs-full), to the digit. Derived
figures (gross profit from revenue − COGS, TTM aggregation) belong in `metrics/`,
not here — normalization stays a faithful mapping, and every canonical fact keeps
`source_concept` so any number traces back to the exact tag it came from.

Cross-currency comparison uses ECB reference rates under IAS 21 (balance-sheet
items at the period-end rate, income/cash-flow items at the period average) —
never a single spot rate, and never a vendor's FX.

### Metrics, briefly

`scout metrics` computes every screening number in code — no LLM touches this
path (design rule 1). The set is chosen from the evidence, not convention:

- **Valuation** — EV/EBIT (the most comparable multiple across accounting
  standards), EV/Sales, P/B, net-cash-to-market-cap, FCF yield, and Graham NCAV
  with a net-net flag.
- **Quality** — Novy-Marx GP/A (the best-evidenced quality metric), ROIC,
  Sloan accruals (as a flag, not a factor — the anomaly itself has decayed), and
  the 9-signal Piotroski F-Score, designed for exactly the low-coverage value
  stocks this targets.
- **Risk** — Beneish M-Score (caught 71% of famous frauds ahead of disclosure),
  the book-value Altman Z″ (no market data, works globally), the share-issuance
  rate (dilution — the single most important microcap red flag), and cash runway.
- **Liquidity** — ADV, quoted spread, position capacity, and a hard
  liquidity-floor flag, because a name you can't exit isn't a candidate.

Everything degrades honestly: a metric with a missing input returns
`ok=False` with a plain reason, never a guessed or zero value — a missing
denominator and a real zero are different facts. Each result carries the exact
inputs it was built from. Cross-period metrics (Piotroski, Beneish, dilution)
require two consecutive annual filings and refuse to run otherwise; valuation
multiples need a price and are simply absent until one is supplied. Golden-tested
to the digit against real 3M (us-gaap) and a Ukrainian microcap (ifrs-full),
cross-checked against independent hand calculation.

### Research, briefly

`scout research` is where the harness meets the screen — and the only place the
LLM runs. For each candidate that already survived the screen, the pipeline
chains five stages in the one order that keeps the trust guarantee intact:

```
evidence  →  extract  →  VERIFY citations  →  debate  →  memo
```

- **Evidence** (no LLM): pull the primary document's narrative from the archived
  filing and select the red-flag passages (going-concern, toxic convertibles,
  related-party, dilution, reverse splits, auditor changes) plus the code-computed
  metrics. The company name and tickers are redacted to `[COMPANY]`/`[TICKER]` —
  anonymized inputs remove both look-ahead and a "distraction" effect and, per
  Glasserman & Lin, actually score better.
- **Extract**: the model is a *forensic reader*, not a decider. Every finding
  must carry a **verbatim quote** and the accession it came from.
- **Verify** (no LLM — the linchpin): each quote is checked against the exact
  text the model was shown. A finding whose quote isn't really there is **dropped
  before the debate can see it**. Instruction isn't enforcement; this is. A high
  drop rate is itself surfaced as a signal that the extractor is confabulating.
- **Debate**: bull and bear argue from the verified findings only; a skeptic
  (ideally a *different model family*) tries to refute and makes the
  disqualifying call, caution-by-default for microcaps.
- **Memo**: the verdict is decided in **code** (`decide_verdict`) — a confirmed
  critical finding or a skeptic disqualification vetoes. The model writes the
  prose but cannot overturn the gate, cannot invent a number (all injected from
  `metrics/`), and cannot recommend buying. It can **veto** a candidate the
  screen chose; it can never **promote** one (design rule 1).

Needs a model in `.env` (any OpenAI-compatible endpoint, local or hosted, or
Anthropic). The whole pipeline is tested end to end with a scripted fake client,
including the two guarantees that matter: a fabricated citation is dropped, and
the model cannot escape a code-decided veto.

### The screen, briefly

`scout screen` is the first step that produces an actual watchlist, and it is
deterministic — no LLM runs until a name has already survived it. Order matters:

1. **Hard excludes first**, because avoiding dilution machines, shells and
   going-concerns is where the expected value is (PLAN.md rule 2). Checkable
   today: >20% share dilution, cash runway under 12 months, recent late filings,
   recent name/ticker change (shell-hijack), and shells (no revenue *and*
   non-operating — a pre-revenue biotech is deliberately *not* excluded here).
   The excludes that still need filing-text or external data (reverse splits,
   toxic convertibles, paid promotion) report `INSUFFICIENT` and are **listed as
   blind spots**, never silently passed — "we didn't check" and "this is fine"
   are different claims.
2. **Within-cohort ranking.** Survivors are ranked on a cheap × quality × safety
   composite, but only against peers in the same (country × accounting-standard ×
   sector) cohort — a Japanese JGAAP microcap and a US biotech are not peers, and
   P/E across standards is not comparable. Ranking z-scores each metric within
   the cohort (winsorized against microcap fat tails), and a name missing the
   price-dependent cheap block is ranked on quality+safety rather than penalized.
   Under-followed is a tie-breaker, never a factor — the neglected-firm premium
   disappears once you control for size.

Sector and several excludes need the SEC submissions data that `scout enrich`
fetches (free): SIC → sector and the financials/utilities exclusion, `formerNames`
→ name-change detection, filing history → delinquency. That enrichment already
caught a real trap: SEC lists a company's *current* name inside `formerNames`
with a rolling date, which naively reads as a same-day name change and would
false-exclude the likes of Equifax — so an entry equal to the current name is
ignored.

The screen is meant to be **eyeballed on its own** before any research runs: if
the ranking is junk, no LLM will fix it. At the current tiny data scale it is
demonstrating the machinery; it becomes a real microcap watchlist as the daily
harvest accumulates small-cap filers and a price feed is wired in.

### Ledger and forward scoring, briefly

`scout pick` and `scout score` are the honesty layer. No retail vendor sells
point-in-time fundamentals outside the US (see the harvest warning above), so the
screen cannot be honestly backtested — which means **forward paper trading is the
only credible evidence** the picks are any good. It only counts if the pick is
recorded *before* the outcome is known, so `scout pick` pre-registers a
timestamped, append-only book for every strategy at once:

- the **agent** (survived the screen *and* the research pipeline without a veto),
- the **screen** itself (the composite rank, no LLM),
- and the three dumb baselines it has to beat — the **equal-weighted universe**,
  the **EV/EBIT decile** within each cohort, and a **gradient-boosted tree** on
  the same tabular features (Levy 2026 found a GBDT beats the best commercial LLM
  with no look-ahead).

`scout score` grades them all on the identical forward window. Three disciplines
carried over verbatim in spirit from the old bot's `score.js`:

- **The distribution, not the hit rate.** Returns are so right-skewed
  (Bessembinder 2018) that a strategy can be right 55% of the time and still lose
  money, so every strategy reports its full quantile spread with the median
  stated next to the mean; the hit rate is shown but never alone. The
  agent-vs-baseline verdict cites *both* the mean and median delta, and calls a
  result "inconclusive" when they disagree in sign — the fingerprint of a
  skew-driven win one big holding is carrying.
- **Say when the sample is too small.** Below ~30 scored picks the verdict is
  "insufficient evidence" — a 70% hit rate on a handful of picks is what a coin
  flip returns routinely, and the report says so loudly.
- **Name the bar.** If the agent does not beat the EV/EBIT decile, the evaluation
  states plainly that the LLM research layer is cost, not signal.

Everything degrades honestly, the same as the layers beneath it. There is no
price feed yet (a documented deferral), so prices are supplied by hand
(`--price ID=VALUE` or a `--prices` JSON file, validated finite and positive) and
`evaluate.py` takes them as an input rather than reaching for a vendor — a name
with no price is recorded ungradeable, never guessed to zero. Because a forward
return only means something over one holding window, `scout score` grades a
single pick vintage at a time (the latest by default, `--vintage`/`--run-id` to
pick another) rather than pooling picks from different dates against one price
snapshot. The GBDT baseline needs labeled forward-return
history to train, which does not exist until picks have been scored, so it
reports INSUFFICIENT (and its `lightgbm` dependency is an optional `gbdt` extra)
until the forward archive is deep enough. The ledger is one JSON line per pick —
the old `logger.js` append-only pattern — so it is trivially parseable, survives
a crash mid-run, and never rewrites the history a pre-registration record depends
on. A write that fails, or a corrupt line on read, raises rather than silently
losing a pick.

### The harness, briefly

The design rule that shapes everything: **the LLM is a reader, not a calculator,
and not a decider.** Numbers are computed in code and unit-tested; the model
reads unstructured filing text and extracts qualitative facts with citations;
ranking is deterministic. Of every open-source LLM investing project surveyed,
the only one with peer-reviewed evidence that it works is the one where the LLM
never makes the call.

No agent framework. For a static-shape pipeline (fan-out → debate → synthesize)
they are net-negative, and the genuinely hard parts here — reasoning-field
normalization across heterogeneous backends, JSON-schema subset intersection,
provenance, deterministic replay — are precisely the parts none of them solve.
The orchestrator is `asyncio.gather` with a `Semaphore` and
`return_exceptions=True`.

Notable pieces:

- **`reasoning.py`** resolves `reasoning_content` (vLLM, LiteLLM) →
  `reasoning` (OpenRouter) → `thinking` (Ollama) → inline `<think>` tags
  (llama.cpp, LM Studio), including unterminated and orphan-closing tags. Its
  load-bearing rule: **empty content plus non-empty reasoning is an error, not a
  success** — that is a hybrid-thinking model with thinking left on, and the fix
  (`REASONING_EFFORT=none`) is named in the exception.
- **`structured.py`** walks NATIVE_SCHEMA → STRICT_TOOL → JSON_OBJECT → TEXT,
  dropping one rung per `UnsupportedParameterError`, then repairs by quoting the
  Pydantic error back to the model. Anthropic's JSON Schema subset silently
  drops `minimum`/`maximum`/`minLength`/`maxLength`, so constraints must live in
  Pydantic validators — the wire schema is a hint, `model_validate()` is the gate.
- **`cache.py`** keys on model, provider, every sampling parameter, the schema
  hash and the prompt version — not just prompt text, which is the standard way
  to end up serving subtly wrong cached results. It buys resumability, offline
  iteration on later stages, and deterministic prompt tests.

## Development

```powershell
uv run pytest -q
uv run ruff check src tests
```

572 tests, no network — every source is exercised through `respx` mocks.

One lesson already learned the hard way: mocked tests only prove the code
matches the fixture. Two real bugs in the SEC parser (a missing `edgar/` URL
segment and a wrongly-dashed date column) passed a green suite because the
fixture was fabricated. The fixtures are now built from live captures, and
`scout harvest --limit 4` against a real source is part of the definition of
done for any new source.
