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
| 4. Screen | not started |
| 5. Research agents | not started |
| 6. Ledger + forward scoring | not started |

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

204 tests, no network — every source is exercised through `respx` mocks.

One lesson already learned the hard way: mocked tests only prove the code
matches the fixture. Two real bugs in the SEC parser (a missing `edgar/` URL
segment and a wrongly-dashed date column) passed a green suite because the
fixture was fabricated. The fixtures are now built from live captures, and
`scout harvest --limit 4` against a real source is part of the definition of
done for any new source.
