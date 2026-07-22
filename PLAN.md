# Plan: global deep-research equity scout (Python)

**What it is.** A side/educational project with two co-equal goals:

1. **The harness** — build a genuinely well-engineered LLM research pipeline by hand: provider
   abstraction, reasoning normalization, structured output, provenance, replay, cost accounting.
2. **The application** — find globally-listed small/microcap companies that are cheap *and*
   healthy *and* under-followed, and produce a cited memo for each.

Both are deliverables. Where they conflict, the harness wins — this is a project to learn from,
so we write the interesting parts ourselves rather than importing a framework that hides them.

**Not a goal.** Order execution. Output is a ranked watchlist plus memos. Nothing here is
investment advice.

**Agreed design rules** (from the first round of research, confirmed):

> **Rule 1 — the LLM is a reader, not a calculator, and not a decider.** Every number is computed
> in code from XBRL and unit-tested. The LLM reads unstructured filing text and extracts
> qualitative facts *with citations*. Ranking is deterministic. The LLM can **veto** a candidate;
> it can never **promote** one.
>
> **Rule 2 — the red-flag filter is worth more than the ranking model.** Avoiding dilution
> machines, shells, pumps and untradeable roach motels is where the expected value is. Build the
> excludes before the score.
>
> **Rule 3 — diversify and hold.** 40–60 names minimum, capped position sizes, quarterly
> rebalancing at most.

**Changes from the first draft:** global rather than US-only; Python rather than Node; and the
harness promoted from an implementation detail to a first-class section (§4).

---

## 1. Going global strengthens the case — and adds four hard problems

### 1.1 The evidence actually gets better outside the US

This was the pleasant surprise. Two findings from the methodology research point the same way:

- **Jacobs & Müller (2020, *JFE*)**, "Anomalies across the globe: Once public, no longer
  existent?" — 241 anomalies across 39 markets, 2M+ anomaly-country-months. **The US is the only
  country with a reliable post-publication decline.** Elsewhere, published anomalies largely
  persist.
- **Piotroski's F-Score international evidence** (*Journal of Asset Management*, 2020) — 20
  developed non-US plus 15 emerging markets, measured over the **post-publication** window
  2000–2018, high-minus-low F-Score ≈ **10%/yr**, preserved across size segments. Compare the US,
  where practitioner backtests find the screen has *inverted*.

Add that Novy-Marx's gross profitability replicates across 19 developed markets and Asness et
al.'s quality-controlled size premium is robust internationally, and the global version of this
strategy rests on firmer ground than the US-only one. Japan in particular is the deepest microcap
value market on earth, and the Nordics are the best-covered.

There is also a **genuine new job for the LLM**: Japanese and Korean filings are published in
Japanese and Korean only, with no official English developer documentation. Reading them is a
language task, which is exactly what rule 1 says the LLM is for. In the US-only design the LLM was
arguably decorative; here it does real work no library does for you.

### 1.2 Problem A — the free EU data layer misses precisely our universe

ESEF (the mandatory iXBRL format for EU annual reports) applies to **regulated markets only**.
AIM, Euronext Growth, Nasdaq First North, Scale and Euronext Growth Oslo/Milan are **MTFs and are
exempt** — and that is where European microcaps actually list. So the free structured-data layer
systematically covers the companies we don't want and misses the ones we do.

Consequences: Europe below the regulated-market line needs either a vendor (EODHD/Börsdata) or
per-venue work. And **ESAP**, the pan-EU access point everyone points to, is not usable — phase 1
began collecting from national bodies on 10 July 2026, twelve days ago, and **public access is
July 2027**. Build an adapter seam for it; ship nothing that depends on it.

### 1.3 Problem B — no retail point-in-time data exists outside the US

Every retail vendor (EODHD, FMP, Börsdata, TIKR) serves **latest-restated** data. None store
as-first-reported. The reference standards — Compustat PIT, Datastream, FactSet — start around
$20k/yr. This is not a budget problem we can solve; it is a structural fact to design around.

Two consequences, and the second is urgent:

1. **We cannot produce a trustworthy backtest of a global microcap screen.** So the screen is a
   *candidate generator* verified against primary filings, not a strategy with a claimed Sharpe.
   Forward paper trading becomes the only credible evidence — which was already the plan.
2. **Build our own PIT archive, starting day one.** Harvest raw filings daily, stamp ingest date
   and filing date, never overwrite. In 2–3 years that is a genuine point-in-time non-US database
   nobody sells at retail. **It cannot be constructed retroactively** — the free sources purge:
   Companies House daily accounts vanish after **60 days**, TDnet after **~30**. The archive
   starts the day the collector is written, and every day we delay is a day permanently lost.

### 1.4 Problem C — accounting heterogeneity breaks naive cross-country ranking

The overwhelming majority of Japanese microcaps file **JGAAP**, not IFRS. JGAAP amortizes goodwill
over up to 20 years where IFRS impairment-tests only, retains extraordinary items that IFRS
abolished, and defines operating income differently. Korea uses K-IFRS, India Ind AS, China CAS —
all "IFRS-converged", none identical. ESEF filings carry **issuer-specific extension taxonomies**
where no standard element fits, inconsistently anchored to IFRS elements in practice.

And the UK has a nasty trap: listed groups report under UK-adopted IFRS, while the *same
companies'* Companies House filings are FRS 102/105 statutory entity accounts. Different numbers,
different scope. Never mix them in one field.

**Design rule:** no single global ranking. Rank by **percentile within (country × accounting
standard × sector) cohorts**. Prefer metrics that survive translation — EV/EBIT, EV/Sales, P/B,
net-cash-to-market-cap, FCF yield. **P/E, ROE and EBITDA multiples are not comparable JGAAP↔IFRS**
and must never be ranked across cohorts.

### 1.5 Problem D — tradeability, timing, currency and tax

- **Filing lag ranges from 45 days to 9 months** by venue (Australian quarterly 4C cash-flow
  reports at one end, Euronext Growth annuals at the other). Every record needs an explicit
  `filing_date` and `as_of`, and the screener must **refuse** to rank a 45-day-old figure against
  a 9-month-old one.
- **FX per IAS 21**: balance-sheet items at period-end rate, income-statement items at
  period-average rate. Applying one spot rate to everything silently corrupts multi-year growth
  and margin series. Use ECB reference rates (free, no auth), never the vendor's FX.
- **Reporting currency ≠ trading currency ≠ listing currency** is common and is reportedly the
  single most frequent silent bug in global screeners. Store all three, always.
- **Withholding tax** on dividends: Switzerland 35%, France 30%, Germany 26.375%, Japan 20%
  statutory (treaties commonly cut EU rates to 15%); UK, Hong Kong and Singapore are 0%. A
  yield-driven thesis in CH/FI/FR/DE carries a 15–35% income haircut.
- **Restrict the universe to 8–12 venues we can actually trade cheaply.** Via IBKR that means US,
  Canada (TSXV/CSE), UK (LSE + AIM), Germany, Euronext, Italy, the Nordics incl. First North,
  Japan (TSE Prime/Standard/Growth — the best global microcap venue), Australia, Hong Kong,
  Singapore and Korea (launched May 2026). **India is not accessible** to non-resident foreigners
  at all, and China A-shares only via Stock Connect, whose eligible list structurally excludes
  microcaps. Per-venue minimum commissions also make sub-$5k positions uneconomic. A screen
  returning 400 names across 30 exchanges, half untradeable, is not a product.

---

## 2. Data sources

### Tier 0 — free, programmatically usable, and covering the best microcap markets

| Source | Endpoint | Notes |
|---|---|---|
| **SEC EDGAR** (US) | `data.sec.gov/api/xbrl/companyfacts/…`, `frames/…`, `efts.sec.gov` full-text | Free, public domain, no redistribution limits. `User-Agent: App contact@email` mandatory, ≤10 rps. |
| **filings.xbrl.org** (EU + UK + UA) | `https://filings.xbrl.org/api` — JSON:API, `?filter[country]=GB` | **Best free pan-European source.** No auth, and the licence says *"no restrictions on the ways that the data can be used"* — the only redistributable source here. Serves pre-normalized **xBRL-JSON**. ⚠️ **Germany and Ireland are absent**, and it inherits the MTF gap from §1.2. |
| **UK Companies House** | `api.company-information.service.gov.uk` + bulk `download.companieshouse.gov.uk/en_accountsdata.html` | Free key. **600 req / 5 min.** Bulk daily iXBRL accounts — **purged after 60 days**, so harvesting starts now. Remember the FRS 102 vs IFRS trap (§1.4). |
| **Japan EDINET v2** | `api.edinet-fsa.go.jp/api/v2/` | Free key. `type=5` returns a **CSV conversion of the XBRL** — a large time-saver. Spec is Japanese-only, 3–5s between requests, and the list endpoint is date-indexed so universe building means iterating day by day. |
| **Korea OpenDART** | `opendart.fss.or.kr` | Free key issued instantly, ~20k requests/day. Closest non-US equivalent to EDGAR. Responses (including account names) are **in Korean** — needs a mapping layer. |
| **ECB FX** | `data-api.ecb.europa.eu/service/data/EXR/…` | Free, no auth, daily. |
| **GLEIF LEI** | `api.gleif.org/api/v1/lei-records` | Free, bulk files. The only cross-jurisdiction entity identifier that works — **essential** for joining filings to vendor tickers. |
| **OpenFIGI** | `api.openfigi.com/v3/mapping` | Free; 25 req/min anonymous, 250 with a key. ISIN ↔ ticker ↔ exchange. |
| **FINRA** (US short interest) | `api.finra.org/data/group/otcMarket/name/consolidatedShortInterest` | Free, no credentials, 1,200 req/min. |

Together these cover the US, Japan, Korea, the UK and EU-regulated markets — including two of the
three best microcap hunting grounds.

### Not usable (documented so we don't re-litigate it)

**ESAP** — public access July 2027. **SEDAR+** (Canada) — no API, bot-protected, all PDFs, no
XBRL mandate; partial workaround is that Canadian issuers filing MJDS/40-F appear in EDGAR.
**ASX** — ToS explicitly prohibits spiders and scrapers; buy Australian fundamentals instead.
**India** — no official API, and untradeable for us anyway. **HKEX / CNINFO** — PDFs, no
financial-statement XBRL mandate, systematic download restricted.

### Tier 1 — paid, when the free tier binds (~$120–160/mo, and not needed for months)

**EODHD Fundamentals $59.99/mo** (or All-In-One $99.99) is the best value for global small-cap
fundamentals: 150k+ tickers, 70+ exchanges, bulk-per-exchange CSV endpoint. Caveats: non-US
history is shallow (~6–10 years for smaller names), no PIT, and its delisted coverage carries
fundamentals only for post-2018 delistings. **Börsdata Pro+ €59/mo** is the specialist supplement
for Nordic microcaps, where EODHD is weakest and the opportunity is richest — note the API is
Pro+-only and its ToS forbids *any* redistribution of API data, commercial or not. **TIKR Plus
$24.95/mo** is worth it as a *human* cross-check before committing capital; it has no API, and the
GitHub scrapers for it violate its ToS.

**Ruled out:** SimFin (US-only despite the marketing), Koyfin and TIKR as pipeline inputs (no
API), Intrinio (international is enterprise-only), twelvedata (global fundamentals at $999/mo),
LSEG/Refinitiv (no retail option).

**ToS note:** every paid plan here is personal/non-professional and forbids redistribution. Fine
for a private screener; relevant the moment any output embeds vendor data.

---

## 3. Architecture

Six stages, each independently runnable and testable — stages 1–2 are cheap and deterministic,
stage 3 is slow and costs money.

```
  ┌─ 0. UNIVERSE + ARCHIVE ──────────────────────────────────────┐
  │ daily raw-filing harvest (never overwrite) → own PIT archive  │
  │ GLEIF LEI + OpenFIGI entity resolution across jurisdictions   │
  │ venue whitelist: 8–12 tradeable exchanges                     │
  └───────────────────────────┬───────────────────────────────────┘
                              ▼
  ┌─ 1. INGEST ──────────────────────────────────────────────────┐
  │ edgartools (US) · Arelle/xbrl-filings-api (ESEF) · EDINET     │
  │ · OpenDART · Companies House → DuckDB, normalized + FX-tagged │
  └───────────────────────────┬───────────────────────────────────┘
                              ▼
  ┌─ 2. DETERMINISTIC SCREEN (code only, no LLM) ────────────────┐
  │ hard excludes → cheap × quality × liquidity                   │
  │ ranked WITHIN (country × standard × sector) cohorts           │
  │ ~15,000 filers  ───────────────────────────►  ~60 candidates  │
  └───────────────────────────┬───────────────────────────────────┘
                              ▼
  ┌─ 3. DEEP RESEARCH (LLM fan-out, one task per candidate) ─────┐
  │ dilution filings, going concern, related-party, toxic         │
  │ convertibles, promotion, insider buys — and translation for   │
  │ JP/KR. Every claim carries a filing ID + quoted span.         │
  └───────────────────────────┬───────────────────────────────────┘
                              ▼
  ┌─ 4. ADVERSARIAL REVIEW ──────────────────────────────────────┐
  │ Bull · Bear · Skeptic (different model family) · citation     │
  │ check. Unsupported claim → dropped. Red flag → vetoed.        │
  └───────────────────────────┬───────────────────────────────────┘
                              ▼
  ┌─ 5. OUTPUT + LEDGER ─────────────────────────────────────────┐
  │ ranked watchlist · cited memo per name · paper-trade ledger   │
  └───────────────────────────┬───────────────────────────────────┘
                              ▼
  ┌─ 6. EVALUATION ──────────────────────────────────────────────┐
  │ forward scoring vs 3 dumb baselines · cost model ·            │
  │ memorization probe · full return distribution, not hit rate   │
  └───────────────────────────────────────────────────────────────┘
```

### Layout

```
src/scout/
  cli.py                    # typer: harvest, ingest, screen, research, report, score
  config.py                 # env-driven, same pattern as the old JS config
  harness/                  # ── §4. THE LLM HARNESS. Hand-written on purpose. ──
    protocol.py             # LLMClient Protocol + ModelResponse
    adapters/
      anthropic.py
      openai_compat.py      # OpenAI, Ollama, LM Studio, vLLM, OpenRouter
    reasoning.py            # the four-convention normalizer (§4.2)
    structured.py           # tier ladder + validate→re-ask repair loop (§4.3)
    cache.py                # content-addressed replay cache (§4.4)
    cost.py                 # token/price rollup per stage
    prompts/                # versioned prompt files, hashed into the cache key
  data/
    archive.py              # daily raw harvest, ingest+filing stamped, append-only
    entity.py               # GLEIF/OpenFIGI resolution, currency triple
    sources/
      sec.py                # edgartools
      esef.py               # xbrl-filings-api + Arelle
      edinet.py             # Japan (type=5 CSV path)
      opendart.py           # Korea (+ Korean account-name mapping)
      companies_house.py    # UK bulk accounts
      fx.py                 # ECB
      vendors/eodhd.py      # optional, behind the same interface
  metrics/                  # ── ALL DETERMINISTIC. ALL UNIT-TESTED. NO LLM. ──
    normalize.py            # XBRL/JGAAP/IFRS tags → canonical statements
    valuation.py            # EV/EBIT, EV/Sales, P/B, net-cash/mcap, FCF yield
    quality.py              # GP/A, ROIC, accruals, Piotroski F
    risk.py                 # Beneish M, Altman Z″, dilution rate, cash runway
    liquidity.py            # ADV, spread proxy, position capacity, venue costs
  screen/
    excludes.py             # hard rejects
    cohorts.py              # country × standard × sector bucketing
    rank.py                 # within-cohort percentile composite
  research/
    evidence.py             # assemble the per-candidate evidence pack
    extract.py              # LLM structured extraction, citations mandatory
    analysts.py             # bull / bear / skeptic
    verify.py               # citation validation, cross-family skeptic
    memo.py
    models.py               # Pydantic schemas (Anthropic∩OpenAI subset — §4.3)
  portfolio/
    ledger.py               # paper positions, JSONL
    evaluate.py             # forward scoring vs baselines
  output/
    console.py, memo.py
```

**Storage: DuckDB.** Columnar, embedded, reads Parquet directly, and handles a few million
fact-rows across 15k filers without a server. The raw-filing archive stays as files on disk
(compressed, partitioned by source/date) with DuckDB as the queryable index over it.

**Tooling:** `uv` for dependency management, `pytest`, `ruff`. Pin exact versions — this ecosystem
moves weekly.

### The old repo

The Node code goes, but not the lessons. `llm-client.js` already solved a real slice of §4.2 —
graceful degradation on `reasoning_effort`/JSON-mode rejection, and detecting a hybrid-thinking
model that diverted its answer into a nonstandard `reasoning` field. That file is the **spec** for
`harness/reasoning.py`, and it turns out to have been an early encounter with what the research
confirms is a four-way mess. `score.js`'s honest sample-size warning carries over verbatim in
spirit. Everything under `src/tradingview/` and `puppeteer-core` is deleted. The package should
probably be renamed off `tradingview-ai-bot` at the same time.

---

## 4. The harness

Promoted to its own section because it's half the point. The guiding principle from the research:
**for a static-shape pipeline (fan-out → debate → synthesize), every agent framework on the market
is net-negative.** They're built for dynamic agent loops and durable long-running state; we have
neither. And the parts of this problem that are genuinely hard — reasoning normalization across
heterogeneous backends, schema-subset intersection, provenance, replay — are exactly the parts no
framework solves.

### 4.1 Provider abstraction — write it, ~200 lines

```
LLMClient (Protocol)
  ├── AnthropicAdapter    → anthropic SDK, native Messages API
  └── OpenAICompatAdapter → openai SDK + base_url
                            (OpenAI, Ollama, LM Studio, vLLM, OpenRouter)
```

Two dependencies (`anthropic`, `openai`), because the OpenAI SDK with `base_url=` already reaches
every local server. Each adapter returns a normalized `ModelResponse(text, reasoning, usage, raw)`.

**Deliberately not LiteLLM.** It's an impressive gateway, but as an in-process SDK its
normalization layer is leaky in precisely the place we need it: it returns `reasoning_content =
None` unconditionally on Ollama (issue #27956, because it routes to `/api/generate` which doesn't
emit the field), drops it on vLLM streaming (#20246), and collapses graded reasoning effort for
some models (#27439). Worse, the near-universally recommended `drop_params=True` **silently
discards `reasoning_effort` and `response_format`** on providers that don't declare support — the
call succeeds and quietly returns garbage.

### 4.2 Reasoning normalization — the highest learning density in the project, ~40 lines

There are **four incompatible conventions** live right now:

| Surface | Response field |
|---|---|
| LiteLLM / vLLM | `message.reasoning_content` |
| OpenRouter, some Ollama paths | `message.reasoning` |
| Ollama native `/api/chat` | `message.thinking` |
| llama.cpp, LM Studio, unconfigured vLLM | inline `<think>…</think>` inside `content` |

Resolution order: `reasoning_content` → `reasoning` → strip inline `<think>` → and then the rule
that matters most: **`content` empty while reasoning is populated is a hard error, not a success.**
That's the exact failure the old JS client already caught, and it's still unhandled by the
libraries for local backends.

Request-side is equally forked: OpenAI takes `reasoning_effort` (`none` now available), Anthropic
takes `thinking: {type, budget_tokens}`, OpenRouter takes `reasoning: {...}`, vLLM needs
`--reasoning-parser` set **at server launch**, and Ollama takes `think: bool | "low"|"medium"|"high"`.
Normalize to one internal `effort` enum and translate per adapter.

*(If you want to see this done well before writing it: Pydantic AI 2.x has the most complete
implementation, including a configurable `openai_chat_thinking_field` for exactly this case.)*

### 4.3 Structured output — write the ladder and the repair loop, ~120 lines

Native structured output is now genuinely good on both frontiers, and **XGrammar is the default
constrained-decoding backend across vLLM, SGLang and Ollama**, so "local model can't do JSON
schema" is mostly a legacy problem. The ladder:

```
1. native json_schema / output_config     if adapter.supports_native_schema
2. strict tool call                       if adapter.supports_strict_tools
3. json_object mode + schema in prompt    local fallback
4. prompt + fenced-block extraction       last resort
   ↓ every tier → pydantic model_validate()
   ↓ ValidationError → re-ask with the error text, max 2 retries, then fail loud
```

**One constraint to absorb up front:** Anthropic's JSON Schema subset is narrower than OpenAI's —
no recursive schemas, no `minimum`/`maximum`, no `minLength`/`maxLength`, no array constraints
beyond `minItems: 0|1`, and `additionalProperties: false` is required. So write the Pydantic models
to the **intersection** of both subsets and enforce numeric/length constraints as **validators**,
not schema keywords — a `Field(ge=...)` silently vanishes from the emitted schema on Anthropic.
That deserves a comment in the code.

We do **not** adopt `instructor`. Its remaining value is one API across providers plus the
validate→re-ask loop — and writing that loop is the point. Same reasoning for BAML: real
engineering, but adopting a DSL that hides the harness is the wrong trade here.

### 4.4 Replay cache — highest practical ROI, ~50 lines

```
key = sha256(model + every sampling param + serialized messages + schema hash + prompt version)
→ ./cache/{key}.json
```

The critical detail is that **the key includes model, temperature, seed, schema hash and every
sampling param** — not just the prompt text. Getting that wrong is the standard way people end up
with subtly incorrect cached results.

This one component buys crash resumability, offline iteration on stages 4–5 without re-running
stage 3, and deterministic prompt regression tests. It is also most of what we'd otherwise adopt
LangGraph for.

Separately, exploit **provider-side prompt caching** for the fan-out: Anthropic `cache_control`
reads at 0.1× base input (break-even at two reads), and OpenAI caches automatically above a
~1,024-token stable prefix. Sixty candidates sharing one long system prompt and rubric is close to
the ideal case. This makes **prefix stability a design constraint** — put anything varying
(candidate ID, filing text, timestamps) at the *end* of the prompt.

### 4.5 Orchestration — ~30 lines, no framework

```python
sem = asyncio.Semaphore(8)

async def bounded(coro):
    async with sem:
        return await coro

results = await asyncio.gather(
    *(bounded(research(c)) for c in candidates),
    return_exceptions=True,          # one bad filing must not kill the run
)
```

A `Semaphore`, `return_exceptions=True`, and the replay cache give partial-failure tolerance and
resumability — the two things LangGraph would have been for. Skipped for cause: **LangGraph**
(durability we don't have), **CrewAI** (hype; it actively hides control flow), **Google ADK** (GCP
gravity), **OpenAI Agents SDK** (still 0.x, and agent-loop-shaped where we need a DAG). The
**Claude Agent SDK** deserves a specific note: it's excellent, but it *is* a harness — adopting it
means delegating the thing we set out to learn.

**Pydantic AI 2.15** is the one worth re-evaluating *later*. Once the harness exists, diffing our
reasoning normalizer against theirs is the best available check on whether we got it right, and
it's the cleanest exit if we ever stop wanting to maintain it.

### 4.6 Provenance — the piece no framework will do for us

```python
class Claim(BaseModel):
    text: str
    source_id: str        # accession / docID / OAM filing id
    quoted_span: str      # verbatim, must be findable in the source document
    page_anchor: str | None
    confidence: float
```

**No claim reaches synthesis without one**, and `verify.py` checks the quoted span actually appears
in the cited document before the memo is written. This is the entire premise of a cited-analysis
product and it is not delegable.

### 4.7 What we deliberately don't write

HTTP/retry/backoff/429 handling (the provider SDKs do it correctly); **XBRL parsing** —
`edgartools` 5.43.0 for SEC (typed filing objects, multi-period stitching, and concept
standardization learned from 32k real filings; note bus factor of one, so pin it) and `Arelle`
2.42.1 for ESEF, because reimplementing XBRL teaches you about XBRL, not about harnesses;
grammar-constrained decoding (XGrammar is already in vLLM/Ollama); Pydantic validation; the trace
viewer — emit **OTel GenAI** spans (semconv 1.40.0, client spans now stable) into a self-hosted
**Phoenix** container, which is one `docker run` versus Langfuse's ClickHouse+Redis+S3; and prompt
optimization, where **DSPy 3.2 / GEPA** is a research-grade implementation worth pointing at a
single weak prompt in phase two, offline, without contaminating the runtime.

---

## 5. The screen

### Universe

Listed on one of the 8–12 whitelisted tradeable venues (§1.5). Market cap **$25M – $2B**, tilted
low. Median daily dollar volume ≥ 20× intended position size, after applying the venue's minimum
commission. Financials and utilities excluded from EV/EBIT ranking. **US OTC Expert Market, Pink
No-Information and Grey Market excluded entirely** — post-15c2-11, non-disclosing issuers went from
~6 market makers to under 3 and two-sided quotes collapsed from ~90% to under 15%; you cannot exit
those names. Eraker & Ready (2015, *JFE*) found OTC returns are **negative on average** as a class.

### Hard excludes — cheap to run, and most of the value

| Rule | Why |
|---|---|
| Share count grew **>20% YoY** (or >50% over 3y) without an acquisition | Pontiff & Woodgate (2008, *JF*): share issuance predicts the cross-section more significantly than size, B/M or momentum. The dilution machine's fingerprint. |
| Floating/discount-to-market convertibles — "variable conversion price", "look-back", "lowest closing price of the prior N days" | Death-spiral financing; the lender is structurally incentivized to short into conversion |
| Reverse split within 24 months | −34% to −54% three-year abnormal returns, consistent across 24 markets |
| Going-concern opinion **and** cash < 12 months of burn | The opinion alone is noisy; opinion plus runway is not |
| Delinquent filings, or filing lag beyond the venue's norm | |
| Auditor changed within 2 years following a dismissal | Opinion shopping; the market doesn't price it |
| No revenue and no product (shell / blank-check) | SEC's Operation Shell-Expel suspended 800+ such names — over 8% of the OTC market |
| Ticker/name/business changed in last 24 months | Classic shell-hijack pattern |
| Detectable paid promotion | Sponsored promos give +11.8% in 5 days then **fully revert within 30**, scaling with the fee paid, not audience size. An LLM reading enthusiastic microcap news is reading paid promotion. |

### Ranking — within cohort, never across

Three deterministic blocks, each percentile-ranked **inside its (country × accounting standard ×
sector) cohort**, then combined:

1. **Cheap** — EV/EBIT primary; EV/Sales, P/B, net-cash-to-market-cap as cross-checks; NCAV where
   it exists (Japan is where it still does).
2. **Quality** — GP/A weighted highest (Novy-Marx: roughly as powerful as book-to-market,
   replicated across 19 markets), ROIC, Piotroski F-Score applied *within* the cheap cohort as its
   author intended, accruals as a flag rather than a factor.
3. **Safety** — Beneish M-Score, Altman Z″ (skipped for pre-revenue firms, where it has no
   discriminating power), share-count trend, cash runway.

**Under-followed is a conditioning variable, never a factor.** Beard & Sias (1997, *FAJ*) found
the neglected-firm premium disappears once you control for market cap. Prefer low coverage and low
institutional ownership *among names that already pass cheap × quality* — the defensible mechanism
is slower information diffusion, not a free premium.

---

## 6. Evaluation

**Forward paper trading is the primary evidence**, and given §1.3 it's now the *only* credible
evidence. Stage 5 writes a timestamped ledger; stage 6 grades it. Six months of 60 pre-registered
picks beats any backtest we could construct from restated retail data.

Benchmark against **three dumb baselines**, not against zero:

1. the equal-weighted screened universe,
2. a plain EV/EBIT within-cohort decile screen with no LLM at all,
3. a gradient-boosted tree (LightGBM) on the same tabular features.

Levy (2026, *JAR*) found a GBDT beats the best commercial LLM by 2.7pp of accuracy while
guaranteeing no look-ahead. **If the agent doesn't beat baseline #2, the LLM layer is cost rather
than signal** — and the project should say so out loud.

**Anonymize before judgment.** Strip company name, ticker and dates from the evidence pack.
Glasserman & Lin (2023) found anonymized inputs *outperform*, removing both look-ahead and a
"distraction effect" where the model's general knowledge of the company interferes. If picks
degrade sharply under anonymization, the analysis was recall.

**Run the memorization probe.** Ask the model with no context what the company's FY20XX revenue
was; accurate answers flag that observation as contaminated (Didisheim et al., 2025). Global
microcaps are far less memorized than US mega-caps — an advantage here, and a measurable one.

**Report the full return distribution, not the hit rate.** Bessembinder (2018): four of seven US
stocks lifetime-underperform T-bills and the best 4% account for all net wealth creation. That skew
means a strategy can be right 55% of the time and lose money, or right 35% and make money. Median
outcome matters more than mean. Cost model: 3–6% round-trip plus impact scaled to ADV, plus the
venue's minimum commission and any withholding drag.

---

## 7. Build order

Branch per phase (`feature/…`), conventional commits, PR to main.

| Phase | Deliverable |
|---|---|
| **0. Archive first** | `scout harvest` — daily raw-filing collector for SEC, EDINET, filings.xbrl.org and Companies House bulk, append-only with ingest+filing stamps. **Ship this before anything else**; the 60-day and 30-day purge windows mean every delayed day is permanently lost data. Also: strip the Node code, `uv init`, DuckDB. |
| **1. Harness** | §4 end to end: Protocol + two adapters, reasoning normalizer, structured-output ladder, replay cache, cost rollup, OTel→Phoenix. Test against one frontier model and one local Qwen3.x — the local one is where the interesting bugs are. Independently useful and independently portfolio-worthy. |
| **2. Data layer** | edgartools/Arelle/EDINET/OpenDART ingestion into DuckDB. Entity resolution via LEI+FIGI. FX per IAS 21. The currency triple. **Tag normalization across JGAAP/IFRS/US-GAAP is the sleeper task — budget real time for it.** |
| **3. Metrics** | `metrics/` with unit tests and golden files over 5–6 hand-checked companies spanning US/JP/UK and including one pre-revenue biotech and one profitable JGAAP microcap. |
| **4. Screen** | `scout screen` → ranked table, no LLM. **Usable on its own.** Eyeball the top 20 against reality before going further; if the screen is junk, no LLM will fix it. |
| **5. Research agents** | `scout research --top 20` → evidence packs, extraction with mandatory citations, bull/bear/skeptic, memos. Extraction first; debate only once citation verification is reliable. |
| **6. Ledger + scoring** | `scout pick` / `scout score` against the three baselines. Start logging picks the day phase 4 works — the forward-evidence clock starts then. |

Phases 0 and 1 are the ones to do properly; 2–4 are where correctness is won; 5 is the fun part
and the least likely to add measurable value. Treat it as a memo generator that saves *you* reading
time, and let phase 6 decide whether it deserves more.

---

## 8. Honest expectations

A well-built version of this is a **research assistant that compresses a weekend of filing-reading
into an hour**, across markets you couldn't otherwise cover because you don't read Japanese — plus
a red-flag filter that keeps you out of dilution machines and pumps. Both are genuinely useful, and
the harness underneath is a better portfolio piece than the screener sitting on top of it.

What the evidence does not support is alpha. Published anomaly returns are inflated precisely by
the microcap, equal-weighted, cost-free construction we operate in; Hou, Xue & Zhang found 65% of
452 anomalies fail at t>1.96 once you stop letting microcaps drive the portfolio. The LLM layer has
no credible track record at deciding — the only real-money microcap LLM experiment lost half its
capital in six months. And with no retail point-in-time data outside the US, we cannot even
honestly backtest our way to a different conclusion.

Which is fine. The measurement harness gets built alongside the agent rather than after it, and
"the deterministic screen was the whole product" remains an acceptable — arguably the most likely —
outcome.
