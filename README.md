# TradingView AI Trading Bot

A local, automated bridge between the **TradingView Desktop app** and **any LLM**.
The bot connects to the desktop app over the Chrome DevTools Protocol (CDP),
reads exactly what is on your active chart (symbol, timeframe, price, and every
plotted indicator), asks an LLM acting as a Senior Technical Analyst for a full
structured analysis, prints a terminal dashboard, and logs everything for
backtesting.

```
┌────────────────────┐   CDP (localhost:9222)   ┌──────────────────┐
│ TradingView Desktop│ ───────────────────────► │  DOM Extractor   │
│  (your live chart) │                          │  (puppeteer-core)│
└────────────────────┘                          └────────┬─────────┘
                                                         │ chart snapshot
                                                         ▼
┌────────────────────┐   OpenAI-compatible API  ┌──────────────────┐
│  Any LLM provider  │ ◄──────────────────────► │ Analysis Engine  │
│ LM Studio / Ollama │      (openai npm pkg)    │ (Senior TA prompt│
│ OpenAI / Anthropic │                          │  + strict JSON)  │
└────────────────────┘                          └────────┬─────────┘
                                                         │ validated verdict
                                                         ▼
                                    ┌─────────────────────────────────┐
                                    │ Terminal dashboard + trades.log │
                                    └─────────────────────────────────┘
```

## Project structure

```
TradingBot/
├── package.json
├── .env.example            # copy to .env — ALL configuration lives here
├── trades.log              # JSONL log of every interaction (created on first run)
└── src/
    ├── index.js            # orchestrator (single-shot & watch mode)
    ├── config.js           # .env loading + validation
    ├── tradingview/
    │   ├── cdp-client.js   # connects to localhost:9222, finds the chart window
    │   └── extractor.js    # DOM scraping: symbol, timeframe, price, indicators
    ├── ai/
    │   ├── prompts.js      # Senior Technical Analyst system prompt + snapshot prompt
    │   └── llm-client.js   # provider-agnostic OpenAI SDK wrapper + JSON validation
    ├── output/
    │   ├── dashboard.js    # colored terminal dashboard
    │   └── logger.js       # JSONL appender for backtesting
    └── tools/
        ├── check-cdp.js    # diagnostic: lists what the CDP port exposes
        ├── test-llm.js     # diagnostic: LLM pipeline test, no TradingView needed
        ├── show-log.js     # read back past analyses from trades.log
        └── score.js        # score past signals against real price history
```

## Setup

### 1. Install

```powershell
npm install
copy .env.example .env
```

### 2. Launch TradingView Desktop with the debug port

TradingView Desktop is an Electron app, so it exposes CDP when launched with a flag:

```powershell
& "$env:LOCALAPPDATA\Programs\TradingView\TradingView.exe" --remote-debugging-port=9222
```

(Adjust the path if you installed it elsewhere. For everyday use, edit your
TradingView shortcut: **Properties → Target** and append
` --remote-debugging-port=9222` after the closing quote. TradingView must be
**fully closed** first — the flag is ignored if an instance is already running.)

Verify the bridge:

```powershell
npm run check:cdp     # lists the windows the debugger can see
npm run dry-run       # extracts + prints your chart data, no LLM involved
```

### 3. Pick your LLM provider (edit `.env`)

| Provider        | OPENAI_BASE_URL                 | OPENAI_API_KEY | MODEL_NAME (example)      |
|-----------------|---------------------------------|----------------|---------------------------|
| LM Studio       | `http://127.0.0.1:1234/v1`      | `lm-studio`    | whatever LM Studio serves |
| Ollama          | `http://127.0.0.1:11434/v1`     | `ollama`       | `llama3.1:8b`             |
| Ollama over LAN | `http://<host-ip>:11434/v1`     | any string     | as reported by `/v1/models` |
| OpenAI          | *(leave empty)*                 | real key       | `gpt-4o-mini`             |
| Anthropic       | `https://api.anthropic.com/v1/` | real key       | `claude-sonnet-5`         |

Switching providers is purely a `.env` edit — no code changes.

Verify the LLM side end-to-end, without TradingView running:

```powershell
npm run check:llm     # pushes a synthetic snapshot through the whole pipeline
```

### Hybrid-thinking models: set `REASONING_EFFORT=none`

Qwen3.x, DeepSeek-R1 and similar models **reason before answering by default**.
For structured output like this bot's JSON verdict that is pure overhead, and the
cost is not subtle. Measured against a local Qwen3.6-35B-A3B over the LAN, same
request, JSON mode both times:

| `reasoning_effort` | latency    | completion tokens |
|--------------------|-----------|-------------------|
| *(omitted)*        | **270.7 s** | 505               |
| `none`             | **1.4 s**   | 13                |

That is a 193× penalty for an identical verdict. In `--watch` mode it would stack
analyses on top of each other. Some servers also divert the answer into a
non-standard `reasoning` field, leaving `content` empty — the bot detects that case
and names the fix in the error message.

Set `REASONING_EFFORT=none` for such models, and leave it **empty** for `gpt-4o`,
Claude and others that reject the parameter. The client degrades gracefully either
way: on a `400`/`422` it retries without JSON mode, then without `reasoning_effort`.

## Usage

```powershell
npm start                 # one analysis of the active chart, then exit
npm run watch             # re-analyze every 60s (or set POLL_INTERVAL_MS in .env)
npm run dry-run           # extraction only + raw snapshot dump (debugging)
npm run check:llm         # LLM-only test on a synthetic snapshot
npm start -- --debug      # full run, also dumps the raw snapshot

npm run log               # table of every signal recorded so far
npm run log -- --full 3   # replay analysis #3 as a full dashboard
npm run score             # check past signals against what price actually did
```

### Watch mode analyzes once per bar, not once per minute

Polling the chart is cheap; analyzing is not. Watch mode looks every
`WATCH_POLL_MS` (20s) but only sends an analysis when the current bar is at
least `ANALYZE_AT_BAR_PCT` (90%) formed, and only once per bar.

Two reasons. **Analytically**, a bar that just opened has provisional
extremes and a volume near zero — a verdict drawn from it is mostly noise.
**Practically**, each analysis occupies the model host for ~20s; re-asking
every minute about the same unfinished bar yields near-identical verdicts
while keeping a fanless laptop pinned.

When the verdict changes between bars (`HOLD` → `BUY`), it is announced
loudly with a terminal bell. Unchanged bars print a quiet one-line status.
Use `--every-tick` for the old analyze-every-poll behaviour.

### Scoring: is the model actually any good?

```powershell
npm run score
npm run score -- --horizon 8 --threshold 0.25
```

Takes the price the model saw as the entry, looks up the close N bars later
from public exchange data (Bitstamp first, since that is where the chart's
BTCUSD comes from; Binance as fallback), and asks whether the call pointed
the right way. `BUY` needs a rise beyond the threshold, `SELL` a fall,
`HOLD` a move that stays inside the band.

It measures direction only — no sizing, no stops, no fees. And it will tell
you loudly when the sample is too small to mean anything, which below ~100
signals it always is. **A 70% hit rate on 5 samples is noise.**

The bot only analyzes what it can *see*: indicators hidden with the legend
"eye" toggle are excluded, and any extraction fallbacks are surfaced as
warnings both on screen and to the LLM.

## Logging / backtesting

Every run appends one JSON line to `trades.log`:

```js
{ timestamp, snapshot, analysis, model, usage, raw_response }
```

Parse it later with:

```js
const entries = fs.readFileSync('trades.log', 'utf8').trim().split('\n').map(JSON.parse);
```

## Notes & limitations

- The extractor reads TradingView's legend. Legend values follow your
  crosshair — keep the mouse off the chart (or on the latest bar) so the
  values reflect the live bar.
- TradingView's DOM uses hashed class names that change between releases; the
  extractor targets stable `data-name` attributes with several fallbacks and
  reports a warning whenever a fallback was used. If a TradingView update
  breaks a field, run `npm run dry-run` to see what is still being captured.
- Number parsing assumes dot-decimal formatting (TradingView's default).
- The legend reports the bar being built *right now*, so its high, low,
  close and volume are provisional. The bot computes how far into the bar it
  is and tells the model explicitly — without that, a 30-second-old candle's
  near-zero volume gets read as "no conviction in the market". Bar
  completeness is only inferable for intraday timeframes, which align to the
  UTC clock; daily and above depend on the exchange session.
- The snapshot is a single instant, not a price history. The model sees each
  indicator's current value but not its previous ones, so it cannot observe
  slope, crossovers or divergences over time. The prompt forbids claiming
  them; treat any such statement in the output as a hallucination.
- `npm run score` compares against Bitstamp/Binance data. For a symbol on
  another venue the prices will not tick-for-tick match your chart.
- **This is a research tool, not financial advice.** It has no demonstrated
  edge. An LLM reading a handful of numbers off a chart legend is doing
  pattern-matching on text, not forecasting, and it will make arithmetic
  slips about numbers it can see. Run `npm run score` before believing any
  of it, and always apply your own risk management.
