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
        └── test-llm.js     # diagnostic: LLM pipeline test, no TradingView needed
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
```

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
- **This is a research tool, not financial advice. Signals come from an LLM
  reading chart values — always apply your own risk management.**
