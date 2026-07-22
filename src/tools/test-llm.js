import { loadConfig } from '../config.js';
import { createLLMClient } from '../ai/llm-client.js';
import { renderDashboard, renderError } from '../output/dashboard.js';

/**
 * Diagnostic helper: runs a SYNTHETIC chart snapshot through the full
 * analysis pipeline (prompt -> LLM -> JSON validation -> dashboard) without
 * needing TradingView Desktop running.
 *
 * Use it to verify the LLM endpoint, model name, and reasoning settings:
 *   npm run check:llm
 */

/* Fixture only -- these are NOT live market values. */
const FIXTURE = {
  extractedAt: new Date().toISOString(),
  symbol: 'BTCUSD (SYNTHETIC FIXTURE)',
  timeframe: '15m',
  price: 64123.45,
  ohlc: { open: 64010.0, high: 64280.5, low: 63955.2, close: 64123.45 },
  changeText: '+0.18%',
  indicators: [
    { name: 'EMA 20 close', values: [{ label: null, value: '64,050.12', numeric: 64050.12 }] },
    { name: 'EMA 50 close', values: [{ label: null, value: '63,780.44', numeric: 63780.44 }] },
    { name: 'EMA 200 close', values: [{ label: null, value: '62,910.08', numeric: 62910.08 }] },
    { name: 'RSI 14', values: [{ label: null, value: '58.3', numeric: 58.3 }] },
    {
      name: 'MACD 12 26 close 9',
      values: [
        { label: null, value: '42.10', numeric: 42.1 },
        { label: null, value: '128.55', numeric: 128.55 },
        { label: null, value: '86.45', numeric: 86.45 },
      ],
    },
    { name: 'Volume', values: [{ label: 'Vol', value: '1.24K', numeric: 1240 }] },
  ],
  hiddenIndicatorCount: 0,
  pageTitle: 'BTCUSD 64,123.45 TradingView',
  url: 'synthetic://fixture',
  warnings: ['SYNTHETIC FIXTURE — not live chart data.'],
};

let config;
try {
  config = loadConfig();
} catch (err) {
  renderError(err);
  process.exit(1);
}

console.log(`\n  Endpoint : ${config.llm.baseURL ?? 'https://api.openai.com/v1 (default)'}`);
console.log(`  Model    : ${config.llm.model}`);
console.log(`  Reasoning: ${config.llm.reasoningEffort ?? '(parameter not sent)'}`);
console.log('\n  Sending synthetic snapshot for analysis ...');

const llm = createLLMClient(config.llm);
const started = Date.now();
try {
  const { analysis, usage } = await llm.analyze(FIXTURE);
  const elapsed = ((Date.now() - started) / 1000).toFixed(1);

  renderDashboard(FIXTURE, analysis, { model: config.llm.model });

  console.log(`  Round trip: ${elapsed}s` + (usage ? `  |  completion tokens: ${usage.completion_tokens}` : ''));
  if (Number(elapsed) > 60) {
    console.log('  WARNING: that is very slow for a structured response. If this is a');
    console.log('  hybrid-thinking model (Qwen3.x, DeepSeek-R1), set REASONING_EFFORT=none.');
  }
  console.log('  LLM pipeline OK.\n');
} catch (err) {
  renderError(err);
  process.exit(1);
}
