import { loadConfig } from './config.js';
import { connectToTradingView, disconnect } from './tradingview/cdp-client.js';
import { extractChartData } from './tradingview/extractor.js';
import { createLLMClient } from './ai/llm-client.js';
import { renderDashboard, renderError } from './output/dashboard.js';
import { logInteraction } from './output/logger.js';

const args = new Set(process.argv.slice(2));
const DRY_RUN = args.has('--dry-run'); // extract + print, skip the LLM
const DEBUG = args.has('--debug');     // dump the raw snapshot as JSON
const WATCH = args.has('--watch');

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** One full cycle: connect -> extract -> analyze -> render -> log. */
async function runOnce(config, llm) {
  const { browser, page } = await connectToTradingView(config.cdp);
  let snapshot;
  try {
    snapshot = await extractChartData(page);
  } finally {
    // Detach only — never close the user's TradingView app.
    disconnect(browser);
  }

  if (DEBUG) {
    console.log('\n--- RAW SNAPSHOT ---');
    console.dir(snapshot, { depth: null });
  }

  if (DRY_RUN) {
    console.log(
      `\n[dry-run] Extracted ${snapshot.symbol ?? '?'} @ ${snapshot.price ?? '?'} ` +
      `(${snapshot.timeframe ?? '?'}) with ${snapshot.indicators.length} indicator(s). LLM skipped.`
    );
    return;
  }

  console.log(
    `  Analyzing ${snapshot.symbol ?? '?'} ${snapshot.timeframe ?? ''} ` +
    `with ${config.llm.model} ...`
  );
  const { analysis, rawResponse, usage } = await llm.analyze(snapshot);

  renderDashboard(snapshot, analysis, { model: config.llm.model });

  await logInteraction(config.logFile, {
    timestamp: new Date().toISOString(),
    snapshot,
    analysis,
    model: config.llm.model,
    usage,
    raw_response: rawResponse,
  });
  console.log(`  Logged to ${config.logFile}\n`);
}

async function main() {
  let config;
  try {
    config = loadConfig();
  } catch (err) {
    renderError(err);
    process.exitCode = 1;
    return;
  }

  const llm = createLLMClient(config.llm);
  const intervalMs =
    config.pollIntervalMs > 0 ? config.pollIntervalMs : WATCH ? 60_000 : 0;

  if (intervalMs === 0) {
    try {
      await runOnce(config, llm);
    } catch (err) {
      renderError(err);
      process.exitCode = 1;
    }
    return;
  }

  console.log(`  Watch mode: analyzing every ${Math.round(intervalMs / 1000)}s. Ctrl+C to stop.`);
  for (;;) {
    try {
      await runOnce(config, llm);
    } catch (err) {
      // In watch mode a failed cycle (app closed, LLM offline...) is not
      // fatal — report it and try again on the next tick.
      renderError(err);
    }
    await sleep(intervalMs);
  }
}

main();
