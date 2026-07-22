import { loadConfig } from './config.js';
import { connectToTradingView, disconnect } from './tradingview/cdp-client.js';
import { extractChartData } from './tradingview/extractor.js';
import { createLLMClient } from './ai/llm-client.js';
import { renderDashboard, renderError, renderSignalChange } from './output/dashboard.js';
import { logInteraction } from './output/logger.js';

const args = new Set(process.argv.slice(2));
const DRY_RUN = args.has('--dry-run');     // extract + print, skip the LLM
const DEBUG = args.has('--debug');         // dump the raw snapshot as JSON
const WATCH = args.has('--watch');
const EVERY_TICK = args.has('--every-tick'); // analyze every poll, not once per bar

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** Reads the chart. Always detaches -- never closes the user's browser. */
async function grabSnapshot(cdpConfig) {
  const { browser, page } = await connectToTradingView(cdpConfig);
  try {
    return await extractChartData(page);
  } finally {
    disconnect(browser);
  }
}

/** Sends a snapshot for analysis, prints it, and appends it to the log. */
async function analyzeAndReport(snapshot, config, llm) {
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
  return analysis;
}

/** One-shot run: read the chart, analyze it once, exit. */
async function runOnce(config, llm) {
  const snapshot = await grabSnapshot(config.cdp);

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
  await analyzeAndReport(snapshot, config, llm);
}

/**
 * Watch mode.
 *
 * Polls the chart often but analyzes rarely: by default once per bar, once
 * that bar is nearly closed. Two reasons. Analytically, a bar that has just
 * opened has provisional extremes and near-zero volume, so a verdict drawn
 * from it is mostly noise. Practically, each analysis occupies the model
 * host for ~20s, and re-asking every minute about the same unfinished bar
 * produces near-identical verdicts while keeping a fanless laptop pinned.
 */
async function watchLoop(config, llm) {
  const { pollMs, analyzeAtBarPct, bell } = config.watch;
  const perBar = !EVERY_TICK && analyzeAtBarPct > 0;

  console.log(
    perBar
      ? `  Watch mode: checking every ${Math.round(pollMs / 1000)}s, analyzing once per bar ` +
        `at >=${analyzeAtBarPct}% formed. Ctrl+C to stop.`
      : `  Watch mode: analyzing every ${Math.round(pollMs / 1000)}s. Ctrl+C to stop.`
  );

  let lastAnalyzedBar = null;
  let lastSignal = null;

  for (;;) {
    try {
      const snapshot = await grabSnapshot(config.cdp);
      const bc = snapshot.barCompletion;

      /* Identify the bar by its start time so each one is analyzed once.
         Without barCompletion (daily bars, unknown timeframe) fall back to
         analyzing every poll -- there is no bar boundary to align to. */
      let shouldAnalyze = true;
      let barKey = null;
      if (perBar && bc) {
        const nowSec = Math.floor(Date.now() / 1000);
        barKey = `${snapshot.symbol}|${snapshot.timeframe}|${nowSec - bc.elapsedSeconds}`;
        shouldAnalyze = bc.percentComplete >= analyzeAtBarPct && barKey !== lastAnalyzedBar;
      }

      if (!shouldAnalyze) {
        const pct = bc ? `${bc.percentComplete}%` : '?';
        process.stdout.write(
          `\r  ${new Date().toLocaleTimeString()}  ${snapshot.symbol ?? '?'} ` +
          `${snapshot.price ?? '?'}  bar ${pct} formed  ` +
          `${lastSignal ? `(last verdict: ${lastSignal})` : '(waiting for bar to mature)'}   `
        );
      } else {
        process.stdout.write('\n');
        const analysis = await analyzeAndReport(snapshot, config, llm);
        if (barKey) lastAnalyzedBar = barKey;

        if (lastSignal !== null && analysis.final_signal !== lastSignal) {
          renderSignalChange(lastSignal, analysis.final_signal, snapshot, { bell });
        }
        lastSignal = analysis.final_signal;
      }
    } catch (err) {
      /* A failed cycle (browser closed, model host asleep) is not fatal in
         watch mode -- report it and try again on the next tick. */
      process.stdout.write('\n');
      renderError(err);
    }
    await sleep(pollMs);
  }
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
  const watching = WATCH || config.pollIntervalMs > 0;

  if (!watching) {
    try {
      await runOnce(config, llm);
    } catch (err) {
      renderError(err);
      process.exitCode = 1;
    }
    return;
  }

  // An explicit POLL_INTERVAL_MS overrides the default look-rate.
  if (config.pollIntervalMs > 0) config.watch.pollMs = config.pollIntervalMs;
  await watchLoop(config, llm);
}

main();
