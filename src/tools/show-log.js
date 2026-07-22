import { readFile } from 'node:fs/promises';
import { loadConfig } from '../config.js';
import { renderDashboard } from '../output/dashboard.js';

/**
 * Reads back past analyses from trades.log.
 *
 *   npm run log            -- compact table of every signal so far
 *   npm run log -- --full  -- replay the most recent as a full dashboard
 *   npm run log -- --full 3   replay entry #3
 */

const argv = process.argv.slice(2);
const wantFull = argv.includes('--full');
const indexArg = argv.find((a) => /^\d+$/.test(a));

let logFile = 'trades.log';
try {
  logFile = loadConfig().logFile;
} catch {
  // .env may be missing/invalid -- the log is still readable at the default path.
}

let raw;
try {
  raw = await readFile(logFile, 'utf8');
} catch {
  console.log(`\n  No log file at "${logFile}" yet. Run "npm start" to create one.\n`);
  process.exit(0);
}

const entries = [];
raw.split('\n').forEach((line, i) => {
  if (!line.trim()) return;
  try {
    entries.push(JSON.parse(line));
  } catch {
    console.error(`  (skipping unparseable line ${i + 1})`);
  }
});

if (entries.length === 0) {
  console.log(`\n  "${logFile}" is empty. Run "npm start" to record an analysis.\n`);
  process.exit(0);
}

if (wantFull) {
  const idx = indexArg ? Number(indexArg) - 1 : entries.length - 1;
  const entry = entries[idx];
  if (!entry) {
    console.error(`\n  No entry #${idx + 1}. The log holds ${entries.length}.\n`);
    process.exit(1);
  }
  console.log(`\n  Replaying entry ${idx + 1} of ${entries.length}  (${entry.timestamp})`);
  renderDashboard(entry.snapshot, entry.analysis, { model: entry.model ?? 'unknown' });
  process.exit(0);
}

/* ---- compact table ---- */
const useColor = Boolean(process.stdout.isTTY) && !process.env.NO_COLOR;
const paint = (code) => (s) => (useColor ? `\x1b[${code}m${s}\x1b[0m` : String(s));
const dim = paint('2');
const bold = paint('1');
const SIGNAL_COLOR = { BUY: paint('32'), SELL: paint('31'), HOLD: paint('33') };

console.log('');
console.log(bold(`  ${entries.length} analysis/analyses in ${logFile}`));
console.log('');
console.log(dim('  #   WHEN (local)       SYMBOL      TF     PRICE       SIGNAL  CONF  RISK'));
console.log(dim('  ' + '─'.repeat(74)));

entries.forEach((e, i) => {
  const a = e.analysis ?? {};
  const s = e.snapshot ?? {};
  const when = new Date(e.timestamp).toLocaleString(undefined, {
    month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit',
  });
  const signal = a.final_signal ?? '?';
  const paintSignal = SIGNAL_COLOR[signal] ?? ((x) => x);
  console.log(
    `  ${String(i + 1).padEnd(3)} ${when.padEnd(18)} ${String(s.symbol ?? '?').padEnd(11)} ` +
    `${String(s.timeframe ?? '?').padEnd(6)} ${String(s.price ?? '?').padEnd(11)} ` +
    `${paintSignal(signal.padEnd(6))}  ${String(a.confidence ?? '?').padStart(3)}%  ${a.risk_level ?? '?'}`
  );
});

console.log('');
console.log(dim(`  Full detail:  npm run log -- --full <n>     (e.g. --full ${entries.length})`));
console.log('');
