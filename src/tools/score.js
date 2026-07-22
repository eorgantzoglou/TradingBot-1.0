import { readFile } from 'node:fs/promises';
import { loadConfig } from '../config.js';
import { timeframeToSeconds } from '../tradingview/extractor.js';

/**
 * Scores past signals in trades.log against what price ACTUALLY did.
 *
 *   npm run score
 *   npm run score -- --horizon 8 --threshold 0.25
 *
 * For each logged analysis it takes the price the model saw as the entry,
 * then looks up the close N bars later from public exchange data, and asks
 * whether the signal pointed the right way.
 *
 * This is deliberately a blunt instrument. It measures direction only: no
 * position sizing, no stops, no fees, no compounding. It answers one
 * question -- "is this model's directional call better than a coin flip?"
 * -- and nothing more. Read the sample-size warning before believing any
 * number it prints.
 */

/* ------------------------- args ------------------------- */
const argv = process.argv.slice(2);
const argVal = (name, fallback) => {
  const i = argv.indexOf(`--${name}`);
  if (i === -1 || i + 1 >= argv.length) return fallback;
  const n = Number(argv[i + 1]);
  return Number.isFinite(n) ? n : fallback;
};
const HORIZON = Math.max(1, Math.round(argVal('horizon', 4)));   // bars ahead
const THRESHOLD = argVal('threshold', 0.15);                     // percent

/* ------------------------- colour ------------------------- */
const useColor = Boolean(process.stdout.isTTY) && !process.env.NO_COLOR;
const paint = (c) => (s) => (useColor ? `\x1b[${c}m${s}\x1b[0m` : String(s));
const bold = paint('1');
const dim = paint('2');
const red = paint('31');
const green = paint('32');
const yellow = paint('33');

/* ------------------------- price sources ------------------------- */

/** Bitstamp only accepts these candle widths (seconds). */
const BITSTAMP_STEPS = [60, 180, 300, 900, 1800, 3600, 7200, 14400, 21600, 43200, 86400, 259200];
const BINANCE_INTERVALS = {
  60: '1m', 180: '3m', 300: '5m', 900: '15m', 1800: '30m', 3600: '1h',
  7200: '2h', 14400: '4h', 21600: '6h', 28800: '8h', 43200: '12h', 86400: '1d',
};

const cache = new Map();

async function fetchJson(url) {
  const res = await fetch(url, { headers: { 'User-Agent': 'TradingBot-score' } });
  if (!res.ok) throw new Error(`HTTP ${res.status} from ${new URL(url).host}`);
  return res.json();
}

/**
 * Returns candles [{ t, close }] at `step` seconds starting at `startSec`.
 * Tries Bitstamp first because that is the exchange the chart reads from;
 * falls back to Binance, which covers far more symbols but is a different
 * venue and so will not tick-for-tick match the chart.
 */
async function getCandles(symbol, step, startSec, count) {
  const key = `${symbol}|${step}|${startSec}|${count}`;
  if (cache.has(key)) return cache.get(key);

  const bare = String(symbol).replace(/^[A-Za-z]+:/, '').replace(/[^A-Za-z0-9]/g, '');
  let candles = null;
  let source = null;

  if (BITSTAMP_STEPS.includes(step)) {
    try {
      const url = `https://www.bitstamp.net/api/v2/ohlc/${bare.toLowerCase()}/`
                + `?step=${step}&limit=${count}&start=${startSec}`;
      const j = await fetchJson(url);
      const rows = j?.data?.ohlc;
      if (Array.isArray(rows) && rows.length > 0) {
        candles = rows.map((r) => ({ t: Number(r.timestamp), close: Number(r.close) }));
        source = 'bitstamp';
      }
    } catch { /* fall through to Binance */ }
  }

  if (!candles && BINANCE_INTERVALS[step]) {
    try {
      const sym = /USD$/i.test(bare) && !/USDT$/i.test(bare) ? `${bare}T` : bare; // BTCUSD -> BTCUSDT
      const url = `https://api.binance.com/api/v3/klines?symbol=${sym.toUpperCase()}`
                + `&interval=${BINANCE_INTERVALS[step]}&startTime=${startSec * 1000}&limit=${count}`;
      const rows = await fetchJson(url);
      if (Array.isArray(rows) && rows.length > 0) {
        candles = rows.map((r) => ({ t: Math.floor(r[0] / 1000), close: Number(r[4]) }));
        source = 'binance';
      }
    } catch { /* unsupported symbol */ }
  }

  const result = candles ? { candles, source } : null;
  cache.set(key, result);
  return result;
}

/* ------------------------- load the log ------------------------- */
let logFile = 'trades.log';
try { logFile = loadConfig().logFile; } catch { /* default is fine */ }

let raw;
try {
  raw = await readFile(logFile, 'utf8');
} catch {
  console.log(`\n  No log file at "${logFile}". Run "npm start" a few times first.\n`);
  process.exit(0);
}

const entries = raw.split('\n')
  .filter((l) => l.trim())
  .map((l) => { try { return JSON.parse(l); } catch { return null; } })
  .filter(Boolean);

if (entries.length === 0) {
  console.log(`\n  "${logFile}" holds no readable entries.\n`);
  process.exit(0);
}

/* ------------------------- evaluate ------------------------- */
console.log('');
console.log(bold(`  Scoring ${entries.length} logged signal(s)`));
console.log(dim(`  horizon ${HORIZON} bars  |  flat band +/-${THRESHOLD}%  |  entry = price the model saw`));
console.log('');

const results = [];
const nowSec = Math.floor(Date.now() / 1000);

for (const e of entries) {
  const snap = e.snapshot ?? {};
  const signal = e.analysis?.final_signal;
  const symbol = snap.symbol;
  const entryPrice = Number(snap.price);
  const step = timeframeToSeconds(snap.timeframe);
  const tSec = Math.floor(new Date(snap.extractedAt ?? e.timestamp).getTime() / 1000);

  const base = { when: snap.extractedAt ?? e.timestamp, symbol, tf: snap.timeframe, signal, entryPrice };

  if (!signal || !symbol || !Number.isFinite(entryPrice) || !step) {
    results.push({ ...base, status: 'skipped', reason: 'incomplete log entry' });
    continue;
  }

  const barStart = Math.floor(tSec / step) * step;
  const exitBar = barStart + HORIZON * step;
  // The exit bar must have CLOSED, otherwise its close is still moving.
  if (exitBar + step > nowSec) {
    const mins = Math.ceil((exitBar + step - nowSec) / 60);
    results.push({ ...base, status: 'pending', reason: `${mins}m until horizon closes` });
    continue;
  }

  let data;
  try {
    data = await getCandles(symbol, step, barStart, HORIZON + 2);
  } catch (err) {
    results.push({ ...base, status: 'skipped', reason: err.message });
    continue;
  }
  if (!data) {
    results.push({ ...base, status: 'skipped', reason: 'no price data for this symbol' });
    continue;
  }

  const exit = data.candles.find((c) => c.t === exitBar)
            ?? data.candles.filter((c) => c.t <= exitBar).pop();
  if (!exit) {
    results.push({ ...base, status: 'skipped', reason: 'exit bar missing from feed' });
    continue;
  }

  const movePct = ((exit.close - entryPrice) / entryPrice) * 100;
  let correct;
  if (signal === 'BUY') correct = movePct > THRESHOLD;
  else if (signal === 'SELL') correct = movePct < -THRESHOLD;
  else correct = Math.abs(movePct) <= THRESHOLD;   // HOLD = price stayed flat

  results.push({ ...base, status: 'scored', exitPrice: exit.close, movePct, correct, source: data.source });
}

/* ------------------------- report ------------------------- */
console.log(dim('  WHEN               SYMBOL    TF    SIGNAL  ENTRY      EXIT       MOVE      VERDICT'));
console.log(dim('  ' + '-'.repeat(84)));

for (const r of results) {
  const when = new Date(r.when).toLocaleString(undefined, { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit' });
  const head = `  ${when.padEnd(18)} ${String(r.symbol ?? '?').padEnd(9)} ${String(r.tf ?? '?').padEnd(5)} ${String(r.signal ?? '?').padEnd(7)}`;
  if (r.status === 'scored') {
    const move = `${r.movePct >= 0 ? '+' : ''}${r.movePct.toFixed(2)}%`;
    const verdict = r.correct ? green('RIGHT') : red('WRONG');
    console.log(`${head} ${String(r.entryPrice).padEnd(10)} ${String(r.exitPrice).padEnd(10)} ${move.padStart(8)}  ${verdict}`);
  } else {
    console.log(`${head} ${String(r.entryPrice ?? '?').padEnd(10)} ${dim(`-- ${r.status}: ${r.reason}`)}`);
  }
}

const scored = results.filter((r) => r.status === 'scored');
const pending = results.filter((r) => r.status === 'pending');
const skipped = results.filter((r) => r.status === 'skipped');

console.log('');
console.log(bold('  SUMMARY'));
console.log(`    scored: ${scored.length}   pending: ${pending.length}   skipped: ${skipped.length}`);

if (scored.length === 0) {
  console.log('');
  console.log(yellow('    Nothing scoreable yet. Signals need to age past the horizon'));
  console.log(yellow(`    (${HORIZON} bars) before price can be checked against them.`));
  console.log('');
  process.exit(0);
}

const hits = scored.filter((r) => r.correct).length;
const rate = (hits / scored.length) * 100;
console.log(`    hit rate: ${bold(`${hits}/${scored.length} = ${rate.toFixed(1)}%`)}`);

for (const sig of ['BUY', 'SELL', 'HOLD']) {
  const subset = scored.filter((r) => r.signal === sig);
  if (subset.length === 0) continue;
  const h = subset.filter((r) => r.correct).length;
  const avg = subset.reduce((a, r) => a + r.movePct, 0) / subset.length;
  console.log(`      ${sig.padEnd(5)} ${h}/${subset.length} correct   avg move ${avg >= 0 ? '+' : ''}${avg.toFixed(2)}%`);
}

if (scored.some((r) => r.source === 'binance')) {
  console.log(dim('    note: some prices came from Binance, a different venue than the chart.'));
}

/* The most important line this tool prints. A 70% hit rate on 6 samples is
   noise, and presenting it without this caveat would be actively misleading. */
console.log('');
if (scored.length < 30) {
  console.log(yellow(`    !! ${scored.length} samples is FAR too few to mean anything.`));
  console.log(yellow('       A coin flip returns 70%+ on small samples routinely. Treat this'));
  console.log(yellow('       as a plumbing check, not evidence. Aim for 100+ before judging.'));
} else if (rate < 55) {
  console.log(yellow('    At this rate the model is not demonstrably better than chance.'));
  console.log(yellow('    Directional calls near 50% carry no edge once fees are paid.'));
} else {
  console.log(`    Above chance on ${scored.length} samples. Worth continuing to measure --`);
  console.log('    then check it holds on a different symbol and market regime.');
}
console.log('');
