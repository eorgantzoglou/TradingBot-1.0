/**
 * Terminal dashboard rendering. Plain ANSI escapes (no extra dependencies);
 * colors are disabled automatically when not writing to a TTY or when
 * NO_COLOR is set.
 */

const useColor = Boolean(process.stdout.isTTY) && !process.env.NO_COLOR;
const paint = (code) => (s) => (useColor ? `\x1b[${code}m${s}\x1b[0m` : String(s));

const bold = paint('1');
const dim = paint('2');
const red = paint('31');
const green = paint('32');
const yellow = paint('33');
const cyan = paint('36');

const WIDTH = 72;
const RULE = '─'.repeat(WIDTH);
const DOUBLE_RULE = '═'.repeat(WIDTH);

const SIGNAL_COLOR = { BUY: green, SELL: red, HOLD: yellow };
const RISK_COLOR = { LOW: green, MEDIUM: yellow, HIGH: red };
const TREND_COLOR = { BULLISH: green, BEARISH: red, RANGING: yellow };

function wrap(textValue, width, indent) {
  const words = String(textValue).split(/\s+/).filter(Boolean);
  const lines = [];
  let line = '';
  for (const word of words) {
    if (line && line.length + word.length + 1 > width) {
      lines.push(line);
      line = word;
    } else {
      line = line ? `${line} ${word}` : word;
    }
  }
  if (line) lines.push(line);
  return lines.map((l, i) => (i === 0 ? l : `${' '.repeat(indent)}${l}`)).join('\n');
}

const row = (label, value) => `  ${dim(label.padEnd(14))}${value}`;

export function renderDashboard(snapshot, analysis, { model }) {
  const out = [];
  out.push('');
  out.push(DOUBLE_RULE);
  out.push(bold(cyan('  TRADINGVIEW SNAPSHOT')) + dim(`   ${snapshot.extractedAt}`));
  out.push(DOUBLE_RULE);
  out.push(row('Symbol', bold(snapshot.symbol ?? 'UNKNOWN')));
  out.push(row('Timeframe', snapshot.timeframe ?? 'UNKNOWN'));
  out.push(row('Price', bold(String(snapshot.price ?? 'UNKNOWN')) + (snapshot.changeText ? dim(`  (${snapshot.changeText})`) : '')));

  const { open, high, low, close } = snapshot.ohlc;
  if ([open, high, low, close].some((v) => v !== null)) {
    out.push(row('Last bar', `O ${open ?? '?'}  H ${high ?? '?'}  L ${low ?? '?'}  C ${close ?? '?'}`));
  }

  out.push('');
  out.push(bold('  Indicators on chart'));
  if (snapshot.indicators.length === 0) {
    out.push(dim('    (none plotted — price action only)'));
  } else {
    for (const ind of snapshot.indicators) {
      const vals = ind.values.map((v) => (v.label ? `${v.label}: ${v.value}` : v.value)).join('  |  ');
      out.push(`    • ${ind.name.padEnd(28)} ${cyan(vals)}`);
    }
  }
  if (snapshot.hiddenIndicatorCount > 0) {
    out.push(dim(`    (+${snapshot.hiddenIndicatorCount} hidden indicator(s) excluded)`));
  }

  if (snapshot.warnings.length > 0) {
    out.push('');
    out.push(yellow('  Extractor warnings'));
    for (const w of snapshot.warnings) out.push(yellow(`    ! ${w}`));
  }

  out.push('');
  out.push(RULE);
  out.push(bold(cyan('  AI ANALYSIS')) + dim(`   model: ${model}`));
  out.push(RULE);

  const ta = analysis.trend_assessment;
  const trendPaint = (t) => (TREND_COLOR[t] ?? yellow)(t);
  out.push(row('Macro trend', trendPaint(ta.macro_trend)));
  out.push(row('Micro trend', trendPaint(ta.micro_trend)));
  if (ta.summary) out.push(row('Summary', wrap(ta.summary, WIDTH - 16, 16)));

  if (analysis.key_observations.length > 0) {
    out.push('');
    out.push(bold('  Key observations'));
    analysis.key_observations.forEach((obs, i) => {
      out.push(`    ${i + 1}. ${wrap(obs, WIDTH - 7, 7)}`);
    });
  }

  const { support, resistance } = analysis.key_levels;
  if (support.length > 0 || resistance.length > 0) {
    out.push('');
    out.push(bold('  Key levels'));
    if (support.length > 0) out.push(row('  Support', green(support.join('  '))));
    if (resistance.length > 0) out.push(row('  Resistance', red(resistance.join('  '))));
  }

  out.push('');
  out.push(row('Risk level', (RISK_COLOR[analysis.risk_level] ?? yellow)(analysis.risk_level)));
  out.push(row('Confidence', `${analysis.confidence}%`));

  for (const note of analysis.validation_notes ?? []) {
    out.push(yellow(`    ! ${note}`));
  }

  const signal = analysis.final_signal;
  const sPaint = SIGNAL_COLOR[signal] ?? yellow;
  const label = `  SIGNAL:  ${signal}  `;
  out.push('');
  out.push(`  ┌${'─'.repeat(label.length)}┐`);
  out.push(`  │${sPaint(bold(label))}│`);
  out.push(`  └${'─'.repeat(label.length)}┘`);
  out.push('');

  console.log(out.join('\n'));
}

export function renderError(err) {
  console.error('');
  console.error(red(bold('  ✖ ERROR')));
  console.error(red(`  ${String(err?.message ?? err).split('\n').join('\n  ')}`));
  console.error('');
}
