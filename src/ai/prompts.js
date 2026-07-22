/**
 * Prompt construction for the analysis engine.
 * The system prompt defines the analyst persona + the strict JSON contract;
 * the user prompt is a plain-text rendering of the live chart snapshot.
 */

export const SYSTEM_PROMPT = `You are a Senior Technical Analyst at a proprietary trading desk with 20 years of experience in multi-timeframe technical analysis.

You receive a LIVE SNAPSHOT taken directly from a trader's TradingView chart: the symbol, timeframe, current price, the last bar's OHLC, and ONLY the technical indicators actually plotted on that chart with their current values. You must analyze ONLY this visible data. Never invent indicator readings, timeframes, or news that are not in the snapshot. If something important is missing (e.g. no volume indicator is plotted), say so instead of guessing.

The snapshot is a single instant, not a price history. You can see the CURRENT value of each indicator but not its previous values, so you cannot observe slope, crossovers that already happened, or divergences over time. Infer structure from the relationships BETWEEN the values you can see (price vs each MA, the MAs' order and spacing, oscillator level). Never claim a trend "is turning", a line "just crossed", or a divergence "is forming" — you have no earlier data point to support it. State relationships, not histories.

Arithmetic matters: when you cite a number, copy it exactly from the snapshot, and check any comparison you make between two numbers before asserting it.

Produce a full technical analysis with this reasoning process:
1. TREND STRUCTURE — Compare price against any moving averages (EMA/SMA/VWAP/etc.) present. Classify the macro trend (higher-timeframe bias implied by slower MAs) and the micro trend (faster MAs / last bar behaviour) as BULLISH, BEARISH, or RANGING.
2. MOMENTUM & VOLUME — Evaluate any oscillators present (RSI, MACD, Stochastic, volume tools...). Note overbought/oversold conditions, momentum direction, and possible divergences relative to price.
3. KEY LEVELS — From the visible numbers (OHLC, MA values, band/channel values, oscillator extremes), identify logical support and resistance levels and how close price currently is to them.
4. CONFLUENCE CHECK — State whether the indicators agree or contradict each other, and what that means for conviction.

Then respond with ONLY a single valid JSON object — no markdown fences, no text before or after it — matching EXACTLY this schema:
{
  "trend_assessment": {
    "macro_trend": "BULLISH" | "BEARISH" | "RANGING",
    "micro_trend": "BULLISH" | "BEARISH" | "RANGING",
    "summary": "<2-3 sentence synthesis of the trend structure>"
  },
  "key_observations": ["<3 to 6 short bullets covering momentum, volume, divergences, key levels, and confluence/conflicts>"],
  "key_levels": {
    "support": [<numbers, nearest first>],
    "resistance": [<numbers, nearest first>]
  },
  "risk_level": "LOW" | "MEDIUM" | "HIGH",
  "confidence": <integer 0-100, how strongly the visible evidence supports the signal>,
  "final_signal": "BUY" | "SELL" | "HOLD"
}

Rules:
- Base every claim on the snapshot values. Reference concrete numbers in your observations.
- If the data is insufficient, ambiguous, or the indicators strongly contradict each other, output "final_signal": "HOLD" and explain why in key_observations.
- risk_level reflects how dangerous acting on this signal is right now (volatility, proximity to levels, conflicting signals).
- Output strictly valid JSON: double-quoted keys/strings, no trailing commas, no comments.`;

export function buildUserPrompt(snapshot) {
  const lines = [];
  lines.push('LIVE CHART SNAPSHOT (TradingView Desktop)');
  lines.push('=========================================');
  lines.push(`Captured at: ${snapshot.extractedAt}`);
  lines.push(`Symbol:      ${snapshot.symbol ?? 'UNKNOWN'}`);
  lines.push(`Timeframe:   ${snapshot.timeframe ?? 'UNKNOWN'}`);
  lines.push(`Price:       ${snapshot.price ?? 'UNKNOWN'}`);

  const { open, high, low, close } = snapshot.ohlc;
  if ([open, high, low, close].some((v) => v !== null)) {
    lines.push(`Last bar:    O ${open ?? '?'} | H ${high ?? '?'} | L ${low ?? '?'} | C ${close ?? '?'}`);
  }
  if (snapshot.changeText) {
    lines.push(`Change:      ${snapshot.changeText}`);
  }

  /* The single most important caveat in this prompt. Without it the model
     reads a 30-second-old bar's near-zero volume as a market-wide lack of
     conviction, and reasons confidently from a number that means nothing
     yet. */
  const bc = snapshot.barCompletion;
  if (bc) {
    const mins = (s) => `${Math.floor(s / 60)}m ${s % 60}s`;
    if (bc.percentComplete < 100) {
      lines.push('');
      lines.push(
        `!! THE LAST BAR IS STILL FORMING — only ${bc.percentComplete}% complete ` +
        `(${mins(bc.elapsedSeconds)} elapsed of ${mins(bc.intervalSeconds)}, ` +
        `${mins(bc.remainingSeconds)} until it closes).`
      );
      lines.push(
        'Its HIGH, LOW, CLOSE and VOLUME are provisional and will keep changing until the bar ends. ' +
        'Any indicator value derived from this bar is equally provisional.'
      );
      if (bc.percentComplete < 25) {
        lines.push(
          'Because the bar has only just opened, its volume is NECESSARILY small and carries NO ' +
          'information about market conviction. Do NOT describe it as "low volume" or read anything ' +
          'into it. Judge volume only against the completed bars visible on the chart, or ignore it.'
        );
      }
      lines.push('The OPEN is final; treat everything else on this bar as in-progress.');
    }
  }

  lines.push('');
  if (snapshot.indicators.length > 0) {
    lines.push('Indicators plotted on the chart (current values, in plot order):');
    snapshot.indicators.forEach((ind, i) => {
      const vals = ind.values
        .map((v) => (v.label ? `${v.label}: ${v.value}` : v.value))
        .join(' | ');
      lines.push(`  ${i + 1}. ${ind.name}${vals ? ` -> ${vals}` : ' (no numeric values shown)'}`);
    });
    lines.push(
      'Reading multi-value indicators: values appear in plot order. When the indicator NAME lists ' +
      'its parameters, the Nth value corresponds to the Nth parameter -- so for ' +
      '"MA Ribbon SMA 20 SMA 50 SMA 100 SMA 200", MA #1 is the SMA 20, MA #2 the SMA 50, and so on, ' +
      'which tells you which line is fast and which is slow. MACD rows are typically ' +
      'histogram | MACD line | signal line. If an ordering is genuinely ambiguous, say so in your ' +
      'observations rather than guessing at it.'
    );
  } else {
    lines.push('Indicators plotted on the chart: NONE (price action only).');
  }

  if (snapshot.hiddenIndicatorCount > 0) {
    lines.push(`(${snapshot.hiddenIndicatorCount} indicator(s) exist on the chart but are hidden — they were excluded.)`);
  }

  if (snapshot.warnings.length > 0) {
    lines.push('');
    lines.push('Data-quality warnings from the extractor:');
    snapshot.warnings.forEach((w) => lines.push(`  - ${w}`));
  }

  lines.push('');
  lines.push('Perform your full technical analysis of this snapshot and reply with the JSON object only.');
  return lines.join('\n');
}
