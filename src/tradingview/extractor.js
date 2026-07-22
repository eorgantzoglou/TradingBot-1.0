/**
 * TradingView chart DOM extractor.
 *
 * Runs a self-contained scraping function inside the TradingView window via
 * CDP and normalizes the result into a clean "chart snapshot".
 *
 * TradingView's CSS class names are build-hashed (e.g. "valueValue-l31H9iuA"),
 * so every lookup targets stable `data-name` attributes / element ids first,
 * then partial class matches (`[class*="valueValue"]`), and finally falls back
 * to parsing the window title. Every fallback taken is recorded in
 * `snapshot.warnings` so you can see exactly how reliable each field is.
 */

export async function extractChartData(page) {
  const raw = await page.evaluate(scrapeChartDOM);
  return normalizeSnapshot(raw);
}

/* ===================================================================== */
/*  Browser-side scraper.                                                */
/*  Serialized by puppeteer and executed INSIDE the TradingView window,  */
/*  so it must be fully self-contained (no imports, no outer scope).     */
/* ===================================================================== */
function scrapeChartDOM() {
  const clean = (s) => (s || '').replace(/\u00a0/g, ' ').replace(/\s+/g, ' ').trim();
  const text = (el) => (el ? clean(el.textContent) : '');
  const q = (sel, root) => (root || document).querySelector(sel);
  const qa = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  /* In multi-chart layouts only the active chart matters. TradingView tags
     it with the "active" class; with a single chart there is exactly one. */
  const chartRoot =
    q('.chart-container.active') ||
    q('.chart-container') ||
    document;

  /* ---------------- Symbol (header toolbar) ---------------- */
  let symbol = '';
  const symbolBtn = q('#header-toolbar-symbol-search');
  if (symbolBtn) {
    symbol = text(q('.js-button-text', symbolBtn)) || text(symbolBtn);
  }

  /* ---------------- Timeframe (header toolbar) ---------------- */
  /* Two toolbar layouts exist: favorite intervals rendered as a radio group
     (the active one has aria-checked="true"), or a single dropdown button
     whose label holds the current interval. */
  let timeframeRaw = '';
  const intervalsRoot = q('#header-toolbar-intervals');
  if (intervalsRoot) {
    const checked = q('[aria-checked="true"]', intervalsRoot);
    if (checked) {
      timeframeRaw = checked.getAttribute('aria-label') || text(checked);
    } else {
      const menuBtn =
        q('button[aria-haspopup]', intervalsRoot) || q('button', intervalsRoot);
      if (menuBtn) {
        timeframeRaw = text(q('.js-button-text', menuBtn)) || text(menuBtn);
      }
    }
  }

  /* ---------------- Legend: main series + indicators ---------------- */
  const readLegendItem = (item) => {
    const titleParts = qa('[data-name="legend-source-title"]', item)
      .map(text)
      .filter(Boolean);
    let title = titleParts.join(' ');
    if (!title) {
      title = text(q('[class*="titleWrapper"]', item)) || text(q('[class*="title"]', item));
    }

    let values = qa('[class*="valueItem"]', item)
      .map((vi) => ({
        label: text(q('[class*="valueTitle"]', vi)),
        value: text(q('[class*="valueValue"]', vi)),
      }))
      .filter((v) => v.value !== '');
    if (values.length === 0) {
      values = qa('[class*="valueValue"]', item)
        .map((el) => ({ label: '', value: text(el) }))
        .filter((v) => v.value !== '');
    }

    /* Studies hidden via the legend "eye" toggle get a disabled/hidden class
       somewhere on the row — those are NOT visible on the chart. */
    const cls = typeof item.className === 'string' ? item.className : '';
    const hidden =
      /disabled|hidden/i.test(cls) ||
      Boolean(q('[class*="disabled"][class*="title"]', item));

    return { title, values, hidden };
  };

  const legendItems = qa('[data-name="legend-source-item"]', chartRoot);
  const explicitSeries = q('[data-name="legend-series-item"]', chartRoot);

  let seriesItem = null;
  const studyItems = [];
  for (const item of legendItems) {
    const isSeries =
      (explicitSeries && (item === explicitSeries || item.contains(explicitSeries))) ||
      (!explicitSeries && !seriesItem && item === legendItems[0]);
    if (isSeries && !seriesItem) seriesItem = item;
    else studyItems.push(item);
  }
  if (!seriesItem && explicitSeries) seriesItem = explicitSeries;

  return {
    url: location.href,
    pageTitle: document.title,
    symbol,
    timeframeRaw,
    series: seriesItem ? readLegendItem(seriesItem) : null,
    studies: studyItems.map(readLegendItem),
    legendItemCount: legendItems.length,
  };
}

/* ===================================================================== */
/*  Node-side normalization                                              */
/* ===================================================================== */

/** Parses TradingView-formatted numbers: "64,123.45", "\u2212150.2" (Unicode
 *  minus), "1.2K", "3.4M". Returns null when the text is not numeric. */
export function parseNumber(textValue) {
  if (typeof textValue !== 'string' || textValue.trim() === '') return null;
  const t = textValue
    .replace(/\u2212/g, '-') // Unicode minus sign
    .replace(/[,\s\u00a0']/g, '')
    .replace(/^\+/, '');
  const suffixMatch = t.match(/^(-?\d+(?:\.\d+)?)([KMBT])$/i);
  if (suffixMatch) {
    const mult = { K: 1e3, M: 1e6, B: 1e9, T: 1e12 }[suffixMatch[2].toUpperCase()];
    return Number(suffixMatch[1]) * mult;
  }
  if (!/^-?\d+(\.\d+)?$/.test(t)) return null;
  const n = Number(t);
  return Number.isFinite(n) ? n : null;
}

/** Normalizes the many ways TradingView expresses an interval into a
 *  compact form: "15m", "1h", "4h", "1D", "1W", "1M".
 *  Handles: "15", "60", "15m", "1D", "D", and aria-labels like "15 minutes". */
export function normalizeTimeframe(raw) {
  if (!raw) return null;
  const t = raw.trim();

  // aria-label style: "15 minutes", "1 hour", "1 day" ...
  const words = t.match(/^(\d+)\s*(second|minute|hour|day|week|month)s?$/i);
  if (words) {
    const unit = { second: 's', minute: 'm', hour: 'h', day: 'D', week: 'W', month: 'M' }[
      words[2].toLowerCase()
    ];
    return `${words[1]}${unit}`;
  }

  // Bare minutes: "15", "60", "240" (legend/legacy toolbar style)
  if (/^\d+$/.test(t)) {
    const n = Number(t);
    if (n >= 60 && n % 60 === 0) return `${n / 60}h`;
    return `${n}m`;
  }

  // Bare period letter: "D", "W", "M"
  if (/^[DWM]$/.test(t)) return `1${t}`;

  // Already compact ("15m", "1h", "1D", "12M", "1s") — keep case as-is,
  // since lowercase m = minutes but uppercase M = months.
  if (/^\d+[smhDWM]$/.test(t)) return t;

  return t; // Unknown format: pass through untouched.
}

const OHLC_LABELS = new Set(['O', 'H', 'L', 'C']);

function normalizeSnapshot(raw) {
  const warnings = [];

  /* ---- Symbol ---- */
  let symbol = raw.symbol || '';
  if (!symbol && raw.series?.title) {
    symbol = raw.series.title.split(/[\s\u00b7,]+/)[0] || '';
    if (symbol) warnings.push('Symbol read from chart legend (header toolbar not found).');
  }
  if (!symbol && raw.pageTitle) {
    const m = raw.pageTitle.replace(/^\(\d+\)\s*/, '').match(/^([A-Z0-9:!._/-]+)\s/);
    if (m) {
      symbol = m[1];
      warnings.push('Symbol parsed from the window title (toolbar and legend not found).');
    }
  }
  if (!symbol) warnings.push('Could not determine the ticker symbol.');

  /* ---- Timeframe ---- */
  let timeframe = normalizeTimeframe(raw.timeframeRaw);
  if (!timeframe && raw.series?.title) {
    const m = raw.series.title.match(/(?:^|[\s\u00b7,])(\d+[smhDWM]?|[DWM])(?:[\s\u00b7,]|$)/);
    if (m) {
      timeframe = normalizeTimeframe(m[1]);
      warnings.push('Timeframe read from chart legend (interval toolbar not found).');
    }
  }
  if (!timeframe) warnings.push('Could not determine the chart timeframe.');

  /* ---- OHLC + current price from the main series legend row ---- */
  const ohlc = { open: null, high: null, low: null, close: null };
  let price = null;
  const seriesValues = raw.series?.values ?? [];

  const labeled = seriesValues.filter((v) => OHLC_LABELS.has(v.label.toUpperCase()));
  if (labeled.length >= 2) {
    for (const v of labeled) {
      const key = { O: 'open', H: 'high', L: 'low', C: 'close' }[v.label.toUpperCase()];
      ohlc[key] = parseNumber(v.value);
    }
    price = ohlc.close;
  } else {
    // Unlabeled values: candles render as [O, H, L, C(, change...)],
    // line-style charts render a single value = last price.
    const numeric = seriesValues.map((v) => parseNumber(v.value)).filter((n) => n !== null);
    if (numeric.length >= 4) {
      [ohlc.open, ohlc.high, ohlc.low, ohlc.close] = numeric;
      price = ohlc.close;
      warnings.push('OHLC inferred from unlabeled legend values (assumed O/H/L/C order).');
    } else if (numeric.length >= 1) {
      price = numeric[0];
      warnings.push('Single legend value used as price (line-style chart?).');
    }
  }

  /* ---- Price fallback: the window title ("BTCUSD 64,123.45 \u25b2 +0.5% …") ---- */
  let changeText = null;
  if (raw.pageTitle) {
    const title = raw.pageTitle.replace(/^\(\d+\)\s*/, '');
    const m = title.match(/^\S+\s+([\d.,]+)\s*(?:[\u25b2\u25bc])?\s*([+\-\u2212][\d.,]+%)?/);
    if (m) {
      const titlePrice = parseNumber(m[1]);
      if (price === null && titlePrice !== null) {
        price = titlePrice;
        warnings.push('Price parsed from the window title (legend values not found).');
      }
      if (m[2]) changeText = m[2].replace(/\u2212/g, '-');
    }
  }
  if (price === null) warnings.push('Could not determine the current price.');

  /* ---- Indicators (visible studies only) ---- */
  const hiddenStudies = (raw.studies ?? []).filter((s) => s.hidden);
  const indicators = (raw.studies ?? [])
    .filter((s) => !s.hidden && s.title)
    .map((s) => ({
      name: s.title,
      values: s.values.map((v) => ({
        label: v.label || null,
        value: v.value,
        numeric: parseNumber(v.value),
      })),
    }));

  if (raw.legendItemCount === 0) {
    warnings.push('No legend items found — chart layout may have changed or no chart is open.');
  }

  return {
    extractedAt: new Date().toISOString(),
    symbol: symbol || null,
    timeframe: timeframe || null,
    price,
    ohlc,
    changeText,
    indicators,
    hiddenIndicatorCount: hiddenStudies.length,
    pageTitle: raw.pageTitle,
    url: raw.url,
    warnings,
  };
}
