import puppeteer from 'puppeteer-core';

/**
 * Connects to the TradingView Desktop app over the Chrome DevTools Protocol
 * (the desktop app is Electron/Chromium, so it speaks CDP when launched with
 * --remote-debugging-port) and locates the window that hosts the chart.
 *
 * IMPORTANT: we only ever `disconnect()` from the app, never `close()` it —
 * closing would kill the user's TradingView session.
 */

const CHART_MARKER_SELECTOR =
  '[data-name="legend"], .chart-container, .chart-markup-table, #header-toolbar-symbol-search';

export async function connectToTradingView({ host, port }) {
  const browserURL = `http://${host}:${port}`;
  let browser;
  try {
    browser = await puppeteer.connect({
      browserURL,
      defaultViewport: null,
      protocolTimeout: 30_000,
    });
  } catch (err) {
    throw new Error(
      `Could not reach TradingView's debugging port at ${browserURL}.\n` +
      'Make sure TradingView Desktop is running and was launched with:\n' +
      `  TradingView.exe --remote-debugging-port=${port}\n` +
      `(underlying error: ${err.message})`
    );
  }

  const page = await findChartPage(browser);
  if (!page) {
    disconnect(browser);
    throw new Error(
      'Connected to TradingView via CDP, but no chart window was found. ' +
      'Open a chart in the desktop app, then try again. ' +
      '(Run "npm run check:cdp" to list what the debugger can see.)'
    );
  }
  return { browser, page };
}

/**
 * Finds the page (window/webContents) that actually renders the chart.
 * Prefers tradingview.com URLs, but falls back to probing every page for
 * chart DOM markers — the Electron shell can host content under other URLs.
 */
async function findChartPage(browser) {
  let pages;
  try {
    pages = await browser.pages();
  } catch {
    return null;
  }

  const byUrl = pages.filter((p) => /tradingview/i.test(p.url()));
  const ordered = [...byUrl, ...pages.filter((p) => !byUrl.includes(p))];

  for (const page of ordered) {
    try {
      const hasChart = await page.evaluate(
        (sel) => Boolean(document.querySelector(sel)),
        CHART_MARKER_SELECTOR
      );
      if (hasChart) return page;
    } catch {
      // Some targets (devtools, service workers, background pages) cannot be
      // evaluated — skip them silently.
    }
  }
  return null;
}

export function disconnect(browser) {
  try {
    browser.disconnect();
  } catch {
    // Already disconnected — nothing to do.
  }
}
