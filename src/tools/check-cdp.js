import 'dotenv/config';

/**
 * Diagnostic helper: lists every target TradingView Desktop exposes on the
 * CDP port. Use this when the bot cannot find the chart window.
 *
 *   npm run check:cdp
 */
const host = process.env.CDP_HOST || '127.0.0.1';
const port = Number(process.env.CDP_PORT) || 9222;
const base = `http://${host}:${port}`;

try {
  const version = await (await fetch(`${base}/json/version`)).json();
  console.log(`Connected to: ${version.Browser ?? 'unknown browser'} (${base})\n`);

  const targets = await (await fetch(`${base}/json/list`)).json();
  if (!Array.isArray(targets) || targets.length === 0) {
    console.log('No debuggable targets found.');
  } else {
    console.log(`Found ${targets.length} target(s):`);
    for (const t of targets) {
      console.log(`  [${t.type}] ${t.title || '(untitled)'}`);
      console.log(`         ${t.url}`);
    }
  }
  console.log('\nThe bot needs at least one "page" target containing the TradingView chart.');
} catch (err) {
  console.error(`Could not reach ${base} — ${err.message}`);
  console.error('\nLaunch TradingView Desktop with the debugging port enabled, e.g.:');
  console.error(`  & "$env:LOCALAPPDATA\\Programs\\TradingView\\TradingView.exe" --remote-debugging-port=${port}`);
  process.exitCode = 1;
}
