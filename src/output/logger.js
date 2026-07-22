import { appendFile } from 'node:fs/promises';

/**
 * Appends one interaction to the log file as a single JSON line (JSONL),
 * so the log is trivially parseable later for backtesting:
 *   const entries = fs.readFileSync('trades.log', 'utf8')
 *     .trim().split('\n').map(JSON.parse);
 */
export async function logInteraction(logFile, entry) {
  try {
    await appendFile(logFile, JSON.stringify(entry) + '\n', 'utf8');
  } catch (err) {
    // Logging must never crash the bot mid-analysis.
    console.error(`  (warning: could not write to ${logFile}: ${err.message})`);
  }
}
