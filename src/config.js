import 'dotenv/config';

function toNumber(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

/**
 * Loads and validates all configuration from the environment (.env).
 * Nothing is hardcoded: the LLM provider is entirely dictated by
 * OPENAI_API_KEY / OPENAI_BASE_URL / MODEL_NAME.
 */
export function loadConfig() {
  const config = {
    llm: {
      apiKey: (process.env.OPENAI_API_KEY || '').trim(),
      // Empty/unset baseURL -> the openai package defaults to api.openai.com
      baseURL: (process.env.OPENAI_BASE_URL || '').trim() || undefined,
      model: (process.env.MODEL_NAME || '').trim(),
      temperature: toNumber(process.env.TEMPERATURE, 0.2),
      // Hybrid-thinking models (Qwen3.x, etc.) reason by default, which is
      // enormously slow for routine structured output. Sending
      // reasoning_effort=none disables it. Left unset the parameter is not
      // sent at all, so providers that reject it are unaffected.
      reasoningEffort: (process.env.REASONING_EFFORT || '').trim() || undefined,
      timeoutMs: toNumber(process.env.REQUEST_TIMEOUT_MS, 180_000),
    },
    cdp: {
      host: (process.env.CDP_HOST || '127.0.0.1').trim(),
      port: toNumber(process.env.CDP_PORT, 9222),
    },
    pollIntervalMs: toNumber(process.env.POLL_INTERVAL_MS, 0),
    logFile: (process.env.LOG_FILE || 'trades.log').trim(),
    watch: {
      /* How often to LOOK at the chart in watch mode. Looking is cheap (a
         DOM read); analyzing is not, so these are separate concerns. */
      pollMs: toNumber(process.env.WATCH_POLL_MS, 20_000),
      /* Analyze once per bar, when the bar is at least this % formed.
         Reading a bar that has just opened means judging provisional
         extremes and a volume near zero; waiting until it is nearly closed
         gives the analyst near-final data. Set 0 to analyze every poll. */
      analyzeAtBarPct: toNumber(process.env.ANALYZE_AT_BAR_PCT, 90),
      /* Ring the terminal bell when the verdict changes. */
      bell: (process.env.ALERT_BELL || 'true').trim().toLowerCase() !== 'false',
    },
  };

  const problems = [];
  if (!config.llm.model) {
    problems.push('MODEL_NAME is required (the model id your provider serves, e.g. "gpt-4o-mini" or a local model name).');
  }
  if (!config.llm.apiKey) {
    problems.push('OPENAI_API_KEY is required (for local servers like LM Studio/Ollama any placeholder string works).');
  }
  if (problems.length > 0) {
    throw new Error(
      `Invalid configuration:\n  - ${problems.join('\n  - ')}\n` +
      'Copy .env.example to .env and fill in the values.'
    );
  }
  return config;
}
