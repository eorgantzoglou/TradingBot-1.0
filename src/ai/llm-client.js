import OpenAI from 'openai';
import { SYSTEM_PROMPT, buildUserPrompt } from './prompts.js';

const SIGNALS = new Set(['BUY', 'SELL', 'HOLD']);
const RISKS = new Set(['LOW', 'MEDIUM', 'HIGH']);
const TRENDS = new Set(['BULLISH', 'BEARISH', 'RANGING']);

/**
 * Provider-agnostic LLM client. Uses the official `openai` package, so any
 * OpenAI-compatible endpoint works (OpenAI, LM Studio, Ollama, Anthropic's
 * compatibility layer, vLLM, ...). Everything comes from .env — nothing is
 * hardcoded.
 */
export function createLLMClient({ apiKey, baseURL, model, temperature, reasoningEffort, timeoutMs }) {
  const client = new OpenAI({ apiKey, baseURL, maxRetries: 2, timeout: timeoutMs ?? 180_000 });

  async function complete(messages, { jsonMode, effort }) {
    return client.chat.completions.create({
      model,
      temperature,
      messages,
      ...(jsonMode ? { response_format: { type: 'json_object' } } : {}),
      ...(effort ? { reasoning_effort: effort } : {}),
    });
  }

  /**
   * Sends the chart snapshot for analysis and returns
   * { analysis, rawResponse, usage }.
   *
   * Providers disagree about which optional parameters they accept, so the
   * request degrades one capability at a time on a 4xx rather than failing:
   * full -> drop json mode -> drop reasoning_effort.
   */
  async function analyze(snapshot) {
    const messages = [
      { role: 'system', content: SYSTEM_PROMPT },
      { role: 'user', content: buildUserPrompt(snapshot) },
    ];

    const attempts = [
      { jsonMode: true, effort: reasoningEffort },
      ...(reasoningEffort ? [{ jsonMode: false, effort: reasoningEffort }] : []),
      { jsonMode: false, effort: undefined },
    ];

    let completion;
    let lastErr;
    for (const opts of attempts) {
      try {
        completion = await complete(messages, opts);
        break;
      } catch (err) {
        lastErr = err;
        // Only a parameter-rejection is worth retrying with fewer features.
        if (err?.status !== 400 && err?.status !== 422) {
          throw decorateProviderError(err, baseURL);
        }
      }
    }
    if (!completion) throw decorateProviderError(lastErr, baseURL);

    const message = completion?.choices?.[0]?.message ?? {};
    const content = message.content ?? '';
    if (!content.trim()) {
      // Hybrid-thinking models divert their output into a non-standard
      // `reasoning` field when thinking is left enabled, leaving content empty.
      const divertedToReasoning = Boolean(message.reasoning || message.reasoning_content);
      throw new Error(
        divertedToReasoning
          ? 'The LLM returned an empty response: its output went to the non-standard ' +
            '"reasoning" field instead of "content". This model is a hybrid-thinking ' +
            'model with thinking enabled — set REASONING_EFFORT=none in your .env.'
          : 'The LLM returned an empty response.'
      );
    }
    const analysis = validateAnalysis(extractJson(content));
    return { analysis, rawResponse: content, usage: completion.usage ?? null };
  }

  return { analyze };
}

function decorateProviderError(err, baseURL) {
  if (err?.status === 401) {
    return new Error(`The LLM endpoint rejected the API key (401). Check OPENAI_API_KEY. (${err.message})`);
  }
  if (err?.code === 'ECONNREFUSED' || /ECONNREFUSED|fetch failed|Connection error/i.test(err?.message ?? '')) {
    return new Error(
      `Could not reach the LLM endpoint at ${baseURL ?? 'https://api.openai.com/v1'}. ` +
      `Is your local model server (LM Studio/Ollama) running? (${err.message})`
    );
  }
  return err;
}

/**
 * Extracts a JSON object from an LLM response that may include reasoning
 * (<think> blocks from local reasoning models), markdown fences, or prose
 * around the JSON.
 */
export function extractJson(content) {
  let t = content
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/```(?:json)?/gi, '')
    .trim();

  try {
    return JSON.parse(t);
  } catch {
    // Fall through to balanced-brace scanning.
  }

  const start = t.indexOf('{');
  if (start === -1) {
    throw new Error(`The LLM response contained no JSON object. Response was:\n${content}`);
  }
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < t.length; i++) {
    const ch = t[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '{') depth++;
    else if (ch === '}') {
      depth--;
      if (depth === 0) {
        try {
          return JSON.parse(t.slice(start, i + 1));
        } catch (err) {
          throw new Error(`Found a JSON-like block in the LLM response but it failed to parse (${err.message}).\nResponse was:\n${content}`);
        }
      }
    }
  }
  throw new Error(`The LLM response contained an unterminated JSON object. Response was:\n${content}`);
}

/**
 * Coerces the model output into the strict schema. Enum violations degrade
 * safely: an unknown signal becomes HOLD, an unknown risk becomes HIGH —
 * never let a malformed response generate a trade.
 */
export function validateAnalysis(parsed) {
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('The LLM response JSON is not an object.');
  }
  const notes = [];

  const asTrend = (v) => {
    const t = String(v ?? '').toUpperCase();
    return TRENDS.has(t) ? t : 'RANGING';
  };
  const ta = parsed.trend_assessment;
  const trend_assessment =
    ta && typeof ta === 'object'
      ? {
          macro_trend: asTrend(ta.macro_trend),
          micro_trend: asTrend(ta.micro_trend),
          summary: typeof ta.summary === 'string' ? ta.summary : '',
        }
      : { macro_trend: 'RANGING', micro_trend: 'RANGING', summary: typeof ta === 'string' ? ta : '' };

  const key_observations = Array.isArray(parsed.key_observations)
    ? parsed.key_observations.filter((o) => typeof o === 'string' && o.trim() !== '')
    : [];

  const levels = parsed.key_levels && typeof parsed.key_levels === 'object' ? parsed.key_levels : {};
  const asNumbers = (arr) =>
    Array.isArray(arr) ? arr.map(Number).filter((n) => Number.isFinite(n)) : [];
  const key_levels = { support: asNumbers(levels.support), resistance: asNumbers(levels.resistance) };

  let risk_level = String(parsed.risk_level ?? '').toUpperCase();
  if (!RISKS.has(risk_level)) {
    notes.push(`Model returned invalid risk_level "${parsed.risk_level}" — defaulted to HIGH.`);
    risk_level = 'HIGH';
  }

  let final_signal = String(parsed.final_signal ?? '').toUpperCase();
  if (!SIGNALS.has(final_signal)) {
    notes.push(`Model returned invalid final_signal "${parsed.final_signal}" — defaulted to HOLD.`);
    final_signal = 'HOLD';
  }

  let confidence = Number(parsed.confidence);
  if (!Number.isFinite(confidence)) confidence = 0;
  confidence = Math.max(0, Math.min(100, Math.round(confidence)));

  return { trend_assessment, key_observations, key_levels, risk_level, confidence, final_signal, validation_notes: notes };
}
