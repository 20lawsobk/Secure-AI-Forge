/**
 * Trending Context Service — Part 3: lightweight trend short-circuit.
 *
 * Produces 2–3 trending tokens (topics/hashtags) that are folded into the
 * `awareness` context sent to MaxCore. Trending is deliberately routed through
 * the awareness channel rather than concatenated onto the raw topic/idea:
 * ScriptAgent already extracts `#hashtags` and industry signals from awareness
 * to condition the prompt (see script_agent.py), whereas polluting the literal
 * topic corrupts the user-facing caption.
 *
 * Source: the existing content-awareness signal set (real RSS/industry-derived
 * trends), which is this platform's live trend feed. It is cached, so calling it
 * here adds negligible latency. Platform-specific signals are preferred when the
 * request targets particular platforms, falling back to the global trend list.
 */
import { contentAwarenessService } from "./contentAwarenessService.js";

const TREND_TIMEOUT_MS = 2_000;
const MAX_TOKENS = 3;

function withTimeout<T>(p: Promise<T>, fallback: T): Promise<T> {
  return Promise.race([
    p,
    new Promise<T>((resolve) => setTimeout(() => resolve(fallback), TREND_TIMEOUT_MS)),
  ]);
}

/**
 * Return up to 3 short trending tokens relevant to the requested platforms.
 * Never throws — returns [] on any error or timeout.
 */
export async function getTrendingTokens(platforms: string[]): Promise<string[]> {
  try {
    const ctx = await withTimeout(
      contentAwarenessService.getContextForMode("social").catch(() => null),
      null,
    );
    if (!ctx) return [];

    const platformSet = new Set(platforms.map((p) => p.toLowerCase()));
    const tokens: string[] = [];

    // Prefer platform-matched signal trends when platforms were requested.
    if (platformSet.size) {
      for (const sig of ctx.platformSignals ?? []) {
        if (sig.platform && platformSet.has(sig.platform.toLowerCase())) {
          if (sig.trend && sig.trend.trim()) tokens.push(sig.trend.trim());
        }
      }
    }

    // Fill remaining slots with global trending topics.
    for (const t of ctx.trendingTopics ?? []) {
      if (t && t.trim()) tokens.push(t.trim());
    }

    // Last resort: any platform-signal trend, regardless of platform.
    if (tokens.length < MAX_TOKENS) {
      for (const sig of ctx.platformSignals ?? []) {
        if (sig.trend && sig.trend.trim()) tokens.push(sig.trend.trim());
      }
    }

    return [...new Set(tokens)].filter(Boolean).slice(0, MAX_TOKENS);
  } catch {
    return [];
  }
}
