/**
 * Music Industry Context Filter
 *
 * Transforms live industry intelligence signals (industryMonitorService) into
 * structured generation context that content, melody, music, and social
 * services can inject into their AI calls.
 *
 * Signal flow:
 *   industryMonitorService (RSS + Search)
 *       └──► musicIndustryContextFilter
 *               ├──► unifiedAIController?.generateContent()  → extraContext injection
 *               ├──► melodyPatternService?.generate*()       → genre / style hints (sync)
 *               ├──► musicGenerationService?.parseText*()    → mood / tempo hints (sync)
 *               └──► routes/songwriting?.ts  ai-assist       → lyric theme context
 *
 * The scanning layer (industryMonitorService) answers: "What changed in the industry?"
 * This filter answers:  "What does that mean for what we generate right now?"
 *
 * All generation services remain fully backward-compatible — if the filter
 * has no cached data, behaviour is identical to before.  Context enrichment
 * is purely additive and never blocks generation.
 */

import {
  industryMonitor,
  type LiveIndustryChange,
} from "./industryMonitorService.js";
import { logger } from "../logger.js";

// ─── Public types ──────────────────────────────────────────────────────────────

export type GenerationMode =
  | "social"
  | "melody"
  | "music"
  | "songwriting"
  | "content"
  | "advertising";

export interface PlatformTrendSignal {
  platform: string;
  trend: string;
  strength: "strong" | "moderate" | "emerging";
}

export interface GenerationHints {
  suggestedGenre?: string;
  suggestedMood?: string;
  tempoBias: "up" | "down" | "neutral";
  productionKeywords: string[];
  contentAngles: string[];
  hashtagContext: string;
}

export interface MusicIndustryContext {
  trendingGenres: string[];
  trendingMoods: string[];
  productionStyles: string[];
  platformSignals: PlatformTrendSignal[];
  viralHookPatterns: string[];
  lyricThemes: string[];
  generationHints: GenerationHints;
  /** Compact string ready to append to extraContext in any MaxCore call */
  contextString: string;
  signalCount: number;
  /** 0–1 confidence — scales with signal count and freshness */
  confidence: number;
  freshness: Date;
}

// ─── Internal keyword tables ───────────────────────────────────────────────────

interface KW<T extends string> {
  pattern: RegExp;
  value: T;
  weight: number;
}

const GENRE_KW: KW<string>[] = [
  { pattern: /\btrap\b/i, value: "trap", weight: 2.0 },
  { pattern: /\bphonk\b/i, value: "phonk", weight: 2.0 },
  { pattern: /\bafrobeats?\b/i, value: "afrobeats", weight: 2.0 },
  { pattern: /\bamapiano\b/i, value: "amapiano", weight: 2.0 },
  { pattern: /\bdrill\b/i, value: "drill", weight: 2.0 },
  { pattern: /\bhyperpop\b/i, value: "hyperpop", weight: 2.0 },
  { pattern: /\blo[.\s-]?fi\b/i, value: "lo-fi", weight: 2.0 },
  { pattern: /\bbedroom pop\b/i, value: "bedroom pop", weight: 2.0 },
  { pattern: /\bk[.\s-]?pop\b/i, value: "K-pop", weight: 2.0 },
  { pattern: /\bneo[.\s-]?soul\b/i, value: "neo-soul", weight: 2.0 },
  { pattern: /\balt[.\s-]?pop\b/i, value: "alt-pop", weight: 1.5 },
  { pattern: /\breggaeton\b/i, value: "reggaeton", weight: 2.0 },
  { pattern: /\bcloud rap\b/i, value: "cloud rap", weight: 2.0 },
  { pattern: /\bsynthwave\b/i, value: "synthwave", weight: 2.0 },
  { pattern: /\bvaporwave\b/i, value: "vaporwave", weight: 2.0 },
  { pattern: /\bindietronica\b/i, value: "indietronica", weight: 1.5 },
  { pattern: /\bpop[.\s-]?punk\b/i, value: "pop punk", weight: 2.0 },
  { pattern: /\br&b\b/i, value: "R&B", weight: 1.5 },
  { pattern: /\bhip[.\s-]?hop\b/i, value: "hip-hop", weight: 1.5 },
  { pattern: /\bhouse\b/i, value: "house", weight: 1.0 },
  { pattern: /\btechno\b/i, value: "techno", weight: 1.0 },
  { pattern: /\bambient\b/i, value: "ambient", weight: 1.5 },
  { pattern: /\bedm\b/i, value: "EDM", weight: 1.0 },
  { pattern: /\blatin\b/i, value: "latin", weight: 1.5 },
  { pattern: /\bindies?\b/i, value: "indie", weight: 1.0 },
  { pattern: /\belectronic\b/i, value: "electronic", weight: 0.8 },
  { pattern: /\bpop\b/i, value: "pop", weight: 0.5 },
];

const MOOD_KW: KW<string>[] = [
  { pattern: /\bmelanchol\w*\b/i, value: "melancholic", weight: 2.0 },
  { pattern: /\beuphori\w*\b/i, value: "euphoric", weight: 2.0 },
  { pattern: /\bnostalgic\b/i, value: "nostalgic", weight: 2.0 },
  { pattern: /\bdark\b/i, value: "dark", weight: 1.0 },
  { pattern: /\buplifting\b/i, value: "uplifting", weight: 1.5 },
  { pattern: /\benergetic\b/i, value: "energetic", weight: 1.5 },
  { pattern: /\brelax\w*\b/i, value: "relaxed", weight: 1.0 },
  { pattern: /\bcalm\b/i, value: "calm", weight: 1.0 },
  { pattern: /\baggressive\b/i, value: "aggressive", weight: 1.5 },
  { pattern: /\bvulnerab\w*\b/i, value: "vulnerable", weight: 2.0 },
  { pattern: /\bempow\w*\b/i, value: "empowering", weight: 2.0 },
  { pattern: /\bmotivational\b/i, value: "motivational", weight: 1.5 },
  { pattern: /\bchill\b/i, value: "chill", weight: 1.0 },
  { pattern: /\braw\b/i, value: "raw", weight: 1.0 },
  { pattern: /\bintense\b/i, value: "intense", weight: 1.5 },
  { pattern: /\bsoulful\b/i, value: "soulful", weight: 1.5 },
  { pattern: /\bplayful\b/i, value: "playful", weight: 1.5 },
  { pattern: /\bdriven\b/i, value: "driven", weight: 1.0 },
];

const PRODUCTION_KW: KW<string>[] = [
  { pattern: /\b808\b/i, value: "808 bass", weight: 2.0 },
  { pattern: /\bspatial audio\b/i, value: "spatial audio", weight: 2.0 },
  { pattern: /\bdolby atmos\b/i, value: "Dolby Atmos", weight: 2.0 },
  { pattern: /\bstem separat\w*\b/i, value: "stem separation", weight: 2.0 },
  { pattern: /\bai mast\w*\b/i, value: "AI mastering", weight: 2.0 },
  { pattern: /\bai mix\w*\b/i, value: "AI mixing", weight: 2.0 },
  { pattern: /\blo[.\s-]?fi\b/i, value: "lo-fi aesthetic", weight: 2.0 },
  { pattern: /\blossless\b/i, value: "lossless audio", weight: 1.5 },
  { pattern: /\bcinematic\b/i, value: "cinematic", weight: 1.5 },
  { pattern: /\bminimalist\b/i, value: "minimalist", weight: 1.5 },
  { pattern: /\bvintage\b/i, value: "vintage", weight: 1.0 },
  { pattern: /\bretro\b/i, value: "retro", weight: 1.0 },
  { pattern: /\blayered\b/i, value: "layered", weight: 1.0 },
  { pattern: /\braw production\b/i, value: "raw production", weight: 2.0 },
  {
    pattern: /\bhigh[.\s-]?fidelity\b|\bhi[.\s-]?fi\b/i,
    value: "hi-fi",
    weight: 1.5,
  },
];

interface PlatformEntry {
  platform: string;
  detect: RegExp;
  trendSignals: Array<{ pattern: RegExp; trend: string }>;
}

const PLATFORM_ENTRIES: PlatformEntry[] = [
  {
    platform: "TikTok",
    detect: /\btiktok\b/i,
    trendSignals: [
      { pattern: /\bchallenge\b/i, trend: "challenge/trend format" },
      { pattern: /\bfyp\b/i, trend: "FYP algorithm push" },
      { pattern: /\bviral sound\b/i, trend: "viral sound placement" },
      { pattern: /\bshort[.\s-]?form\b/i, trend: "short-form hook priority" },
      { pattern: /\bcreator\b/i, trend: "creator-first content" },
      { pattern: /\balgorithm\b/i, trend: "algorithm change" },
    ],
  },
  {
    platform: "Instagram",
    detect: /\binstagram\b|\breels?\b/i,
    trendSignals: [
      { pattern: /\baesthetic\b/i, trend: "aesthetic-driven Reels" },
      { pattern: /\breels?\b/i, trend: "Reels reach boost" },
      { pattern: /\bengagement\b/i, trend: "engagement optimisation" },
      { pattern: /\balgorithm\b/i, trend: "algorithm change" },
      { pattern: /\bcollaborat\w*\b/i, trend: "collab posts rewarded" },
    ],
  },
  {
    platform: "YouTube",
    detect: /\byoutube\b|\bshorts?\b/i,
    trendSignals: [
      { pattern: /\bshorts?\b/i, trend: "Shorts discovery push" },
      { pattern: /\bmonetiz\w*\b/i, trend: "monetisation update" },
      { pattern: /\bsuggested\b/i, trend: "suggested feed change" },
      { pattern: /\balgorithm\b/i, trend: "algorithm change" },
    ],
  },
  {
    platform: "Spotify",
    detect: /\bspotify\b/i,
    trendSignals: [
      { pattern: /\bplaylist\b/i, trend: "editorial playlist activity" },
      { pattern: /\balgorithm\b/i, trend: "Discovery algorithm signal" },
      { pattern: /\bpayout\b|\broyalt\w*\b/i, trend: "royalty/payout change" },
      { pattern: /\bdiscover\w*\b/i, trend: "Discovery algorithm signal" },
      { pattern: /\bcanvas\b/i, trend: "Canvas visual format" },
    ],
  },
  {
    platform: "Apple Music",
    detect: /\bapple music\b/i,
    trendSignals: [
      {
        pattern: /\batmos\b|\bspatial\b/i,
        trend: "Spatial Audio / Atmos push",
      },
      { pattern: /\beditorial\b/i, trend: "editorial playlist activity" },
      { pattern: /\bradio\b/i, trend: "radio / Beats 1 feature" },
    ],
  },
];

const HOOK_KW: Array<{ pattern: RegExp; hook: string }> = [
  { pattern: /\bstorytell\w*\b/i, hook: "storytelling narrative" },
  { pattern: /\bconfessional\b/i, hook: "confessional / personal" },
  { pattern: /\bchallenge\b/i, hook: "trend / challenge format" },
  { pattern: /\bbehind[.\s-]?the[.\s-]?scenes\b/i, hook: "behind-the-scenes" },
  {
    pattern: /\bcreative process\b|\bprocess\b/i,
    hook: "creative process reveal",
  },
  { pattern: /\bauthentic\w*\b/i, hook: "authenticity / rawness" },
  { pattern: /\bvulnerab\w*\b/i, hook: "vulnerability / emotional honesty" },
  { pattern: /\brelatable\b/i, hook: "relatable moment" },
  { pattern: /\bcollaborat\w*\b/i, hook: "collaboration reveal" },
  { pattern: /\bfirst[.\s]?listen\b/i, hook: "first-listen reaction" },
  { pattern: /\bcomeback\b/i, hook: "comeback / return story" },
  { pattern: /\bunderdog\b/i, hook: "underdog narrative" },
];

const LYRIC_KW: Array<{ pattern: RegExp; theme: string }> = [
  { pattern: /\bauthenticit\w*\b/i, theme: "authenticity" },
  {
    pattern: /\bstruggle\b|\bhard[.\s]?time\b/i,
    theme: "struggle and resilience",
  },
  {
    pattern: /\bsuccess\b|\bmade[.\s]?it\b/i,
    theme: "success and achievement",
  },
  { pattern: /\blove\b|\brelationship\b/i, theme: "love and relationships" },
  {
    pattern: /\bmental health\b|\banxiet\w*\b|\bdepression\b/i,
    theme: "mental health",
  },
  { pattern: /\bidentit\w*\b|\bself[.\s-]?discover\w*\b/i, theme: "identity" },
  { pattern: /\bcommunity\b|\bbelonging\b/i, theme: "community" },
  {
    pattern: /\bfreedom\b|\bindepend\w*\b/i,
    theme: "freedom and independence",
  },
  { pattern: /\bhustle\b|\bgrind\b/i, theme: "hustle culture" },
  { pattern: /\bnostalg\w*\b/i, theme: "nostalgia" },
  { pattern: /\bempow\w*\b/i, theme: "empowerment" },
  { pattern: /\bcelebrat\w*\b|\bparty\b/i, theme: "celebration and joy" },
];

// ─── Service ───────────────────────────────────────────────────────────────────

const CACHE_TTL_MS = 30 * 60 * 1000; // 30 minutes
const MAX_ITEMS = 5;

class MusicIndustryContextFilterService {
  private cache: { ctx: MusicIndustryContext; builtAt: number } | null = null;

  // ── Async API (for services that can await) ────────────────────────────────

  /**
   * Returns a fully formatted MusicIndustryContext for the given generation mode.
   * The `contextString` field is ready to append to `extraContext` in MaxCore calls.
   * Never throws — returns an empty zero-confidence context on any error.
   */
  async getContextForMode(mode: GenerationMode): Promise<MusicIndustryContext> {
    try {
      const base = await this?.getOrBuild();
      return this?.applyMode(base, mode);
    } catch (err) {
      logger?.warn(
        "[IndustryFilter] Context unavailable — returning empty context:",
        (err as Error).message,
      );
      return this?.empty();
    }
  }

  // ── Sync API (for services that cannot await, reads warm cache only) ───────

  /** Returns the top trending genre from the last cached fetch, or undefined. */
  getSuggestedGenreSync(): string | undefined {
    return this?.cache?.ctx?.generationHints.suggestedGenre;
  }

  /** Returns the top trending mood from the last cached fetch, or undefined. */
  getSuggestedMoodSync(): string | undefined {
    return this?.cache?.ctx?.generationHints.suggestedMood;
  }

  /** Returns tempo bias from the last cached fetch ('neutral' when cold). */
  getTempoBiasSync(): "up" | "down" | "neutral" {
    return this?.cache?.ctx?.generationHints.tempoBias ?? "neutral";
  }

  /** Returns top production style keywords from the last cached fetch. */
  getProductionKeywordsSync(): string[] {
    return this?.cache?.ctx?.generationHints.productionKeywords ?? [];
  }

  /** Returns all platform trend signals from the last cached fetch. */
  getPlatformSignalsSync(): PlatformTrendSignal[] {
    return this?.cache?.ctx?.platformSignals ?? [];
  }

  /** Returns current confidence level (0 when cold). */
  getConfidenceSync(): number {
    return this?.cache?.ctx?.confidence ?? 0;
  }

  clearCache(): void {
    this.cache = null;
  }

  // ── Internal ───────────────────────────────────────────────────────────────

  private async getOrBuild(): Promise<MusicIndustryContext> {
    if (this?.cache && Date?.now() - this?.cache.builtAt < CACHE_TTL_MS) {
      return this?.cache.ctx;
    }
    const signals = await industryMonitor?.fetchLiveChanges();
    const ctx = this?.build(signals);
    this.cache = { ctx, builtAt: Date.now() };
    logger?.info(
      `[IndustryFilter] Built context from ${signals?.length} signals — ` +
        `confidence=${ctx?.confidence.toFixed(2)} ` +
        `genres=[${ctx?.trendingGenres.slice(0, 3).join(",")}] ` +
        `moods=[${ctx?.trendingMoods.slice(0, 3).join(",")}]`,
    );
    return ctx;
  }

  private build(signals: LiveIndustryChange[]): MusicIndustryContext {
    const genreScores = new Map<string, number>();
    const moodScores = new Map<string, number>();
    const prodScores = new Map<string, number>();
    const hookSet = new Set<string>();
    const themeSet = new Set<string>();
    const platMap = new Map<string, Set<string>>();

    for (const sig of signals) {
      const text = `${sig?.title} ${sig?.description}`;
      const age = this?.recencyFactor(sig?.detectedAt);
      const boost =
        sig?.urgency === "critical"
          ? 1.4
          : sig?.urgency === "high"
            ? 1.2
            : sig?.urgency === "medium"
              ? 1.0
              : 0.8;

      for (const kw of GENRE_KW) {
        if (kw?.pattern.test(text)) {
          genreScores?.set(
            kw?.value,
            (genreScores?.get(kw?.value) ?? 0) + kw?.weight * age * boost,
          );
        }
      }
      for (const kw of MOOD_KW) {
        if (kw?.pattern.test(text)) {
          moodScores?.set(
            kw?.value,
            (moodScores?.get(kw?.value) ?? 0) + kw?.weight * age * boost,
          );
        }
      }
      for (const kw of PRODUCTION_KW) {
        if (kw?.pattern.test(text)) {
          prodScores?.set(
            kw?.value,
            (prodScores?.get(kw?.value) ?? 0) + kw?.weight * age * boost,
          );
        }
      }
      for (const kw of HOOK_KW) {
        if (kw?.pattern.test(text)) hookSet?.add(kw?.hook);
      }
      for (const kw of LYRIC_KW) {
        if (kw?.pattern.test(text)) themeSet?.add(kw?.theme);
      }
      for (const pe of PLATFORM_ENTRIES) {
        if (pe?.detect.test(text)) {
          const set = platMap?.get(pe?.platform) ?? new Set<string>();
          for (const ts of pe?.trendSignals) {
            if (ts?.pattern.test(text)) set?.add(ts?.trend);
          }
          if (set?.size) platMap?.set(pe?.platform, set);
        }
      }
    }

    const trendingGenres = this?.topN(genreScores, MAX_ITEMS);
    const trendingMoods = this?.topN(moodScores, MAX_ITEMS);
    const productionStyles = this?.topN(prodScores, MAX_ITEMS);
    const viralHookPatterns = [...hookSet].slice(0, 4);
    const lyricThemes = [...themeSet].slice(0, 4);

    const platformSignals: PlatformTrendSignal[] = [];
    for (const [platform, trends] of platMap) {
      let i = 0;
      for (const trend of trends) {
        const strength = i === 0 ? "strong" : i === 1 ? "moderate" : "emerging";
        platformSignals?.push({
          platform,
          trend,
          strength: strength as PlatformTrendSignal["strength"],
        });
        i++;
      }
    }

    const confidence = Math?.min(1, signals?.length / 40);
    const hints = this?.buildHints(
      trendingGenres,
      trendingMoods,
      productionStyles,
      platformSignals,
      viralHookPatterns,
    );

    return {
      trendingGenres,
      trendingMoods,
      productionStyles,
      platformSignals,
      viralHookPatterns,
      lyricThemes,
      generationHints: hints,
      contextString: "",
      signalCount: signals.length,
      confidence,
      freshness: new Date(),
    };
  }

  private applyMode(
    ctx: MusicIndustryContext,
    mode: GenerationMode,
  ): MusicIndustryContext {
    if (ctx?.confidence < 0.05 || ctx?.signalCount === 0) {
      return { ...ctx, contextString: "" };
    }

    const parts: string[] = [];

    switch (mode) {
      case "social":
      case "content":
      case "advertising":
        parts?.push("[Music Industry Context]");
        if (ctx?.trendingGenres.length)
          parts?.push(`Trending: ${ctx?.trendingGenres.slice(0, 3).join(", ")}`);
        if (ctx?.platformSignals.length) {
          const top = ctx?.platformSignals
            .slice(0, 3)
            .map((s) => `${s?.platform}: ${s?.trend}`)
            .join("; ");
          parts?.push(`Platform: ${top}`);
        }
        if (ctx?.viralHookPatterns.length)
          parts?.push(`Hooks: ${ctx?.viralHookPatterns.slice(0, 2).join(", ")}`);
        if (ctx?.lyricThemes.length)
          parts?.push(`Themes: ${ctx?.lyricThemes.slice(0, 2).join(", ")}`);
        break;

      case "melody":
      case "music":
        parts?.push("[Industry Music Context]");
        if (ctx?.trendingGenres.length)
          parts?.push(
            `Trending genres: ${ctx?.trendingGenres.slice(0, 3).join(", ")}`,
          );
        if (ctx?.trendingMoods.length)
          parts?.push(
            `Resonant moods: ${ctx?.trendingMoods.slice(0, 3).join(", ")}`,
          );
        if (ctx?.productionStyles.length)
          parts?.push(
            `Production: ${ctx?.productionStyles.slice(0, 2).join(", ")}`,
          );
        if (ctx?.generationHints.tempoBias !== "neutral")
          parts?.push(`Tempo bias: ${ctx?.generationHints.tempoBias}`);
        break;

      case "songwriting":
        parts?.push("[Songwriting Industry Context]");
        if (ctx?.lyricThemes.length)
          parts?.push(
            `Resonant themes: ${ctx?.lyricThemes.slice(0, 3).join(", ")}`,
          );
        if (ctx?.trendingGenres.length)
          parts?.push(
            `Trending genres: ${ctx?.trendingGenres.slice(0, 3).join(", ")}`,
          );
        if (ctx?.viralHookPatterns.length)
          parts?.push(
            `Hook patterns: ${ctx?.viralHookPatterns.slice(0, 2).join(", ")}`,
          );
        if (ctx?.trendingMoods.length)
          parts?.push(
            `Audience mood: ${ctx?.trendingMoods.slice(0, 2).join(", ")}`,
          );
        break;
    }

    return { ...ctx, contextString: parts.join(" | ") };
  }

  private buildHints(
    genres: string[],
    moods: string[],
    production: string[],
    platformSignals: PlatformTrendSignal[],
    hooks: string[],
  ): GenerationHints {
    const upMoods = new Set([
      "energetic",
      "aggressive",
      "euphoric",
      "driven",
      "intense",
      "motivational",
      "empowering",
      "playful",
    ]);
    const downMoods = new Set([
      "melancholic",
      "calm",
      "relaxed",
      "chill",
      "nostalgic",
      "vulnerable",
      "raw",
      "soulful",
    ]);
    let up = 0,
      down = 0;
    for (const m of moods) {
      if (upMoods?.has(m)) up++;
      if (downMoods?.has(m)) down++;
    }
    const tempoBias: "up" | "down" | "neutral" =
      up > down ? "up" : down > up ? "down" : "neutral";

    const hashtagContext = platformSignals
      .filter((s) => s?.strength === "strong")
      .map((s) => `#${s?.platform.toLowerCase().replace(/\s+/g, "")}`)
      .join(" ");

    return {
      suggestedGenre: genres[0],
      suggestedMood: moods[0],
      tempoBias,
      productionKeywords: production.slice(0, 3),
      contentAngles: hooks.slice(0, 3),
      hashtagContext,
    };
  }

  private topN(scores: Map<string, number>, n: number): string[] {
    return [...scores?.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, n)
      .map(([k]) => k);
  }

  /**
   * Recency decay:
   *   ≤ 24 h  → 1.0   (freshest signal, full weight)
   *   ≤ 72 h  → 0.8
   *   ≤ 168 h → 0.5
   *   older   → 0.2
   */
  private recencyFactor(date: Date): number {
    const h = (Date?.now() - date?.getTime()) / 3_600_000;
    if (h <= 24) return 1.0;
    if (h <= 72) return 0.8;
    if (h <= 168) return 0.5;
    return 0.2;
  }

  private empty(): MusicIndustryContext {
    return {
      trendingGenres: [],
      trendingMoods: [],
      productionStyles: [],
      platformSignals: [],
      viralHookPatterns: [],
      lyricThemes: [],
      generationHints: {
        tempoBias: "neutral",
        productionKeywords: [],
        contentAngles: [],
        hashtagContext: "",
      },
      contextString: "",
      signalCount: 0,
      confidence: 0,
      freshness: new Date(),
    };
  }
}

export const musicIndustryContextFilter =
  new MusicIndustryContextFilterService();
