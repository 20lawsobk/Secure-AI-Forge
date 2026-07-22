/**
 * CONTENT GENERATION AWARENESS SERVICE LAYER
 *
 * Merges and transforms IndustryMonitorService + MusicIndustryContextFilter
 * into a unified real-time awareness layer for ALL content generation pipelines.
 *
 * Signal flow:
 *   Live RSS Feeds (10 music/creator sources)
 *   + Tavily Web Search (real-time news intelligence)
 *   + Exa Semantic Search (deep trend discovery)
 *       └──► ContentGenerationAwarenessService
 *               ├──► Social post generation    → platform hooks, hashtags, CTAs
 *               ├──► Ad copy generation        → campaign angles, offers, urgency
 *               ├──► Video script generation   → narrative arcs, hooks, pacing
 *               ├──► Email campaign generation → subject lines, preview text, themes
 *               ├──► Press release generation  → newsworthiness, quotes, angles
 *               ├──► Blog post generation      → SEO angles, headlines, structure
 *               ├──► Melody generation         → genre, mood, tempo, key hints
 *               ├──► Music generation          → production style, arrangement
 *               └──► Songwriting assistance    → lyric themes, hook patterns
 *
 * All generation services remain backward-compatible.
 * Context enrichment is additive and never blocks generation.
 * Cache TTL: 30 minutes per build cycle.
 */

import { XMLParser } from "fast-xml-parser";
import crypto from "crypto";

// ─── Types ────────────────────────────────────────────────────────────────────

export type ContentGenerationMode =
  | "social"
  | "ad_copy"
  | "video_script"
  | "email"
  | "press_release"
  | "blog"
  | "melody"
  | "music"
  | "songwriting"
  | "content"
  | "advertising"
  | "distribution";

export type SignalUrgency = "critical" | "high" | "medium" | "low";
export type SignalSource =
  | "competitor"
  | "streaming_platform"
  | "social_media"
  | "security"
  | "regulation"
  | "technology"
  | "content_trend"
  | "audience_behavior"
  | "platform_algorithm";

export interface LiveIndustrySignal {
  id: string;
  source: SignalSource;
  category:
    | "feature"
    | "api_change"
    | "standard"
    | "optimization"
    | "security_patch"
    | "ux_pattern"
    | "trend"
    | "algorithm_change"
    | "audience_shift";
  title: string;
  description: string;
  detectedAt: Date;
  urgency: SignalUrgency;
  affectedModules: string[];
  competitiveImpact: number;
  implementationComplexity:
    | "trivial"
    | "simple"
    | "moderate"
    | "complex"
    | "major";
  estimatedImplementationHours: number;
  sourceUrl?: string;
  feedSource?: string;
}

export interface PlatformTrendSignal {
  platform: string;
  trend: string;
  strength: "strong" | "moderate" | "emerging";
  contentFormat?: string;
}

export interface AudiencePsychologySignal {
  trigger: string;
  pattern: string;
  strength: number;
}

export interface ContentFormatTrend {
  format: string;
  platform: string;
  momentum: "rising" | "peak" | "declining";
}

/** Hints that feed into ANY content generator call */
export interface ContentGenerationHints {
  suggestedGenre?: string;
  suggestedMood?: string;
  tempoBias: "up" | "down" | "neutral";
  productionKeywords: string[];
  contentAngles: string[];
  hashtagContext: string;
  ctaPatterns: string[];
  emotionalTriggers: string[];
  contentFormats: ContentFormatTrend[];
  audiencePsychology: AudiencePsychologySignal[];
  trendingTopics: string[];
  platformAlgorithmNotes: string[];
}

/** Full awareness context returned to any generation service */
export interface ContentAwarenessContext {
  trendingGenres: string[];
  trendingMoods: string[];
  productionStyles: string[];
  platformSignals: PlatformTrendSignal[];
  viralHookPatterns: string[];
  lyricThemes: string[];
  contentAngles: string[];
  ctaPatterns: string[];
  emotionalTriggers: string[];
  contentFormats: ContentFormatTrend[];
  audiencePsychology: AudiencePsychologySignal[];
  trendingTopics: string[];
  platformAlgorithmNotes: string[];
  generationHints: ContentGenerationHints;
  /** Ready-to-inject string for MaxCore extraContext */
  contextString: string;
  signalCount: number;
  /** 0–1 confidence score — scales with signal count and freshness */
  confidence: number;
  freshness: Date;
}

// ─── RSS Feed Sources ─────────────────────────────────────────────────────────

const RSS_FEEDS: Array<{ url: string; name: string; followRedirect: boolean }> =
  [
    {
      url: "https://www.musicbusinessworldwide.com/feed/",
      name: "Music Business Worldwide",
      followRedirect: false,
    },
    {
      url: "https://www.digitalmusicnews.com/feed/",
      name: "Digital Music News",
      followRedirect: false,
    },
    {
      url: "https://musically.com/feed/",
      name: "Music Ally",
      followRedirect: false,
    },
    {
      url: "https://www.musicradar.com/rss",
      name: "MusicRadar",
      followRedirect: true,
    },
    {
      url: "https://newsroom.spotify.com/rss/",
      name: "Spotify Newsroom",
      followRedirect: true,
    },
    {
      url: "https://about.fb.com/news/tag/creators/feed/",
      name: "Meta Creators",
      followRedirect: true,
    },
    {
      url: "https://www.hypebot.com/hypebot/atom.xml",
      name: "Hypebot",
      followRedirect: true,
    },
    {
      url: "https://blog.landr.com/feed/",
      name: "Landr Blog",
      followRedirect: true,
    },
    {
      url: "https://blog.distrokid.com/feed",
      name: "DistroKid Blog",
      followRedirect: true,
    },
    {
      url: "https://www.tunecore.com/blog/feed/",
      name: "TuneCore Blog",
      followRedirect: true,
    },
  ];

// ─── Search Query Suites ───────────────────────────────────────────────────────

const SEARCH_QUERIES = {
  music: [
    "streaming platform API changes music 2024 2025",
    "Spotify Apple Music algorithm change artists",
    "TikTok Instagram music creator policy update",
    "AI music technology announcement 2025",
    "independent artist music career management platform feature launch 2025",
    "indie artist viral marketing TikTok Instagram strategy winning 2025",
    "music industry AI tools artists competitive advantage 2025",
  ],
  content: [
    "social media content trends creators 2025",
    "Instagram TikTok algorithm update content reach 2025",
    "YouTube Shorts monetization update creators 2025",
    "viral content format trends short-form video 2025",
    "content marketing best practices engagement rate 2025",
    "creator economy platform update brand deals 2025",
  ],
  advertising: [
    "digital advertising trends ROI 2025",
    "Meta Ads platform update targeting 2025",
    "TikTok Ads creator monetization update 2025",
    "email marketing open rate trends 2025",
    "influencer marketing new platform tools 2025",
  ],
};

const TAVILY_API = "https://api.tavily.com/search";
const EXA_API = "https://api.exa.ai/search";
/** Base TTL floor (10 min) when the Python corpus is fresh/empty and needs live signals most. */
const CACHE_TTL_MIN_MS = 10 * 60 * 1000;
/** TTL ceiling (45 min) when the Python corpus is self-sufficient and external signals matter less. */
const CACHE_TTL_MAX_MS = 45 * 60 * 1000;
const FETCH_TIMEOUT_MS = 10_000;

// ─── Python corpus maturity probe ────────────────────────────────────────────
// We query the Python AI server's self-sufficiency status and use it to scale
// the awareness cache TTL: when the model's own phrase corpus is small
// (buffer_weight near 1.0 → still learning from external signals), we use a
// shorter TTL so the TS layer refreshes more often and feeds richer live data.
// As the corpus matures (buffer_weight → 0), the Python layer leans on its own
// corpus and we relax the TTL to reduce unnecessary RSS/search fetches.

interface CorpusMaturity {
  bufferWeight: number;  // 0–1; 1 = fully external, 0 = self-sufficient
  ownCorpus: number;
  retireThreshold: number;
  retired: boolean;
  fetchedAt: number;
}

let _corpusMaturityCache: CorpusMaturity | null = null;
const CORPUS_PROBE_TTL_MS = 5 * 60 * 1000; // re-probe Python every 5 min

async function fetchCorpusMaturity(): Promise<CorpusMaturity> {
  const now = Date.now();
  if (_corpusMaturityCache && now - _corpusMaturityCache.fetchedAt < CORPUS_PROBE_TTL_MS) {
    return _corpusMaturityCache;
  }
  try {
    const port = process.env.MODEL_API_PORT ?? "9878";
    const apiKey =
      process.env.AI_SERVER_KEY ||
      process.env.AI_TRAINING_KEY_PROD ||
      process.env.ADMIN_KEY ||
      "";
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3_000);
    try {
      const res = await fetch(`http://127.0.0.1:${port}/api/awareness/quality/status`, {
        signal: controller.signal,
        headers: apiKey ? { "X-Api-Key": apiKey } : {},
      });
      if (res.ok) {
        const data = (await res.json()) as {
          buffer_weight?: number;
          own_corpus?: number;
          retire_threshold?: number;
          retired?: boolean;
        };
        const maturity: CorpusMaturity = {
          bufferWeight: typeof data.buffer_weight === "number" ? data.buffer_weight : 1.0,
          ownCorpus: typeof data.own_corpus === "number" ? data.own_corpus : 0,
          retireThreshold: typeof data.retire_threshold === "number" ? data.retire_threshold : 500,
          retired: !!data.retired,
          fetchedAt: now,
        };
        _corpusMaturityCache = maturity;
        return maturity;
      }
    } finally {
      clearTimeout(timeout);
    }
  } catch {
    // Never block — fall back to "model is immature" so we refresh aggressively
  }
  const fallback: CorpusMaturity = {
    bufferWeight: 1.0,
    ownCorpus: 0,
    retireThreshold: 500,
    retired: false,
    fetchedAt: now,
  };
  _corpusMaturityCache = fallback;
  return fallback;
}

/**
 * Dynamic cache TTL:
 *   buffer_weight=1.0 (fresh/empty corpus) → 10 min   (need live external data)
 *   buffer_weight=0.0 (self-sufficient)    → 45 min   (model carries its own)
 */
function dynamicCacheTtl(bufferWeight: number): number {
  const w = Math.max(0, Math.min(1, bufferWeight));
  return CACHE_TTL_MIN_MS + (1 - w) * (CACHE_TTL_MAX_MS - CACHE_TTL_MIN_MS);
}

// ─── Keyword Tables ───────────────────────────────────────────────────────────

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
  { pattern: /\bhi[.\s-]?fi\b/i, value: "hi-fi", weight: 1.5 },
];

// ── Content generation-specific keyword tables (NEW) ──────────────────────────

const CTA_KW: Array<{ pattern: RegExp; cta: string; weight: number }> = [
  { pattern: /\bstream now\b/i, cta: "Stream Now", weight: 2.0 },
  { pattern: /\bpre[.\s-]?save\b/i, cta: "Pre-Save", weight: 2.0 },
  { pattern: /\blink in bio\b/i, cta: "Link in Bio", weight: 1.5 },
  { pattern: /\bexclusive\b/i, cta: "Exclusive Access", weight: 1.5 },
  { pattern: /\blimited time\b/i, cta: "Limited Time Offer", weight: 2.0 },
  { pattern: /\bfree\b/i, cta: "Free Download", weight: 1.5 },
  { pattern: /\bdrops?\b/i, cta: "New Drop Alert", weight: 2.0 },
  { pattern: /\bmerch\b/i, cta: "Shop Now", weight: 1.5 },
  { pattern: /\bcollaborat\w*\b/i, cta: "Collab Announcement", weight: 1.5 },
  {
    pattern: /\bbehind[.\s-]?the[.\s-]?scenes\b/i,
    cta: "See Behind the Scenes",
    weight: 1.5,
  },
  { pattern: /\btickets?\b/i, cta: "Get Tickets", weight: 2.0 },
  { pattern: /\bfollow\b/i, cta: "Follow for Updates", weight: 1.0 },
  { pattern: /\bchallenge\b/i, cta: "Join the Challenge", weight: 2.0 },
  {
    pattern: /\bdonate\b|\bsupport\b/i,
    cta: "Support the Artist",
    weight: 1.5,
  },
];

const EMOTIONAL_TRIGGER_KW: Array<{
  pattern: RegExp;
  trigger: string;
  weight: number;
}> = [
  { pattern: /\bnostalg\w*\b/i, trigger: "nostalgia", weight: 2.0 },
  { pattern: /\bfomo\b|\bmissing out\b/i, trigger: "FOMO", weight: 2.5 },
  { pattern: /\binspir\w*\b/i, trigger: "inspiration", weight: 1.5 },
  {
    pattern: /\bstrugg\w*\b|\bovercom\w*\b/i,
    trigger: "triumph over struggle",
    weight: 2.0,
  },
  { pattern: /\bexclusive\b|\bvip\b/i, trigger: "exclusivity", weight: 2.0 },
  {
    pattern: /\bcommunity\b|\bbelonging\b/i,
    trigger: "community belonging",
    weight: 1.5,
  },
  {
    pattern: /\bauthentic\w*\b|\breal\b/i,
    trigger: "authenticity",
    weight: 1.5,
  },
  {
    pattern: /\bsurprise\b|\bunexpected\b/i,
    trigger: "surprise and delight",
    weight: 2.0,
  },
  {
    pattern: /\bpride\b|\bachiev\w*\b/i,
    trigger: "pride and achievement",
    weight: 1.5,
  },
  {
    pattern: /\brebellion\b|\banti[.\s-]?establishment\b/i,
    trigger: "rebellion",
    weight: 2.0,
  },
  {
    pattern: /\blove\b|\bromatic\w*\b/i,
    trigger: "love and connection",
    weight: 1.0,
  },
  {
    pattern: /\burgen\w*\b|\bnow\b|\btoday\b/i,
    trigger: "urgency",
    weight: 1.5,
  },
  {
    pattern: /\bsocial proof\b|\bever(?:yone|ybody)\b/i,
    trigger: "social proof",
    weight: 2.0,
  },
];

const CONTENT_FORMAT_KW: Array<{
  pattern: RegExp;
  format: string;
  platform: string;
  weight: number;
}> = [
  {
    pattern: /\bshorts?\b/i,
    format: "Shorts",
    platform: "YouTube",
    weight: 2.0,
  },
  {
    pattern: /\breels?\b/i,
    format: "Reels",
    platform: "Instagram",
    weight: 2.0,
  },
  {
    pattern: /\btiktok\b/i,
    format: "TikTok Video",
    platform: "TikTok",
    weight: 2.0,
  },
  {
    pattern: /\bcarousel\b/i,
    format: "Carousel",
    platform: "Instagram",
    weight: 1.5,
  },
  {
    pattern: /\bstory\b|\bstories\b/i,
    format: "Stories",
    platform: "Instagram",
    weight: 1.5,
  },
  {
    pattern: /\bthread\b/i,
    format: "Thread",
    platform: "X/Twitter",
    weight: 1.5,
  },
  {
    pattern: /\bpodcast\b/i,
    format: "Podcast",
    platform: "Spotify",
    weight: 1.5,
  },
  {
    pattern: /\blive\b/i,
    format: "Live Stream",
    platform: "Multiple",
    weight: 2.0,
  },
  {
    pattern: /\bshort[.\s-]?form\b/i,
    format: "Short-Form Video",
    platform: "TikTok/Reels/Shorts",
    weight: 2.0,
  },
  {
    pattern: /\bnewsletter\b/i,
    format: "Newsletter",
    platform: "Email",
    weight: 1.5,
  },
  {
    pattern: /\bblog\b|\barticle\b/i,
    format: "Blog Article",
    platform: "Web",
    weight: 1.0,
  },
  {
    pattern: /\bpress release\b/i,
    format: "Press Release",
    platform: "Media",
    weight: 1.5,
  },
];

const PLATFORM_ALGORITHM_KW: Array<{
  pattern: RegExp;
  note: string;
  weight: number;
}> = [
  {
    pattern: /\bfyp\b|\bfor you\b/i,
    note: "TikTok FYP algorithm shift detected",
    weight: 2.5,
  },
  {
    pattern: /\binstagram algorithm\b/i,
    note: "Instagram algorithm change detected",
    weight: 2.5,
  },
  {
    pattern: /\byoutube algorithm\b/i,
    note: "YouTube recommendation algorithm update",
    weight: 2.5,
  },
  {
    pattern: /\bspotify algorithm\b|\bdiscovery\b/i,
    note: "Spotify discovery algorithm active",
    weight: 2.0,
  },
  {
    pattern: /\borganic reach\b/i,
    note: "Organic reach fluctuation detected",
    weight: 2.0,
  },
  {
    pattern: /\bshadowban\b/i,
    note: "Shadowban / reduced visibility risk noted",
    weight: 2.5,
  },
  {
    pattern: /\bengagement rate\b/i,
    note: "Engagement rate weighting shift",
    weight: 1.5,
  },
  {
    pattern: /\bwatch time\b/i,
    note: "Watch time signal boosted",
    weight: 2.0,
  },
  {
    pattern: /\bsave rate\b|\bsaves\b/i,
    note: "Save rate being weighted by algorithm",
    weight: 2.0,
  },
  {
    pattern: /\bshares?\b|\breposts?\b/i,
    note: "Share/repost signal rewarded",
    weight: 1.5,
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
  { pattern: /\bday in the life\b/i, hook: "day-in-the-life vlog" },
  { pattern: /\breaction\b/i, hook: "reaction / response" },
  { pattern: /\bcontroversial\b/i, hook: "hot take / controversy" },
  { pattern: /\btutorial\b|\bhow[.\s-]?to\b/i, hook: "tutorial / how-to" },
  { pattern: /\bq&a\b|\bquestions\b/i, hook: "Q&A / fan questions" },
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
  { pattern: /\bbetrayal\b|\bheartbreak\b/i, theme: "heartbreak and betrayal" },
  { pattern: /\bfaith\b|\bspiritual\w*\b/i, theme: "faith and spirituality" },
  { pattern: /\brebellion\b|\bdefiant\w*\b/i, theme: "rebellion and defiance" },
];

const TRENDING_TOPIC_KW: Array<{
  pattern: RegExp;
  topic: string;
  weight: number;
}> = [
  {
    pattern: /\bai[.\s]?generated\b/i,
    topic: "AI-generated content",
    weight: 2.0,
  },
  {
    pattern: /\bcollaborat\w*\b/i,
    topic: "artist collaborations",
    weight: 1.5,
  },
  {
    pattern: /\bweb3\b|\bnft\b|\bblockchain\b/i,
    topic: "Web3 / NFTs",
    weight: 2.0,
  },
  {
    pattern: /\bexclusive drop\b|\bdrop\b/i,
    topic: "exclusive drops",
    weight: 2.0,
  },
  { pattern: /\bsocial commerce\b/i, topic: "social commerce", weight: 1.5 },
  {
    pattern: /\bcommunity[.\s-]?led\b/i,
    topic: "community-led growth",
    weight: 1.5,
  },
  {
    pattern: /\bunfiltered\b|\braw\b/i,
    topic: "unfiltered / lo-fi content",
    weight: 2.0,
  },
  {
    pattern: /\bmicro[.\s-]?content\b/i,
    topic: "micro-content strategy",
    weight: 1.5,
  },
  {
    pattern: /\buser[.\s-]?generated\b|\bugc\b/i,
    topic: "user-generated content",
    weight: 2.0,
  },
  {
    pattern: /\bwellness\b|\bself[.\s-]?care\b/i,
    topic: "wellness and self-care",
    weight: 1.5,
  },
  {
    pattern: /\bsustainab\w*\b/i,
    topic: "sustainability messaging",
    weight: 1.5,
  },
  {
    pattern: /\bnostalgia\b|\b(?:90s|80s|70s|2000s)\b/i,
    topic: "nostalgia marketing",
    weight: 2.0,
  },
];

interface PlatformEntry {
  platform: string;
  detect: RegExp;
  trendSignals: Array<{
    pattern: RegExp;
    trend: string;
    contentFormat?: string;
  }>;
}

const PLATFORM_ENTRIES: PlatformEntry[] = [
  {
    platform: "TikTok",
    detect: /\btiktok\b/i,
    trendSignals: [
      {
        pattern: /\bchallenge\b/i,
        trend: "challenge/trend format",
        contentFormat: "Duet/Stitch",
      },
      {
        pattern: /\bfyp\b/i,
        trend: "FYP algorithm push",
        contentFormat: "Short Video",
      },
      {
        pattern: /\bviral sound\b/i,
        trend: "viral sound placement",
        contentFormat: "Sound-on",
      },
      {
        pattern: /\bshort[.\s-]?form\b/i,
        trend: "short-form hook priority",
        contentFormat: "15-30s",
      },
      {
        pattern: /\bcreator\b/i,
        trend: "creator-first content",
        contentFormat: "Authentic",
      },
      {
        pattern: /\balgorithm\b/i,
        trend: "algorithm change",
        contentFormat: "Trend-riding",
      },
    ],
  },
  {
    platform: "Instagram",
    detect: /\binstagram\b|\breels?\b/i,
    trendSignals: [
      {
        pattern: /\baesthetic\b/i,
        trend: "aesthetic-driven Reels",
        contentFormat: "Reels",
      },
      {
        pattern: /\breels?\b/i,
        trend: "Reels reach boost",
        contentFormat: "Reels",
      },
      {
        pattern: /\bengagement\b/i,
        trend: "engagement optimisation",
        contentFormat: "Carousel",
      },
      {
        pattern: /\balgorithm\b/i,
        trend: "algorithm change",
        contentFormat: "Saves-focused",
      },
      {
        pattern: /\bcollaborat\w*\b/i,
        trend: "collab posts rewarded",
        contentFormat: "Collab Post",
      },
    ],
  },
  {
    platform: "YouTube",
    detect: /\byoutube\b|\bshorts?\b/i,
    trendSignals: [
      {
        pattern: /\bshorts?\b/i,
        trend: "Shorts discovery push",
        contentFormat: "Shorts",
      },
      {
        pattern: /\bmonetiz\w*\b/i,
        trend: "monetisation update",
        contentFormat: "Long-form",
      },
      {
        pattern: /\bsuggested\b/i,
        trend: "suggested feed change",
        contentFormat: "Thumbnail-optimised",
      },
      {
        pattern: /\balgorithm\b/i,
        trend: "algorithm change",
        contentFormat: "Watch-time focus",
      },
    ],
  },
  {
    platform: "Spotify",
    detect: /\bspotify\b/i,
    trendSignals: [
      {
        pattern: /\bplaylist\b/i,
        trend: "editorial playlist activity",
        contentFormat: "Audio",
      },
      {
        pattern: /\balgorithm\b/i,
        trend: "Discovery algorithm signal",
        contentFormat: "Audio",
      },
      {
        pattern: /\bpayout\b|\broyalt\w*\b/i,
        trend: "royalty/payout change",
        contentFormat: "Audio",
      },
      {
        pattern: /\bdiscover\w*\b/i,
        trend: "Discovery algorithm signal",
        contentFormat: "Audio",
      },
      {
        pattern: /\bcanvas\b/i,
        trend: "Canvas visual format",
        contentFormat: "Canvas",
      },
    ],
  },
  {
    platform: "X/Twitter",
    detect: /\btwitter\b|\bx\.com\b/i,
    trendSignals: [
      {
        pattern: /\bthread\b/i,
        trend: "thread format engagement",
        contentFormat: "Thread",
      },
      {
        pattern: /\btrending\b/i,
        trend: "trending topic opportunity",
        contentFormat: "Reply/Quote",
      },
      {
        pattern: /\bblue\b/i,
        trend: "X Premium content boost",
        contentFormat: "Long-form post",
      },
    ],
  },
  {
    platform: "Facebook",
    detect: /\bfacebook\b|\bfb\b|\bmeta\b/i,
    trendSignals: [
      {
        pattern: /\breels?\b/i,
        trend: "Facebook Reels reach push",
        contentFormat: "Reels",
      },
      {
        pattern: /\bgroup\b/i,
        trend: "Groups community engagement",
        contentFormat: "Group Post",
      },
      {
        pattern: /\bevent\b/i,
        trend: "Events / show promotion",
        contentFormat: "Event Post",
      },
      {
        pattern: /\bstory\b|\bstories\b/i,
        trend: "Stories format active",
        contentFormat: "Stories",
      },
      {
        pattern: /\balgorithm\b/i,
        trend: "News Feed algorithm change",
        contentFormat: "Native Video",
      },
      {
        pattern: /\bads?\b|\bboost\b/i,
        trend: "Paid boost / ad reach shift",
        contentFormat: "Boosted Post",
      },
      {
        pattern: /\bwatch\b/i,
        trend: "Facebook Watch engagement",
        contentFormat: "Watch Video",
      },
      {
        pattern: /\bcreator\b/i,
        trend: "Creator monetisation update",
        contentFormat: "Creator Studio",
      },
    ],
  },
  {
    platform: "LinkedIn",
    detect: /\blinkedin\b/i,
    trendSignals: [
      {
        pattern: /\barticle\b|\blog\b/i,
        trend: "Long-form article reach boost",
        contentFormat: "Article",
      },
      {
        pattern: /\bvideo\b/i,
        trend: "Native video prioritised",
        contentFormat: "Native Video",
      },
      {
        pattern: /\bthought leadership\b|\bopinion\b/i,
        trend: "Thought leadership rewarded",
        contentFormat: "Text Post",
      },
      {
        pattern: /\bpoll\b/i,
        trend: "Polls driving engagement",
        contentFormat: "Poll",
      },
      {
        pattern: /\balgorithm\b/i,
        trend: "LinkedIn feed algorithm update",
        contentFormat: "Text + Media",
      },
      {
        pattern: /\bcreator mode\b/i,
        trend: "Creator Mode visibility boost",
        contentFormat: "Newsletter",
      },
      {
        pattern: /\bcollaborat\w*\b/i,
        trend: "Collaborative articles feature",
        contentFormat: "Collab Article",
      },
      {
        pattern: /\bmusic\b|\bartist\b|\bindustry\b/i,
        trend: "Music industry professional content",
        contentFormat: "Industry Post",
      },
    ],
  },
];

// ─── Signal Classifier ────────────────────────────────────────────────────────

function classifySignal(
  title: string,
  description: string,
): {
  source: LiveIndustrySignal["source"];
  category: LiveIndustrySignal["category"];
  urgency: SignalUrgency;
  modules: string[];
  impact: number;
  complexity: LiveIndustrySignal["implementationComplexity"];
  hours: number;
} | null {
  const text = `${title} ${description}`.toLowerCase();

  if (
    /vulnerabilit|security patch|breach|exploit|cve-|zero.?day|ransomware|malware|data.?leak/.test(
      text,
    )
  ) {
    return {
      source: "security",
      category: "security_patch",
      urgency: "critical",
      modules: ["security"],
      impact: 95,
      complexity: "moderate",
      hours: 8,
    };
  }
  if (
    /copyright|dmca|gdpr|ccpa|royalt|compulsory license|mechanical license|eu.?copyright|mma|music modernization/.test(
      text,
    )
  ) {
    return {
      source: "regulation",
      category: "standard",
      urgency: "high",
      modules: ["distribution", "monetization"],
      impact: 80,
      complexity: "moderate",
      hours: 24,
    };
  }
  if (
    /spotify|apple music|amazon music|youtube music|tidal|deezer|soundcloud|pandora|audiomack/.test(
      text,
    )
  ) {
    if (/api|deprecat|endpoint|oauth|sdk|developer/.test(text)) {
      return {
        source: "streaming_platform",
        category: "api_change",
        urgency: "critical",
        modules: ["distribution"],
        impact: 90,
        complexity: "moderate",
        hours: 20,
      };
    }
    if (
      /algorithm|discovery|playlist|recommend|stream count|royalt|payout/.test(
        text,
      )
    ) {
      return {
        source: "streaming_platform",
        category: "optimization",
        urgency: "high",
        modules: ["distribution", "analytics"],
        impact: 85,
        complexity: "simple",
        hours: 6,
      };
    }
    return {
      source: "streaming_platform",
      category: "standard",
      urgency: "medium",
      modules: ["distribution"],
      impact: 55,
      complexity: "simple",
      hours: 6,
    };
  }
  if (
    /tiktok|instagram|reels|shorts|youtube|twitter|x\.com|facebook|threads|creator|influencer/.test(
      text,
    )
  ) {
    if (/api|deprecat|rate.?limit|oauth|token|developer/.test(text)) {
      return {
        source: "social_media",
        category: "api_change",
        urgency: "high",
        modules: ["social"],
        impact: 80,
        complexity: "moderate",
        hours: 16,
      };
    }
    if (
      /algorithm|reach|organic|viral|engagement|views|impressions/.test(text)
    ) {
      return {
        source: "platform_algorithm",
        category: "algorithm_change",
        urgency: "high",
        modules: ["social", "advertising"],
        impact: 85,
        complexity: "simple",
        hours: 8,
      };
    }
    if (/music|sound|audio|song|track|artist|monetiz/.test(text)) {
      return {
        source: "social_media",
        category: "feature",
        urgency: "medium",
        modules: ["social", "distribution"],
        impact: 70,
        complexity: "simple",
        hours: 10,
      };
    }
    if (/content trend|viral|format|challenge|stitch|duet/.test(text)) {
      return {
        source: "content_trend",
        category: "trend",
        urgency: "high",
        modules: ["social", "content"],
        impact: 78,
        complexity: "simple",
        hours: 4,
      };
    }
    if (/facebook|meta/.test(text)) {
      if (/ads?|boost|paid|campaign|targeting/.test(text)) {
        return {
          source: "social_media",
          category: "optimization",
          urgency: "high",
          modules: ["advertising", "social"],
          impact: 82,
          complexity: "simple",
          hours: 8,
        };
      }
      if (/group|community|event/.test(text)) {
        return {
          source: "audience_behavior",
          category: "audience_shift",
          urgency: "medium",
          modules: ["social", "content"],
          impact: 65,
          complexity: "simple",
          hours: 4,
        };
      }
    }
  }
  if (/\blinkedin\b/.test(text)) {
    if (/api|deprecat|rate.?limit|oauth|developer/.test(text)) {
      return {
        source: "social_media",
        category: "api_change",
        urgency: "high",
        modules: ["social"],
        impact: 75,
        complexity: "moderate",
        hours: 14,
      };
    }
    if (
      /algorithm|reach|organic|engagement|impressions|newsletter/.test(text)
    ) {
      return {
        source: "platform_algorithm",
        category: "algorithm_change",
        urgency: "high",
        modules: ["social", "content"],
        impact: 78,
        complexity: "simple",
        hours: 6,
      };
    }
    if (
      /thought leadership|article|creator mode|professional|b2b|industry/.test(
        text,
      )
    ) {
      return {
        source: "content_trend",
        category: "trend",
        urgency: "medium",
        modules: ["social", "content", "advertising"],
        impact: 70,
        complexity: "simple",
        hours: 5,
      };
    }
    if (/music|artist|label|tour|industry/.test(text)) {
      return {
        source: "social_media",
        category: "feature",
        urgency: "low",
        modules: ["social"],
        impact: 60,
        complexity: "trivial",
        hours: 3,
      };
    }
    return {
      source: "social_media",
      category: "standard",
      urgency: "low",
      modules: ["social"],
      impact: 55,
      complexity: "trivial",
      hours: 3,
    };
  }
  if (
    /fl studio|ableton|logic pro|pro tools|studio one|cubase|reaper|garageband|bitwig|reason|bandlab|splice|landr/.test(
      text,
    )
  ) {
    const isAI = /ai |artificial intelligence|machine learning|neural/.test(
      text,
    );
    return {
      source: "competitor",
      category: "feature",
      urgency: isAI ? "high" : "medium",
      modules: ["studio"],
      impact: isAI ? 80 : 60,
      complexity: isAI ? "complex" : "moderate",
      hours: isAI ? 60 : 30,
    };
  }
  if (
    /stem separat|vocal remov|ai master|ai mix|ai composi|neural audio|music generat|ai produc|plugin|vst|spatial audio/.test(
      text,
    )
  ) {
    return {
      source: "technology",
      category: "feature",
      urgency: "medium",
      modules: ["studio"],
      impact: 70,
      complexity: "complex",
      hours: 50,
    };
  }
  if (
    /distrokid|tunecore|cd baby|awal|unitedmasters|amuse|stem\.is|routenote|ditto music|onerpm|believe digital/.test(
      text,
    )
  ) {
    if (
      /new feature|launch|release|announce|update|add|introduce|now offer|partnership|integrat/.test(
        text,
      )
    ) {
      return {
        source: "competitor",
        category: "feature",
        urgency: "high",
        modules: ["distribution", "analytics", "monetization"],
        impact: 92,
        complexity: "moderate",
        hours: 20,
      };
    }
    return {
      source: "competitor",
      category: "optimization",
      urgency: "medium",
      modules: ["distribution"],
      impact: 72,
      complexity: "simple",
      hours: 8,
    };
  }
  if (
    /audience behavior|consumer trend|listening habit|streaming behavior/.test(
      text,
    )
  ) {
    return {
      source: "audience_behavior",
      category: "audience_shift",
      urgency: "medium",
      modules: ["analytics", "content"],
      impact: 68,
      complexity: "simple",
      hours: 6,
    };
  }
  if (
    /content marketing|email marketing|newsletter|seo|brand deal|influencer marketing/.test(
      text,
    )
  ) {
    return {
      source: "content_trend",
      category: "trend",
      urgency: "low",
      modules: ["advertising", "social"],
      impact: 60,
      complexity: "simple",
      hours: 4,
    };
  }
  if (
    /ai |artificial intelligence|machine learning|generative|gpt|llm/.test(
      text,
    ) &&
    /music|audio|song|track|artist|production|content|creat/.test(text)
  ) {
    return {
      source: "technology",
      category: "feature",
      urgency: "medium",
      modules: ["studio", "analytics", "content"],
      impact: 72,
      complexity: "complex",
      hours: 45,
    };
  }
  return null;
}

// ─── RSS Parser ───────────────────────────────────────────────────────────────

interface RssItem {
  title?: string;
  description?: string;
  link?: string;
  pubDate?: string;
  guid?: string | { "#text": string };
}

function stripHtml(html: string): string {
  return html
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

// ─── Signal Monitor ───────────────────────────────────────────────────────────

interface SignalCache {
  signals: LiveIndustrySignal[];
  fetchedAt: number;
}

class ContentSignalMonitor {
  private cache: SignalCache | null = null;
  private seenIds = new Set<string>();
  private parser = new XMLParser({
    ignoreAttributes: false,
    attributeNamePrefix: "@_",
  });

  async fetchLiveSignals(): Promise<LiveIndustrySignal[]> {
    if (this.cache && Date.now() - this.cache.fetchedAt < CACHE_TTL_MIN_MS) {
      const fresh = this.cache.signals.filter((s) => !this.seenIds.has(s.id));
      for (const s of fresh) this.seenIds.add(s.id);
      return fresh;
    }

    const [rssSignals, searchSignals] = await Promise.all([
      this.fetchAllRssFeeds(),
      this.fetchSearchIntelligence(),
    ]);

    const all = [...rssSignals, ...searchSignals];
    const unique = this.dedup(all);
    this.cache = { signals: unique, fetchedAt: Date.now() };

    const fresh = unique.filter((s) => !this.seenIds.has(s.id));
    for (const s of fresh) this.seenIds.add(s.id);
    return fresh;
  }

  private async fetchAllRssFeeds(): Promise<LiveIndustrySignal[]> {
    const results = await Promise.allSettled(
      RSS_FEEDS.map((f) => this.fetchRssFeed(f.url, f.name)),
    );
    const all: LiveIndustrySignal[] = [];
    for (const r of results) {
      if (r.status === "fulfilled") all.push(...r.value);
    }
    return all;
  }

  private async fetchRssFeed(
    url: string,
    feedName: string,
  ): Promise<LiveIndustrySignal[]> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
    try {
      const res = await fetch(url, {
        signal: controller.signal,
        headers: { "User-Agent": "MaxBooster-ContentAwareness/2.0" },
        redirect: "follow",
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const xml = await res.text();
      return this.parseRss(xml, feedName, url);
    } finally {
      clearTimeout(timeout);
    }
  }

  private parseRss(
    xml: string,
    feedName: string,
    _url: string,
  ): LiveIndustrySignal[] {
    const signals: LiveIndustrySignal[] = [];
    try {
      const parsed = this.parser.parse(xml);
      const channel = parsed?.rss?.channel;
      if (!channel) return signals;
      const items: RssItem[] = Array.isArray(channel.item)
        ? channel.item
        : channel.item
          ? [channel.item]
          : [];

      for (const item of items.slice(0, 20)) {
        const title = stripHtml(String(item.title || "")).trim();
        const description = stripHtml(String(item.description || ""))
          .trim()
          .slice(0, 500);
        const link = String(item.link || "");
        const pubDate = item.pubDate
          ? new Date(String(item.pubDate))
          : new Date();
        if (!title || title.length < 5) continue;

        const cls = classifySignal(title, description);
        if (!cls) continue;

        const rawId =
          typeof item.guid === "object"
            ? item.guid["#text"]
            : item.guid || link || title;
        const id = `rss_${crypto.createHash("sha256").update(String(rawId)).digest("hex").slice(0, 16)}`;

        signals.push({
          id,
          source: cls.source,
          category: cls.category,
          urgency: cls.urgency,
          affectedModules: cls.modules,
          competitiveImpact: cls.impact,
          implementationComplexity: cls.complexity,
          estimatedImplementationHours: cls.hours,
          title: `[${feedName}] ${title}`,
          description: description || title,
          detectedAt: isNaN(pubDate.getTime()) ? new Date() : pubDate,
          sourceUrl: link,
          feedSource: feedName,
        });
      }
    } catch {
      // Non-fatal — individual feeds may fail
    }
    return signals;
  }

  private async fetchSearchIntelligence(): Promise<LiveIndustrySignal[]> {
    const tavilyKey = process.env.TAVILY_API_KEY;
    const exaKey = process.env.EXA_API_KEY;
    if (!tavilyKey && !exaKey) return [];

    const allQueries = [
      ...SEARCH_QUERIES.music,
      ...SEARCH_QUERIES.content,
      ...SEARCH_QUERIES.advertising,
    ];

    const [tavilyResults, exaResults] = await Promise.all([
      tavilyKey
        ? Promise.allSettled(
            allQueries.map((q) => this.tavilySearch(q, tavilyKey)),
          )
        : Promise.resolve([] as PromiseSettledResult<LiveIndustrySignal[]>[]),
      exaKey
        ? Promise.allSettled(allQueries.map((q) => this.exaSearch(q, exaKey)))
        : Promise.resolve([] as PromiseSettledResult<LiveIndustrySignal[]>[]),
    ]);

    const signals: LiveIndustrySignal[] = [];
    for (const r of [...tavilyResults, ...exaResults]) {
      if (r.status === "fulfilled") signals.push(...r.value);
    }
    return signals;
  }

  private async tavilySearch(
    query: string,
    apiKey: string,
  ): Promise<LiveIndustrySignal[]> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
    try {
      const res = await fetch(TAVILY_API, {
        method: "POST",
        signal: controller.signal,
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          query,
          max_results: 5,
          search_depth: "basic",
          topic: "news",
        }),
      });
      if (!res.ok) throw new Error(`Tavily HTTP ${res.status}`);
      const data = (await res.json()) as {
        results?: Array<{ title: string; content: string; url: string }>;
      };
      return (data.results || [])
        .map((r) => {
          const cls = classifySignal(r.title, r.content);
          if (!cls) return null;
          const id = `tavily_${crypto.createHash("sha256").update(r.url).digest("hex").slice(0, 16)}`;
          return {
            id,
            source: cls.source,
            category: cls.category,
            urgency: cls.urgency,
            affectedModules: cls.modules,
            competitiveImpact: cls.impact,
            implementationComplexity: cls.complexity,
            estimatedImplementationHours: cls.hours,
            title: `[Search] ${r.title}`,
            description: r.content.slice(0, 500),
            detectedAt: new Date(),
            sourceUrl: r.url,
            feedSource: "Tavily",
          };
        })
        .filter((x) => x !== null) as LiveIndustrySignal[];
    } finally {
      clearTimeout(timeout);
    }
  }

  private async exaSearch(
    query: string,
    apiKey: string,
  ): Promise<LiveIndustrySignal[]> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
    try {
      const res = await fetch(EXA_API, {
        method: "POST",
        signal: controller.signal,
        headers: { "Content-Type": "application/json", "x-api-key": apiKey },
        body: JSON.stringify({
          query,
          numResults: 5,
          type: "neural",
          useAutoprompt: true,
        }),
      });
      if (!res.ok) throw new Error(`Exa HTTP ${res.status}`);
      const data = (await res.json()) as {
        results?: Array<{ title: string; text?: string; url: string }>;
      };
      return (data.results || [])
        .map((r) => {
          const text = r.text || "";
          const cls = classifySignal(r.title, text);
          if (!cls) return null;
          const id = `exa_${crypto.createHash("sha256").update(r.url).digest("hex").slice(0, 16)}`;
          return {
            id,
            source: cls.source,
            category: cls.category,
            urgency: cls.urgency,
            affectedModules: cls.modules,
            competitiveImpact: cls.impact,
            implementationComplexity: cls.complexity,
            estimatedImplementationHours: cls.hours,
            title: `[Search] ${r.title}`,
            description: text.slice(0, 500),
            detectedAt: new Date(),
            sourceUrl: r.url,
            feedSource: "Exa",
          };
        })
        .filter((x) => x !== null) as LiveIndustrySignal[];
    } finally {
      clearTimeout(timeout);
    }
  }

  private dedup(signals: LiveIndustrySignal[]): LiveIndustrySignal[] {
    const seen = new Set<string>();
    return signals.filter((s) => {
      if (seen.has(s.id)) return false;
      seen.add(s.id);
      return true;
    });
  }
}

// ─── Context Builder ──────────────────────────────────────────────────────────

class ContentGenerationContextBuilder {
  build(signals: LiveIndustrySignal[]): ContentAwarenessContext {
    const genreScores = new Map<string, number>();
    const moodScores = new Map<string, number>();
    const prodScores = new Map<string, number>();
    const ctaScores = new Map<string, number>();
    const triggerScores = new Map<string, number>();
    const topicScores = new Map<string, number>();
    const algoNotes = new Set<string>();
    const hookSet = new Set<string>();
    const themeSet = new Set<string>();
    const platMap = new Map<string, Set<string>>();
    const platFormatMap = new Map<string, string>();
    const formatMap = new Map<string, { platform: string; weight: number }>();

    for (const sig of signals) {
      const text = `${sig.title} ${sig.description}`;
      const age = this.recencyFactor(sig.detectedAt);
      const boost =
        sig.urgency === "critical"
          ? 1.4
          : sig.urgency === "high"
            ? 1.2
            : sig.urgency === "medium"
              ? 1.0
              : 0.8;
      const w = age * boost;

      for (const kw of GENRE_KW) {
        if (kw.pattern.test(text))
          genreScores.set(
            kw.value,
            (genreScores.get(kw.value) ?? 0) + kw.weight * w,
          );
      }
      for (const kw of MOOD_KW) {
        if (kw.pattern.test(text))
          moodScores.set(
            kw.value,
            (moodScores.get(kw.value) ?? 0) + kw.weight * w,
          );
      }
      for (const kw of PRODUCTION_KW) {
        if (kw.pattern.test(text))
          prodScores.set(
            kw.value,
            (prodScores.get(kw.value) ?? 0) + kw.weight * w,
          );
      }
      for (const kw of CTA_KW) {
        if (kw.pattern.test(text))
          ctaScores.set(kw.cta, (ctaScores.get(kw.cta) ?? 0) + kw.weight * w);
      }
      for (const kw of EMOTIONAL_TRIGGER_KW) {
        if (kw.pattern.test(text))
          triggerScores.set(
            kw.trigger,
            (triggerScores.get(kw.trigger) ?? 0) + kw.weight * w,
          );
      }
      for (const kw of TRENDING_TOPIC_KW) {
        if (kw.pattern.test(text))
          topicScores.set(
            kw.topic,
            (topicScores.get(kw.topic) ?? 0) + kw.weight * w,
          );
      }
      for (const kw of PLATFORM_ALGORITHM_KW) {
        if (kw.pattern.test(text)) algoNotes.add(kw.note);
      }
      for (const kw of HOOK_KW) {
        if (kw.pattern.test(text)) hookSet.add(kw.hook);
      }
      for (const kw of LYRIC_KW) {
        if (kw.pattern.test(text)) themeSet.add(kw.theme);
      }
      for (const kw of CONTENT_FORMAT_KW) {
        if (kw.pattern.test(text)) {
          const existing = formatMap.get(kw.format);
          formatMap.set(kw.format, {
            platform: kw.platform,
            weight: (existing?.weight ?? 0) + kw.weight * w,
          });
        }
      }
      for (const pe of PLATFORM_ENTRIES) {
        if (pe.detect.test(text)) {
          const set = platMap.get(pe.platform) ?? new Set<string>();
          for (const ts of pe.trendSignals) {
            if (ts.pattern.test(text)) {
              set.add(ts.trend);
              if (ts.contentFormat)
                platFormatMap.set(
                  `${pe.platform}:${ts.trend}`,
                  ts.contentFormat,
                );
            }
          }
          if (set.size) platMap.set(pe.platform, set);
        }
      }
    }

    const MAX = 5;
    const trendingGenres = this.topN(genreScores, MAX);
    const trendingMoods = this.topN(moodScores, MAX);
    const productionStyles = this.topN(prodScores, MAX);
    const ctaPatterns = this.topN(ctaScores, 4);
    const emotionalTriggers = this.topN(triggerScores, 4);
    const trendingTopics = this.topN(topicScores, 5);
    const platformAlgorithmNotes = [...algoNotes].slice(0, 5);
    const viralHookPatterns = [...hookSet].slice(0, 4);
    const lyricThemes = [...themeSet].slice(0, 4);
    const contentAngles = viralHookPatterns.slice(0, 3);

    const platformSignals: PlatformTrendSignal[] = [];
    for (const [platform, trends] of platMap) {
      let i = 0;
      for (const trend of trends) {
        const strength = i === 0 ? "strong" : i === 1 ? "moderate" : "emerging";
        const contentFormat = platFormatMap.get(`${platform}:${trend}`);
        platformSignals.push({
          platform,
          trend,
          strength: strength as PlatformTrendSignal["strength"],
          contentFormat,
        });
        i++;
      }
    }

    const contentFormats: ContentFormatTrend[] = [...formatMap.entries()]
      .sort((a, b) => b[1].weight - a[1].weight)
      .slice(0, 6)
      .map(([format, { platform }]) => ({
        format,
        platform,
        momentum: "rising" as const,
      }));

    const audiencePsychology: AudiencePsychologySignal[] = [
      ...triggerScores.entries(),
    ]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 4)
      .map(([trigger, strength]) => ({
        trigger,
        pattern: this.triggerPattern(trigger),
        strength: Math.min(1, strength / 10),
      }));

    const confidence = Math.min(1, signals.length / 40);
    const hints = this.buildHints(
      trendingGenres,
      trendingMoods,
      productionStyles,
      platformSignals,
      viralHookPatterns,
      ctaPatterns,
      emotionalTriggers,
    );

    return {
      trendingGenres,
      trendingMoods,
      productionStyles,
      platformSignals,
      viralHookPatterns,
      lyricThemes,
      contentAngles,
      ctaPatterns,
      emotionalTriggers,
      contentFormats,
      audiencePsychology,
      trendingTopics,
      platformAlgorithmNotes,
      generationHints: hints,
      contextString: "",
      signalCount: signals.length,
      confidence,
      freshness: new Date(),
    };
  }

  applyMode(
    ctx: ContentAwarenessContext,
    mode: ContentGenerationMode,
  ): ContentAwarenessContext {
    if (ctx.confidence < 0.05 || ctx.signalCount === 0) {
      return { ...ctx, contextString: "" };
    }

    // strength → signal priority tag understood by Python parsers
    const tag = (s: string) =>
      s === "strong" ? "[HIGH]" : s === "moderate" ? "[MEDIUM]" : "[LOW]";

    const lines: string[] = ["=== LIVE INDUSTRY SIGNALS ==="];

    // Platform signals as [HIGH/MEDIUM/LOW] tagged lines
    for (const sig of ctx.platformSignals.slice(0, 6)) {
      const fmt = sig.contentFormat ? ` (${sig.contentFormat})` : "";
      lines.push(`${tag(sig.strength)} ${sig.platform}: ${sig.trend}${fmt}`);
    }

    // Mode-specific Action lines — drive hook/CTA parsing in Python agents
    const modeActions: Partial<Record<ContentGenerationMode, string[]>> = {
      social: ctx.ctaPatterns.slice(0, 3),
      ad_copy: ctx.ctaPatterns.slice(0, 3),
      advertising: ctx.ctaPatterns.slice(0, 3),
      video_script: [
        ...ctx.viralHookPatterns.slice(0, 2),
        ...ctx.ctaPatterns.slice(0, 1),
      ],
      songwriting: [
        ...ctx.lyricThemes.slice(0, 2),
        ...ctx.ctaPatterns.slice(0, 1),
      ],
      music: ctx.lyricThemes.slice(0, 3),
      melody: ctx.lyricThemes.slice(0, 3),
      content: ctx.ctaPatterns.slice(0, 2),
      distribution: [
        ...ctx.viralHookPatterns.slice(0, 1),
        ...ctx.ctaPatterns.slice(0, 2),
      ],
    };
    for (const action of modeActions[mode] ?? ctx.ctaPatterns.slice(0, 2)) {
      lines.push(`Action: ${action}`);
    }

    // Hook patterns and emotional triggers as bullet points
    for (const hook of ctx.viralHookPatterns.slice(0, 3)) {
      lines.push(`• Hook pattern: ${hook}`);
    }
    for (const trigger of ctx.emotionalTriggers.slice(0, 3)) {
      lines.push(`• Emotional pull: ${trigger}`);
    }
    for (const angle of ctx.contentAngles.slice(0, 2)) {
      lines.push(`• Content angle: ${angle}`);
    }

    // Trending topics section — drives #hashtag extraction in Python parsers
    const hashtagSet = new Set<string>();
    for (const raw of (ctx.generationHints.hashtagContext ?? "").split(/\s+/)) {
      if (raw.startsWith("#") && raw.length > 1) hashtagSet.add(raw);
    }
    for (const genre of ctx.trendingGenres.slice(0, 5)) {
      hashtagSet.add(`#${genre.toLowerCase().replace(/[\s\-/]+/g, "")}`);
    }
    for (const mood of ctx.trendingMoods.slice(0, 3)) {
      hashtagSet.add(`#${mood.toLowerCase().replace(/[\s\-/]+/g, "")}`);
    }
    for (const topic of ctx.trendingTopics.slice(0, 3)) {
      const w = topic.split(/\s+/)[0];
      if (w) hashtagSet.add(`#${w.toLowerCase().replace(/\W/g, "")}`);
    }
    if (hashtagSet.size) {
      lines.push("=== TRENDING TOPICS ===");
      lines.push([...hashtagSet].slice(0, 12).join(" "));
    }

    // Platform algorithm notes (may include timing cues Python timing parser scans)
    if (ctx.platformAlgorithmNotes.length) {
      lines.push("=== PLATFORM NOTES ===");
      for (const note of ctx.platformAlgorithmNotes.slice(0, 4)) {
        lines.push(`• ${note}`);
      }
    }

    // Distribution-specific release window signals — scanned by Python timing parser
    if (mode === "distribution") {
      lines.push("=== DISTRIBUTION WINDOW ===");
      lines.push("• Playlist pitch lead: 7+ days before release date");
      lines.push("• Release window: Friday drops maximise chart eligibility");
      lines.push("• Editorial submission: via distributor 7 days pre-release");
      const streamingSignals = ctx.platformSignals.filter((s) =>
        ["spotify", "apple music", "youtube", "tidal"].includes(
          s.platform.toLowerCase(),
        ),
      );
      for (const sig of streamingSignals.slice(0, 3)) {
        lines.push(`• ${sig.platform}: ${sig.trend}`);
      }
      const peakTimes = ctx.platformAlgorithmNotes.filter((n) =>
        /peak|morning|evening|afternoon|9am|6pm|7pm|noon/i.test(n),
      );
      for (const t of peakTimes.slice(0, 2)) {
        lines.push(`• Timing signal: ${t}`);
      }
    }

    // Rich trailing context block — all modes benefit from this
    lines.push("=== TRENDING CONTEXT ===");
    if (ctx.trendingGenres.length)
      lines.push(
        `Trending genres: ${ctx.trendingGenres.slice(0, 5).join(", ")}`,
      );
    if (ctx.trendingMoods.length)
      lines.push(`Trending moods: ${ctx.trendingMoods.slice(0, 4).join(", ")}`);
    if (ctx.lyricThemes.length)
      lines.push(`Lyric themes: ${ctx.lyricThemes.slice(0, 4).join(", ")}`);
    if (ctx.productionStyles.length)
      lines.push(
        `Production styles: ${ctx.productionStyles.slice(0, 3).join(", ")}`,
      );
    if (ctx.trendingTopics.length)
      lines.push(
        `Trending topics: ${ctx.trendingTopics.slice(0, 4).join(", ")}`,
      );
    if (ctx.generationHints.tempoBias !== "neutral")
      lines.push(`Tempo bias: ${ctx.generationHints.tempoBias}`);

    // Corpus maturity bridge — consumed by Python quality_awareness.py and
    // script_agent.py to calibrate how heavily they lean on the live buffer
    // vs the model's own grown corpus.  buffer_weight=1.0 means Python should
    // weigh external signals heavily; 0.0 means the model is self-sufficient.
    const maturity = (
      ctx as ContentAwarenessContext & { _maturity?: CorpusMaturity }
    )._maturity;
    if (maturity) {
      lines.push("=== CORPUS MATURITY ===");
      lines.push(`buffer_weight: ${maturity.bufferWeight.toFixed(3)}`);
      lines.push(
        `own_corpus: ${maturity.ownCorpus}/${maturity.retireThreshold}`,
      );
      lines.push(`external_signal_priority: ${maturity.retired ? "low" : maturity.bufferWeight > 0.7 ? "high" : "medium"}`);
    }

    return { ...ctx, contextString: lines.join("\n") };
  }

  private buildHints(
    genres: string[],
    moods: string[],
    production: string[],
    platformSignals: PlatformTrendSignal[],
    hooks: string[],
    ctas: string[],
    triggers: string[],
  ): ContentGenerationHints {
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
      if (upMoods.has(m)) up++;
      if (downMoods.has(m)) down++;
    }
    const tempoBias: "up" | "down" | "neutral" =
      up > down ? "up" : down > up ? "down" : "neutral";
    const hashtagContext = platformSignals
      .filter((s) => s.strength === "strong")
      .map((s) => `#${s.platform.toLowerCase().replace(/[\s/]+/g, "")}`)
      .join(" ");

    return {
      suggestedGenre: genres[0],
      suggestedMood: moods[0],
      tempoBias,
      productionKeywords: production.slice(0, 3),
      contentAngles: hooks.slice(0, 3),
      hashtagContext,
      ctaPatterns: ctas.slice(0, 3),
      emotionalTriggers: triggers.slice(0, 3),
      contentFormats: [],
      audiencePsychology: [],
      trendingTopics: [],
      platformAlgorithmNotes: [],
    };
  }

  private topN(scores: Map<string, number>, n: number): string[] {
    return [...scores.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, n)
      .map(([k]) => k);
  }

  private recencyFactor(date: Date): number {
    const h = (Date.now() - date.getTime()) / 3_600_000;
    if (h <= 24) return 1.0;
    if (h <= 72) return 0.8;
    if (h <= 168) return 0.5;
    return 0.2;
  }

  private triggerPattern(trigger: string): string {
    const patterns: Record<string, string> = {
      nostalgia: "Reference past eras, throwback moments, 'remember when'",
      FOMO: "Limited time, exclusive access, 'don't miss out'",
      inspiration: "Transformation story, before/after, 'what's possible'",
      "triumph over struggle": "Overcoming odds, real talk, journey reveal",
      exclusivity: "VIP access, members-only, early release",
      "community belonging": "Fan shoutout, 'join us', shared identity",
      authenticity: "No filter, raw moment, honest take",
      "surprise and delight": "Unexpected reveal, plot twist, bonus",
      "pride and achievement": "Milestone drop, chart performance, award",
      rebellion: "Anti-mainstream, doing it differently, against the grain",
      "love and connection": "Relationship anthem, fan love, collab chemistry",
      urgency: "Today only, act now, limited quantity",
      "social proof": "Viral stats, fan reactions, press quotes",
    };
    return patterns[trigger] ?? "Apply this trigger to opening hook and CTA";
  }

  empty(): ContentAwarenessContext {
    return {
      trendingGenres: [],
      trendingMoods: [],
      productionStyles: [],
      platformSignals: [],
      viralHookPatterns: [],
      lyricThemes: [],
      contentAngles: [],
      ctaPatterns: [],
      emotionalTriggers: [],
      contentFormats: [],
      audiencePsychology: [],
      trendingTopics: [],
      platformAlgorithmNotes: [],
      generationHints: {
        tempoBias: "neutral",
        productionKeywords: [],
        contentAngles: [],
        hashtagContext: "",
        ctaPatterns: [],
        emotionalTriggers: [],
        contentFormats: [],
        audiencePsychology: [],
        trendingTopics: [],
        platformAlgorithmNotes: [],
      },
      contextString: "",
      signalCount: 0,
      confidence: 0,
      freshness: new Date(),
    };
  }
}

// ─── Unified Service ──────────────────────────────────────────────────────────

class ContentGenerationAwarenessService {
  private monitor = new ContentSignalMonitor();
  private builder = new ContentGenerationContextBuilder();
  private cache: { ctx: ContentAwarenessContext; builtAt: number } | null =
    null;

  /**
   * Primary API — returns mode-specific awareness context.
   * The `contextString` field is ready to inject into any MaxCore `extraContext`.
   * Never throws — returns an empty zero-confidence context on any error.
   */
  async getContextForMode(
    mode: ContentGenerationMode,
  ): Promise<ContentAwarenessContext> {
    try {
      const base = await this.getOrBuild();
      return this.builder.applyMode(base, mode);
    } catch {
      return this.builder.empty();
    }
  }

  // ── Sync getters (read warm cache only — zero latency) ─────────────────────

  getSuggestedGenreSync(): string | undefined {
    return this.cache?.ctx.generationHints.suggestedGenre;
  }
  getSuggestedMoodSync(): string | undefined {
    return this.cache?.ctx.generationHints.suggestedMood;
  }
  getTempoBiasSync(): "up" | "down" | "neutral" {
    return this.cache?.ctx.generationHints.tempoBias ?? "neutral";
  }
  getProductionKeywordsSync(): string[] {
    return this.cache?.ctx.generationHints.productionKeywords ?? [];
  }
  getCtaPatternsSync(): string[] {
    return this.cache?.ctx.ctaPatterns ?? [];
  }
  getEmotionalTriggersSync(): string[] {
    return this.cache?.ctx.emotionalTriggers ?? [];
  }
  getPlatformSignalsSync(): PlatformTrendSignal[] {
    return this.cache?.ctx.platformSignals ?? [];
  }
  getTrendingTopicsSync(): string[] {
    return this.cache?.ctx.trendingTopics ?? [];
  }
  getPlatformAlgorithmNotesSync(): string[] {
    return this.cache?.ctx.platformAlgorithmNotes ?? [];
  }
  getContentFormatsSync(): ContentFormatTrend[] {
    return this.cache?.ctx.contentFormats ?? [];
  }
  getConfidenceSync(): number {
    return this.cache?.ctx.confidence ?? 0;
  }
  clearCache(): void {
    this.cache = null;
  }

  // ── Internal ───────────────────────────────────────────────────────────────

  private async getOrBuild(): Promise<ContentAwarenessContext> {
    // Probe corpus maturity first (cached for 5 min, never blocks)
    const maturity = await fetchCorpusMaturity().catch(() => null);
    const bufferWeight = maturity?.bufferWeight ?? 1.0;
    const ttl = dynamicCacheTtl(bufferWeight);

    if (this.cache && Date.now() - this.cache.builtAt < ttl) {
      return this.cache.ctx;
    }
    const signals = await this.monitor.fetchLiveSignals();
    const ctx = this.builder.build(signals);
    // Store corpus maturity alongside the context for use in applyMode
    (ctx as ContentAwarenessContext & { _maturity?: CorpusMaturity })._maturity =
      maturity ?? undefined;
    this.cache = { ctx, builtAt: Date.now() };
    return ctx;
  }
}

// ─── Singleton Export ──────────────────────────────────────────────────────────

export const contentAwarenessService = new ContentGenerationAwarenessService();

/**
 * Convenience re-export of the full service type for dependency injection.
 */
export type { ContentGenerationAwarenessService };
