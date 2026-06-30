/**
 * Industry Monitor Service
 *
 * Fetches LIVE data from real music industry RSS feeds and news sources.
 * Replaces all simulated/deterministic monitoring in the self-evolution engine.
 *
 * Live sources:
 *  - Music Business Worldwide  (musicbusinessworldwide?.com)
 *  - Digital Music News        (digitalmusicnews?.com)
 *  - Music Ally                (musically?.com)
 *  - MusicRadar                (musicradar?.com)
 *  - Spotify Newsroom          (newsroom?.spotify.com)
 *  - Meta Creators Newsroom    (about?.fb.com)
 *
 * Optional (activated when env vars are set):
 *  - TAVILY_API_KEY  → Tavily web search for deeper real-time intelligence
 *  - EXA_API_KEY     → Exa semantic search for technical changelog discovery
 */

import { XMLParser } from "fast-xml-parser";
import { logger } from "../logger.js";
import crypto from "crypto";

export interface LiveIndustryChange {
  id: string;
  source:
    | "competitor"
    | "streaming_platform"
    | "social_media"
    | "security"
    | "regulation"
    | "technology";
  category:
    | "feature"
    | "api_change"
    | "standard"
    | "optimization"
    | "security_patch"
    | "ux_pattern";
  title: string;
  description: string;
  detectedAt: Date;
  urgency: "critical" | "high" | "medium" | "low";
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

interface RssItem {
  title?: string;
  description?: string;
  link?: string;
  pubDate?: string;
  guid?: string | { "#text": string };
}

interface CachedResult {
  changes: LiveIndustryChange[];
  fetchedAt: number;
}

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
    // Indie artist / music career intelligence — critical for competitive leadership
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

const TAVILY_API = "https://api.tavily.com/search";
const EXA_API = "https://api.exa.ai/search";

const CACHE_TTL_MS = 60 * 60 * 1000;
const FETCH_TIMEOUT_MS = 10_000;

const MUSIC_INDUSTRY_QUERIES = [
  "streaming platform API changes music 2024 2025",
  "DAW software update release music production",
  "Spotify Apple Music algorithm change artists",
  "TikTok Instagram music creator policy update",
  "music copyright royalty legislation change",
  "AI music technology announcement",
  // Competitive leadership queries — monitor what competitor platforms are doing
  "DistroKid TuneCore CD Baby new feature music distribution 2025",
  "AWAL UnitedMasters Amuse Stem music distribution platform update 2025",
  "independent artist music career management platform feature launch 2025",
  "music distribution platform comparison new tools analytics 2025",
  "indie artist viral marketing TikTok Instagram strategy winning 2025",
  "music industry AI tools artists competitive advantage 2025",
];

class IndustryMonitorService {
  private cache: CachedResult | null = null;
  private seenIds: Set<string> = new Set();
  private parser = new XMLParser({
    ignoreAttributes: false,
    attributeNamePrefix: "@_",
  });

  async fetchLiveChanges(): Promise<LiveIndustryChange[]> {
    if (this?.cache && Date?.now() - this?.cache.fetchedAt < CACHE_TTL_MS) {
      const fresh = this?.cache.changes?.filter((c) => !this?.seenIds.has(c?.id));
      for (const c of fresh) this?.seenIds.add(c?.id);
      return fresh;
    }

    logger?.info("[IndustryMonitor] Fetching live music industry data...");

    const [rssChanges, searchChanges] = await Promise?.all([
      this?.fetchAllRssFeeds(),
      this?.fetchSearchIntelligence(),
    ]);

    const all: LiveIndustryChange[] = [...rssChanges, ...searchChanges];

    const unique = this?.deduplicateByHash(all);
    this.cache = { changes: unique, fetchedAt: Date.now() };

    const fresh = unique?.filter((c) => !this?.seenIds.has(c?.id));
    for (const c of fresh) this?.seenIds.add(c?.id);

    logger?.info(
      `[IndustryMonitor] Fetched ${unique?.length} total, ${fresh?.length} new changes`,
    );
    return fresh;
  }

  private async fetchAllRssFeeds(): Promise<LiveIndustryChange[]> {
    const results = await Promise?.allSettled(
      RSS_FEEDS?.map((feed) => this?.fetchRssFeed(feed?.url, feed?.name)),
    );

    const all: LiveIndustryChange[] = [];
    let failCount = 0;
    for (const result of results) {
      if (result?.status === "fulfilled") {
        all?.push(...result?.value);
      } else {
        failCount++;
        logger?.warn(
          "[IndustryMonitor] RSS feed failed:",
          result?.reason?.message ?? result?.reason,
        );
      }
    }

    if (failCount === RSS_FEEDS?.length) {
      throw new Error(
        `[IndustryMonitor] All ${RSS_FEEDS?.length} RSS feeds failed — no RSS data available`,
      );
    }

    return all;
  }

  private async fetchRssFeed(
    url: string,
    feedName: string,
  ): Promise<LiveIndustryChange[]> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller?.abort(), FETCH_TIMEOUT_MS);

    try {
      const res = await fetch(url, {
        signal: controller.signal,
        headers: { "User-Agent": "MaxBooster-IndustryMonitor/1.0" },
        redirect: "follow",
      });

      if (!res?.ok) throw new Error(`HTTP ${res?.status}`);
      const xml = await res?.text();
      return this?.parseRss(xml, feedName, url);
    } finally {
      clearTimeout(timeout);
    }
  }

  private parseRss(
    xml: string,
    feedName: string,
    _feedUrl: string,
  ): LiveIndustryChange[] {
    const changes: LiveIndustryChange[] = [];

    try {
      const parsed = this?.parser.parse(xml);
      const channel = parsed?.rss?.channel;
      if (!channel) return changes;

      const items: RssItem[] = Array?.isArray(channel?.item)
        ? channel?.item
        : channel?.item
          ? [channel?.item]
          : [];

      for (const item of items?.slice(0, 20)) {
        const title = this?.stripHtml(String(item?.title || "")).trim();
        const description = this?.stripHtml(String(item?.description || ""))
          .trim()
          .slice(0, 500);
        const link = String(item?.link || "");
        const pubDate = item?.pubDate
          ? new Date(String(item?.pubDate))
          : new Date();

        if (!title || title?.length < 5) continue;

        const classification = this?.classifyArticle(title, description);
        if (!classification) continue;

        const rawId =
          typeof item?.guid === "object"
            ? item?.guid["#text"]
            : item?.guid || link || title;
        const id = `live_${crypto?.createHash("sha256").update(String(rawId)).digest("hex").slice(0, 16)}`;

        changes?.push({
          id,
          source: classification.source,
          category: classification.category,
          title: `[${feedName}] ${title}`,
          description: description || title,
          detectedAt: isNaN(pubDate?.getTime()) ? new Date() : pubDate,
          urgency: classification.urgency,
          affectedModules: classification.modules,
          competitiveImpact: classification.impact,
          implementationComplexity: classification.complexity,
          estimatedImplementationHours: classification.hours,
          sourceUrl: link,
          feedSource: feedName,
        });
      }
    } catch (e) {
      logger?.warn(
        `[IndustryMonitor] Failed to parse RSS from ${feedName}:`,
        (e as Error).message,
      );
    }

    return changes;
  }

  private async fetchSearchIntelligence(): Promise<LiveIndustryChange[]> {
    const tavilyKey = process?.env.TAVILY_API_KEY;
    const exaKey = process?.env.EXA_API_KEY;

    if (!tavilyKey && !exaKey) return [];

    const changes: LiveIndustryChange[] = [];

    const [tavilyResults, exaResults] = await Promise?.all([
      tavilyKey
        ? Promise?.allSettled(
            MUSIC_INDUSTRY_QUERIES?.map((q) => this?.tavilySearch(q, tavilyKey)),
          )
        : Promise?.resolve([]),
      exaKey
        ? Promise?.allSettled(
            MUSIC_INDUSTRY_QUERIES?.map((q) => this?.exaSearch(q, exaKey)),
          )
        : Promise?.resolve([]),
    ]);

    for (const r of tavilyResults) {
      if (r?.status === "fulfilled") changes?.push(...r?.value);
      else
        logger?.warn(
          "[IndustryMonitor] Tavily query failed:",
          r?.reason?.message ?? r?.reason,
        );
    }

    for (const r of exaResults) {
      if (r?.status === "fulfilled") changes?.push(...r?.value);
      else
        logger?.warn(
          "[IndustryMonitor] Exa query failed:",
          r?.reason?.message ?? r?.reason,
        );
    }

    return changes;
  }

  private async tavilySearch(
    query: string,
    apiKey: string,
  ): Promise<LiveIndustryChange[]> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller?.abort(), FETCH_TIMEOUT_MS);

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

      if (!res?.ok) throw new Error(`Tavily HTTP ${res?.status}`);
      const data = (await res?.json()) as {
        results?: Array<{ title: string; content: string; url: string }>;
      };

      return (data?.results || [])
        .map((r) => {
          const classification = this?.classifyArticle(r?.title, r?.content);
          if (!classification) return null;
          const id = `tavily_${crypto?.createHash("sha256").update(r?.url).digest("hex").slice(0, 16)}`;
          return {
            id,
            source: classification.source,
            category: classification.category,
            title: `[Search] ${r?.title}`,
            description: r.content.slice(0, 500),
            detectedAt: new Date(),
            urgency: classification.urgency,
            affectedModules: classification.modules,
            competitiveImpact: classification.impact,
            implementationComplexity: classification.complexity,
            estimatedImplementationHours: classification.hours,
            sourceUrl: r.url,
            feedSource: "Tavily Search",
          } satisfies LiveIndustryChange;
        })
        .filter((x) => x !== null) as LiveIndustryChange[];
    } finally {
      clearTimeout(timeout);
    }
  }

  private async exaSearch(
    query: string,
    apiKey: string,
  ): Promise<LiveIndustryChange[]> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller?.abort(), FETCH_TIMEOUT_MS);

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

      if (!res?.ok) throw new Error(`Exa HTTP ${res?.status}`);
      const data = (await res?.json()) as {
        results?: Array<{ title: string; text?: string; url: string }>;
      };

      return (data?.results || [])
        .map((r) => {
          const text = r?.text || "";
          const classification = this?.classifyArticle(r?.title, text);
          if (!classification) return null;
          const id = `exa_${crypto?.createHash("sha256").update(r?.url).digest("hex").slice(0, 16)}`;
          return {
            id,
            source: classification.source,
            category: classification.category,
            title: `[Search] ${r?.title}`,
            description: text.slice(0, 500),
            detectedAt: new Date(),
            urgency: classification.urgency,
            affectedModules: classification.modules,
            competitiveImpact: classification.impact,
            implementationComplexity: classification.complexity,
            estimatedImplementationHours: classification.hours,
            sourceUrl: r.url,
            feedSource: "Exa Search",
          } satisfies LiveIndustryChange;
        })
        .filter((x) => x !== null) as LiveIndustryChange[];
    } finally {
      clearTimeout(timeout);
    }
  }

  private classifyArticle(
    title: string,
    description: string,
  ): {
    source: LiveIndustryChange["source"];
    category: LiveIndustryChange["category"];
    urgency: LiveIndustryChange["urgency"];
    modules: string[];
    impact: number;
    complexity: LiveIndustryChange["implementationComplexity"];
    hours: number;
  } | null {
    const text = `${title} ${description}`.toLowerCase();

    // Security — highest priority, check first
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

    // Regulation — music-specific copyright/licensing laws
    if (
      /copyright|dmca|gdpr|ccpa|royalt|compulsory license|mechanical license|neighbouring rights|eu.?copyright|digital services act|legislation|collect.?society|mma|music modernization/.test(
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

    // Streaming platforms — Spotify, Apple Music, DSPs
    if (
      /spotify|apple music|amazon music|youtube music|tidal|deezer|soundcloud|pandora|audiomack|boomplay|napster|qobuz/.test(
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
        /loudness|lufs|normali[sz]|codec|format|bitrate|lossless|dolby|spatial audio|atmos/.test(
          text,
        )
      ) {
        return {
          source: "streaming_platform",
          category: "standard",
          urgency: "high",
          modules: ["studio", "distribution"],
          impact: 75,
          complexity: "simple",
          hours: 8,
        };
      }
      if (
        /algorithm|discovery|playlist|recommend|stream count|royalt|payout|per.?stream/.test(
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
      if (/metadata|isrc|upc|artwork|release|label|distributor/.test(text)) {
        return {
          source: "streaming_platform",
          category: "standard",
          urgency: "medium",
          modules: ["distribution"],
          impact: 60,
          complexity: "trivial",
          hours: 4,
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

    // Social media — music-relevant creator/API changes
    if (
      /tiktok|instagram|reels|shorts|youtube|twitter|x\.com|facebook|threads|snapchat|creator|influencer/.test(
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
          source: "social_media",
          category: "optimization",
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
    }

    // Competitor DAWs and music production tools
    if (
      /fl studio|ableton|logic pro|pro tools|studio one|cubase|reaper|garageband|bitwig|reason|bandlab|soundtrap|splice|landr/.test(
        text,
      )
    ) {
      if (/update|release|version|feature|add|launch|new/.test(text)) {
        const isAI = /ai |artificial intelligence|machine learning|neural/.test(
          text,
        );
        return {
          source: "competitor",
          category: isAI ? "feature" : "feature",
          urgency: isAI ? "high" : "medium",
          modules: ["studio"],
          impact: isAI ? 80 : 60,
          complexity: isAI ? "complex" : "moderate",
          hours: isAI ? 60 : 30,
        };
      }
    }

    // Music production technology broadly
    if (
      /stem separat|vocal remov|ai master|ai mix|ai composi|neural audio|music generat|ai produc|plugin|vst|audio codec|midi 2|spatial audio|dolby atmos|sony 360/.test(
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

    // Royalty/revenue tech
    if (
      /royalt|blockchain|nft|web3|smart contract|publish|sync licens|revenue|payout|collect|distributor|aggregator/.test(
        text,
      )
    ) {
      return {
        source: "technology",
        category: "feature",
        urgency: "medium",
        modules: ["monetization", "distribution"],
        impact: 65,
        complexity: "moderate",
        hours: 30,
      };
    }

    // General AI in music
    if (
      /ai |artificial intelligence|machine learning|generative|gpt|llm/.test(
        text,
      ) &&
      /music|audio|song|track|artist|production|sound/.test(text)
    ) {
      return {
        source: "technology",
        category: "feature",
        urgency: "medium",
        modules: ["studio", "analytics"],
        impact: 72,
        complexity: "complex",
        hours: 45,
      };
    }

    // Music distribution competitors
    if (
      /distrokid|tunecore|cd baby|cdbaby|awal|unitedmasters|amuse|stem\.is|landr|bandcamp|routenote|ditto music|onerpm|believe digital|vydia|soundrop/.test(
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
      if (/ai |analytics|dashboard|reporting|insight|automat/.test(text)) {
        return {
          source: "competitor",
          category: "feature",
          urgency: "high",
          modules: ["analytics", "distribution"],
          impact: 90,
          complexity: "moderate",
          hours: 16,
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

    // Beat marketplace competitors
    if (
      /beatstars|airbit|soundclick|traktrain|beatbrokerz|soundee|rocbattle|soundgine/.test(
        text,
      )
    ) {
      if (
        /new feature|launch|release|announce|update|add|partnership/.test(text)
      ) {
        return {
          source: "competitor",
          category: "feature",
          urgency: "high",
          modules: ["marketplace", "monetization"],
          impact: 90,
          complexity: "moderate",
          hours: 18,
        };
      }
      if (/ai |analytics|pricing|automat|algorithm/.test(text)) {
        return {
          source: "competitor",
          category: "optimization",
          urgency: "high",
          modules: ["marketplace", "analytics"],
          impact: 88,
          complexity: "moderate",
          hours: 14,
        };
      }
      return {
        source: "competitor",
        category: "optimization",
        urgency: "medium",
        modules: ["marketplace"],
        impact: 70,
        complexity: "simple",
        hours: 8,
      };
    }

    // AI music creation competitors
    if (
      /suno ai|suno\.ai|udio\.com| udio |boomy|aiva\.ai| aiva |soundraw|beatoven|mubert|loudly\.com/.test(
        text,
      )
    ) {
      if (
        /new feature|launch|release|update|model|generate|announce/.test(text)
      ) {
        return {
          source: "competitor",
          category: "feature",
          urgency: "high",
          modules: ["studio", "analytics"],
          impact: 88,
          complexity: "complex",
          hours: 40,
        };
      }
      return {
        source: "competitor",
        category: "optimization",
        urgency: "medium",
        modules: ["studio"],
        impact: 75,
        complexity: "complex",
        hours: 30,
      };
    }

    // AI social media management competitors
    if (
      /hootsuite|sprout social|later\.com| buffer |metricool|planoly|vista social|publer/.test(
        text,
      )
    ) {
      if (/ai |new feature|launch|music|artist|creator/.test(text)) {
        return {
          source: "competitor",
          category: "feature",
          urgency: "high",
          modules: ["social", "advertising"],
          impact: 85,
          complexity: "moderate",
          hours: 16,
        };
      }
      return {
        source: "competitor",
        category: "optimization",
        urgency: "medium",
        modules: ["social"],
        impact: 68,
        complexity: "simple",
        hours: 8,
      };
    }

    // Music marketing tool competitors
    if (
      /submithub|groover|feature\.fm|hypeddit|linkfire|chartmetric|soundcharts|reverbnation|toneden|promoly/.test(
        text,
      )
    ) {
      if (/new feature|launch|release|update|ai |announce/.test(text)) {
        return {
          source: "competitor",
          category: "feature",
          urgency: "high",
          modules: ["analytics", "social", "distribution"],
          impact: 85,
          complexity: "moderate",
          hours: 14,
        };
      }
      return {
        source: "competitor",
        category: "optimization",
        urgency: "medium",
        modules: ["analytics", "social"],
        impact: 65,
        complexity: "simple",
        hours: 10,
      };
    }

    // DAW competitors — new AI features are highest priority
    if (
      /fl studio|ableton live|ableton|logic pro|pro tools|studio one|presonus|cubase|steinberg|reaper|bitwig|reason studios|garageband|cakewalk|adobe audition|soundtrap|bandlab|splice/.test(
        text,
      )
    ) {
      if (
        /ai |artificial intelligence|machine learning|neural|generate|stem|mastering|mixing/.test(
          text,
        )
      ) {
        return {
          source: "competitor",
          category: "feature",
          urgency: "high",
          modules: ["studio"],
          impact: 85,
          complexity: "complex",
          hours: 50,
        };
      }
      if (/new feature|update|release|version|launch/.test(text)) {
        return {
          source: "competitor",
          category: "feature",
          urgency: "medium",
          modules: ["studio"],
          impact: 68,
          complexity: "moderate",
          hours: 30,
        };
      }
      return {
        source: "competitor",
        category: "optimization",
        urgency: "low",
        modules: ["studio"],
        impact: 55,
        complexity: "simple",
        hours: 10,
      };
    }

    // Independent artist career tools broadly (catch anything we missed above)
    if (
      /independent artist|indie artist|unsigned artist|music career|artist platform|artist tool/.test(
        text,
      )
    ) {
      if (/feature|tool|launch|new|update/.test(text)) {
        return {
          source: "competitor",
          category: "feature",
          urgency: "medium",
          modules: ["analytics", "social", "distribution"],
          impact: 65,
          complexity: "simple",
          hours: 12,
        };
      }
    }

    return null;
  }

  private deduplicateByHash(
    changes: LiveIndustryChange[],
  ): LiveIndustryChange[] {
    const seen = new Set<string>();
    return changes?.filter((c) => {
      if (seen?.has(c?.id)) return false;
      seen?.add(c?.id);
      return true;
    });
  }

  private stripHtml(html: string): string {
    return html
      .replace(/<!\[CDATA\[(.*?)\]\]>/gs, "$1")
      .replace(/<[^>]+>/g, " ")
      .replace(/&amp;/g, "&")
      .replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">")
      .replace(/&quot;/g, '"')
      .replace(/&#039;/g, "'")
      .replace(/&nbsp;/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  clearCache(): void {
    this.cache = null;
  }

  /**
   * Returns the subset of cached changes that are competitor-sourced,
   * sorted by competitive impact descending.  Used by the evolution engine's
   * Phase 0 competitive leadership check.
   */
  getCompetitiveIntelligence(): LiveIndustryChange[] {
    if (!this.cache) return [];
    return this.cache.changes
      .filter((c) => c.source === "competitor")
      .sort((a, b) => b.competitiveImpact - a.competitiveImpact);
  }

  getStatus(): Record<string, unknown> {
    const competitive = this.getCompetitiveIntelligence();
    return {
      cacheAge: this.cache
        ? Math.round((Date.now() - this.cache.fetchedAt) / 1000) + "s"
        : "empty",
      cachedItems: this.cache?.changes?.length ?? 0,
      competitorItems: competitive.length,
      topCompetitorThreat: competitive[0]?.title ?? null,
      seenIds: this.seenIds.size,
      feeds: RSS_FEEDS.length,
      tavilyEnabled: !!process?.env.TAVILY_API_KEY,
      exaEnabled: !!process?.env.EXA_API_KEY,
    };
  }
}

export const industryMonitor = new IndustryMonitorService();
