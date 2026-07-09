/**
 * Auto Post Generator — generation-time enrichment assembler (Parts 1 & 2).
 *
 * Gathers the artist's real context from the platform data layer (MaxCore pdim
 * storage) and composes it into a structured `awareness` block for the MaxCore
 * generation call:
 *   1. Artist name, current single/album, audience age/geo   (artist profile)
 *   2. Release / streaming context                            (artist releases)
 *   3. Top-performing historical hook patterns                (curriculum feedback)
 *   4. Trending tokens                                        (trendingContextService)
 *
 * Everything is routed through the `awareness` channel — the one input MaxCore's
 * ScriptAgent actually conditions on — rather than the raw topic/idea, which
 * would corrupt the user-facing caption. Sections with no backing data are
 * omitted entirely: this never fabricates enrichment.
 */
import { Agent, fetch as undiciFetch } from "undici";
import { MAXCORE_URL, MAXCORE_API_KEY } from "../config/maxcore.js";
import { getTrendingTokens } from "./trendingContextService.js";

const _enrichPool = new Agent({
  keepAliveTimeout: 30_000,
  keepAliveMaxTimeout: 60_000,
  connections: 8,
  headersTimeout: 15_000,
  bodyTimeout: 15_000,
});

const ENRICH_TIMEOUT_MS = 2_500;

export interface GenerationEnrichment {
  /** Structured context block prepended to the awareness contextString. */
  awarenessBlock: string;
  hasData: boolean;
}

interface ArtistProfile {
  artist_name?: string;
  current_single?: string;
  current_album?: string;
  audience_age?: string;
  audience_geo?: string;
}

interface ReleaseRecord {
  title?: string;
  kind?: string;
  release_date?: string;
  streaming_url?: string;
  status?: string;
  platforms?: string[];
}

interface ArtistEnrichmentResponse {
  profile?: ArtistProfile;
  releases?: ReleaseRecord[];
}

interface CurriculumResponse {
  top_performers?: Array<{
    style_tags?: string[];
    platform?: string;
    engagement_rate?: number;
    content_type?: string;
  }>;
}

const EMPTY: GenerationEnrichment = { awarenessBlock: "", hasData: false };

async function fetchJson<T>(url: string): Promise<T | null> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), ENRICH_TIMEOUT_MS);
    try {
      const res = await undiciFetch(url, {
        headers: { "X-Api-Key": MAXCORE_API_KEY },
        dispatcher: _enrichPool,
        signal: controller.signal,
      });
      if (!res.ok) return null;
      return (await res.json()) as T;
    } finally {
      clearTimeout(timer);
    }
  } catch {
    return null;
  }
}

/**
 * Build generation-time enrichment for a request. Never throws; returns an
 * empty (no-op) enrichment when nothing is available.
 */
export async function buildGenerationEnrichment(params: {
  userId: string;
  artistProfileId?: string;
  platforms: string[];
}): Promise<GenerationEnrichment> {
  const { userId, artistProfileId, platforms } = params;

  try {
    const [artist, curriculum, trendingTokens] = await Promise.all([
      artistProfileId
        ? fetchJson<ArtistEnrichmentResponse>(
            `${MAXCORE_URL}/storage/artist/${encodeURIComponent(artistProfileId)}`,
          )
        : Promise.resolve(null),
      userId
        ? fetchJson<CurriculumResponse>(
            `${MAXCORE_URL}/storage/curriculum/${encodeURIComponent(userId)}`,
          )
        : Promise.resolve(null),
      getTrendingTokens(platforms).catch(() => [] as string[]),
    ]);

    const sections: string[] = [];

    // ── 1. Artist context ────────────────────────────────────────────────────
    const profile = artist?.profile;
    if (
      profile &&
      (profile.artist_name ||
        profile.current_single ||
        profile.current_album ||
        profile.audience_age ||
        profile.audience_geo)
    ) {
      const lines = ["=== ARTIST CONTEXT ==="];
      if (profile.artist_name) lines.push(`Artist: ${profile.artist_name}`);
      if (profile.current_single) lines.push(`Current single: ${profile.current_single}`);
      if (profile.current_album) lines.push(`Current album: ${profile.current_album}`);
      const audienceBits: string[] = [];
      if (profile.audience_age) audienceBits.push(`age ${profile.audience_age}`);
      if (profile.audience_geo) audienceBits.push(profile.audience_geo);
      if (audienceBits.length) lines.push(`Audience: ${audienceBits.join(", ")}`);
      sections.push(lines.join("\n"));
    }

    // ── 2. Release / streaming context ───────────────────────────────────────
    const releases = (artist?.releases ?? []).filter((r) => r && r.title);
    if (releases.length) {
      const lines = ["=== RELEASE CONTEXT ==="];
      for (const r of releases.slice(0, 5)) {
        const meta: string[] = [];
        if (r.kind) meta.push(r.kind);
        if (r.status) meta.push(r.status);
        if (r.release_date) meta.push(r.release_date);
        let line = meta.length ? `${r.title} (${meta.join(", ")})` : `${r.title}`;
        if (r.streaming_url) line += ` — ${r.streaming_url}`;
        lines.push(`- ${line}`);
      }
      sections.push(lines.join("\n"));
    }

    // ── 3. Top-performing hook patterns (this artist's post history) ──────────
    const performers = curriculum?.top_performers ?? [];
    if (performers.length) {
      const platformSet = new Set(platforms);
      const relevant = performers.filter((p) => !p.platform || platformSet.has(p.platform));
      const chosen = (relevant.length ? relevant : performers).slice(0, 5);
      const tagSet = new Set<string>();
      for (const p of chosen) {
        for (const t of p.style_tags ?? []) {
          if (t && t.trim()) tagSet.add(t.trim());
        }
      }
      const tags = [...tagSet].slice(0, 10);
      if (tags.length) {
        sections.push(
          [
            "=== TOP-PERFORMING HOOK PATTERNS (this artist) ===",
            `Proven styles: ${tags.join(", ")}`,
          ].join("\n"),
        );
      }
    }

    // ── 4. Trending (Part 3) ─────────────────────────────────────────────────
    if (trendingTokens.length) {
      const hashtags = trendingTokens
        .map((t) => `#${t.replace(/[^A-Za-z0-9]/g, "")}`)
        .filter((h) => h.length > 1);
      sections.push(
        ["=== TRENDING NOW ===", `Trending: ${trendingTokens.join(", ")}`, hashtags.join(" ")]
          .filter(Boolean)
          .join("\n"),
      );
    }

    if (!sections.length) return EMPTY;
    return { awarenessBlock: sections.join("\n\n"), hasData: true };
  } catch {
    return EMPTY;
  }
}
