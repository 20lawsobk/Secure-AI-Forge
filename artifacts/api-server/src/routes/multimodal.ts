import { Router, type IRouter, type Request, type Response } from "express";
import { randomUUID } from "crypto";
import { Agent, fetch as undiciFetch } from "undici";
import platformRules from "../platform_rules.json";
import {
  contentAwarenessService,
  type ContentAwarenessContext,
} from "../services/contentAwarenessService.js";
import {
  buildGenerationEnrichment,
  type GenerationEnrichment,
} from "../services/autoPostGenerator.js";
import { MAXCORE_URL, MAXCORE_API_KEY } from "../config/maxcore.js";

const router: IRouter = Router();

// ─── Model connection pool ──────────────────────────────────────────────────
// Multimodal generation fans out to several upstream model calls per request
// (analyze + N per-asset generations). Under concurrent load a single asset
// can legitimately sit in-flight for minutes. The bare global `fetch` (undici)
// applies a default 300s headersTimeout, which aborts valid in-flight model
// calls and surfaces as a 500. This pool raises that ceiling well past the
// slowest observed multimodal render so honest, still-working requests are
// never killed mid-flight. Same keep-alive pattern used by model-proxy.ts.
const _modelPool = new Agent({
  keepAliveTimeout: 30_000,
  keepAliveMaxTimeout: 60_000,
  connections: 32,
  pipelining: 1,
  headersTimeout: 900_000,
  bodyTimeout: 900_000,
});

// ─── Types ────────────────────────────────────────────────────────────────────

type InputModality = "text" | "url" | "image" | "audio" | "video";
type OutputModality = "text" | "image" | "audio" | "video";

type Platform =
  | "facebook"
  | "instagram"
  | "threads"
  | "tiktok"
  | "youtube"
  | "google_business"
  | "linkedin";

type PackId =
  | "singlereleasefull_pack"
  | "announcement_pack"
  | "tourdates_pack"
  | "evergreenbrand_pack";

interface PlatformAssetSpec {
  id: string;
  platform: Platform;
  modality: OutputModality;
  purpose: string;
}

interface GenerationRequest {
  id: string;
  userId: string;
  artistProfileId?: string;
  input: {
    modality: InputModality;
    payload: string;
    metadata?: Record<string, unknown>;
  };
  platforms: Platform[];
  packId?: PackId;
  intent?: string;
  constraints?: {
    length?: "short" | "medium" | "long";
    styleTags?: string[];
    language?: string;
  };
}

interface GeneratedAsset {
  id: string;
  modality: OutputModality;
  payload: string;
  platform?: Platform;
  slotId?: string;
  metadata?: Record<string, unknown>;
}

interface TaskStep {
  id: string;
  type: "analyze" | "generate";
  worker: "text" | "image" | "audio" | "video";
  inputFrom: "normalizedInput" | string[];
  params?: Record<string, unknown>;
}

interface TaskPlan {
  requestId: string;
  steps: TaskStep[];
}

interface MultimodalPackage {
  requestId: string;
  assets: GeneratedAsset[];
  plan: TaskPlan;
}

// ─── Pack Definitions ─────────────────────────────────────────────────────────

const PACK_DEFINITIONS: Record<PackId, PlatformAssetSpec[]> = {
  singlereleasefull_pack: [
    {
      id: "fb_post",
      platform: "facebook",
      modality: "text",
      purpose: "Main FB post copy",
    },
    {
      id: "ig_caption",
      platform: "instagram",
      modality: "text",
      purpose: "IG feed caption",
    },
    {
      id: "threads_post",
      platform: "threads",
      modality: "text",
      purpose: "Threads announcement",
    },
    {
      id: "tt_caption",
      platform: "tiktok",
      modality: "text",
      purpose: "TikTok caption + hashtags",
    },
    {
      id: "yt_description",
      platform: "youtube",
      modality: "text",
      purpose: "YouTube description",
    },
    {
      id: "yt_title",
      platform: "youtube",
      modality: "text",
      purpose: "YouTube title options",
    },
    {
      id: "gb_post",
      platform: "google_business",
      modality: "text",
      purpose: "Google Business update",
    },
    {
      id: "li_post",
      platform: "linkedin",
      modality: "text",
      purpose: "Professional angle post",
    },
    {
      id: "cover_image",
      platform: "instagram",
      modality: "image",
      purpose: "Cover/thumbnail cross-platform",
    },
    {
      id: "story_background",
      platform: "instagram",
      modality: "image",
      purpose: "Story background art",
    },
    {
      id: "tt_voiceover_audio",
      platform: "tiktok",
      modality: "audio",
      purpose: "Voiceover audio for TikTok short",
    },
    {
      id: "yt_voiceover_audio",
      platform: "youtube",
      modality: "audio",
      purpose: "Voiceover for teaser video",
    },
    {
      id: "tt_short_video",
      platform: "tiktok",
      modality: "video",
      purpose: "Vertical short teaser",
    },
    {
      id: "yt_short_video",
      platform: "youtube",
      modality: "video",
      purpose: "YouTube Short teaser",
    },
  ],

  announcement_pack: [
    {
      id: "fb_post",
      platform: "facebook",
      modality: "text",
      purpose: "FB announcement copy",
    },
    {
      id: "ig_caption",
      platform: "instagram",
      modality: "text",
      purpose: "IG announcement caption",
    },
    {
      id: "threads_post",
      platform: "threads",
      modality: "text",
      purpose: "Threads announcement",
    },
    {
      id: "tt_caption",
      platform: "tiktok",
      modality: "text",
      purpose: "TikTok caption",
    },
    {
      id: "yt_description",
      platform: "youtube",
      modality: "text",
      purpose: "YouTube description",
    },
    {
      id: "li_post",
      platform: "linkedin",
      modality: "text",
      purpose: "LinkedIn announcement",
    },
    {
      id: "cover_image",
      platform: "instagram",
      modality: "image",
      purpose: "Announcement visual",
    },
  ],

  tourdates_pack: [
    {
      id: "fb_post",
      platform: "facebook",
      modality: "text",
      purpose: "Tour dates FB post",
    },
    {
      id: "ig_caption",
      platform: "instagram",
      modality: "text",
      purpose: "Tour dates IG caption",
    },
    {
      id: "threads_post",
      platform: "threads",
      modality: "text",
      purpose: "Tour dates Threads post",
    },
    {
      id: "tt_caption",
      platform: "tiktok",
      modality: "text",
      purpose: "TikTok tour hype caption",
    },
    {
      id: "gb_post",
      platform: "google_business",
      modality: "text",
      purpose: "Google Business event post",
    },
    {
      id: "tour_poster",
      platform: "instagram",
      modality: "image",
      purpose: "Tour poster — cross-platform",
    },
    {
      id: "fb_event_image",
      platform: "facebook",
      modality: "image",
      purpose: "Facebook event cover image",
    },
    {
      id: "tt_hype_video",
      platform: "tiktok",
      modality: "video",
      purpose: "Short hype clip for tour",
    },
  ],

  evergreenbrand_pack: [
    {
      id: "fb_post",
      platform: "facebook",
      modality: "text",
      purpose: "Evergreen brand story",
    },
    {
      id: "ig_caption",
      platform: "instagram",
      modality: "text",
      purpose: "Brand aesthetic caption",
    },
    {
      id: "threads_post",
      platform: "threads",
      modality: "text",
      purpose: "Conversational brand post",
    },
    {
      id: "li_post",
      platform: "linkedin",
      modality: "text",
      purpose: "Professional brand statement",
    },
    {
      id: "brand_image",
      platform: "instagram",
      modality: "image",
      purpose: "Brand visual identity",
    },
    {
      id: "yt_thumbnail",
      platform: "youtube",
      modality: "image",
      purpose: "YouTube channel art",
    },
    {
      id: "brand_audio",
      platform: "youtube",
      modality: "audio",
      purpose: "Brand voiceover / intro",
    },
  ],
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

type PlatformRules = typeof platformRules;

function getPlatformRules(
  platform: string,
): PlatformRules[keyof PlatformRules] | null {
  return (
    (platformRules as Record<string, PlatformRules[keyof PlatformRules]>)[
      platform
    ] ?? null
  );
}

function safeExtractJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    const match = text.match(/\{[\s\S]*\}/);
    if (match) {
      try {
        return JSON.parse(match[0]);
      } catch {
        /* fall through */
      }
    }
    return null;
  }
}

function validateTaskPlan(raw: unknown, requestId: string): TaskPlan {
  if (
    raw &&
    typeof raw === "object" &&
    "steps" in raw &&
    Array.isArray((raw as TaskPlan).steps)
  ) {
    return raw as TaskPlan;
  }
  return { requestId, steps: [] };
}

async function maxcorePost(path: string, body: unknown): Promise<unknown> {
  const res = await undiciFetch(`${MAXCORE_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Api-Key": MAXCORE_API_KEY,
    },
    body: JSON.stringify(body),
    dispatcher: _modelPool,
  });
  if (!res.ok) {
    const err = await res.text().catch(() => res.statusText);
    throw new Error(`maxcore ${path} → ${res.status}: ${err}`);
  }
  return res.json();
}

// ─── Step 1: Normalize input via maxcore /analyze ─────────────────────────────

async function normalizeInput(
  req: GenerationRequest,
  enrichment: GenerationEnrichment,
): Promise<unknown> {
  return maxcorePost("/analyze", {
    modality: req.input.modality,
    payload: req.input.payload,
    artistProfileId: req.artistProfileId,
    platforms: req.platforms,
    intent: req.intent,
    ...(enrichment.awarenessBlock
      ? { awareness: enrichment.awarenessBlock }
      : {}),
  });
}

// ─── Step 2: Plan tasks via maxcore /generate/text (mode=planner) ─────────────

async function planTasks(
  normalized: unknown,
  req: GenerationRequest,
): Promise<TaskPlan> {
  const packSpec = req.packId ? PACK_DEFINITIONS[req.packId] : null;

  const raw = await maxcorePost("/generate/text", {
    mode: "planner",
    input: {
      normalized,
      request: req,
      packSpec,
    },
  });

  const plan = validateTaskPlan(raw, req.id);

  if (plan.steps.length === 0 && packSpec) {
    return buildFallbackPlan(req.id, normalized, req, packSpec);
  }
  return plan;
}

function buildFallbackPlan(
  requestId: string,
  normalized: unknown,
  req: GenerationRequest,
  packSpec: PlatformAssetSpec[],
): TaskPlan {
  const steps: TaskStep[] = [
    {
      id: "analysis_step",
      type: "analyze",
      worker: "text",
      inputFrom: "normalizedInput",
      params: { intent: req.intent ?? "engagement" },
    },
  ];

  const byModality: Record<string, PlatformAssetSpec[]> = {};
  for (const slot of packSpec) {
    (byModality[slot.modality] ??= []).push(slot);
  }

  for (const [modality, slots] of Object.entries(byModality)) {
    steps.push({
      id: `step_${modality}`,
      type: "generate",
      worker: modality as TaskStep["worker"],
      inputFrom: ["analysis_step"],
      params: {
        slots,
        platforms: slots.map((s) => s.platform),
        constraints: req.constraints ?? {},
      },
    });
  }

  return { requestId, steps };
}

// ─── Workers ──────────────────────────────────────────────────────────────────

// The final awareness string sent to MaxCore (enrichment block + live signals).
type WorkerAwareness = string | null;

const textWorker = {
  async run(
    step: TaskStep,
    inputs: unknown,
    awareness: WorkerAwareness,
  ): Promise<GeneratedAsset[]> {
    const result = (await maxcorePost("/generate/text", {
      mode: "content",
      step,
      inputs,
      ...(awareness ? { awareness } : {}),
    })) as {
      outputs: Array<{
        text: string;
        platform: Platform;
        slotId: string;
        meta: Record<string, unknown>;
      }>;
    };

    return (result.outputs ?? []).map((o) => ({
      id: randomUUID(),
      modality: "text" as OutputModality,
      payload: o.text,
      platform: o.platform,
      slotId: o.slotId,
      metadata: o.meta ?? {},
    }));
  },
};

const imageWorker = {
  async run(
    step: TaskStep,
    inputs: unknown,
    awareness: WorkerAwareness,
  ): Promise<GeneratedAsset[]> {
    const result = (await maxcorePost("/generate/image", {
      step,
      inputs,
      ...(awareness ? { awareness } : {}),
    })) as {
      outputs: Array<{
        url: string;
        platform: Platform;
        slotId: string;
        meta: Record<string, unknown>;
      }>;
    };

    return (result.outputs ?? []).map((o) => ({
      id: randomUUID(),
      modality: "image" as OutputModality,
      payload: o.url,
      platform: o.platform,
      slotId: o.slotId,
      metadata: o.meta ?? {},
    }));
  },
};

const audioWorker = {
  async run(
    step: TaskStep,
    inputs: unknown,
    awareness: WorkerAwareness,
  ): Promise<GeneratedAsset[]> {
    const result = (await maxcorePost("/generate/audio", {
      step,
      inputs,
      ...(awareness ? { awareness } : {}),
    })) as {
      outputs: Array<{
        url: string;
        platform: Platform;
        slotId: string;
        meta: Record<string, unknown>;
      }>;
    };

    return (result.outputs ?? []).map((o) => ({
      id: randomUUID(),
      modality: "audio" as OutputModality,
      payload: o.url,
      platform: o.platform,
      slotId: o.slotId,
      metadata: o.meta ?? {},
    }));
  },
};

const videoWorker = {
  async run(
    step: TaskStep,
    inputs: unknown,
    awareness: WorkerAwareness,
  ): Promise<GeneratedAsset[]> {
    const result = (await maxcorePost("/generate/video", {
      step,
      inputs,
      ...(awareness ? { awareness } : {}),
    })) as {
      outputs: Array<{
        url: string;
        platform: Platform;
        slotId: string;
        meta: Record<string, unknown>;
      }>;
    };

    return (result.outputs ?? []).map((o) => ({
      id: randomUUID(),
      modality: "video" as OutputModality,
      payload: o.url,
      platform: o.platform,
      slotId: o.slotId,
      metadata: o.meta ?? {},
    }));
  },
};

const workers: Record<
  string,
  {
    run: (
      step: TaskStep,
      inputs: unknown,
      awareness: WorkerAwareness,
    ) => Promise<GeneratedAsset[]>;
  }
> = {
  text: textWorker,
  image: imageWorker,
  audio: audioWorker,
  video: videoWorker,
};

// ─── Orchestrator ─────────────────────────────────────────────────────────────

async function handleGeneration(
  req: GenerationRequest,
): Promise<MultimodalPackage> {
  // Assemble generation-time enrichment (artist profile, releases, proven hook
  // patterns, trending) from the platform data layer. Guarded so it never
  // blocks or fails the pipeline; absent data is simply omitted.
  const enrichment = await buildGenerationEnrichment({
    userId: req.userId,
    artistProfileId: req.artistProfileId,
    platforms: req.platforms,
  }).catch<GenerationEnrichment>(() => ({
    awarenessBlock: "",
    hasData: false,
  }));

  const normalized = await normalizeInput(req, enrichment);
  const plan = await planTasks(normalized, req);

  // Fetch per-modality awareness contexts in parallel, racing against a 3 s
  // guard so a cold-cache RSS fetch never delays the generation pipeline.
  const AWARENESS_TIMEOUT_MS = 3_000;
  const awarenessRace = <T>(p: Promise<T>): Promise<T | null> =>
    Promise.race([
      p,
      new Promise<null>((resolve) =>
        setTimeout(() => resolve(null), AWARENESS_TIMEOUT_MS),
      ),
    ]);

  const [textAwareness, imageAwareness, audioAwareness, videoAwareness] =
    await Promise.all([
      awarenessRace(
        contentAwarenessService.getContextForMode("social").catch(() => null),
      ),
      awarenessRace(
        contentAwarenessService.getContextForMode("content").catch(() => null),
      ),
      awarenessRace(
        contentAwarenessService.getContextForMode("music").catch(() => null),
      ),
      awarenessRace(
        contentAwarenessService
          .getContextForMode("video_script")
          .catch(() => null),
      ),
    ]);

  // Merge the enrichment block with live per-modality awareness into a single
  // awareness string per worker. Enrichment leads so artist/release/hook
  // context is the first signal the model conditions on.
  const mergeAwareness = (
    ctx: ContentAwarenessContext | null,
  ): WorkerAwareness => {
    const ctxStr = ctx?.confidence ? ctx.contextString : "";
    const merged = [enrichment.awarenessBlock, ctxStr]
      .filter(Boolean)
      .join("\n\n");
    return merged || null;
  };

  const awarenessMap: Record<string, WorkerAwareness> = {
    text: mergeAwareness(textAwareness),
    image: mergeAwareness(imageAwareness),
    audio: mergeAwareness(audioAwareness),
    video: mergeAwareness(videoAwareness),
  };

  const stepOutputs = new Map<string, GeneratedAsset[]>();

  // Separate steps into dependency tiers so independent work runs in parallel.
  // A step is "ready" when all its inputFrom dependencies are resolved.
  const pending = plan.steps.filter((s) => s.type !== "analyze");
  const completed = new Set<string>(["normalizedInput"]);
  // Analyze steps are folded into the normalized input and excluded from
  // `pending`; mark their ids resolved so generate steps that declare them in
  // `inputFrom` (e.g. ["analysis_step"]) become ready instead of deadlocking.
  for (const step of plan.steps) {
    if (step.type === "analyze") completed.add(step.id);
  }

  while (pending.length > 0) {
    // Find all steps whose inputs are fully resolved
    const ready = pending.filter((step) => {
      const deps =
        step.inputFrom === "normalizedInput"
          ? []
          : (step.inputFrom as string[]);
      return deps.every((d) => completed.has(d));
    });

    if (ready.length === 0) break; // avoid infinite loop on malformed plans

    // Run all ready steps concurrently
    await Promise.all(
      ready.map(async (step) => {
        const worker = workers[step.worker];
        if (!worker) {
          completed.add(step.id);
          return;
        }

        const inputs =
          step.inputFrom === "normalizedInput"
            ? { normalized }
            : {
                normalized,
                prior: (step.inputFrom as string[]).flatMap(
                  (id) => stepOutputs.get(id) ?? [],
                ),
              };

        const assets = await worker.run(
          step,
          inputs,
          awarenessMap[step.worker] ?? null,
        );
        stepOutputs.set(step.id, assets);
        completed.add(step.id);
      }),
    );

    // Remove completed steps from pending
    for (const step of ready) {
      const idx = pending.indexOf(step);
      if (idx !== -1) pending.splice(idx, 1);
    }
  }

  return {
    requestId: req.id,
    assets: Array.from(stepOutputs.values()).flat(),
    plan,
  };
}

// ─── pdim helpers ─────────────────────────────────────────────────────────────

async function fetchCurriculumStyleTags(
  userId: string,
  platforms: Platform[],
): Promise<string[]> {
  try {
    const res = await undiciFetch(
      `${MAXCORE_URL}/storage/curriculum/${encodeURIComponent(userId)}`,
      {
        headers: { "X-Api-Key": MAXCORE_API_KEY },
        dispatcher: _modelPool,
      },
    );
    if (!res.ok) return [];
    const data = (await res.json()) as {
      top_performers?: Array<{ style_tags?: string[]; platform?: string }>;
    };
    const topPerformers = data.top_performers ?? [];
    const platformSet = new Set(platforms as string[]);
    const relevant = topPerformers.filter(
      (p) => !p.platform || platformSet.has(p.platform),
    );
    const tags = (relevant.length > 0 ? relevant : topPerformers)
      .flatMap((p) => p.style_tags ?? [])
      .filter(Boolean);
    return [...new Set(tags)].slice(0, 8);
  } catch {
    return [];
  }
}

function recordGenerationToFlywheel(
  pkg: MultimodalPackage,
  req: GenerationRequest,
): void {
  const seen = new Set<string>();
  for (const asset of pkg.assets) {
    if (!asset.platform) continue;
    const key = `${asset.platform}:${asset.modality}`;
    if (seen.has(key)) continue;
    seen.add(key);
    undiciFetch(`${MAXCORE_URL}/storage/feedback`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Api-Key": MAXCORE_API_KEY,
      },
      body: JSON.stringify({
        user_id: req.userId,
        platform: asset.platform,
        engagement_rate: 0.5,
        content_type: asset.modality,
        style_tags: [
          ...(req.constraints?.styleTags ?? []),
          req.intent ?? "engagement",
          asset.modality,
        ].filter(Boolean),
      }),
      dispatcher: _modelPool,
    }).catch(() => {});
  }
}

// ─── Routes ───────────────────────────────────────────────────────────────────

router.get("/multimodal/packs", (_req: Request, res: Response) => {
  const summary: Record<
    string,
    { slotCount: number; modalities: string[]; platforms: string[] }
  > = {};
  for (const [packId, slots] of Object.entries(PACK_DEFINITIONS)) {
    summary[packId] = {
      slotCount: slots.length,
      modalities: [...new Set(slots.map((s) => s.modality))],
      platforms: [...new Set(slots.map((s) => s.platform))],
    };
  }
  res.json({
    packs: summary,
    packIds: Object.keys(PACK_DEFINITIONS),
    platformRules: Object.keys(platformRules),
  });
});

router.post("/multimodal/generate", async (req: Request, res: Response) => {
  const body = req.body as Partial<GenerationRequest>;

  if (
    !body.id ||
    !body.userId ||
    !body.input?.payload ||
    !body.platforms?.length
  ) {
    res.status(400).json({
      error: "Missing required fields: id, userId, input.payload, platforms",
    });
    return;
  }

  const curriculumTags = await fetchCurriculumStyleTags(
    body.userId,
    body.platforms,
  );

  const genReq: GenerationRequest = {
    id: body.id,
    userId: body.userId,
    artistProfileId: body.artistProfileId,
    input: {
      modality: body.input.modality ?? "text",
      payload: body.input.payload,
      metadata: body.input.metadata,
    },
    platforms: body.platforms,
    packId: body.packId,
    intent: body.intent,
    constraints: {
      ...body.constraints,
      styleTags: [
        ...new Set([...(body.constraints?.styleTags ?? []), ...curriculumTags]),
      ],
    },
  };

  try {
    const pkg = await handleGeneration(genReq);
    recordGenerationToFlywheel(pkg, genReq);
    res.json(pkg);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    res.status(500).json({ error: "Generation failed", detail: message });
  }
});

export default router;
