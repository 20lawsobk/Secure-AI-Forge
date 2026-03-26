import { Router, type IRouter, type Request, type Response } from "express";
import { randomUUID } from "crypto";
import platformRules from "../platform_rules.json";

const router: IRouter = Router();

const MAXCORE_URL = `http://localhost:${process.env.MODEL_API_PORT || "9878"}`;
const MAXCORE_API_KEY = process.env.ADMIN_KEY || "mbs_8a3edbac97ff333dda5068410227267e6d85b14a4c9caee279fbb18ddfb47edc";

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
    { id: "fb_post",            platform: "facebook",       modality: "text",  purpose: "Main FB post copy" },
    { id: "ig_caption",         platform: "instagram",      modality: "text",  purpose: "IG feed caption" },
    { id: "threads_post",       platform: "threads",        modality: "text",  purpose: "Threads announcement" },
    { id: "tt_caption",         platform: "tiktok",         modality: "text",  purpose: "TikTok caption + hashtags" },
    { id: "yt_description",     platform: "youtube",        modality: "text",  purpose: "YouTube description" },
    { id: "yt_title",           platform: "youtube",        modality: "text",  purpose: "YouTube title options" },
    { id: "gb_post",            platform: "google_business",modality: "text",  purpose: "Google Business update" },
    { id: "li_post",            platform: "linkedin",       modality: "text",  purpose: "Professional angle post" },
    { id: "cover_image",        platform: "instagram",      modality: "image", purpose: "Cover/thumbnail cross-platform" },
    { id: "story_background",   platform: "instagram",      modality: "image", purpose: "Story background art" },
    { id: "tt_voiceover_audio", platform: "tiktok",         modality: "audio", purpose: "Voiceover audio for TikTok short" },
    { id: "yt_voiceover_audio", platform: "youtube",        modality: "audio", purpose: "Voiceover for teaser video" },
    { id: "tt_short_video",     platform: "tiktok",         modality: "video", purpose: "Vertical short teaser" },
    { id: "yt_short_video",     platform: "youtube",        modality: "video", purpose: "YouTube Short teaser" },
  ],

  announcement_pack: [
    { id: "fb_post",        platform: "facebook",       modality: "text",  purpose: "FB announcement copy" },
    { id: "ig_caption",     platform: "instagram",      modality: "text",  purpose: "IG announcement caption" },
    { id: "threads_post",   platform: "threads",        modality: "text",  purpose: "Threads announcement" },
    { id: "tt_caption",     platform: "tiktok",         modality: "text",  purpose: "TikTok caption" },
    { id: "yt_description", platform: "youtube",        modality: "text",  purpose: "YouTube description" },
    { id: "li_post",        platform: "linkedin",       modality: "text",  purpose: "LinkedIn announcement" },
    { id: "cover_image",    platform: "instagram",      modality: "image", purpose: "Announcement visual" },
  ],

  tourdates_pack: [
    { id: "fb_post",        platform: "facebook",       modality: "text",  purpose: "Tour dates FB post" },
    { id: "ig_caption",     platform: "instagram",      modality: "text",  purpose: "Tour dates IG caption" },
    { id: "threads_post",   platform: "threads",        modality: "text",  purpose: "Tour dates Threads post" },
    { id: "tt_caption",     platform: "tiktok",         modality: "text",  purpose: "TikTok tour hype caption" },
    { id: "gb_post",        platform: "google_business",modality: "text",  purpose: "Google Business event post" },
    { id: "tour_poster",    platform: "instagram",      modality: "image", purpose: "Tour poster — cross-platform" },
    { id: "fb_event_image", platform: "facebook",       modality: "image", purpose: "Facebook event cover image" },
    { id: "tt_hype_video",  platform: "tiktok",         modality: "video", purpose: "Short hype clip for tour" },
  ],

  evergreenbrand_pack: [
    { id: "fb_post",        platform: "facebook",       modality: "text",  purpose: "Evergreen brand story" },
    { id: "ig_caption",     platform: "instagram",      modality: "text",  purpose: "Brand aesthetic caption" },
    { id: "threads_post",   platform: "threads",        modality: "text",  purpose: "Conversational brand post" },
    { id: "li_post",        platform: "linkedin",       modality: "text",  purpose: "Professional brand statement" },
    { id: "brand_image",    platform: "instagram",      modality: "image", purpose: "Brand visual identity" },
    { id: "yt_thumbnail",   platform: "youtube",        modality: "image", purpose: "YouTube channel art" },
    { id: "brand_audio",    platform: "youtube",        modality: "audio", purpose: "Brand voiceover / intro" },
  ],
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

type PlatformRules = typeof platformRules;

function getPlatformRules(platform: string): PlatformRules[keyof PlatformRules] | null {
  return (platformRules as Record<string, PlatformRules[keyof PlatformRules]>)[platform] ?? null;
}

function safeExtractJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    const match = text.match(/\{[\s\S]*\}/);
    if (match) {
      try { return JSON.parse(match[0]); } catch { /* fall through */ }
    }
    return null;
  }
}

function validateTaskPlan(raw: unknown, requestId: string): TaskPlan {
  if (raw && typeof raw === "object" && "steps" in raw && Array.isArray((raw as TaskPlan).steps)) {
    return raw as TaskPlan;
  }
  return { requestId, steps: [] };
}

async function maxcorePost(path: string, body: unknown): Promise<unknown> {
  const res = await fetch(`${MAXCORE_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Api-Key": MAXCORE_API_KEY,
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.text().catch(() => res.statusText);
    throw new Error(`maxcore ${path} → ${res.status}: ${err}`);
  }
  return res.json();
}

// ─── Step 1: Normalize input via maxcore /analyze ─────────────────────────────

async function normalizeInput(req: GenerationRequest): Promise<unknown> {
  return maxcorePost("/analyze", {
    modality: req.input.modality,
    payload: req.input.payload,
    artistProfileId: req.artistProfileId,
    platforms: req.platforms,
    intent: req.intent,
  });
}

// ─── Step 2: Plan tasks via maxcore /generate/text (mode=planner) ─────────────

async function planTasks(normalized: unknown, req: GenerationRequest): Promise<TaskPlan> {
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

const textWorker = {
  async run(step: TaskStep, inputs: unknown): Promise<GeneratedAsset[]> {
    const result = (await maxcorePost("/generate/text", {
      mode: "content",
      step,
      inputs,
    })) as { outputs: Array<{ text: string; platform: Platform; slotId: string; meta: Record<string, unknown> }> };

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
  async run(step: TaskStep, inputs: unknown): Promise<GeneratedAsset[]> {
    const result = (await maxcorePost("/generate/image", {
      step,
      inputs,
    })) as { outputs: Array<{ url: string; platform: Platform; slotId: string; meta: Record<string, unknown> }> };

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
  async run(step: TaskStep, inputs: unknown): Promise<GeneratedAsset[]> {
    const result = (await maxcorePost("/generate/audio", {
      step,
      inputs,
    })) as { outputs: Array<{ url: string; platform: Platform; slotId: string; meta: Record<string, unknown> }> };

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
  async run(step: TaskStep, inputs: unknown): Promise<GeneratedAsset[]> {
    const result = (await maxcorePost("/generate/video", {
      step,
      inputs,
    })) as { outputs: Array<{ url: string; platform: Platform; slotId: string; meta: Record<string, unknown> }> };

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

const workers: Record<string, { run: (step: TaskStep, inputs: unknown) => Promise<GeneratedAsset[]> }> = {
  text: textWorker,
  image: imageWorker,
  audio: audioWorker,
  video: videoWorker,
};

// ─── Orchestrator ─────────────────────────────────────────────────────────────

async function handleGeneration(req: GenerationRequest): Promise<MultimodalPackage> {
  const normalized = await normalizeInput(req);
  const plan = await planTasks(normalized, req);

  const stepOutputs = new Map<string, GeneratedAsset[]>();

  for (const step of plan.steps) {
    if (step.type === "analyze") continue;

    const worker = workers[step.worker];
    if (!worker) continue;

    const inputs =
      step.inputFrom === "normalizedInput"
        ? { normalized }
        : {
            normalized,
            prior: (step.inputFrom as string[]).flatMap((id) => stepOutputs.get(id) ?? []),
          };

    const assets = await worker.run(step, inputs);
    stepOutputs.set(step.id, assets);
  }

  return {
    requestId: req.id,
    assets: Array.from(stepOutputs.values()).flat(),
    plan,
  };
}

// ─── Routes ───────────────────────────────────────────────────────────────────

router.get("/multimodal/packs", (_req: Request, res: Response) => {
  const summary: Record<string, { slotCount: number; modalities: string[]; platforms: string[] }> = {};
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

  if (!body.id || !body.userId || !body.input?.payload || !body.platforms?.length) {
    res.status(400).json({
      error: "Missing required fields: id, userId, input.payload, platforms",
    });
    return;
  }

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
    constraints: body.constraints,
  };

  try {
    const pkg = await handleGeneration(genReq);
    res.json(pkg);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    res.status(500).json({ error: "Generation failed", detail: message });
  }
});

export default router;
