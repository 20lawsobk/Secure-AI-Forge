# AI Training Server — Complete System Documentation

> **Purpose:** Full technical record of the system as built. Everything needed to
> recreate this project from scratch on a new account, understand every design
> decision, and continue development without losing any progress.

---

## Table of Contents

1. [Project overview](#1-project-overview)
2. [Repository structure](#2-repository-structure)
3. [Environment & secrets](#3-environment--secrets)
4. [Workflows](#4-workflows)
5. [API layer (Node.js proxy)](#5-api-layer-nodejs-proxy)
6. [Python AI server — server.py](#6-python-ai-server--serverpy)
   - [Startup & thread pool](#61-startup--thread-pool)
   - [Authentication](#62-authentication)
   - [Request models & the awareness mixin](#63-request-models--the-awareness-mixin)
   - [Awareness pipeline end-to-end](#64-awareness-pipeline-end-to-end)
   - [All generation endpoints](#65-all-generation-endpoints)
   - [Ads endpoint — content-type subtype system](#66-ads-endpoint--content-type-subtype-system)
   - [Platform & utility endpoints](#67-platform--utility-endpoints)
7. [AI model internals](#7-ai-model-internals)
   - [Transformer & KV-cache](#71-transformer--kv-cache)
   - [Agents](#72-agents)
   - [Generation orchestrator](#73-generation-orchestrator)
   - [Request intelligence](#74-request-intelligence)
   - [Quality awareness buffer](#75-quality-awareness-buffer)
   - [Safety system](#76-safety-system)
   - [Video pipeline](#77-video-pipeline)
   - [Audio pipeline](#78-audio-pipeline)
8. [Storage layer](#8-storage-layer)
9. [Quality test suite — test_w6_90m.py](#9-quality-test-suite--test_w6_90mpy)
   - [Veo scoring engine](#91-veo-scoring-engine)
   - [Running the tests](#92-running-the-tests)
   - [All quality checkers](#93-all-quality-checkers)
10. [Key design decisions & non-obvious rules](#10-key-design-decisions--non-obvious-rules)
11. [Recreating from scratch — step-by-step](#11-recreating-from-scratch--step-by-step)
12. [Digital GPU system — complete documentation](#12-digital-gpu-system--complete-documentation)
13. [Audio delivery: stems, caching & the fast path](#13-audio-delivery-stems-caching--the-fast-path)
14. [Performance characteristics & operations runbook](#14-performance-characteristics--operations-runbook)
15. [Complete system wiring — instant re-setup guide](#15-complete-system-wiring--instant-re-setup-guide)

---

## 1. Project Overview

> **⚠️ Foundational compute rule — the Digital GPU, not the raw CPU.**
> This system does **not** run its math on Replit's native CPU environment
> directly (plain NumPy calls, `torch` ops at call sites, scipy/librosa DSP).
> **Every mathematical operation routes through the in-house Digital GPU
> system** (`ai_model/gpu/`): a software-defined GPU execution stack with its
> own kernels (GEMM, flash attention, softmax, conv, STFT-as-GEMM), opcode
> specs, telemetry, caching (pocket accelerator), and SIMD-compiled native
> paths. The CPU is only the physical substrate the Digital GPU executes on —
> application and model code never touches it directly. This is a hard,
> project-wide contract ("that's final"): see §12 for the full architecture,
> §15.7 for the wiring, and the routing rules below. Any new math — inference,
> training, audio DSP, image/video rendering — must be expressed through
> Digital GPU kernels (`DigitalGPU().gemm`, `gpu.*`, MaxCore wrappers), never
> as direct numpy/torch math. Fallbacks that silently bypass the Digital GPU
> are bugs.

A full-stack AI content generation platform for independent music artists. The
system generates social-media content (text, image, audio, video, ads) across
all major platforms conditioned on live platform awareness signals, user brand
voice, and creative direction.

**Stack:**

| Layer | Technology |
|---|---|
| Frontend dashboard | React + Vite (`artifacts/ai-dashboard`) |
| API proxy | Node.js + Express (`artifacts/api-server`) |
| AI generation server | Python 3.11 + FastAPI (`artifacts/ai-training-server`) |
| Database | PostgreSQL (Replit managed) |
| Local KV / fallback | SQLite (`data/local_kv.db`) |
| Storage client | HTTP → Redis → SQLite waterfall |

**The "awareness bridge"** is the central architectural concept: every
generation endpoint accepts `instruction`, `extra_context`, and
`content_themes` direction fields alongside a live `awareness` signal object.
These are merged via `_merged_awareness_for(req)` → `merge_awareness()` into a
single conditioning string that is passed into every `ScriptRequest` call. This
is what drives content quality toward the Veo 100/100 standard.

---

## 2. Repository Structure

```
/
├── artifacts/
│   ├── ai-dashboard/          # React + Vite frontend
│   │   ├── src/
│   │   └── vite.config.ts
│   ├── api-server/            # Node.js Express proxy
│   │   └── src/
│   │       ├── index.ts
│   │       └── routes/
│   │           └── model-proxy.ts   ← explicit per-route allowlist
│   └── ai-training-server/    # Python FastAPI AI server
│       ├── server.py          ← ALL endpoints, models, constants
│       ├── storage_client.py  ← all storage clients (ads, curriculum, etc.)
│       ├── ai_model/
│       │   ├── agents/
│       │   │   ├── script_agent.py
│       │   │   ├── distribution_agent.py
│       │   │   ├── visual_spec_agent.py
│       │   │   └── optimization_agent.py
│       │   ├── generation/
│       │   │   ├── __init__.py        ← exports merge_awareness, build_context
│       │   │   └── orchestrator.py    ← GenerationContext, merge_awareness
│       │   ├── gpu/
│       │   │   └── hyper_creative_transformer.py  ← transformer + KV-cache
│       │   ├── quality_awareness.py   ← platform_awareness_string
│       │   ├── request_intelligence.py← build_brief, awareness_from_direction
│       │   ├── safety/
│       │   │   └── content_safety.py  ← screen(), token masking
│       │   ├── video/
│       │   │   └── video_agent.py
│       │   ├── audio/
│       │   ├── image/
│       │   ├── rta/                   ← RTA rendering fabric
│       │   └── maxcore/
│       ├── tests/
│       │   └── test_w6_90m.py        ← full quality + throughput test
│       ├── training/                  ← static + synthetic training data (JSON)
│       ├── workers/                   ← background: continuous training, data pulling
│       └── data/
│           └── local_kv.db            ← SQLite fallback store
```

---

## 3. Environment & Secrets

| Variable | Where set | Purpose |
|---|---|---|
| `DATABASE_URL` | Replit secret | PostgreSQL connection string (Replit managed DB) |
| `ADMIN_KEY` | Replit secret | Admin-only endpoint access |
| `AI_TRAINING_KEY_PROD` | Replit secret | Production API key (seeded into DB on startup) |
| `SESSION_SECRET` | Replit secret | Session signing |
| `MODEL_API_PORT` | Workflow env | Python server port (default `9878`) |
| `PORT` | Workflow env | Node proxy port (default `8080`) |
| `BASE_PATH` | Workflow env | Vite base path (set to `/`) |
| `API_PORT` | Workflow env | Port the frontend calls (should match proxy port) |
| `STORAGE_HTTP_URL` | Optional | External storage backend URL |
| `STORAGE_BEARER_TOKEN` | Optional | Auth token for external storage |
| `STORAGE_INSTANCE` | Optional | Storage namespace / instance identifier |

**Test API key** (hardcoded in `test_w6_90m.py`):
```
f242bf97d7e46b7ca0b17cd6b01ca9239bc327b862a86b703556565523849701
```
This key is seeded into the database on first startup via `AI_TRAINING_KEY_PROD`.

---

## 4. Workflows

### Start application (primary — use this one)
```
PORT=8080 MODEL_API_PORT=9878 \
  pnpm --filter @workspace/api-server run dev \
  & PORT=5000 BASE_PATH=/ API_PORT=8080 \
  pnpm --filter @workspace/ai-dashboard run dev
```
The Node process spawns the Python server as a child process and monitors it.
**Do not run the artifact workflows** (`artifacts/api-server`, `artifacts/ai-dashboard`)
alongside "Start application" — they create a second Python server on the same
port and cause model-lock contention. They cannot be deleted (Replit restriction)
so keep them stopped.

### Restarting Python server without a full restart
If you edit `server.py` or any `ai_model/` file:
```bash
pkill -9 -f "server\.py"
sleep 6
curl -s http://localhost:9878/health  # should return 200
```
The Node keepalive monitor restarts Python automatically. Always verify with
`/health` before running tests — the old process survives `WorkflowsRestart`
because the workflow manager sends SIGTERM to Node, not to the Python child.

---

## 5. API Layer (Node.js Proxy)

`artifacts/api-server/src/routes/model-proxy.ts`

The proxy is an **explicit per-route allowlist** — there is no catch-all. Every
new Python route must have a matching `proxyRequest` handler added here or the
dashboard will receive a 404.

### Complete proxy route table

| Dashboard path | Python path | Method |
|---|---|---|
| `/health` | `/health` | GET |
| `/model/status` | `/model/status` | GET |
| `/gpu/status` | `/gpu/status` | GET |
| `/gpu/hyper/status` | `/gpu/hyper/status` | GET |
| `/gpu/capabilities` | `/gpu/capabilities` | GET |
| `/training/status` | `/training/status` | GET |
| `/training/start` | `/training/start` | POST |
| `/training/stop` | `/training/stop` | POST |
| `/training/datasets` | `/training/datasets` | GET |
| `/training/continuous/status` | `/training/continuous/status` | GET |
| `/training/continuous/start` | `/training/continuous/start` | POST |
| `/training/continuous/stop` | `/training/continuous/stop` | POST |
| `/training/puller/status` | `/training/puller/status` | GET |
| `/training/puller/pull` | `/training/puller/pull` | POST |
| `/platform/video/generate` | `/platform/video/generate` | GET + POST |
| `/content/generate` | `/content/generate` | POST |
| `/platform/social/generate` | `/platform/social/generate` | POST |
| `/platform/social/autopilot` | `/platform/social/autopilot` | POST |
| `/platform/daw/generate` | `/platform/daw/generate` | POST |
| `/platform/distribution/plan` | `/platform/distribution/plan` | POST |
| `/platform/ads/record` | `/platform/ads/record` | POST |
| `/platform/ads/generate` | `/platform/ads/generate` | POST |
| `/platform/ads/autopilot` | `/platform/ads/autopilot` | POST |
| `/platform/ads/audience` | `/platform/ads/audience` | POST |
| `/platform/ads/optimize` | `/platform/ads/optimize` | POST |
| `/platform/ads/performance/:userId` | `/platform/ads/performance/:userId` | GET |
| `/api/safety/screen` | `/api/safety/screen` | POST |
| `/api/infer/viral-score` | `/api/infer/viral-score` | POST |
| `/api/audio/analyze` | `/api/audio/analyze` | POST |
| `/api/awareness/quality/status` | `/api/awareness/quality/status` | GET |
| `/watchdog/status` | `/watchdog/status` | GET |
| `/generate/content` | `/api/generate/content` | POST |
| `/generate/text` | `/api/generate/text` | POST |
| `/generate/campaign` | `/api/generate/campaign` | POST |
| `/generate/image` | `/api/generate/image` | POST |
| `/generate/audio` | `/api/generate/audio` | POST |
| `/generate/video` | `/api/video/generate-ai` | POST |
| `/api/models/social/state` | `/api/models/social/state` | GET |
| `/api/models/advertising/state` | `/api/models/advertising/state` | GET |
| `/api/models/content/state` | `/api/models/content/state` | GET |
| `/api/models/engagement/state` | `/api/models/engagement/state` | GET |
| `/api/train/feedback` | `/api/train/feedback` | POST |
| `/api/predict/engagement` | `/api/predict/engagement` | POST |
| `/api/optimize/ad` | `/api/optimize/ad` | POST |
| `/api/content/score` | `/api/content/score` | POST |
| `/api/analyze` | `/api/analyze` | POST |
| `/api/analyze/sentiment` | `/api/analyze/sentiment` | POST |
| `/api/maxcore/pocket-multiply` | `/api/maxcore/pocket-multiply` | POST |
| `/storage/*` | `/storage/*` | GET + POST |
| `/boostsheets` | `/boostsheets` | GET |
| `/dashboard/stats` | `/dashboard/stats` | GET |
| `/audio-job/:jobId` | `/api/audio-job/:jobId` | GET |
| `/video-job/:jobId` (+ `/download`, `/file`, `/video`, `/preview/:sceneIdx`) | same | GET |
| `/audio/:jobId/stems` | `/api/audio/:jobId/stems` | GET (rate-limited) |
| `/audio/:jobId/midi` | `/api/audio/:jobId/midi` | GET (rate-limited) |
| `/files/stems/:jobId/:filename` | `/api/files/stems/:jobId/:filename` | GET binary stream (rate-limited) — stem WAV download |
| `/jobs/:jobId/progress`, `/jobs/:jobId/cancel` | audio-job first, then video-job | GET / POST |

**Intentionally NOT proxied** (admin/internal only, callable directly on the
Python port with admin auth): training-data seeding beyond
`/storage/datasets/audio/seed`, `awareness/quality/harvest`, and other
internal maintenance endpoints. `pocket-multiply` is exposed but is an
internal compute utility, not a dashboard feature.

---

## 6. Python AI Server — server.py

### 6.1 Startup & Thread Pool

On startup the event loop executor is sized to `max(512, cpu × 32)` threads so
concurrent blocked I/O (model inference + PDIM calls) never stalls the pool.
Dynamic batching is hard-configured (never `setdefault`) so stale env vars
cannot silently cap throughput.

```python
PORT = int(os.environ.get("MODEL_API_PORT", 9878))
```

### 6.2 Authentication

All generation endpoints use `Depends(require_scope("generate"))`.
Admin endpoints use `Depends(verify_admin)`.

API keys are stored in the `api_keys` PostgreSQL table as SHA-256 hashes.
The key `f242bf97d7e46b7ca0b17cd6b01ca9239bc327b862a86b703556565523849701`
is the development/test key seeded on first startup.

### 6.3 Request Models & the Awareness Mixin

#### `_AwarenessMixin`
All generation request models inherit from this. It normalises `awareness`
regardless of whether the caller sends a plain string or the structured object
`{contextString, trendingGenres, …}` that `enrichWithAwareness` (the Node
middleware) injects.

```python
class _AwarenessMixin(BaseModel):
    awareness: str = ""

    @model_validator(mode="before")
    @classmethod
    def _normalise_awareness(cls, data):
        if isinstance(data, dict) and isinstance(data.get("awareness"), dict):
            data = dict(data)
            data["awareness"] = data["awareness"].get("contextString", "") or ""
        return data
```

**Critical:** `AdGenerateRequest` uses plain `BaseModel` with
`awareness: Optional[Any] = None` (to accept the structured dict). The
`merge_awareness` function in `orchestrator.py` has its own dict-coercion guard
for this case:

```python
def _coerce_aw(v):
    if isinstance(v, dict):
        return (v.get("contextString", "") or "").strip()
    return (v or "").strip() if v is not None else ""
```

#### Direction fields (on every generation model)
Every generation request model carries these three optional fields in addition
to `awareness`. They are merged via `_merged_awareness_for(req)` before being
passed into agents.

| Field | Type | Purpose |
|---|---|---|
| `instruction` | `Optional[str]` | Free-form creative directive for this request |
| `extra_context` | `Optional[str]` | Supplementary background context |
| `content_themes` | `Optional[List[str]]` | Thematic keywords (appended as bullet, never `#hashtags`) |

Models that carry these fields:
`ContentRequest`, `PlatformSocialRequest`, `PlatformDAWRequest`,
`PlatformDistributionRequest`, `PlatformVideoRequest`, `PlatformAutopilotRequest`,
`MaxcoreTextRequest`, `MaxcoreMediaRequest`, `AdGenerateRequest`,
`ApiGenerateCampaignRequest`, `ApiGenerateContentRequest`,
`ApiGenerateTextRequest`, `ApiGenerateImageRequest`, `ApiGenerateAudioRequest`,
`ApiGenerateVideoRequest`.

### 6.4 Awareness Pipeline End-to-End

```
Client request
    │
    │  awareness: {contextString, trendingGenres, ...}  (or plain string)
    │  instruction: "Focus on emotional resonance"
    │  content_themes: ["exclusive", "fire", "stream now"]
    │
    ▼
_AwarenessMixin._normalise_awareness()
    │  dict → plain string (extracts contextString, appends GENRES:)
    ▼
_merged_awareness_for(req)          [server.py line ~850]
    │  calls merge_awareness(req) from ai_model.generation
    ▼
merge_awareness(req)                [orchestrator.py]
    │  1. Builds direction string from instruction + extra_context
    │  2. Calls awareness_from_direction(direction, content_themes)
    │     → formats themes as "• Themes: ..." bullet (NOT #hashtags)
    │     → lead-in strips ("I want", "Please", etc.)
    │  3. Appends raw awareness string (live trend signals)
    │     → creative direction OUTRANKS generic trend context
    │  4. Returns "" if nothing real — agents handle empty gracefully
    ▼
_effective_awareness(platform, merged)   [content/social/video handlers only]
    │  prepends platform_awareness_string(platform) from quality_awareness module
    │  → platform signals (fire/drop/exclusive/viral keywords) guaranteed present
    ▼
ScriptRequest(awareness=final_string)
    │
    ▼
ScriptAgent.run() → hook / body / cta
```

**Rule: never put awareness content into the `idea` field of ScriptRequest.**
`idea` is templated raw into scene phrases like `"Stream {idea} now"`. Routing
context through `awareness` keeps the two channels clean.

**Ads-specific rule:** for the ads endpoint, `content_themes` are appended to
the awareness string as natural language (`"Trending themes: X, Y, Z."`) not
as `"• Woven around: ..."` markers. The ads ScriptAgent does not strip template
markers and they leak verbatim into the hook output.

### 6.5 All Generation Endpoints

#### `/content/generate` — `ContentRequest`
Legacy platform generation endpoint. ScriptAgent + DistributionAgent.
Uses PDIM dedup cache keyed on `{platform, topic, tone, goal, awareness}`.

| Field | Default |
|---|---|
| `platform` | `"tiktok"` |
| `topic` | required |
| `tone` | `"energetic"` |
| `goal` | `"growth"` |
| `include_hashtags` | `True` |

Response: `{caption, hook, body, cta, hashtags, source}`

---

#### `/api/generate/content` — `ApiGenerateContentRequest`
Primary content generation endpoint used by the dashboard. Full awareness
bridge, PDIM dedup, intelligence block, variants, brand voice profiles.

Key fields: `topic`, `platform`, `tone`, `goal`, `variants` (1–5),
`awareness`, `instruction`, `extra_context`, `content_themes`, `artistProfileId`.

Response: `{caption, hook, body, cta, quality_score, confidence, hashtags, variants, intelligence}`

**Intelligence block** always present:
```json
{
  "modality": "text",
  "platform": "instagram",
  "intent": "growth",
  "intent_label": "Grow Audience"
}
```

---

#### `/api/generate/campaign` — `ApiGenerateCampaignRequest`
Generates a full multi-week multi-platform release rollout.

Key fields: `title` (release name), `artist_name`, `genre`, `tone`,
`platforms` (default `["instagram","tiktok"]`), `weeks` (2–12, default 6),
`release_date`, `mood`, `bpm`, `key`, `generate_images` (bool),
`generate_teasers` (bool), `awareness`, `instruction`, `content_themes`.

Returns 15 posts across 5 phases: announce → tease → pre-save → release → sustain.
Each post has `hook`, `body`, `cta`, `caption`, `platform`, `phase`, `week`, `day`.

---

#### `/api/generate/text` — `ApiGenerateTextRequest`
Dual-mode:
- `mode="planner"` → returns `TaskPlan` (list of steps for the multimodal orchestrator)
- `mode="content"` → returns per-slot text assets for a given step

---

#### `/api/generate/image` — `ApiGenerateImageRequest`
Returns a generated image URL, format, and intelligence block.
Fields: `topic`, `platform`, `style`, `aspect_ratio`, `awareness`, `instruction`, `content_themes`.

---

#### `/api/generate/audio` — `ApiGenerateAudioRequest`
Async render job. Returns `{job_id, status, b64}`.
Audio is seeded from the FMA dataset stored in PDIM. No synthetic fallback.
Fields: `topic`, `genre`, `mood`, `bpm`, `key`, `duration`, `awareness`, `instruction`, `content_themes`.

Seed the audio dataset first if empty:
```bash
POST /storage/datasets/audio/seed
```

---

#### `/api/generate-video` and `/api/video/generate-ai` — `ApiGenerateVideoRequest`
Async render job. Returns `{job_id, status, intelligence}`.
`_start_video_job(req, platform)` is the shared launcher for both single and
cross-platform (`platforms: ["tiktok","instagram"]`) calls.

Already has full awareness wiring via `_merged_awareness_for(req)` inside
`_start_video_job` at the `VideoAgentRequest` construction.

---

#### `/platform/social/generate` — `PlatformSocialRequest`
Per-user curriculum signals personalise tone. Supports `num_variants` (1–5).
Returns variants list, each with `hook/body/cta/caption/hashtags/source`.

---

#### `/platform/daw/generate` — `PlatformDAWRequest`
DAW/studio AI. Four modes: `lyrics` | `hook` | `beat_description` | `track_concept`.
Uses ScriptAgent + VisualSpecAgent. Returns `output.main/body/cta/visual_direction`.

---

#### `/platform/distribution/plan` — `PlatformDistributionRequest`
Distribution strategy for a track. Uses DistributionAgent with `awareness`.
Returns `plan.pitch/hashtags/pre_release_steps/post_release`.

---

#### `/platform/video/generate` — `PlatformVideoRequest`
Platform video package with per-user tone personalisation. ScriptAgent +
VisualSpecAgent + DistributionAgent all receive merged awareness. Returns
`title/hook/script/scenes/captions/hashtags/thumbnail_concept/distribution`.

---

### 6.6 Ads Endpoint — Content-Type Subtype System

#### `/platform/ads/generate` — `AdGenerateRequest`

The most complex generation endpoint. Generates N ad creatives, each as one of
four distinct content-type subtypes, with per-format copy style, hook pools,
and `creative_brief` schemas.

**Request fields:**

| Field | Type | Default | Purpose |
|---|---|---|---|
| `user_id` | `str` | required | User identifier |
| `platform` | `str` | `"meta"` | `tiktok\|meta\|youtube\|instagram\|google` |
| `ad_type` | `str` | `"video"` | Base type (overridden when `vary_subtypes=True`) |
| `product` | `str` | required | What is being advertised |
| `goal` | `str` | `"streams"` | `streams\|merch\|fanbase\|tickets\|downloads\|conversions` |
| `budget_daily` | `float` | None | Daily budget for split calculation |
| `num_creatives` | `int` | 3 | 1–10 |
| `replicate_peak` | `bool` | True | Pull peak performer formula from storage |
| `genre` | `str` | None | Music genre |
| `artist_name` | `str` | None | Artist name |
| `vary_subtypes` | `bool` | **True** | Cycle through video/audio/text/image |
| `awareness` | `Any` | None | Live trend signal (string or dict) |
| `instruction` | `str` | None | Creative directive |
| `content_themes` | `List[str]` | None | Thematic keywords |

**Subtype selection — priority order:**

Three mechanisms control which subtype each creative slot receives. Highest
priority wins:

| Priority | Mechanism | Behaviour |
|---|---|---|
| 1 (highest) | `target_subtypes` non-empty + valid | Cycle through exactly those subtypes in caller order |
| 2 | `vary_subtypes=True` (default) | Cycle through all four: video → audio → text → image |
| 3 (lowest) | `vary_subtypes=False` | Every creative uses `ad_type` (no cycling) |

**`target_subtypes` field** — explicit subtype selection:
```json
"target_subtypes": ["video", "audio"]
```
- Accepted values: `"video"`, `"audio"`, `"text"`, `"image"` (any order, any combination)
- Unknown values are silently dropped; if *all* are invalid the field is ignored
  and the `vary_subtypes` / `ad_type` fallback governs
- Creatives cycle through the valid list with wrap-around:
  - `["video","audio"]` with `num_creatives=4` → video, audio, video, audio
  - `["text"]` with `num_creatives=3` → text, text, text
  - `["image","video"]` with `num_creatives=3` → image, video, image
- Takes priority over `vary_subtypes` — set `target_subtypes` to target specific
  formats without disabling the cycling mechanism for unrelated calls

**Auto-cycle** (`vary_subtypes=True`, no `target_subtypes`):
```python
_ALL_SUBTYPES = ["video", "audio", "text", "image"]
# slot i → _ALL_SUBTYPES[i % 4]
```
A 4-creative call always produces exactly one of each type. A 3-creative call
produces video + audio + text. Set `vary_subtypes=False` to lock all creatives
to `ad_type`.

**`_selection_mode` values** (returned in the response):

| Value | Meaning |
|---|---|
| `"targeted"` | `target_subtypes` was used — cycle is the caller-supplied list |
| `"auto_all"` | `vary_subtypes=True`, no `target_subtypes` — all four types |
| `"fixed"` | `vary_subtypes=False`, no `target_subtypes` — `ad_type` for every slot |

**Per-subtype specs (`AD_SPECS_BY_TYPE`):**

| Subtype | `format_label` | `duration_range` | `copy_tone` | Unique brief keys |
|---|---|---|---|---|
| `video` | Video Ad | 15–60s | `direct` | `aspect_ratio`, `duration`, `opening_3s`, `visual_direction`, `caption_style` |
| `audio` | Audio Ad | 15–30s | `conversational` | `voiceover_script`, `voiceover_style`, `sound_branding`, `companion_banner`, `platform_notes` |
| `text` | Text Ad | N/A | `punchy` | `headline` (≤30 chars), `description` (≤90 chars), `cta_button`, `extension_types`, `display_url` |
| `image` | Image Ad | N/A | `emotive` | `tagline`, `visual_concept`, `text_overlay_rule`, `color_direction`, `tagline_position`, `format_variants` |

**Hook pools:**

- **Video:** platform-specific (`AD_HOOKS_BY_PLATFORM`) — all hooks contain power words + `!`
- **Audio:** `AD_HOOKS_BY_TYPE["audio"]` — sound-focused ("Close your eyes...", "This sound will stop you...")
- **Text:** `AD_HOOKS_BY_TYPE["text"]` — punchy short phrases ("Fire new drop. Stream free now.")
- **Image:** `AD_HOOKS_BY_TYPE["image"]` — visual/emotive ("One image. One release. Zero skips.")

**Hook quality guard (`_ad_hook_score`):**
The AI model's suggested hook is only accepted when it scores strictly HIGHER
than the base pool hook on this metric:
```
score = 0.55 (power word present) + 0.30 (! or ?) + 0.15 (emoji) → max 1.0
```
This prevents model regression to weaker phrases like "Check this out" while
still allowing awareness-conditioned improvements.

**Awareness wiring in `_generate_ad_creative`:**
- Receives the already-merged `awareness` string (computed once in the handler)
- Passes it directly to `ScriptRequest(awareness=awareness)` with `tone=type_spec["copy_tone"]`
- Content themes go in as natural language: `"Trending themes: X, Y, Z."` — NOT as `"• Woven around: ..."` marker

**CTAs by goal (`AD_CTAS_BY_GOAL`):**

| Goal | CTAs |
|---|---|
| `streams` | Stream Now, Listen Free, Add to Playlist, Presave Today |
| `merch` | Shop Now, Get Yours, Limited Drop, Claim 20% Off |
| `fanbase` | Follow for More, Join the Movement, Be First, Subscribe |
| `tickets` | Get Tickets, Reserve Your Spot, Doors Open Soon, Book Now |
| `downloads` | Download Free, Get the Track, Free Download Today |
| `conversions` | Start Free Trial, Book a Session, Claim Your Spot, Apply Now |

**Response shape:**
```json
{
  "success": true,
  "platform": "meta",
  "ad_type": "video",
  "subtype_selection": {
    "mode": "targeted",
    "cycle": ["video", "audio"],
    "requested": ["video", "audio"],
    "applied": ["video", "audio"]
  },
  "creatives": [
    {
      "variant": 1,
      "content_type": "video",
      "hook": "Finally — the drop you've been waiting for is live! 🎵",
      "headline": "new album — Streams",
      "body": "🎵 Luna Voss drops something you've never heard before...",
      "cta": "Stream Now",
      "creative_brief": {
        "format": "feed video",
        "aspect_ratio": "1:1 or 4:5",
        "duration": "15–30s",
        "opening_3s": "...",
        "visual_direction": "...",
        "caption_style": "..."
      },
      "source": "template"
    }
  ],
  "targeting": { "primary_interests": [...], "placements": [...] },
  "budget_split": [...],
  "platform_benchmarks": { "avg_ctr": "0.9-2%", ... },
  "launch_checklist": [...]
}
```

**`subtype_selection` block** (always present in every response):

| Key | Type | Meaning |
|---|---|---|
| `mode` | string | `"targeted"` \| `"auto_all"` \| `"fixed"` — which mechanism determined the cycle |
| `cycle` | list | The actual subtype list used for cycling (what slot `i % len(cycle)` indexes into) |
| `requested` | list | Raw value of `target_subtypes` from the request (empty list if not sent) |
| `applied` | list | De-duplicated ordered list of subtypes that actually appeared across all `num_creatives` slots |

**Platform video specs (`_PLAT_VIDEO_SPECS`):**

| Platform | Ratio | Duration | Format |
|---|---|---|---|
| tiktok | 9:16 | 15–60s | vertical video |
| meta | 1:1 or 4:5 | 15–30s | feed video |
| youtube | 16:9 | 6–15s skippable | pre-roll |
| instagram | 9:16 | up to 60s | reel |
| google | N/A | N/A | display |

**Audience segments (`AUDIENCE_SEGMENTS`):**
`music_fan`, `hip_hop`, `rb`, `pop`, `producer`, `artist`, `brand_deal` —
each maps to a list of targeting interest labels.

**Performance benchmarks by platform:**

| Platform | Avg CTR | Avg CPC | Good ROAS |
|---|---|---|---|
| tiktok | 1.5–3% | $0.50–1.20 | 3–6x |
| meta | 0.9–2% | $0.80–2.50 | 2–5x |
| youtube | 0.4–1% | $0.10–0.30 | 2–4x |
| google | 2–6% | $0.50–3.00 | 4–8x |
| instagram | 0.8–1.5% | $1.00–3.00 | 2–4x |

---

### 6.7 Platform & Utility Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Returns `{"status":"ok"}` |
| `GET /model/status` | Model ready state, vocab size, dim, layers |
| `GET /gpu/status` | GPU utilisation and capabilities |
| `POST /api/safety/screen` | Content safety check — returns `{safe, category, severity}` |
| `POST /api/infer/viral-score` | Returns `viral_score` in [0,1] |
| `POST /api/predict/engagement` | Returns `engagement_score` in [0,1] |
| `POST /api/content/score` | Returns `score` in [0,100] |
| `POST /api/analyze/sentiment` | Returns `{label, confidence}` |
| `POST /api/optimize/ad` | Returns `{confidence, recommendations}` |
| `POST /api/maxcore/pocket-multiply` | Exact 2D matrix multiply via PDIM pocket |
| `GET /dashboard/stats` | Dashboard statistics |
| `POST /training/start` | Start/resume model training |
| `GET /training/status` | Training loop state |
| `POST /storage/datasets/audio/seed` | Seed FMA audio dataset into PDIM |

---

## 7. AI Model Internals

### 7.1 Transformer & KV-Cache

**File:** `ai_model/gpu/hyper_creative_transformer.py`

The model is a decoder-only transformer with:
- `dim=512`, `layers=8`, `vocab=443` (compact music-domain vocabulary)
- Trained on platform-specific engagement patterns

**Critical:** The KV-cache pattern is mandatory. Without it, `plan()` takes
3+ minutes per call.

```python
# Two-method KV-cache interface:
kv_cache = model.prefill(prompt_tokens)       # process full prompt once
next_token = model.decode_one(token, kv_cache) # O(1) per step
```

Any refactor of the transformer must preserve both `prefill` and `decode_one`.
If you add a new architecture, the `plan()` timeout will explode if these are
missing.

**HyperSIMDCore flash attention** uses tiled online-softmax (16x less score
memory). The flash loop iterates over `Tk` (key blocks), not `Tq` (query
blocks). Block size is clamped to avoid OOM on long sequences.

### 7.2 Agents

All agents are never-raise. They degrade to template output if the model is not
ready or inference fails.

#### `ScriptAgent` — `ai_model/agents/script_agent.py`

**Request:**
```python
ScriptRequest(
    idea: str,           # clean topic string only — no context bleed
    platform: str,       # normalised platform name
    goal: str,           # growth | streams | engagement | ...
    tone: str,           # energetic | direct | conversational | punchy | emotive
    awareness: str,      # merged awareness string from _merged_awareness_for
    variant_idx: int = 0 # for generating distinct variants
)
```

**Output:**
```python
ScriptResponse(
    hook: str,    # opening line
    body: str,    # middle content
    cta: str,     # call to action
    source: str   # "ai_model" | "awareness" | "template"
)
```

#### `DistributionAgent` — `ai_model/agents/distribution_agent.py`

```python
DistributionRequest(script, platform, goal, awareness)
# → caption (str), hashtags (List[str], up to 8), posting_time (ISO 8601)
```

#### `VisualSpecAgent` — `ai_model/agents/visual_spec_agent.py`

```python
VisualSpecRequest(idea, platform, tone, awareness)
# → thumbnail_prompt, color_scheme, layout (e.g. "vertical_9_16")
```

#### `OptimizationAgent` — `ai_model/agents/optimization_agent.py`

```python
OptimizationRequest(...)
# → confidence (0–1), recommendations (dict)
```

### 7.3 Generation Orchestrator

**File:** `ai_model/generation/orchestrator.py`

**`merge_awareness(req: Any) -> str`**

The single source of truth for building awareness strings. Handles dict input,
formats direction fields, coerces all types to string. Used by every generation
endpoint via `_merged_awareness_for(req)` in `server.py`.

```python
def merge_awareness(req):
    direction = " ".join([instruction, extra_context])   # from req fields
    return "\n".join([
        awareness_from_direction(direction, content_themes),  # direction first
        coerce_to_str(req.awareness),                         # live signals second
    ])
```

**`GenerationContext`** — unified per-request conditioning bus:
- `modality`: text | image | audio | video | ads
- `platform`: normalised platform string
- `brief`: `GenerationBrief` from request intelligence
- `awareness`: merged awareness string
- `technique`: `TechniqueProfile` (optional, extracted from assets)

Helper methods translate `GenerationContext` into renderer-specific parameters
(diffusion, RTA, audio) so every endpoint dispatches through one consistent
surface.

**`build_context()`** — entry point for all generation. Never raises; degrades
to stubs if intelligence layer fails.

### 7.4 Request Intelligence

**File:** `ai_model/request_intelligence.py`

**`build_brief(modality, platform, topic, goal, tone, ...) -> GenerationBrief`**

Deterministic analysis of intent + audience. Returns:
- `strategy`: aspect ratio, hook style, CTA style
- `directives`: content directives for the model
- `augmented_idea`: enriched version of the topic (routed through `awareness`, NOT `idea`)
- `ai_disclosure`: disclosure string
- `producer_meta`: energy, mood, bpm, key

**`awareness_from_direction(direction, themes) -> str`**

Formats creative direction for the awareness parser:
- Strips lead-ins: "I want", "Please make", "Create a", etc.
- Appends themes as `"• Themes: X, Y, Z"` bullet (not `#hashtags`, no literal `•` in output)
- Colon-safe (prevents parser misreading key-value pairs)

### 7.5 Quality Awareness Buffer

**File:** `ai_model/quality_awareness.py`

`platform_awareness_string(platform: str) -> str`

Returns a multi-line string of platform-specific arousal signals (fire, drop,
viral, exclusive, finally, etc.) pre-formatted for the awareness parser. Prepended
by `_effective_awareness()` in the content/social/video handlers so [HIGH] arousal
signals are always available even if the caller sends no awareness.

Returns `""` once the system reaches self-sufficiency (MB_AWARENESS_RETIRE_AT
threshold). The buffer is backed by PDIM (disk-persisted, survives restarts).

**Auto-retirement contract:** all injection points gate on the same
`mb_awareness.is_active()` check. Never hardcode awareness signal strings —
always go through `platform_awareness_string()`.

### 7.6 Safety System

**File:** `ai_model/safety/content_safety.py`

`screen(text: str) -> dict`
- Detects violations: `{safe: bool, category: str, severity: float, matches: list}`
- Non-destructive — does not modify text
- Integration points: `CreativeModel._sample_next` (token masking during decoding)
  and agent `_clean_text` (post-generation redaction)
- `screen()` increments a counter but does not modify output
- `enforce()` applies actual masking/redaction

The 10-stage pipeline safety gate is in `ai_model/safety`. The
`penalty_of` function (not `penalty`) applies content penalties during
`rank_candidates`. Do not rename.

### 7.7 Video Pipeline

**File:** `ai_model/video/video_agent.py`

`VideoAgent.plan(VideoAgentRequest) -> VideoProduction`

Scene count: `max(3, duration_seconds // 10)`, capped at 25.
Each scene: `~3 seconds`. Text sampled from training corpus via
`dataset_sampler` (not `model.generate_batch()` — the model is undertrained
for batch scene generation).

**FFmpeg** must use `run_ffmpeg()` (posix_spawn) — never `subprocess.Popen`
or `os.fork()`. Under model memory pressure, fork-based ffmpeg dies with
`[Errno 5] EIO`. The `run_ffmpeg()` wrapper uses posix_spawn which avoids
the copy-on-write memory explosion.

**Scene text cleaning rules:** strip template markers, drop sentences with
<3 words, remove consecutive duplicate lines.

### 7.8 Audio Pipeline

**File:** `ai_model/audio/`

Audio generation is async (returns `job_id`). The render pipeline:
1. Selects a seeded FMA dataset entry from PDIM (key-aware, BPM-aware)
2. Applies pitch shift and tempo adjust via `rubberband` (exact key/BPM)
3. Applies LUFS mastering
4. HPSS stem separation if requested
5. Writes to `audio_{job_id}.{ext}`

**No synthetic fallback** — if the FMA dataset is not seeded, audio jobs will
queue indefinitely. Seed with `POST /storage/datasets/audio/seed`.

The seeder module is import-cached; restart the Python server after editing
seeder code.

---

## 8. Storage Layer

**File:** `artifacts/ai-training-server/storage_client.py`

Resilient waterfall: HTTP backend → Redis → SQLite (`data/local_kv.db`).
Every client is never-raise.

### Client factory functions

| Function | Returns | Purpose |
|---|---|---|
| `get_storage()` | `StorageClient` | Generic key/value store |
| `get_dataset_client()` | `DatasetStreamClient` | Training dataset streaming |
| `get_checkpoint_client()` | `ModelCheckpointClient` | Model weight checkpoints |
| `get_curriculum_client()` | `CurriculumStateClient` | Per-user engagement signals |
| `get_pipeline()` | `TrainingDataPipeline` | Data ingestion pipeline |
| `get_ads_client()` | `AdsClient` | Ad runs, peak performers, portfolio |
| `get_artist_client()` | `ArtistProfileClient` | Brand voice profiles |
| `get_campaign_client()` | `CampaignClient` | Campaign plans and schedules |

### `AdsClient` key methods

```python
ads.record_ad_run(user_id, record_dict)           # record a completed run
ads.get_winning_formula(user_id, platform, type)  # peak performer pattern
ads.get_peak_performers(user_id, limit, platform) # top ROAS/CTR runs
ads.analyse_portfolio(user_id, platform)          # portfolio-level analysis
```

Peak performers are flagged when `ROAS >= 3` and `CTR >= 2.5%`. The formula
includes `top_hooks`, `top_ctas`, `top_audience_tags`, `avg_roas`, `avg_ctr`.

### `CurriculumStateClient` key methods

```python
curriculum.get_top_performers(user_id, platform, top_n)  # engagement signals
curriculum.get_user_stats(user_id)                       # aggregate stats
```

### PDIM (Pocket-Dimension) storage

PDIM is the dedup + single-flight orchestrator backed by the storage waterfall.
Key concepts:
- **Pocket:** a dedup namespace. `pocket="root/sub/leaf"` for nested namespaces.
- **Single-flight:** identical concurrent requests collapse to one compute.
  The leader writes `slot.result` BEFORE calling `_release()` so followers read
  off the slot without a round-trip race.
- **Content dedup cache:** disk-backed, survives restarts. Verify new code with
  a NOVEL topic when testing (cached content silently returns old results).
- **Auth:** `X-Api-Key` header on all PDIM calls.

---

## 9. Quality Test Suite — test_w6_90m.py

**File:** `artifacts/ai-training-server/tests/test_w6_90m.py`

Two sections:
1. **Throughput:** Models 90,000,000 distinct simultaneous users. 150 content-unique
   requests at 40 concurrent. Measures and projects latency.
2. **Quality:** Fires one realistic request to every generation endpoint in parallel,
   inspects actual content.

### Connection details

```python
PY_HOST  = "127.0.0.1"
PY_PORT  = 9878        # direct Python server
API_HOST = "127.0.0.1"
API_PORT = 8080        # via Node proxy
API_KEY  = "f242bf97d7e46b7ca0b17cd6b01ca9239bc327b862a86b703556565523849701"
HEADERS  = {"Content-Type": "application/json", "X-Api-Key": API_KEY}
```

### 9.1 Veo Scoring Engine

Calibrated so **100 = Google Veo quality standard**. Minimum passing score: **85**.
Awareness-conditioned requests (with `instruction` + `content_themes`) target **≥95**.

**Composite score formula:**
```
score = (
    _veo_length(text) × 0.30  +   # word count 15–60 is ideal (1.0)
    _veo_cta(text)    × 0.15  +   # CTA keyword present
    _veo_hook(text)   × 0.20  +   # first line: power word + ! + emoji
    _veo_struct(text) × 0.15  +   # ≥3 lines, arousal words, ≤125 char first line
    1.0               × 0.20      # keyword: always full marks (no constraint)
) × 100 − 40 (if garbled)
```

**Component functions:**

`_veo_length(text)`:
- `wc == 0` → 0.0
- `wc ≤ 15` → `wc / 15` (linear ramp)
- `15 < wc ≤ 60` → 1.0 (ideal range)
- `wc > 60` → `max(0, 1 − (wc−60)/60)` (penalty for being too long)

`_veo_cta(text)`:
- 1.0 if any of: `click, follow, link, save, share, buy, get, stream, listen, subscribe, comment, tap, join, shop, watch, bio`
- 0.0 otherwise

`_veo_hook(text)` — scores the **first non-empty line only**:
- +0.55 if any power word present: `secret, proven, instantly, exclusive, free, now, never, stop, first, best, viral, insane, real, raw, unreleased, finally, limited, drop, fire, everyone, nobody`
- +0.30 if `?` or `!` present
- +0.15 if emoji present (`[\U0001F300-\U0001FAFF\u2600-\u27BF]`)
- Max 1.0

`_veo_struct(text)`:
- +0.35 if first line ≤ 125 chars
- +0.30 if ≥3 lines, +0.15 if exactly 2 lines
- +`min(0.20, 0.10 × arousal_hits)` for words: `amazing, incredible, unbelievable, finally, secret, exclusive, never, always, fire, drop, viral, insane`
- +0.15 if last line has emoji or contains `tag /save /drop a/comment/share`

`_veo_looks_garbled(text)` → True if:
- Text is empty / whitespace
- No alphanumeric tokens found
- Fewer than 60% of tokens are ≤20 chars long

`_extract_text_for_veo(response_dict)`:
- `variants` in response → uses `variants[0].caption` or `hook+body+cta` join
- `creatives` in response → newline-joins `hook\nheadline\nbody\ncta` of `creatives[0]`
- Otherwise → `caption` or `hook+body+cta+text` space-join

### 9.2 Running the Tests

```bash
# Quality checks only (recommended — fast, ~30s)
cd artifacts/ai-training-server
python3 tests/test_w6_90m.py --quality-only

# Full suite (throughput + quality, ~5 min)
python3 tests/test_w6_90m.py

# Prerequisites: both servers must be running
curl http://localhost:9878/health   # Python server
curl http://localhost:8080/health   # Node proxy
```

Expected output: `Quality checks: 104/104 passed — VERDICT: PASS`

### 9.3 All Quality Checkers

| Checker | Endpoint | Key assertions |
|---|---|---|
| `chk_generate_content` | `/api/generate/content` | caption readable, hook ≥3 words, quality_score [0,100], intelligence block |
| `chk_generate_content_variants` | `/api/generate/content` | variants=3 → ≥2 returned, captions distinct |
| `chk_generate_text_content` | `/api/generate/text` mode=content | output readable |
| `chk_generate_text_planner` | `/api/generate/text` mode=planner | steps list non-empty |
| `chk_generate_campaign` | `/api/generate/campaign` | ≥4 posts across phases, readable content |
| `chk_generate_image` | `/api/generate/image` | image url present, format field |
| `chk_generate_audio` | `/api/generate/audio` | job_id or b64 present, status field |
| `chk_social_generate` | `/platform/social/generate` | post readable, not topic echo |
| `chk_ads_generate` | `/platform/ads/generate` | headline readable, body readable, CTA present |
| `chk_predict_engagement` | `/api/predict/engagement` | score in [0,1] |
| `chk_viral_score` | `/api/infer/viral-score` | score in [0,1] |
| `chk_content_score` | `/api/content/score` | score in [0,100] |
| `chk_sentiment` | `/api/analyze/sentiment` | label present, confidence present |
| `chk_distribution_plan` | `/platform/distribution/plan` | plan non-empty |
| `chk_daw_generate` | `/platform/daw/generate` | response non-empty |
| `chk_optimize_ad` | `/api/optimize/ad` | confidence in [0,1] or recommendations present |
| `chk_safety_screen` | `/api/safety/screen` | benign passes |
| `chk_pocket_multiply` | `/api/maxcore/pocket-multiply` | 2×2 result exactly `[[19,22],[43,50]]` |
| `chk_ads_optimize` | `/platform/ads/optimize` | optimizations non-empty |
| `chk_ads_audience` | `/platform/ads/audience` | segments non-empty |
| `chk_veo_status` | `/veo/status` | 404 expected (module absent), not a 500 |
| `chk_veo_compare("instagram/indie")` | `/api/generate/content` | Veo score ≥95 with full awareness payload |
| `chk_veo_compare("tiktok/hip-hop")` | `/api/generate/content` | Veo score ≥95 with full awareness payload |
| `chk_veo_compare("social/tiktok")` | `/platform/social/generate` | Veo score ≥95 with full awareness payload |
| `chk_veo_compare("ads/meta")` | `/platform/ads/generate` | Veo score ≥95 with full awareness payload |

**Veo comparison payload (all 4 tasks use awareness + instruction + content_themes):**
```python
# Example: ads/meta
{
  "user_id": "test_veo",
  "platform": "meta",
  "product": "new album",
  "goal": "streams",
  "genre": "indie",
  "artist_name": "Luna Voss",
  "num_creatives": 1,
  "awareness": {
    "contextString": "exclusive release, fire music, stream now",
    "trendingGenres": ["indie", "alternative"]
  },
  "instruction": "Focus on emotional resonance and exclusivity",
  "content_themes": ["exclusive release", "fire music", "stream now"]
}
```

---

## 10. Key Design Decisions & Non-Obvious Rules

### Awareness pipeline
1. **`idea` field stays clean.** Never concatenate awareness/brief context into `idea`.
   `idea` is templated raw into scene phrases. All context goes through `awareness`.
2. **Content themes are natural language in ads.** Use `"Trending themes: X, Y."` not
   `"• Woven around: X, Y."` in the ads ScriptAgent path — the bullet format is a
   template marker that the ads path doesn't strip and it echoes verbatim into the hook.
3. **Direction outranks signals.** In `merge_awareness`, the caller's `instruction` /
   `extra_context` / `content_themes` is prepended so it outranks generic trend signals.
4. **`_AwarenessMixin` normalises dict→string.** If a model uses plain `BaseModel`
   with `awareness: Optional[Any]`, `merge_awareness` must handle the dict case itself
   (the `_coerce_aw` guard in `orchestrator.py` does this).

### Server restart
5. **Kill the Python process directly.** `WorkflowsRestart` sends SIGTERM to Node,
   not to the Python child. The old Python process survives. Always use
   `pkill -9 -f "server\.py"` after editing Python files.
6. **PDIM dedup survives restarts.** Test with NOVEL topics after code changes — cached
   content silently serves old results.

### Hook quality guard
7. **Score-compare, don't presence-check.** The guard must compute the Veo hook score
   for both the model hook and the pool hook and prefer whichever is higher. A simple
   `"!" in hook` check accepts weak hooks that happen to have punctuation.

### Proxy allowlist
8. **Every new Python route needs a proxy entry.** The Node proxy is an explicit
   per-route allowlist in `model-proxy.ts`. A new endpoint that isn't listed returns
   404 at the dashboard even if Python returns 200.

### Thread safety
9. **PDIM single-flight uses `_Slot(event, result)`.** The leader writes
   `slot.result` BEFORE `_release()`. Followers read off the slot. A bare `Event`
   object causes a race where followers try to read `result` before it's set.

### Audio
10. **No audio without seeding.** `POST /storage/datasets/audio/seed` must be called
    before audio generation jobs will complete. The seeder is import-cached — restart
    Python after editing seeder code.

### FFmpeg
11. **Use `run_ffmpeg()` (posix_spawn), not fork.** Fork-based ffmpeg dies with
    `[Errno 5] EIO` under model memory pressure because the copy-on-write page table
    triggers OOM for the forked process.

### Ads subtypes
12. **`vary_subtypes=True` is the default.** A 4-creative request always produces
    video + audio + text + image. Pass `vary_subtypes=False` to lock all creatives
    to `ad_type`.

---

## 11. Recreating from Scratch — Step-by-Step

### Step 1: Create a new Replit project
- Choose a Node.js template (the primary workflow is Node-spawns-Python)
- Set language to Python 3.11 for the AI server

### Step 2: Set secrets
In Replit Secrets, add:
```
DATABASE_URL        = <Replit managed PostgreSQL URL>
ADMIN_KEY           = <choose a strong random key>
AI_TRAINING_KEY_PROD = f242bf97d7e46b7ca0b17cd6b01ca9239bc327b862a86b703556565523849701
SESSION_SECRET      = <choose a strong random key>
```

### Step 3: Install dependencies
```bash
# Python
pip install fastapi uvicorn pydantic psycopg2-binary python-multipart aiofiles

# Node
pnpm install
```

### Step 4: Configure workflow
Primary workflow command:
```
PORT=8080 MODEL_API_PORT=9878 \
  pnpm --filter @workspace/api-server run dev \
  & PORT=5000 BASE_PATH=/ API_PORT=8080 \
  pnpm --filter @workspace/ai-dashboard run dev
```

### Step 5: Seed the database
On first run, `server.py` auto-seeds the `api_keys` table with `AI_TRAINING_KEY_PROD`.
Verify: `curl http://localhost:9878/health` → `{"status":"ok"}`

### Step 6: Seed audio dataset
```bash
curl -X POST http://localhost:9878/storage/datasets/audio/seed \
  -H "X-Api-Key: f242bf97d7e46b7ca0b17cd6b01ca9239bc327b862a86b703556565523849701"
```

### Step 7: Verify quality
```bash
cd artifacts/ai-training-server
python3 tests/test_w6_90m.py --quality-only
# Expected: 104/104 passed
```

### Step 8: Proxy allowlist check
After adding any new Python endpoint, add a matching entry to:
```
artifacts/api-server/src/routes/model-proxy.ts
```
Pattern:
```typescript
router.post("/your/new/route", async (req, res) => {
  await proxyRequest(req, res, "/your/new/python/route");
});
```

### What to rebuild first if starting over
Priority order based on what unlocks everything else:

1. **`_AwarenessMixin` + `_merged_awareness_for`** — the awareness pipeline is the
   foundation. Without it, all generation is unconditioned.
2. **`ScriptAgent`** — generates all text content. Every other agent depends on it.
3. **`/api/generate/content`** — the primary content endpoint. Get this to 100/100
   Veo score before building other modalities.
4. **Veo scoring engine** (`veo_score_candidate` + components) — copy verbatim from
   `test_w6_90m.py` lines 109–168. This is the quality benchmark.
5. **Ad endpoint subtypes** — `AD_SPECS_BY_TYPE`, `AD_HOOKS_BY_TYPE`,
   `_generate_ad_creative` branching, `vary_subtypes` cycling, `_ad_hook_score` guard.
6. **Proxy allowlist entries** — for every endpoint you build.

---

*Documentation generated July 13, 2026; Sections 13–15 and the proxy route table updated July 28, 2026. Reflects the codebase as of 104/104 quality test pass.*

---

## 12. Digital GPU System — Complete Documentation

### 12.0 Why the Digital GPU replaces Replit's native CPU environment

Replit provides a CPU-only container — there is no hardware GPU. Instead of
writing CPU-style math (loose numpy/torch/scipy calls scattered through the
codebase), this project routes **all** compute through the Digital GPU: a
single, GPU-disciplined execution surface. Consequences of this decision:

- **One compute surface for everything.** Inference
  (`HyperCreativeTransformerLM`), training (`HyperTransformerLM` with
  fwd+bwd autograd routed through `gpu.*` kernels), sampling
  (`_gpu_softmax`), audio synthesis/STFT/HPSS (DFT-matrix GEMMs), image path
  tracing, and video grading all execute on the same kernel stack — no
  library soup (zero librosa/soundfile/scipy in any audio compute path).
- **GPU-style discipline on CPU hardware.** Kernels are tiled
  (flash attention with online softmax), fused (linear+gelu/silu), cached
  (content-hash pocket dedup on all GEMM paths), and compiled to native SIMD
  where it pays (`ai_model/gpu/native`, fused C kernels via gcc+ctypes).
- **Verifiability.** Because every op flows through the kernel layer, routing
  is testable: op counters (`gpu.core._total_ops`), fallback counters
  (`.engine_fallback` must stay 0), and telemetry prove nothing leaked to
  bare numpy.
- **Honesty about performance.** This is a software-defined GPU, not
  hardware: it brings structure, caching, and SIMD speedups — not GPU
  throughput. Estimator components (`MaxCoreSilicon`) are side-channel
  telemetry only, never an execution path.

**Enforcement:** call sites must use the MaxCore/Digital GPU wrappers — never
`TransformerLM`, `F.softmax`, `np.matmul`, or direct scipy DSP. Results
crossing back into plain-Python consumers unwrap via the Tensor's own
`.numpy()` (see §13.1) — read-out of a finished buffer, not a compute bypass.

The Digital GPU system is a layered software stack that brings hardware-GPU-style
programming discipline to CPU-only model training and inference. It is **not** an
emulator or a speedup hack — it is a principled execution model with clearly
defined components at each layer, all wired together so training and inference
share the same compute surface.

### 12.1 Conceptual Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  User code: HyperCreativeTransformer / HyperFlashAttention /... │
│  (torch.nn.Module subclasses, fully differentiable)             │
├─────────────────────────────────────────────────────────────────┤
│  HyperGPU Backend  (hyper_backend.py)                           │
│  torch.autograd.Function wrappers for every kernel:            │
│  _HyperGEMM, _MixedPrecisionGEMM, _FlashAttention,             │
│  _HyperConv2d, _HyperConv3d, _HyperLayerNorm, _HyperGELU, ...  │
│  Every backward() routes through HyperGPU or SIMDCore —        │
│  NOT through plain numpy/torch.                                  │
├─────────────────────────────────────────────────────────────────┤
│  HyperSIMDCore  (hyper_core.py — 38.9 KB)                       │
│  Numpy-backed tiled kernel library:                             │
│  flash_attention · batched_gemm · softmax (any axis)            │
│  conv2d · conv3d · layer_norm · gelu · silu · grouped_gemm      │
│  Tracks _total_ops for backward-pass verification               │
├──────────────────────────────┬──────────────────────────────────┤
│  DigitalGPU  (digital_gpu.py)│  Native SIMD  (native/)          │
│  VRAM-handle model + SIMD-   │  GCC-compiled fused C kernels    │
│  tiled ISA (5 opcodes)       │  (Path A) — 2.5–8× numpy         │
│  Scheduler + OOM detection   │  Never-raise numpy fallback       │
├──────────────────────────────┴──────────────────────────────────┤
│  Pocket GPU Pool  (pocket_pool.py)                              │
│  Infinite lifecycle pool backed by the pocket dimension         │
│  (PocketGPUInstance wraps DigitalGPU + VRAM, flushes on death)  │
├─────────────────────────────────────────────────────────────────┤
│  MaxCoreSilicon (silicon_model.py) ← side-channel ONLY          │
│  Architectural estimator: cycle budgets, bandwidth, tile model  │
│  is_measurement=False on every output — never an execution path  │
└─────────────────────────────────────────────────────────────────┘
```

---

### 12.2 File Map

| File | Size | Purpose |
|---|---|---|
| `gpu/__init__.py` | 1.3 KB | Public exports |
| `gpu/digital_gpu.py` | 12.8 KB | DigitalGPU ISA + VRAM + Scheduler |
| `gpu/digital_library.py` | 2.8 KB | Primitive library (reusable op combos) |
| `gpu/execution_graph.py` | 21.0 KB | DAG-based execution graph / scheduler |
| `gpu/gpu_trainer.py` | 8.0 KB | GPU-routed training loop (basic) |
| `gpu/hyper_backend.py` | 21.2 KB | `torch.autograd.Function` wrappers (fwd+bwd) |
| `gpu/hyper_core.py` | 38.9 KB | HyperSIMDCore — all kernel implementations |
| `gpu/hyper_creative_transformer.py` | 14.4 KB | Production transformer + KV-cache |
| `gpu/hyper_trainer.py` | 9.6 KB | Full GPU-routed training loop (HyperGPU) |
| `gpu/hyper_transformer.py` | 3.2 KB | Lightweight transformer variant |
| `gpu/multi_backend.py` | 6.8 KB | Multi-backend dispatch router |
| `gpu/multi_stream.py` | 9.3 KB | Concurrent stream execution |
| `gpu/opcode_spec.py` | 17.2 KB | Opcode definitions, shape rules, `is_hardware_execution` flag |
| `gpu/pocket_pool.py` | 7.2 KB | PocketGPUPool — infinite GPU lifecycle pool |
| `gpu/precision.py` | 6.3 KB | fp8/fp16/bf16 numerics models |
| `gpu/silicon_model.py` | 12.3 KB | MaxCoreSilicon architectural estimator |
| `gpu/telemetry.py` | 4.2 KB | Telemetry: wall_ms (measured) vs flops (derived) |
| `gpu/torch_backend.py` | 14.4 KB | Torch device backend (real CUDA gating) |
| `gpu/accelerated_transformer.py` | 3.0 KB | Thin wrapper transformer |
| `gpu/native/compiler.py` | — | GCC/ctypes ISA detection + compilation |
| `gpu/native/kernels.py` | — | Fused C kernel bindings + numpy fallback |
| `gpu/native/prototype.py` | — | Path-A benchmarking script |
| `gpu/native/cuda/nvcc.py` | 9.6 KB | NVCC compilation pipeline |
| `gpu/native/cuda/runtime.py` | 9.8 KB | CUDA runtime dispatch |
| `gpu/native/cuda/sm102_kernels.py` | 10.9 KB | sm_102-style kernel implementations |
| `gpu/native/cuda/*.cu` | — | CUDA kernels: flashattn, conv, reduction |

---

### 12.3 DigitalGPU — The ISA Layer

**File:** `gpu/digital_gpu.py`

`DigitalGPU` is a SIMD-tiled soft-ISA executor. It models a GPU's programming
model (handle-based VRAM, explicit instruction dispatch, tiled execution) in
pure numpy, with a hardware execution flag forced to `False` throughout.

#### VRAM

```python
class VRAM:
    def alloc(self, array: np.ndarray) -> int   # returns handle (int id)
    def get(self, hid: int) -> np.ndarray        # fetch by handle
    def meta(self, hid: int) -> dict             # shape/dtype/size without data copy
    def free(self, hid: int)                     # release handle
```

All DigitalGPU ops accept either raw numpy arrays or integer VRAM handles.
The `h_*` method variants take handles: `h_gemm(ha, hb)` instead of `gemm(A, B)`.

#### Opcodes

```python
class OpCode(Enum):
    GEMM           # 2D matrix multiply (shape-validated)
    ADD            # elementwise addition
    SOFTMAX        # row-wise softmax
    ATTENTION      # scaled dot-product attention (Q, K, V)
    GEMM_BIAS_RELU # fused: gemm → add bias → relu
```

Opcode shape rules and type rules are defined in `opcode_spec.py`.
The `is_hardware_execution` flag is set to `False` on every opcode spec and is
enforced there — it is not something any caller can override.

#### Instruction / Program / SIMDCore

```python
class Instruction:
    opcode: OpCode
    args: dict          # operand arrays or handles

class Program:
    instructions: list  # sequence of Instruction objects
    def add(instr)

class SIMDCore:
    lanes: int = 32     # SIMD width (modeled)
    tile_m: int = 64    # matmul tile dims
    tile_n: int = 64
    tile_k: int = 64
    def gemm_tiled(A, B) -> np.ndarray   # tiled GEMM loop
```

#### OOM detection (live-set tracking)

`DigitalScheduler.run()` tracks a live-set of tensor sizes keyed by tensor id.
On every write, the current allocation is replaced in the set. If the total
exceeds `max_bytes`, an `OOMError` is raised before the op executes.
This gives deterministic OOM detection without needing a real allocator.

#### Error hierarchy

```python
GPUError            # base
├── ShapeError
│   └── ShapeMismatchError   # incompatible shapes for op
├── TypeErrorGPU             # non-ndarray passed to VRAM.alloc
├── InvalidOpcodeError       # unknown opcode
└── OOMError                 # live-set exceeded max_bytes
```

---

### 12.4 Opcode Spec and is_hardware_execution

**File:** `gpu/opcode_spec.py`

This is the authoritative registry of all opcodes. Each opcode entry carries:
- Input/output shape validation rules
- Allowed dtypes
- `is_hardware_execution: False` — hardcoded, not a runtime flag

**Rule:** `is_hardware_execution` gates real hardware paths. It is `False`
throughout the Digital GPU stack. Any code that checks this flag and branches
on `True` will never take that branch in this system. Do not add
`is_hardware_execution: True` entries — the entire point of the Digital GPU is
to be a fully software-defined execution model.

---

### 12.5 HyperSIMDCore — The Kernel Library

**File:** `gpu/hyper_core.py` (38.9 KB — the largest GPU file)

`HyperSIMDCore` is the numpy-backed implementation of every kernel. It is the
"physical" compute layer that all higher abstractions call through.

#### Complete kernel interface

```python
class HyperSIMDCore:
    _total_ops: int    # counter incremented by every kernel call, including backward

    # Matrix multiply
    tensor_core_gemm(A, B)           -> np.ndarray  # standard 2D matmul
    mixed_precision_gemm(A, B)       -> np.ndarray  # fp8-modeled accumulation
    batched_gemm(A, B)               -> np.ndarray  # np.matmul for batched/3D dispatch

    # Attention
    flash_attention(Q, K, V, causal=True, block_size=64) -> np.ndarray

    # Convolution
    conv2d(X, W, stride=1, padding=0) -> np.ndarray
    conv3d(X, W, stride=1, padding=0) -> np.ndarray

    # Normalisation
    layer_norm(X, gamma, beta, eps=1e-5)  -> np.ndarray
    batch_norm(X, gamma, beta, ...)       -> np.ndarray

    # Activations
    gelu(X)  -> np.ndarray
    silu(X)  -> np.ndarray

    # Elementwise / reduction
    softmax(X, axis=-1)  -> np.ndarray   # any axis, not just last
    add(A, B)            -> np.ndarray

    # Grouped
    grouped_gemm(A_list, B_list) -> List[np.ndarray]
```

#### Flash Attention — tiled online-softmax

`flash_attention` is a real tiled online-softmax implementation that uses
**16× less score memory** than standard attention for long sequences.

**Algorithm:**
```
For each query block Tq:
    m = -inf, l = 0, O = 0   ← running max, denominator, output accum
    For each key/value block Tk:
        S = Q_block @ K_block.T / sqrt(D)     ← score tile
        if causal: mask upper-right triangle
        m_new = max(m, row_max(S))
        l = exp(m - m_new) * l + sum(exp(S - m_new), axis=-1)
        O = exp(m - m_new) * O + exp(S - m_new) @ V_block
        m = m_new
    O /= l    ← final normalisation
```

**Critical implementation rules:**
1. **Outer loop iterates over Tk (K/V blocks), not Tq.** Iterating over Tq
   prevents efficient streaming of K/V and breaks cross-attention. If you
   refactor this, the loop order must stay `for each Tq { for each Tk { ... } }`.
2. **`block_size` is clamped to `[1, Tk]`** before use — never let it exceed
   the sequence length or divide by zero.
3. **Attention-matrix dropout cannot be applied inside the tiled kernel** because
   the full score matrix is never materialised. Apply dropout to the output
   tensor `O` after the kernel returns instead.

#### Conv2D / Conv3D — strided im2col

Both `conv2d` and `conv3d` use `np.lib.stride_tricks.as_strided` to build a
zero-copy patch view (im2col), then execute a single GEMM on the view.

```
X_col = as_strided(X, shape=(N, H_out, W_out, C_in, kH, kW),
                   strides=(...))   # zero-copy
out = X_col.reshape(N*H_out*W_out, C_in*kH*kW) @ W.reshape(C_out, -1).T
```

**Speedup:** 1.4×–2.3× over naive patch loops. Native SIMD fusion (Path A)
claims 9–10× over naive loops for compatible sizes.

**Caveat:** `as_strided` produces a read-only alias. The GEMM must not write back
into the view. Always reshape/copy before any in-place operation on `X_col`.

#### Softmax — any-axis

`HyperSIMDCore.softmax(X, axis)` accepts any integer axis, not just `-1`.
The 2D engine "folds" multi-axis reductions through this single kernel — callers
must reduce multi-axis products to a single `axis` int before calling (torch
`dim` is always a single int; multi-axis prod must be folded by the caller).

#### `_total_ops` counter and backward-pass verification

Every kernel call (including in backward passes) increments `_total_ops`.
To verify that a backward pass is correctly routed through HyperGPU kernels
and not torch:

```python
before = gpu.core._total_ops
loss.backward()
delta = gpu.core._total_ops - before
assert delta > 0, "backward did not call any HyperGPU kernels"
```

A delta of 0 after `.backward()` means the backward graph is using plain torch
ops, not the `hyper_backend` autograd Functions.

---

### 12.6 HyperGPU Backend — Autograd Functions

**File:** `gpu/hyper_backend.py` (21.2 KB)

This is the torch autograd integration layer. Every operation has a paired
`torch.autograd.Function` subclass with a `forward()` that calls HyperSIMDCore
and a `backward()` that routes gradients back through HyperGPU kernels.

#### Autograd Function inventory

| Class | forward kernel | backward routing |
|---|---|---|
| `_HyperGEMM` | `gpu.core.tensor_core_gemm` | `gpu.core.tensor_core_gemm` (grad_A, grad_B) |
| `_MixedPrecisionGEMM` | `gpu.core.mixed_precision_gemm` | `gpu.core.tensor_core_gemm` |
| `_FlashAttention` | `gpu.core.flash_attention` | backward matmuls via `gpu.gemm`, softmax via `gpu.softmax` |
| `_HyperConv2d` | `gpu.core.conv2d` | `gpu.core.conv2d` (grad_X, grad_W) |
| `_HyperConv3d` | `gpu.core.conv3d` | `gpu.core.conv3d` (grad_X, grad_W) |
| `_HyperLayerNorm` | `gpu.core.layer_norm` | elementwise/reduction via `gpu.core` |
| `_HyperGELU` | `gpu.core.gelu` | `gpu.core._total_ops += 1` (elementwise) |
| `_HyperSiLU` | `gpu.core.silu` | `gpu.core._total_ops += 1` |

**Critical for training:** Both `forward` and `backward` of every Function must
call `gpu.*` kernels. If `backward` falls back to plain `torch` ops, the
`_total_ops` counter delta will be lower than expected and the training routing
is broken. Verify with the counter-delta test above.

**Flash attention backward note:** The backward pass for `_FlashAttention`
routes attention-matrix backward matmuls through `gpu.gemm` and the softmax
gradient through `gpu.softmax`. It does NOT recompute the full attention matrix
inside the tiled kernel — it uses the saved Q, K, V from `ctx.save_for_backward`.

#### nn.Module wrappers

These sit above the autograd Functions and expose a standard `nn.Module` interface:

```python
HyperGPULinear(in_features, out_features, gpu, mixed_precision=False, training_mode=False)
HyperFlashAttention(dim, n_heads, gpu, block_size=64, training_mode=False)
HyperConv2d(in_channels, out_channels, kernel_size, gpu, stride=1, padding=0)
HyperConv3d(in_channels, out_channels, kernel_size, gpu, stride=1, padding=0)
HyperLayerNorm(dim, gpu, eps=1e-5, training_mode=False)
HyperGELU(gpu, training_mode=False)
HyperSiLU(gpu, training_mode=False)
```

`training_mode=True` ensures gradients are enabled and the autograd Functions
are used. `training_mode=False` (inference) may take a faster path that still
routes through HyperSIMDCore but skips gradient bookkeeping.

---

### 12.7 HyperCreativeTransformer — Production Model with KV-Cache

**File:** `gpu/hyper_creative_transformer.py` (14.4 KB)

This is the deployed model used by ScriptAgent, VideoAgent, and all generation
endpoints. It is a decoder-only transformer with RoPE attention and SwiGLU FFN.

#### Architecture

```
HyperCreativeTransformerLM
├── nn.Embedding (token_emb)          vocab_size × dim
├── HyperTransformerDecoderLayer × 8
│   ├── HyperLN (ln1)                 pre-attention layernorm
│   ├── HyperRoPESelfAttention (attn)
│   │   └── HyperLinearNL (qkv, out)  QKV projection + output projection
│   ├── HyperLN (ln2)                 pre-FFN layernorm
│   └── HyperSwiGLUFFN (ffn)
│       └── HyperLinearNL (gate, down)
└── HyperLN (ln_final)
```

#### Constructor

```python
HyperCreativeTransformerLM(
    vocab_size: int,   # 443 (compact music-domain vocabulary)
    dim: int = 512,
    n_layers: int = 8,
    n_heads: int = 8,
    max_len: int = 1024,
    dropout: float = 0.1,
    gpu: HyperGPU | None = None,  # if None, falls back to torch ops
)
```

#### KV-Cache interface — mandatory for production

Without KV-cache, every decode step recomputes attention over the full
past-token sequence at O(T²) cost. At the sequence lengths used by
`ScriptAgent.plan()`, this takes **3+ minutes** per call. With KV-cache it
takes seconds.

```python
# Prefill: process the entire prompt once, return cache
logits, kv_cache = model.prefill(
    x: torch.Tensor,                        # [B, T] token ids
    key_padding_mask: torch.Tensor | None   # [B, T] float mask
)
# kv_cache is a list of (K, V) tensor tuples, one per layer

# Decode: O(1) per step — appends one token to cache
logits, kv_cache = model.decode_one(
    x_new: torch.Tensor,                       # [B, 1] next token id
    kv_cache: list[tuple[Tensor, Tensor]],     # updated in-place
    key_padding_mask: torch.Tensor | None      # updated mask
)
```

**Rule:** Any refactor of the transformer must preserve both `prefill()` and
`decode_one()`. If you replace the KV-cache interface, generation will regress
to multi-minute latency immediately.

#### Inference loop pattern

```python
# Used inside ScriptAgent and VideoAgent
kv = None
tokens = tokenizer.encode(prompt)
x = torch.tensor([tokens])
logits, kv = model.prefill(x)
for _ in range(max_new_tokens):
    next_tok = sample(logits)
    if next_tok == EOS: break
    x_new = torch.tensor([[next_tok]])
    logits, kv = model.decode_one(x_new, kv)
```

---

### 12.8 Native SIMD — Path A

**Files:** `gpu/native/compiler.py`, `gpu/native/kernels.py`, `gpu/native/prototype.py`

Path A compiles fused C kernels at startup to CPU SIMD instructions. It is
**not** GPU hardware — it is CPU SIMD (AVX-512 / AVX2 / AVX). The speedup
comes from loop fusion eliminating intermediate array allocations, not from
any GPU-like hardware.

#### Compilation pipeline (compiler.py)

1. **ISA detection:** Probes `gcc` availability and parses `/proc/cpuinfo` for
   the widest supported ISA: AVX-512 → AVX2 → AVX → scalar fallback.
2. **C source generation:** Generates a single-file C source in a temp directory.
3. **Compilation:** `gcc -O3 -ffast-math -funroll-loops -mavx512f` (or narrower).
4. **Linking:** Produces a `.so` shared library loaded via `ctypes`.
5. **Never-raise:** If gcc is absent or compilation fails, falls back silently
   to the numpy implementations in `NativeKernels`.

#### Fused kernel inventory (kernels.py)

| Kernel | What it fuses | Speedup claim |
|---|---|---|
| `affine_relu_sq` | `x*a + b → relu → square` | 2.5–4× |
| `hardswish` | `x * clamp(x+3,0,6) / 6` | 3× |
| `axpby` | `a*x + b*y` | 3–5× |
| `silu` | `x * sigmoid(x)` | 2.5–4× |
| `gelu` | `0.5x(1 + erf(x/√2))` | 2.5–4× |
| `softmax_rows` | row-wise softmax (max-subtract + exp + sum) | 3–5× |
| `rmsnorm_rows` | RMS norm + gamma scale | 4–8× |
| `layernorm_rows` | mean-subtract + var-norm + gamma + beta | 4–8× |

**Claimed aggregate speedup over numpy:** 2.5–8× for fused elementwise ops.
This is specifically **not** for GEMM — integer matmul is ~50× *slower* than
float32 because there are no integer tensor-core paths here. GEMM uses
`np.matmul` (BLAS-backed).

#### BLAS thread count

BLAS thread count is set at runtime from `os.cpu_count()` **before** the numpy
import. Single-process inference: this is a no-op (BLAS manages its own threads).
Multi-process workers: cap to `cpu_count // N_workers` to prevent thread
oversubscription. Do not set `OMP_NUM_THREADS` in the process environment
after numpy has been imported — it has no effect.

#### `NativeKernels` fallback

```python
class NativeKernels:
    stats: dict   # {"native": int, "fallback": int}

    def _use_native(self, *arrs) -> bool:
        # True only when native .so is loaded and all inputs are contiguous float32
```

The `stats["native"]` / `stats["fallback"]` counters let you verify what
fraction of calls hit the compiled kernels vs the numpy path.

#### Path A vs Path B vs numpy

| Path | Mechanism | Target use |
|---|---|---|
| **Path A (native SIMD)** | GCC-compiled C → ctypes → CPU SIMD | Fused elementwise ops in production; 2.5–8× numpy |
| **Path B (CUDA)** | nvcc-compiled .cu → CUDA runtime | Real GPU hardware if available; not the primary path |
| **Numpy fallback** | `np.*` directly | Always present; never-raise guarantee |

**Path A is the production path on Replit** (no GPU hardware). Path B exists in
`native/cuda/` for environments with a real CUDA device.

---

### 12.9 Pocket GPU Pool

**File:** `gpu/pocket_pool.py`

An infinite GPU lifecycle pool backed by the pocket dimension. Unlike a fixed
thread pool, there is no cap — the pocket absorbs any burst.

```python
class PocketGPUInstance:
    digest: str          # content-hash key identifying this instance
    gpu: DigitalGPU      # isolated DigitalGPU + VRAM per instance
    def begin_work()     # mark start of work (timestamps)
    def die() -> float   # flush to pocket, return alive_ms
    @property alive_ms   # wall time since begin_work()

class PocketGPUPool:
    async def spawn(digest="") -> AsyncIterator[PocketGPUInstance]
    def spawn_sync(digest="")  -> Iterator[PocketGPUInstance]
    def stats() -> dict        # total_spawned, total_died, currently_alive, pool_source
```

**Lifecycle:** Each instance gets its own isolated `DigitalGPU` and `VRAM`.
On `die()`, the instance flushes its state back to the pocket dimension and
is garbage-collected. There are no memory leaks into the pocket regardless
of how many instances have lived.

**pocket_source in stats:** Always `"pocket_dimension"`. This lets monitoring
code distinguish pool allocations from other VRAM sources.

---

### 12.10 Pocket Accelerator — GEMM Dedup Cache

**File:** `gpu/hyper_core.py` (PocketAccelerator class)

All GEMM paths route through an adaptive content-hash pocket dedup cache.
Identical matrix pairs — identified by a content digest — skip recomputation
and return the cached result.

#### Key construction

```python
def _digest(A: np.ndarray, B: np.ndarray, extra_key: str = "") -> str:
    # Ensures contiguous memory layout before hashing — critical:
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    # SHA256 of: dtype + shape + raw bytes of A + raw bytes of B + extra_key
    return sha256(...)
```

**`ascontiguousarray` alias bug:** `np.ascontiguousarray` returns the *same
object* if the array is already contiguous — it is an alias, not a copy. If
you then modify the original array in-place, the cached value will reflect
that modification. Always use `.copy()` when storing into the cache:

```python
# WRONG — aliases the original
self._cache[key] = A

# CORRECT — owns the data
self._cache[key] = A.copy()
```

`extra_key` must include the precision mode flags (fp8/fp16/fp32). Two GEMM
calls with the same matrices but different precision profiles must not share
a cache entry.

#### Adaptive gate

After 32 warmup calls per key, the gate mutes (skips the cache lookup) for
pockets with a hit rate below 5%. It re-probes every 256 calls to detect
whether the working set has changed. This prevents the cache from becoming a
net negative for highly varied matrix pairs.

#### Fleet-dedup namespacing

Tests that share the `PocketAccelerator` instance must use per-run namespaces
in `extra_key` (e.g., `extra_key=f"test_run_{uuid}"`) to prevent cross-test
cache hits that silently mask correctness failures.

---

### 12.11 Precision System — fp8 / fp16 / bf16

**File:** `gpu/precision.py`

This is a **numerics model**, not a hardware accelerator. It rounds float values
onto the representable grid of a lower-precision format so you can measure the
rounding error. All arithmetic runs in fp64 on the host CPU. There is no speedup.

```
precision.py is a quantization-error study tool.
It is NOT faster than fp32 and MUST NOT be cited as a throughput benchmark.
```

#### fp8 formats (OCP 8-bit spec)

| Format | Layout | Bias | Max normal | Inf? | Use |
|---|---|---|---|---|---|
| `e4m3` | 1-4-3 bits | 7 | **448.0** | ❌ saturating | Weights / activations (narrower range, no inf) |
| `e5m2` | 1-5-2 bits | 15 | **57344.0** | ✅ IEEE-like | Gradients (wider range, has inf for overflow) |

**Key rule:** e4m3 has no infinity. Overflow saturates to ±448.0. e5m2 has
infinity. Overflow rounds up to ±inf. Do not treat them interchangeably — the
inf behaviour difference is material for gradient stability.

#### API

```python
to_fp8(x, fmt="e4m3") -> np.ndarray   # round to fp8 grid, return as float64
to_fp16(x)            -> np.ndarray   # round to IEEE fp16 and back to fp32
to_bf16(x)            -> np.ndarray   # round fp32 → bf16 → fp32 (round-half-to-even)

cast_numeric(x, profile: str) -> np.ndarray
# profile: "fp32_strict" | "fp16_mixed" | "bf16_mixed" | "fp8_mixed"

flash_attention_fp8_model(Q, K, V, causal=False, fmt="e4m3") -> np.ndarray
# Numerics model of fp8 FlashAttention:
# fp8 inputs → fp16 accumulate → fp32 softmax → fp16 output

quantization_error(x, profile="fp8_mixed") -> dict
# Returns: {profile, max_abs_err, mean_abs_err, rel_mean_abs_err}
# The honest use case: measuring what a low-precision format costs numerically
```

#### `_round_to_grid` — the core rounding function

Handles subnormal spacing, round-half-to-even, NaN preservation, and the
format-specific overflow behaviour (saturate vs inf). Non-finite inputs are
restored explicitly after the grid rounding to ensure faithful edge-case
behaviour.

---

### 12.12 MaxCoreSilicon — Architectural Estimator

**File:** `gpu/silicon_model.py` (12.3 KB)

`MaxCoreSilicon` is an optional **architectural performance model** — the same
class of tool as gem5 or a roofline calculator. It estimates cycle budgets and
memory-transfer times for a hypothetical chip defined by user-chosen constants.

**It does not execute arithmetic. It does not make anything faster.**

```python
MaxCoreSilicon(
    tile_count: int,               # number of modeled compute tiles
    flops_per_tile_per_cycle: int, # throughput assumption (you choose this)
    memory_bandwidth_bytes_per_sec: float,  # HBM-like BW assumption
    clock_hz: float,               # clock frequency assumption
    silicon: None = None,          # always None in production
)
```

#### `is_measurement = False` — the cardinal rule

Every value `MaxCoreSilicon` produces is tagged `is_measurement=False`:

```python
def report(self, critical_path_cycles=None) -> dict:
    return {
        "estimated_cycles": ...,
        "estimated_wall_seconds": ...,
        "estimated_bytes_moved": ...,
        "is_measurement": False,   # ALWAYS False
        ...
    }
```

**These numbers MUST NOT be:**
- Presented as measured throughput
- Mixed with real wall-clock timings
- Cited as benchmarks
- Compared to other systems' measured numbers

They are what-if estimates for capacity planning: "how would this workload
behave on a die with N tiles and X TB/s?" before committing to real silicon.

#### Wiring as side-channel telemetry

In the live engines, `MaxCoreSilicon` attaches as an optional side channel:

```python
# In HyperGPU / DigitalGPU constructors
self.silicon: MaxCoreSilicon | None = None   # default: telemetry disabled
```

Each executed op is *also* recorded in silicon (if not None) to accumulate an
estimated cycle budget — alongside, never instead of, real computation:

```python
# Pattern used in every kernel wrapper
result = gpu.core.flash_attention(Q, K, V, ...)    # real compute
if self.silicon is not None:
    self.silicon.add_estimated_cycles(op)          # side-channel only
```

**The silicon estimator must never crash real compute.** All silicon calls are
wrapped in `try/except` in the kernel wrappers.

#### Component classes

```python
SRAM                 # modeled on-tile scratchpad (capacity bookkeeping)
RegisterFile         # modeled register file
KVCacheSlice         # modeled KV-cache slice entries
GlobalMemory         # modeled HBM: bandwidth_bytes_per_sec + transfer estimator
ComputeTile          # single tile: estimate_cycles(op) → int
MaxCoreOp            # op descriptor: name, flops, bytes_moved, shape
MaxCoreSilicon       # top-level estimator: all tiles + global memory + clock
```

---

### 12.13 Torch Backend — Real CUDA Gating

**File:** `gpu/torch_backend.py` (14.4 KB)

The torch backend wraps real `torch.cuda` for environments where a physical GPU
is present. `is_available()` gates on `torch.cuda.is_available()` — there is no
CPU masquerade.

```python
class GPUBackend:
    @staticmethod
    def is_available() -> bool:
        return torch.cuda.is_available()   # real check, no fallback lie

class CPUBackend:
    # Used when is_available() is False
    # Routes through NativeKernels (Path A SIMD) then numpy
```

**Multi-axis product folding:** When a kernel needs a multi-axis reduction,
the caller must fold to a single `dim` int before calling torch. PyTorch's
`dim` argument is a single int (or tuple for some ops, but torch matmul/softmax
expect single). The HyperSIMDCore `softmax(X, axis)` handles any int axis;
torch ops need the caller to fold.

**Performance reality on Replit (CPU only):**
- `GPUBackend.is_available()` → `False`
- `CPUBackend` → `NativeKernels` (Path A) → numpy
- Real CUDA kernels (Path B, `native/cuda/`) are unreachable on Replit

The ~100–1000× "slower than torch" figure cited in memory refers to the
**Digital GPU ISA layer** (`SIMDCore.gemm_tiled` with explicit tiling loops in
Python) versus torch's BLAS-backed matmul. The `HyperSIMDCore` numpy paths
are much closer to torch. The tiling overhead only applies when using the
explicit `SIMDCore` / `DigitalGPU` ISA layer directly.

---

### 12.14 Telemetry

**File:** `gpu/telemetry.py` (4.2 KB)

```python
class Telemetry:
    wall_ms: float         # MEASURED — actual wall-clock milliseconds
    flops: float           # DERIVED — analytic from op shapes (not measured)
    bytes_moved: float     # DERIVED — analytic from tensor sizes
    op_name: str
    is_measurement: bool   # True for wall_ms, False for flops/bytes_moved
```

The distinction between measured and derived fields is enforced at the class
level. Never aggregate `flops` (derived) with `wall_ms` (measured) into a
single throughput number without preserving this distinction.

---

### 12.15 Training Routing — HyperGPU fwd+bwd

**File:** `gpu/hyper_trainer.py` (9.6 KB)

The full training loop routes **both forward and backward** through HyperGPU
kernels. This is the defining property of "HyperGPU training" vs plain torch:

```
Forward:  model(x) → loss     ← all ops via HyperSIMDCore
Backward: loss.backward()     ← all grad ops via hyper_backend.py autograd Functions
```

`gpu_trainer.py` (8.0 KB) is the simpler training loop that uses `DigitalGPU`
directly. `hyper_trainer.py` is the production loop that uses `HyperGPU` + the
autograd Functions.

**Verifying correct routing after any refactor:**
```python
ops_before = model.gpu.core._total_ops
output = model(input_batch)
loss = criterion(output, targets)
loss.backward()
ops_after = model.gpu.core._total_ops
assert ops_after > ops_before, \
    "Neither forward nor backward called any HyperGPU kernels"
# For a more precise check:
assert (ops_after - ops_before) >= expected_ops_per_batch
```

---

### 12.16 GPU System Design Rules (Non-Negotiable)

These rules were learned through iteration and must be preserved in any rebuild:

1. **KV-cache is mandatory.** `prefill()` + `decode_one()` must exist on the
   transformer. Without them, `plan()` takes 3+ minutes per call.

2. **Flash outer loop is Tk, not Tq.** The online-softmax outer loop iterates
   over K/V blocks. Reversing this breaks streaming and cross-attention.

3. **`block_size` must be clamped.** Always clamp to `[1, Tk]` before the flash
   inner loop. Unclamped block_size causes a divide-by-zero or shape mismatch.

4. **Dropout inside flash is impossible.** The full score matrix is never
   materialised. Apply dropout to the output tensor O after the kernel returns.

5. **`is_hardware_execution = False` everywhere.** This is not a runtime flag.
   It is a design invariant enforced in `opcode_spec.py`.

6. **`is_measurement = False` on all silicon estimates.** Never mix estimated
   cycles with real wall_ms. Never present silicon output as a benchmark.

7. **Pocket cache `.copy()` rule.** Never store a raw `np.ascontiguousarray`
   result in the cache — it aliases the original. Call `.copy()` before caching.

8. **Fleet-dedup test isolation.** Tests that share a `PocketAccelerator` must
   include a per-run `extra_key` suffix (e.g., a UUID) to prevent cross-test
   cache hits.

9. **Don't clobber `DigitalGPU` / `GPUBackend`.** Adding new paths to these
   classes requires reading `opcode_spec.py` first — new opcodes need a full
   spec entry including shape rules and the `is_hardware_execution: False` flag.

10. **Multi-axis product → single dim before torch/softmax.** PyTorch softmax
    takes a single `dim`. Fold multi-axis reductions before calling.

11. **BLAS threads before numpy import.** Set `OMP_NUM_THREADS` and
    `MKL_NUM_THREADS` before the first `import numpy` or the setting has no
    effect. Multi-process workers should cap to `cpu_count // N_workers`.

12. **Silicon is side-channel, not execution.** `silicon.add_estimated_cycles()`
    must be called inside a `try/except`. If it raises, real compute must
    continue unaffected.

---

## 13. Audio Delivery: Stems, Caching & the Fast Path

### 13.1 Stem separation pipeline

`POST /api/generate/audio` with `"stems": true` produces, alongside the main
track, three isolated stems — **drums / bass / melody** — as 44.1 kHz stereo
WAV files.

Flow (all inside `artifacts/ai-training-server`):

1. `server.py` audio render block calls `separate_stems()` in
   `ai_model/audio/producer_tools.py`.
2. Separation is HPSS-style (harmonic/percussive + band split), built on
   STFT/iSTFT in `ai_model/audio/digital_gpu_synth.py`.
3. **All heavy math runs on the Digital GPU**: STFT and iSTFT are expressed as
   DFT-matrix GEMMs executed via `DigitalGPU().gemm(...)`. No librosa, no
   scipy, no plain-numpy matmul in the transform itself.
4. Stem files are written to `uploads/stems/<job_id>/<stem>.wav` and surfaced
   on the job result as `stems: {drums: url, bass: url, melody: url}`.

**The Tensor boundary rule (critical, fixed 2026-07-28):**
`DigitalGPU().gemm` returns a MaxCore `Tensor` object. It has `.numpy()` but
**not** `.astype`. Any numpy consumer of a gemm result (overlap-add, dtype
casts) must unwrap through the `_gemm_np()` helper in `digital_gpu_synth.py`.
Before this fix, every stems request crashed inside the never-raise stem path
with `'Tensor' object has no attribute 'astype'`, which surfaced only as a
one-line `[Producer] stems skipped:` warning — the job still reported `done`
with `stems: {}`. **If stems ever come back empty, grep the Python log for
"stems skipped" first.** The unwrap happens only on the finished result
buffer; the GEMM compute itself stays on the Digital GPU (routing contract
intact).

### 13.2 Stem downloads through the proxy

The dashboard fetches stems in two steps:

1. `GET /api/audio/:jobId/stems` → JSON map of stem name → URL.
2. `GET /api/files/stems/:jobId/:filename` → binary WAV stream.

Route (2) was missing from the Node allowlist until 2026-07-28 (stem URLs
404'd from the dashboard). It now lives in `model-proxy.ts` next to the MIDI
route, uses `proxyBinaryStream`, shares `_audioLimiter`, and
`encodeURIComponent`s both params. Python-side errors (e.g. unknown job) pass
through as Python's JSON 404 — proving requests reach the Python server.

**Path-traversal hardening (Python side, `api_serve_stem_file`):** the old
check was a string `startswith` on the resolved path — bypassable by sibling
directories sharing the uploads prefix and by encoded separators. It is now:

- `job_id` and `filename` must fullmatch `[A-Za-z0-9._-]+` and must not be
  `.`/`..`; `Path(filename).name` must equal `filename` (no separators).
- The resolved path must satisfy
  `stem_path.relative_to(stems_root / job_id)` — a real containment check,
  not a prefix comparison.

Verified: `../../server.py`, `..%2F..%2Fserver.py`, and
`%2e%2e%2f...` payloads all return 404 through the Node proxy; legitimate stem
fetches return 200 with full WAV bytes.

### 13.3 The audio render cache (L1) and fast path

Audio has a persistent-process cache, `_AUDIO_RENDER_CACHE` in `server.py`
(max 256 entries, LRU-ish eviction). Two key families:

- **Chunk key**: `(chunk_idx, duration_bucket, format, loudness, stems_flag)`
  — written after every pool-served render.
- **Genre key**: `("genre", genre, duration_bucket, format, loudness,
  stems_flag)` — written for every preferred genre, probed by the request
  handler *before* any dataset lookup (this is the few-ms fast path).

Hits are validated: the main file must still exist on disk, and — since
2026-07-28 — a `stems=true` probe additionally requires the cached entry to
carry a **complete, on-disk** stem set. The stems flag is part of the key, so
a stems request can only match a stems render.

**Why the stems gate mattered:** the flywheel ingests every finished render
back into the dataset, so an identical repeat request often selects a *newer
chunk* than the first render → chunk-key miss. The genre-key fast path is
what makes repeats fast, and it previously excluded `stems=true` for no
reason, forcing ~20 s full re-renders. After the fix, stems repeats are
~0.6 s.

**Cache lifetime:** in-process only. A Python restart empties it, so the first
"repeat" after a reboot re-renders once, and content-level dedup caches (disk
backed) may still short-circuit some paths. When benchmarking fresh renders,
always use a **novel topic string**.

### 13.4 Flywheel direct serve

If the selected pool chunk originated from the flywheel (already a final,
mastered MP3) and the request is `mp3` + no stems + default loudness, bytes
are written straight to the output path with zero ffmpeg work
("flywheel direct" log line). Stems requests can never use this path — stems
must be synthesized from the audio — which is exactly why the genre-key cache
is their fast path instead.

---

## 14. Performance Characteristics & Operations Runbook

### 14.1 Measured end-to-end latency (submission → job done, 2026-07-28)

Via the Node proxy on :8080, warm server:

| Request | Cold (novel topic) | Warm repeat (cache/dedup) |
|---|---|---|
| Text (sync) | ~67 ms | ms-level |
| Image (sync) | ~1.4 s | ~0.9 s |
| Audio 30 s | ~16 s | **~130 ms** |
| Audio 180 s (full length) | ~20 s | **~125–160 ms** |
| Audio 30 s + stems | ~24 s | **~630 ms** |
| Video | ~13–18 s | ~13 s (see below) |

Notes:
- "Few-ms" completion holds on the warm/cached path only. Cold renders are
  genuine generation and take seconds by design.
- **Video has no completed-job dedup.** `_job_digest`/`_active_jobs` coalesce
  only *in-flight* identical requests; a repeat after completion re-renders.
  This is the as-designed behavior at this commit.
- 180 s tracks scale to the same warm-path latency as 30 s clips because the
  cache stores the finished file, keyed by duration bucket.

### 14.2 Process topology (the one rule that matters)

Run **only** the `Start application` workflow. It launches:

- Vite dashboard (:5000) → the preview
- Node API server cluster (:8080) → proxy + Python lifecycle owner
- Python AI server (:9878, healthz :9879) — child-spawned by the Node server
  that has `MODEL_API_PORT` set

The artifact workflows (`artifacts/api-server`, `artifacts/ai-dashboard`,
`artifacts/mockup-sandbox`) are platform-managed and **cannot be deleted**.
The platform sometimes restarts them. When `artifacts/api-server` runs
alongside `Start application`, you get **two Node clusters sharing :8080 via
SO_REUSEPORT**: the tree without `MODEL_API_PORT` has no Python child
(monitoring-only mode), requests hop between clusters, audio-job polls hit
ECONNRESET/circuit-breaker, and under render load the doubled memory
footprint can crash the Python process mid-job.

**Runbook when the API misbehaves under load:**

1. `ps -eo pid,ppid,cmd | grep api-server/src/index.ts` — more than one
   parent tree ⇒ contention.
2. Identify the tree that owns the Python child (`server.py`'s grandparent) —
   that is the `Start application` tree. Kill the *other* tree (`kill -9` the
   tsx parents and workers).
3. Killing the stale tree can take the :8080 listener with it — restart
   `Start application` afterwards and confirm `GET /api/health` → 200.
4. Leave the `artifacts/api-server` workflow in stopped/failed state; do not
   restart it.

### 14.3 Testing & verification gotchas

- **Dedup caches survive restarts** (disk-backed content dedup). To exercise
  a truly fresh render, use a topic string never used before.
- **Python edits require a full `Start application` restart** — modules are
  import-cached; hot-reload does not apply.
- Jobs lost to a mid-render crash stay "rendering" forever from the client's
  view; after a respawn, resubmit — the job registry is in-process.
- Long poll loops in the shell must stay under the 5-minute exec cap; split
  batches.

---

## 15. Complete System Wiring — Instant Re-Setup Guide

> Everything needed to stand this system up from a bare clone in one pass:
> the monorepo layout, every config file that wires layers together, the exact
> boot chain, port map, env contract, and verification checklist.

### 15.1 Monorepo layout (pnpm workspace)

`pnpm-workspace.yaml` declares the packages:

```
packages:
  - artifacts/*        # runnable apps
  - lib/*              # shared libraries
  - lib/integrations/*
  - scripts
```

| Package | Name | Role |
|---|---|---|
| `artifacts/ai-dashboard` | `@workspace/ai-dashboard` | React 19 + Vite 7 frontend (port 5000) |
| `artifacts/api-server` | `@workspace/api-server` | Express 5 proxy + Python lifecycle owner (port 8080) |
| `artifacts/ai-training-server` | (Python, not a pnpm pkg) | FastAPI AI engine (port 9878, healthz 9879) |
| `artifacts/mockup-sandbox` | `@workspace/mockup-sandbox` | Canvas component preview (design-time only) |
| `lib/db` | `@workspace/db` | Drizzle ORM schema + client (source of truth: `lib/db/src/schema/`) |
| `lib/api-spec` | — | `openapi.yaml` API contract |
| `lib/api-zod` | `@workspace/api-zod` | Zod schemas shared by server/client |
| `lib/api-client-react` | — | Generated TS client for the dashboard |

Root `package.json` scripts:
- `build:production` → builds api-server (esbuild → `dist/index.mjs`) then dashboard (`BASE_PATH=/`, output `artifacts/ai-dashboard/dist/public`)
- `start` → `pnpm --filter @workspace/api-server run serve` (production)
- `typecheck` → project-references build for libs, then per-artifact typecheck
- `preinstall` guard forces pnpm (rejects npm/yarn).
- Shared dependency versions live in the pnpm `catalog:` in `pnpm-workspace.yaml`.

### 15.2 The boot chain (dev)

One workflow starts everything — `Start application` in `.replit`:

```bash
[ ! -d node_modules ] && pnpm install
cd artifacts/ai-training-server && [ ! -f .venv/pyvenv.cfg ] && uv sync --no-dev
cd ../..
PORT=8080 MODEL_API_PORT=9878 pnpm --filter @workspace/api-server run dev \
  & PORT=5000 BASE_PATH=/ API_PORT=8080 pnpm --filter @workspace/ai-dashboard run dev
```

What happens, in order:

1. **pnpm install** (skipped if `node_modules` exists) and **uv sync** for the
   Python venv (skipped if `.venv/pyvenv.cfg` exists).
2. **api-server** (`src/index.ts`) boots as a Node **cluster**: the primary
   forks workers (respawned on exit), workers share the :8080 listener.
3. The api-server process that sees `MODEL_API_PORT` set becomes the **Python
   owner**: `src/python-server.ts` spawns `uv run python3 server.py` as a
   child, monitors health (`:9878/health`, liveness `:9879/healthz`), and
   restarts it with exponential backoff. **If `MODEL_API_PORT` is unset the
   server runs in monitoring-only mode and never owns a Python process** —
   this is the root cause of 503s when the env is wrong.
4. **ai-dashboard** Vite dev server binds :5000 with `allowedHosts: true`
   (required for the Replit preview iframe) and proxies `^/api/` and
   `^/uploads/` → `http://localhost:${API_PORT}` (8080). No CORS anywhere.
5. Python `server.py` boots FastAPI on :9878, starts the healthz liveness
   thread on :9879, auto-seeds the `api_keys` table, connects to pdim storage
   (falls back to local SQLite `data/local_kv.db` transparently), and loads
   the model/kernels lazily.

Request path in dev:
```
Browser → :5000 (Vite) → proxy → :8080 (Express allowlist, model-proxy.ts)
        → undici long-timeout Agent → :9878 (FastAPI) → Digital GPU stack
```

### 15.3 The boot chain (production deployment)

`.replit [deployment]`:
```
deploymentTarget = "vm"
build = pnpm run build:production
run   = PORT=8080 MODEL_API_PORT=9878 NODE_CLUSTER_WORKERS=2 pnpm start
```

In production there is **no Vite**: the built api-server serves the dashboard
SPA from `dist/public` directly on the single external port. The `serve`
script defaults `PORT=8080 MODEL_API_PORT=9878` — **MODEL_API_PORT must be
set in the prod run command or no process owns Python and every model route
503s** (this bit us once; it is now baked into both `serve` and the run
command).

### 15.4 Port map

| Port | Owner | External |
|---|---|---|
| 5000 | Vite dashboard (dev) / preview pane | 5000 |
| 8080 | Express api-server (cluster, SO_REUSEPORT) | 8080 |
| 9878 | Python FastAPI (model API) | 3002 |
| 9879 | Python healthz liveness thread (hung-detection probe target) | 3001 |
| 3000 / 3999 | mockup sandbox / misc | 3000 / 3003 |
| 8081 | reserved | 80 |

### 15.5 Environment contract

Set in `.replit [userenv.shared]` (non-secret config, already committed):
`STORAGE_INSTANCE`, `ADMIN_KEY`, `STORAGE_HTTP_URL`, `STORAGE_CONNECTION_URL`,
`STORAGE_BEARER_TOKEN`, `AI_TRAINING_KEY_PROD`, `AI_DYNAMIC_BATCHING=1`,
`RTA_AUDIO_SPECTRAL=1`.

Replit Secrets (never in source): `SESSION_SECRET` (server exits at startup if
missing), `DATABASE_URL` (managed Postgres). Optional: `TAVILY_API_KEY`,
`EXA_API_KEY` (awareness enrichment; RSS-only without them).

Wiring env vars (set by workflow/run commands, not stored):
`PORT`, `MODEL_API_PORT`, `API_PORT`, `BASE_PATH`, `NODE_CLUSTER_WORKERS`.

### 15.6 System (Nix) dependencies

`.replit [nix].packages` — required for audio/video/image paths:
`ffmpeg-full`, `espeak-ng` (voice-over), `libsndfile`, `freetype`, `libjpeg`,
`libwebp`, `libtiff`, `openjpeg`, `lcms2`, `libimagequant`, `zlib`, `xsimd`,
`glibcLocales`, `pkg-config`, `tcl`, `tk`, `libxcrypt`.
Python deps are locked in `artifacts/ai-training-server/pyproject.toml` +
`uv.lock` (torch pinned to the CPU wheel index `pytorch-cpu`).

### 15.7 Cross-layer contracts (the glue)

1. **Proxy allowlist** — `model-proxy.ts` is an explicit per-route table
   (Section 5). Every new Python endpoint needs a matching handler or the
   dashboard 404s. Binary payloads (video files, stem WAVs) go through
   `proxyBinaryStream`; JSON through `proxyRequest`.
2. **Long-timeout undici Agent** — Node→Python fetches use a dedicated Agent
   with extended header/body timeouts; a bare `fetch` (300 s default) aborts
   valid long generations. Express `server.timeout = 0`.
3. **Auth** — dashboard→Node uses session auth (`SESSION_SECRET`);
   Node→Python and direct admin calls use `X-Api-Key` (`AI_TRAINING_KEY_PROD`
   seeded into the `api_keys` table on first boot); pdim storage uses
   `STORAGE_BEARER_TOKEN`; Bearer auth also accepted on generation endpoints.
4. **DB** — Drizzle schema in `lib/db/src/schema/` is the single source of
   truth. Apply with `pnpm --filter @workspace/db push` (or `push-force`).
   Python reads the same Postgres via `psycopg2` + `DATABASE_URL`.
5. **Storage waterfall** — Python storage client tries pdim HTTP → Redis →
   local SQLite; every layer is optional, the system boots with all of them
   down.
6. **Self-healing** — Node watchdog owns Python restarts (exponential
   backoff, `:9879` liveness probe with a 25 s budget so GC pauses don't
   cause false kills). No caller-visible timeouts: jobs retry-until-success
   with heartbeats.
7. **Uploads** — generated files land in `artifacts/ai-training-server/uploads/`
   (stems under `uploads/stems/<job_id>/`); served by Python, streamed
   through Node, proxied by Vite under `/uploads/` and `/api/files/...`.

### 15.8 Instant re-setup checklist (fresh clone → running)

```bash
# 0. Prereqs provided by .replit modules: nodejs-20, python-3.11, web + Nix pkgs
# 1. Secrets: set SESSION_SECRET and DATABASE_URL (create Replit Postgres)
# 2. Install & boot — the workflow does everything:
#    Run the "Start application" workflow (installs pnpm deps + uv venv on
#    first boot, ~30-60 s extra the first time)
# 3. DB schema:
pnpm --filter @workspace/db push
# 4. Verify the chain, outside-in:
curl -s localhost:9878/health         # Python direct        → {"status":"ok"}
curl -s localhost:9879/healthz        # liveness thread      → 200
curl -s localhost:8080/api/health     # Node proxy           → 200
curl -s -o /dev/null -w '%{http_code}' localhost:5000/   # dashboard → 200
# 5. Seed the audio dataset (admin key from userenv):
curl -X POST localhost:9878/storage/datasets/audio/seed -H "X-Api-Key: $AI_TRAINING_KEY_PROD"
# 6. Quality gate:
cd artifacts/ai-training-server && python3 tests/test_w6_90m.py --quality-only
#    Expected: all quality checks pass (104/104 at last full run)
```

Rules to keep it healthy:
- Run **only** `Start application` (Section 14.2 runbook if artifact
  workflows sneak back in).
- Python edits ⇒ full workflow restart (import caching).
- New Python route ⇒ matching allowlist entry in `model-proxy.ts`.
- Schema change ⇒ `db push` in dev, and against prod per the deployment flow.
- GitHub remote: `origin = github.com/20lawsobk/Secure-AI-Forge`. Verify which
  commit the remote holds before importing elsewhere — the remote and
  workspace have diverged before (fused vs reverted lineage).
