# MaxBooster (MaxCore AI)

> Setup note: after import, run `pnpm install` (requires Node.js 22 — installed via the nodejs-22 module) and restart the `Start application` workflow. The Python AI training server auto-installs its deps into `.pythonlibs` on first boot (~30s) and then serves `/health` on port 9878. All required secrets (`DATABASE_URL`, `ADMIN_KEY`, `STORAGE_*`, `AI_TRAINING_KEY_PROD`, `SESSION_SECRET`) were already present in this environment.

A production-grade AI content generation and training platform for music artists that generates multi-platform social media content (text, images, video) powered by a custom in-house Transformer model.

## Run & Operate

- **Main workflow**: `Start application` — starts everything: Vite dashboard (port 5000), Express API server (port 8080), and the Python AI engine (port 9878, child-spawned automatically by the API server). This is the only workflow you need.
- **Artifact workflows** (`artifacts/api-server`, `artifacts/ai-dashboard`, `artifacts/mockup-sandbox`): keep these **stopped** — they conflict with `Start application` and cannot be deleted (platform-managed).
- **DB migration**: `pnpm --filter @workspace/db push` (interactive) or `push-force`
- **Build for production**: `pnpm run build:production`
- **Production start**: `pnpm start` (serves built dashboard via API server)

Required env vars: `DATABASE_URL`, `PORT`, `MODEL_API_PORT`, `ADMIN_KEY`, `STORAGE_CONNECTION_URL`, `STORAGE_HTTP_URL`, `STORAGE_BEARER_TOKEN`, `AI_TRAINING_KEY_PROD`

Required Replit Secrets (must NOT be stored in source or `.replit` env): `SESSION_SECRET`. The server validates this at startup and exits immediately if it is missing — set it in the Replit Secrets panel so it flows into both dev and production automatically.

Optional Replit Secrets for content awareness (set in the Replit Secrets panel):
- `TAVILY_API_KEY` — Tavily web search (https://tavily.com). Enables real-time news intelligence for content generation.
- `EXA_API_KEY` — Exa semantic search (https://exa.ai). Enables deep trend discovery for content generation.

Without these, `ContentGenerationAwarenessService` and `IndustryMonitorService` still function using RSS feeds only — web search enrichment is skipped gracefully. With them, every content generation call is enriched with live music industry signals.

## Stack

- **Frontend**: React 19, Vite 7, Tailwind CSS v4, Radix UI, TanStack Query, Framer Motion, Wouter
- **API Server**: Express 5, TypeScript, tsx (dev) / esbuild (prod), Drizzle ORM — requires **Node.js 22** (undici v8 dependency)
- **AI Engine**: Python 3.11, FastAPI, PyTorch (custom Transformer), NumPy GPU simulation, Pillow, FFmpeg
- **Database**: PostgreSQL via Drizzle ORM (drizzle-kit for migrations)
- **Package manager**: pnpm (workspace monorepo)

## Where things live

- `artifacts/ai-dashboard/` — React frontend
- `artifacts/api-server/` — Express orchestration/proxy layer
- `artifacts/ai-training-server/` — Python FastAPI AI engine
- `lib/db/` — Drizzle schema & DB client (source of truth: `lib/db/src/schema/`)
- `lib/api-client-react/` — Generated TypeScript API client
- `lib/api-spec/openapi.yaml` — API contract
- `artifacts/ai-training-server/ai_model/` — Custom Transformer, GPU sim, video renderer

## Architecture decisions

- **Frontend proxies to API**: Vite dev server proxies `/api` and `/uploads` to the Express API server (port 8080), which in turn proxies to the Python server (port 9878). No CORS issues.
- **API server manages Python lifecycle**: The Node.js API server spawns and monitors the Python AI server as a child process with exponential backoff restarts.
- **No external AI APIs**: All AI inference uses the in-house custom Transformer with Digital GPU simulation (NumPy SIMD) for CPU-safe operation.
- **In-process storage fallback**: If the pdim storage server is offline, the AI server operates in local-only mode transparently.
- **Single external port**: Port 5000 → port 80 externally. In production, the API server serves the built React SPA directly.

## Product

- System overview dashboard with live GPU, model, and training metrics
- API key management (create, rotate, delete)
- Training controls (start/stop, continuous training, data puller)
- Multi-platform content generation (TikTok, Instagram, YouTube, etc.)
- Audio generation (`/api/generate/audio`) renders from REAL music samples (Free Music Archive) seeded into pdim storage — no procedural synthesis
- Video Studio for AI-generated video content
- Generators for social content, DAW scripts, distribution plans
- Watchdog system health monitoring

## User preferences

_Populate as you build_

## First-run setup

After a fresh import or new environment:

1. **Install Node packages**: `pnpm install` (node_modules aren't checked in)
2. **Start the workflow**: `Start application` — the Python AI server auto-provisions its venv on first boot (~1 min cold start)
3. **Seed the audio dataset** — required before `/api/generate/audio` can return results:

   ```bash
   curl -X POST "http://localhost:9878/storage/datasets/audio/seed?count=12" \
     -H "X-Admin-Key: $ADMIN_KEY"
   ```

   This runs in the background. Poll `GET /storage/datasets/audio/status` (same header) until `seeding_now` is `false` and `num_chunks > 0`. Typically takes 1–3 minutes. The seeder pulls real CC-licensed tracks from the Free Music Archive via HuggingFace and falls back to librosa's bundled examples if the HF datasets-server is unavailable.

   > **Note**: `ADMIN_KEY` is available as an environment variable. The seed step is idempotent — re-running adds more tracks without replacing existing ones unless you pass `?replace=true`.

## Setup notes

- Imported from GitHub; first run requires `pnpm install` (node_modules aren't checked in) and lets the Python server auto-provision its venv (torch, etc. — takes ~1 min on cold start). Both are now done and the `Start application` workflow runs cleanly.

## Gotchas

- Always use `pnpm` (not npm/yarn) — the preinstall script enforces this
- The `Start application` workflow starts both API server (bg) and dashboard together
- DB push is interactive by default; use `push-force` flag or pipe `""` to auto-select
- Python typecheck (mypy) is fully green; keep it that way. Mixed-type state dicts (e.g. `self.state`, `_training_state`) should be annotated `dict[str, Any]`; read-only constant config dicts (mixed str/int/float values) use `typing.cast(...)` at the point of use rather than annotation
- Storage warnings about "pdim storage offline" are expected in dev; the system falls back gracefully
- Real audio generation needs the `mb:dataset:audio` dataset seeded into pdim first: `POST /storage/datasets/audio/seed?count=N&replace=` (admin-only, runs in a background thread, pulls CC tracks from FMA-small via the HF datasets-server). If the dataset is empty, `/api/generate/audio` raises explicitly — there is no synth fallback
- The seed worker module is import-cached per server process; after editing `workers/seed_audio_dataset.py` (or storage/env config) restart the workflow(s) that spawn `server.py` so a process with the new code/env wins the `:9878` lock
- pdim storage is multi-instance: `STORAGE_BEARER_TOKEN` must match the instance in `STORAGE_HTTP_URL`. `available:false` + `url_configured:true` = token/instance mismatch (HTTP 403 WRONGPASS). Diagnose via the unauthenticated `GET /api/redis/instances` list + per-instance PING; the app is currently pointed at the `max-booster-agent` instance

## Pointers

- Workflows skill: `.local/skills/workflows/SKILL.md`
- Package management: `.local/skills/package-management/SKILL.md`
- Database skill: `.local/skills/database/SKILL.md`
