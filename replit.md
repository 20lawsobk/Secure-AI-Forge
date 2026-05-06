# MaxBooster (MaxCore AI)

A production-grade AI content generation and training platform for music artists that generates multi-platform social media content (text, images, video) powered by a custom in-house Transformer model.

## Run & Operate

- **Main workflow**: `Start application` — starts both the API server (port 8080) and the Vite dev dashboard (port 5000)
- **AI Training Server**: separate workflow running the Python FastAPI server on port 9878
- **DB migration**: `pnpm --filter @workspace/db push` (interactive) or `push-force`
- **Build for production**: `pnpm run build:production`
- **Production start**: `pnpm start` (serves built dashboard via API server)

Required env vars: `DATABASE_URL`, `PORT`, `MODEL_API_PORT`, `ADMIN_KEY`, `STORAGE_CONNECTION_URL`, `STORAGE_HTTP_URL`, `STORAGE_BEARER_TOKEN`, `AI_TRAINING_KEY_PROD`

## Stack

- **Frontend**: React 19, Vite 7, Tailwind CSS v4, Radix UI, TanStack Query, Framer Motion, Wouter
- **API Server**: Express 5, TypeScript, tsx (dev) / esbuild (prod), Drizzle ORM
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
- Video Studio for AI-generated video content
- Generators for social content, DAW scripts, distribution plans
- Watchdog system health monitoring

## User preferences

_Populate as you build_

## Gotchas

- Always use `pnpm` (not npm/yarn) — the preinstall script enforces this
- The `Start application` workflow starts both API server (bg) and dashboard together
- DB push is interactive by default; use `push-force` flag or pipe `""` to auto-select
- Python typecheck (mypy) has known non-blocking type errors — these are warnings, not blockers
- Storage warnings about "pdim storage offline" are expected in dev; the system falls back gracefully

## Pointers

- Workflows skill: `.local/skills/workflows/SKILL.md`
- Package management: `.local/skills/package-management/SKILL.md`
- Database skill: `.local/skills/database/SKILL.md`
