# MaxBooster AI Training Server

## Overview

Production-grade AI model training server with a custom-built transformer model, GPU simulation engine, multi-platform content generation, and in-house API key management. No external AI APIs used — everything is built from scratch.

Part of a tri-app music artist platform:
1. **This server** — AI training + platform inference API
2. **Storage server** — 7TB training dataset (`pocketdimensionstorage.replit.app`)
3. **Main music platform** — DAW, beat marketplace, social media management, music distribution

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5 (proxy/orchestration layer)
- **AI Server**: Python 3.11 + FastAPI + Uvicorn (custom transformer model)
- **Database**: PostgreSQL + Drizzle ORM (TypeScript) + psycopg2 (Python)
- **Validation**: Zod (zod/v4), drizzle-zod
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)
- **Frontend**: React + Vite (TypeScript)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Main Music Platform (DAW, Beat Marketplace, Social, Distro)     │
│  Calls → /api/platform/* endpoints                               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│  User Browser  (React Dashboard at port 20060 / /)               │
└─────────────────────────────┬───────────────────────────────────┘
                              │ /api/*
┌─────────────────────────────▼───────────────────────────────────┐
│  Express API Server  (port 8080)                                  │
│  - Proxy routes to Python AI server                               │
│  - /api/platform/*, /api/training/*, /api/storage/*, etc.         │
└─────────────────────────────┬───────────────────────────────────┘
                              │ localhost:9878
┌─────────────────────────────▼───────────────────────────────────┐
│  Python FastAPI AI Training Server  (port 9878)                   │
│  - Custom Transformer LM (RoPE attention, dim=512, 8 layers)      │
│  - Platform API (/platform/social, /platform/daw, /platform/dist) │
│  - Creative generation agents (Script, Visual, Distribution)      │
│  - HyperGPU + Digital GPU simulation engine                       │
│  - API key management with PostgreSQL                             │
│  - Training orchestration (background threads)                    │
│  - Storage pipeline (pull 7TB dataset from storage server)        │
│  - Auto-loads latest checkpoint from storage on boot              │
└──────────┬──────────────────────────────────────────────────────┘
           │                                │
┌──────────▼──────────┐     ┌──────────────▼──────────────────────┐
│  PostgreSQL Database │     │  MaxBooster Storage Server           │
│  - api_keys          │     │  pocketdimensionstorage.replit.app   │
│  - training_logs     │     │  Instance: e50d64e610d37dd52ce85711  │
│  - request_logs      │     │  7TB training dataset (pending)      │
└─────────────────────┘     │  Keys: mbs:training:session,         │
                            │        mbs:data, mbs:downloads, etc.  │
                            └──────────────────────────────────────┘
```

### Data Flow: Storage → Training → Platform

```
Storage server (7TB) ──► POST /training/start-from-storage
                               │
                               ▼
                     TrainingDataPipeline.stream_batches()
                     Pulls: mbs:data, mbs:downloads, mbs:session
                               │
                               ▼
                     CurriculumTrainer feeds TransformerLM
                     (with per-user engagement signal bias)
                               │
                               ▼
                     Checkpoint saved to storage + local disk
                               │
                               ▼
                     Main platform calls /platform/* endpoints
                     Model generates personalized content
```

## Structure

```text
artifacts/
├── ai-training-server/         # Python FastAPI AI server
│   ├── server.py               # Main entry point + API key mgmt + platform endpoints
│   ├── storage_client.py       # Storage server client (Redis-like HTTP exec API)
│   │                           # StorageClient, TrainingDataPipeline,
│   │                           # DatasetStreamClient, ModelCheckpointClient, CurriculumStateClient
│   ├── ai_model/               # Custom ML model code
│   │   ├── model/              # Transformer LM, tokenizer, creative model
│   │   ├── agents/             # Script, VisualSpec, Distribution, Optimization agents
│   │   ├── gpu/                # Digital GPU + HyperGPU simulation
│   │   ├── boostsheets/        # BoostSheet CRUD
│   │   └── weights/            # Model weights (model.pt saved here after training)
│   └── admin_key.txt           # Auto-generated admin key (first run)
├── ai-dashboard/               # React dashboard (port 20060 at /)
│   └── src/pages/              # Dashboard, API Keys, Training, GPU, Content, Model
├── api-server/                 # Express proxy server (port 8080 at /api)
│   └── src/routes/model-proxy.ts  # All proxy routes including /platform/*
lib/
├── api-spec/openapi.yaml       # OpenAPI 3.1 contract
├── api-client-react/           # Generated React Query hooks
├── api-zod/                    # Generated Zod schemas
└── db/                         # Drizzle schema (api_keys, training_logs, request_logs)
```

## API Key Management

- Keys stored as SHA-256 hashes in PostgreSQL — raw key never stored
- Admin key pinned via `ADMIN_KEY` env var (same value in dev and production)
- `X-Admin-Key` header for admin operations; `X-Api-Key` for regular access
- Scopes: `read`, `write`, `train`, `admin`, `generate`
- Admin key: `mbs_8a3edbac97ff333dda5068410227267e6d85b14a4c9caee279fbb18ddfb47edc`

## Platform API Endpoints (for main music platform)

All routes accessible via Express proxy at `/api/...`:

### Social Media Management
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/platform/social/generate` | Generate N variants of a social post, personalized by user engagement history |
| POST | `/api/platform/social/autopilot` | Analyse past performance → recommend next content strategy + post schedule |

### DAW / Studio
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/platform/daw/generate` | Generate lyrics, hooks, beat descriptions, or track concepts |

### Music Distribution
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/platform/distribution/plan` | Generate release strategy across Spotify, Apple Music, Tidal, etc. |

### Model Management
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/platform/model/info` | Current model state, training stats, storage connection |
| POST | `/api/platform/model/reload` | Hot-reload latest checkpoint from storage server |

### Training & Storage
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/training/start-from-storage` | Pull 7TB dataset from storage → train model in background |
| GET | `/api/storage/session` | View active 7TB training session from storage server |
| GET | `/api/storage/pipeline/status` | Live training pipeline progress |
| GET | `/api/storage/status` | Storage server connection health |
| POST | `/api/storage/feedback` | Record per-user engagement signal (curriculum training bias) |
| GET | `/api/storage/curriculum/:userId` | Get user's engagement history + top-performing content |

## Storage Server Credentials

- **URL**: `https://pocketdimensionstorage.replit.app/api/redis/instances/e50d64e610d37dd52ce85711/exec`
- **Token**: `ed5ff25a3ad7b54d58f063190723df8b41f08b4a5b8bbc041f4cfa0ea13d9f46`
- **Exec format**: `{"cmd": "KEYS", "args": ["*"]}` (not `{"command": [...]}`)
- **Session**: `sess-1773450967978` — 7TB (`7,696,581,394,432 bytes`), state: `pending`
- Env vars: `STORAGE_HTTP_URL`, `STORAGE_BEARER_TOKEN`, `STORAGE_CONNECTION_URL`

## Custom AI Model

- **Transformer LM** with RoPE (Rotary Position Embeddings) self-attention
- **Creative Model wrapper** — nucleus sampling, top-k/p, temperature, repetition penalty
- **Digital GPU** — custom SIMD compute engine with tiled GEMM, softmax, flash attention
- **HyperGPU** — tensor core simulation (512 lanes, 8 tensor cores, mixed precision)
- **Agents**: ScriptAgent, VisualSpecAgent, DistributionAgent, OptimizationAgent
- **No external AI APIs** — everything runs locally

## Workflows

- `artifacts/ai-dashboard: web` — React dashboard (port 20060, user-facing)
- `artifacts/api-server: API Server` — Express proxy/orchestration (port 8080)
- `AI Training Server` — Python FastAPI model server (port 9878)

## Running

- `pnpm run build` — build everything
- `pnpm run typecheck` — full type check
- `pnpm --filter @workspace/db run push` — push DB schema changes
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API client/zod from OpenAPI spec
