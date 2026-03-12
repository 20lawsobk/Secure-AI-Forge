# MaxBooster AI Training Server

## Overview

Production-grade AI model training server with a custom-built transformer model, GPU simulation engine, multi-platform content generation, and in-house API key management. No external AI APIs used — everything is built from scratch.

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
┌─────────────────────────────────────────────────────────┐
│  User Browser  (React Dashboard at port 20060 / /)      │
└──────────────────────────┬──────────────────────────────┘
                           │ /api/*
┌──────────────────────────▼──────────────────────────────┐
│  Express API Server  (port 8080)                         │
│  - Proxy routes to Python AI server                      │
│  - /api/healthz, /api/api-keys, /api/training, etc.      │
└──────────────────────────┬──────────────────────────────┘
                           │ localhost:9878
┌──────────────────────────▼──────────────────────────────┐
│  Python FastAPI AI Training Server  (port 9878)          │
│  - Custom Transformer LM (RoPE attention)                │
│  - Creative generation agents (Script, Visual, etc.)     │
│  - HyperGPU + Digital GPU simulation engine              │
│  - API key management with PostgreSQL                    │
│  - Training orchestration (background threads)           │
│  - BoostSheet content management                         │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│  PostgreSQL Database                                     │
│  - api_keys table (in-house key management)              │
│  - training_logs table                                   │
│  - request_logs table                                    │
└─────────────────────────────────────────────────────────┘
```

## Structure

```text
artifacts/
├── ai-training-server/         # Python FastAPI AI server
│   ├── server.py               # Main entry point + API key mgmt
│   ├── ai_model/               # Custom ML model code
│   │   ├── api/                # FastAPI endpoints
│   │   ├── model/              # Transformer LM, tokenizer, creative model
│   │   ├── agents/             # Script, visual spec, distribution, optimization agents
│   │   ├── gpu/                # Digital GPU + HyperGPU simulation
│   │   ├── boostsheets/        # BoostSheet CRUD
│   │   ├── video/              # Video generation
│   │   └── weights/            # Model weights + fine-tune datasets
│   ├── workers/                # Background workers
│   └── admin_key.txt           # Auto-generated admin key (first run)
├── ai-dashboard/               # React dashboard (port 20060 at /)
│   └── src/pages/              # Dashboard, API Keys, Training, GPU, Content, Model
├── api-server/                 # Express proxy server (port 8080 at /api)
│   └── src/routes/             # model-proxy.ts forwards to Python server
lib/
├── api-spec/openapi.yaml       # OpenAPI 3.1 contract
├── api-client-react/           # Generated React Query hooks
├── api-zod/                    # Generated Zod schemas
└── db/                         # Drizzle schema (api_keys, training_logs, request_logs)
```

## API Key Management

- Keys are stored as SHA-256 hashes in PostgreSQL — the raw key is never stored
- On first startup, a default admin key is auto-generated and saved to `artifacts/ai-training-server/admin_key.txt`
- Use `X-Admin-Key` header for admin operations (list/create/revoke/rotate keys)
- Use `X-Api-Key` header for regular API access
- Scopes: `read`, `write`, `train`, `admin`, `generate`
- Keys can have optional expiry dates

## Custom AI Model

- **Transformer LM** with RoPE (Rotary Position Embeddings) self-attention
- **Creative Model wrapper** — nucleus sampling, top-k/p, temperature, repetition penalty, beam search
- **Digital GPU** — custom SIMD compute engine with tiled GEMM, softmax, flash attention
- **HyperGPU** — tensor core simulation (512 lanes, 8 tensor cores, mixed precision)
- **Agents**: Script, VisualSpec, Distribution, Optimization
- **No external AI APIs** — everything runs locally on CPU/GPU

## Workflows

- `artifacts/ai-dashboard: web` — React dashboard (user-facing)
- `artifacts/api-server: API Server` — Express proxy/orchestration
- `AI Training Server` — Python FastAPI model server

## Running

- `pnpm run build` — build everything
- `pnpm run typecheck` — full type check
- `pnpm --filter @workspace/db run push` — push DB schema changes
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API client/zod from OpenAPI spec
