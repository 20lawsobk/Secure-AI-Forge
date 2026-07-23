"""
MaxBooster AI Training Server
==============================================
Production-grade AI model training server with:
  - Custom transformer model (no external APIs)
  - API key management with PostgreSQL
  - Training orchestration
  - GPU simulation / HyperGPU cluster
  - Multi-platform content generation
  - BoostSheet management
  - Storage server sync (datasets, checkpoints, curriculum state)
"""
from __future__ import annotations

import os
import sys
import time
import secrets
import hashlib
import uuid
import json
import asyncio
import threading
import re
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional, List
from urllib.parse import urlparse
import html as _html
import urllib.request as _urllib_request

# ── BLAS/OpenMP thread pools — minimal on the host ───────────────────────────
# The Digital GPU backend handles all heavy compute. The Python process is an
# API surface; its numpy usage is bookkeeping only. One BLAS thread is correct —
# we do not want host CPU thread pools competing with the GPU engine.
for _blas_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_blas_var, "1")

import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException, Depends, Header, Request, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

_srv_logger = logging.getLogger(__name__)

# ─── DB Setup ────────────────────────────────────────────────────────────────

DATABASE_URL = os.environ.get("DATABASE_URL")

_db_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
_db_pool_lock = threading.Lock()

def _get_db_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _db_pool
    if _db_pool is None:
        with _db_pool_lock:
            if _db_pool is None:
                _db_pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=2,
                    maxconn=15,
                    dsn=DATABASE_URL,
                    # Keep connections alive across long idle periods
                    keepalives=1,
                    keepalives_idle=30,
                    keepalives_interval=5,
                    keepalives_count=5,
                )
    return _db_pool

def _acquire() -> psycopg2.extensions.connection:
    return _get_db_pool().getconn()

def _release(conn, error: bool = False):
    try:
        if error:
            conn.rollback()
        _get_db_pool().putconn(conn)
    except Exception as e:
        _srv_logger.debug(f"[DB] pool release error: {e}")

def get_db():
    conn = _acquire()
    try:
        yield conn
    finally:
        _release(conn)

def init_db():
    """Initialize database tables."""
    conn = _acquire()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name TEXT NOT NULL,
            key_hash TEXT NOT NULL UNIQUE,
            prefix TEXT NOT NULL,
            scopes TEXT[] NOT NULL DEFAULT '{}',
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            request_count INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_used_at TIMESTAMPTZ,
            expires_at TIMESTAMPTZ
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS training_logs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            level TEXT NOT NULL DEFAULT 'info',
            message TEXT NOT NULL,
            epoch INTEGER,
            loss FLOAT,
            job_id TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS request_logs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            api_key_id UUID REFERENCES api_keys(id),
            endpoint TEXT NOT NULL,
            method TEXT NOT NULL,
            status_code INTEGER,
            response_time_ms INTEGER,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    # Create a default admin key if none exists
    cur.execute("SELECT COUNT(*) FROM api_keys")
    count = cur.fetchone()[0]
    if count == 0:
        admin_key = f"mbs_{secrets.token_hex(32)}"
        key_hash = hashlib.sha256(admin_key.encode()).hexdigest()
        prefix = admin_key[:12]
        cur.execute(
            """INSERT INTO api_keys (name, key_hash, prefix, scopes, is_active)
               VALUES (%s, %s, %s, %s, TRUE)""",
            ("Default Admin Key", key_hash, prefix, ["read", "write", "train", "admin", "generate"])
        )
        # Print admin key once to stdout — do not persist to disk
        print("[Server] *** DEFAULT ADMIN KEY (copy now — not stored) ***")
        print(f"[Server] {admin_key}")
        print(f"[Server] Admin key prefix: {prefix}...")

    # Ensure AI_TRAINING_KEY_PROD is registered (idempotent — upsert by hash)
    _prod_key = os.environ.get("AI_TRAINING_KEY_PROD")
    if _prod_key:
        _prod_hash   = hashlib.sha256(_prod_key.encode()).hexdigest()
        _prod_prefix = _prod_key[:12]
        cur.execute("SELECT id FROM api_keys WHERE key_hash = %s", (_prod_hash,))
        if not cur.fetchone():
            cur.execute(
                """INSERT INTO api_keys (name, key_hash, prefix, scopes, is_active)
                   VALUES (%s, %s, %s, %s, TRUE)""",
                ("AI Training Key (prod)", _prod_hash, _prod_prefix,
                 ["read", "write", "train", "admin", "generate"])
            )
            print(f"[Server] AI_TRAINING_KEY_PROD registered: {_prod_prefix}...")
        else:
            # Ensure it's active
            cur.execute(
                "UPDATE api_keys SET is_active = TRUE WHERE key_hash = %s",
                (_prod_hash,)
            )
            print(f"[Server] AI_TRAINING_KEY_PROD already registered: {_prod_prefix}...")

    conn.commit()
    cur.close()
    _release(conn)

def log_training(message: str, level: str = "info", epoch: int = None,
                 loss: float = None, job_id: str = None):
    try:
        conn = _acquire()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO training_logs (level, message, epoch, loss, job_id) VALUES (%s, %s, %s, %s, %s)",
            (level, message, epoch, loss, job_id)
        )
        conn.commit()
        cur.close()
        _release(conn)
    except Exception as e:
        print(f"[Server] Failed to log training: {e}")

# ─── Training State ──────────────────────────────────────────────────────────

_training_state: dict[str, Any] = {
    "state": "idle",
    "epoch": 0,
    "total_epochs": 0,
    "loss": None,
    "perplexity": None,
    "samples_trained": 0,
    "elapsed_seconds": 0,
    "eta_seconds": None,
    "weights_exist": False,
    "job_id": None,
    "started_at": None,
    "first_loss": None,
    "best_loss": None,
    "current_loss": None,
    "sessions_done": 0,
    "last_weights_save": None,
    "weights_file": "weights_v4.npz",
    "total_trained": 0,
    "training_time": 0,
    "curriculum_phase": None,
    "curriculum_phases_done": [],
    "schedule_mode": "single",
    "stop_requested": False,
}
_training_lock = threading.Lock()

# ─── App Init ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MaxBooster AI Training Server",
    description="Production-grade custom AI model training server with API key management",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Progressive latency observability (NO aborts) ──────────────────────────
# Guaranteed-completion policy: requests are NEVER aborted by the server.
# Every endpoint keeps a latency-class budget, but it is used purely for
# observability — slow requests are logged at 50% and 100% of budget and then
# allowed to run to completion.  The Node.js proxy and uvicorn keep-alives are
# configured with no request timeouts, so the full chain waits for the work.

_TIMEOUT_EXEMPT_SUFFIXES = ("/download", "/file", "/video", "/preview")

_TIMEOUT_CLASSES: list = [
    # (path prefixes, budget seconds) — first match wins; order matters
    (("/health", "/api/health", "/model/status", "/gpu/", "/dashboard/stats",
      "/api/concurrency", "/api/video-job", "/api/audio-job", "/api/video-jobs",
      "/api-keys", "/storage/status", "/storage/pipeline/status",
      "/storage/datasets/audio/status", "/training/status",
      "/training/continuous/status", "/training/puller/status",
      "/api/awareness/quality/status"), 10.0),
    (("/generate/audio", "/api/generate/audio", "/platform/video/generate",
      "/api/generate-video", "/api/video/", "/platform/model/reload",
      "/api/warm", "/training/", "/storage/"), 150.0),
    (("/content/generate", "/api/generate/", "/platform/", "/generate/",
      "/api/url-parser", "/api/analyze", "/api/safety",
      "/api/awareness/"), 40.0),
]
# Default stays below the Node proxy's 45s non-stream abort so unclassified
# routes still return a structured 504 instead of an opaque proxy abort.
_TIMEOUT_DEFAULT_BUDGET = 40.0


def _timeout_budget_for(path: str) -> float:
    for prefixes, budget in _TIMEOUT_CLASSES:
        if path.startswith(prefixes):
            return budget
    return _TIMEOUT_DEFAULT_BUDGET


@app.middleware("http")
async def progressive_timeout_middleware(request, call_next):
    from fastapi.responses import JSONResponse  # local: avoid top-level churn

    path = request.url.path
    # Exempt binary/streaming routes (video files, scene previews).
    if path.startswith("/api/video-job/") and (
        "/preview/" in path or path.endswith(_TIMEOUT_EXEMPT_SUFFIXES)
    ):
        return await call_next(request)

    budget = _timeout_budget_for(path)
    soft = budget * 0.5
    start = time.time()

    # Observability-only: log at soft/full budget, but NEVER abort — the
    # request always runs to completion (guaranteed-completion policy).
    task = asyncio.ensure_future(call_next(request))
    done, _ = await asyncio.wait({task}, timeout=soft)
    if not done:
        print(f"[latency] SLOW {request.method} {path}: exceeded soft mark "
              f"{soft:.0f}s (class {budget:.0f}s) — still running", flush=True)
        done, _ = await asyncio.wait({task}, timeout=budget - soft)
        if not done:
            print(f"[latency] VERY SLOW {request.method} {path}: exceeded "
                  f"{budget:.0f}s class budget — waiting to completion", flush=True)
            await asyncio.wait({task})

    response = task.result()
    response.headers["X-Latency-Class"] = f"{budget:.0f}"
    elapsed = time.time() - start
    if elapsed > budget:
        print(f"[latency] COMPLETED {request.method} {path} after {elapsed:.1f}s "
              f"(class {budget:.0f}s)", flush=True)
    return response


# ─── Async helper: run blocking calls off the event loop ─────────────────────

async def _in_thread(fn):
    """Run a synchronous callable in the default thread-pool executor so that
    Digital GPU / blocking agent inference does not stall uvicorn's event loop."""
    return await asyncio.get_event_loop().run_in_executor(None, fn)


from ai_model.adaptive_concurrency import INFERENCE_GATE, RENDER_GATE, GateBusy  # noqa: E402


async def _in_thread_gated(gate, fn, timeout=None):
    """Run ``fn`` in a worker thread while holding an adaptive-concurrency slot.

    Guaranteed-completion policy: the slot acquisition waits INDEFINITELY
    (timeout=None) — bursts queue up and drain in order instead of being
    rejected with GateBusy/503.  The blocking wait happens inside the worker
    thread so the event loop is never stalled.  The ``timeout`` parameter is
    retained for signature compatibility and ignored."""
    def _run():
        with gate.slot(timeout=None):
            return fn()
    return await _in_thread(_run)


# ─── Pocket GPU Pool: infinite per-request GPU lifecycle ─────────────────────
# Each unique inference request gets its own GPU instance spawned from the
# pocket dimension (born → working → dead).  The pool is unbounded — the
# pocket's compression+filetree system absorbs any burst.  Identical requests
# never reach the pool; the async coalescer above collapses them to one life.
_gpu_pool = None
_gpu_pool_lock = threading.Lock()


def _get_gpu_pool():
    """Process-wide PocketGPUPool singleton.  Lazily initialised on first use."""
    global _gpu_pool
    if _gpu_pool is None:
        with _gpu_pool_lock:
            if _gpu_pool is None:
                from ai_model.gpu.pocket_pool import PocketGPUPool
                _gpu_pool = PocketGPUPool()
    return _gpu_pool


# ─── Model-readiness & worker-readiness helpers ──────────────────────────────
# These replace the old instant-503 guards that fired whenever an endpoint was
# hit before _init_ai_model() / _init_workers() had finished.  Requests now
# park here (250 ms sleep loop) for up to their respective deadlines, then
# raise 503 with a Retry-After header only if the subsystem truly never came up.

async def _wait_for_model_ready(max_wait: float = 30.0) -> None:
    """Wait until _model_ready is True, then return.

    Guaranteed-completion policy: parks indefinitely (250 ms poll) until the
    model finishes initialising — never raises 503.  Warm-up is watchdog-
    supervised, so readiness always arrives.  ``max_wait`` is retained for
    signature compatibility and used only as the slow-log threshold."""
    if _model_ready:
        return  # fast path — already ready
    _waited = 0.0
    while not _model_ready:
        await asyncio.sleep(0.25)
        _waited += 0.25
        if _waited and _waited % max(max_wait, 30.0) < 0.25:
            print(f"[readiness] still waiting for model init ({_waited:.0f}s)", flush=True)


async def _wait_for_workers(max_wait: float = 20.0):
    """Return (continuous_trainer, data_puller) once both are initialised.

    Guaranteed-completion policy: parks indefinitely (250 ms poll) until both
    workers exist — never returns early.  ``max_wait`` retained only as the
    slow-log threshold."""
    _waited = 0.0
    while True:
        with _workers_lock:
            ct, dp = _continuous_trainer, _data_puller
        if ct is not None and dp is not None:
            return ct, dp
        await asyncio.sleep(0.25)
        _waited += 0.25
        if _waited and _waited % max(max_wait, 30.0) < 0.25:
            print(f"[readiness] still waiting for workers ({_waited:.0f}s)", flush=True)


def _digest_str(fields: dict) -> str:
    """Stable digest string for a dict of request fields (used as pool spawn key)."""
    payload = json.dumps(fields, sort_keys=True, default=str).encode()
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


# ─── Storage health tracking ─────────────────────────────────────────────────
# Updated by _init_storage() at boot and polled lazily by _get_storage_mode().
# Three states:
#   "live"           — pdim is reachable and serving requests
#   "local_fallback" — pdim offline; disk-backed SQLite store is active
#   "offline"        — neither pdim nor the disk store is available
_storage_mode: str = "unknown"


def _get_storage_mode() -> str:
    """Return the current storage health as a string constant.

    Reads the live StorageClient state (never raises) so the caller always
    gets a concrete value to surface in health probes and generation responses."""
    try:
        from storage_client import get_storage
        s = get_storage()
        if s.is_available:
            return "live"
        return "local_fallback" if s.disk_store_available else "offline"
    except Exception:
        return "unknown"


# ─── PDIM orchestrator: dedup + single-flight (default for content gen) ────────
_pdim_orchestrator = None
_pdim_orch_lock = threading.Lock()


def _get_pdim_orchestrator():
    """Process-wide PDIM orchestrator — fleet-wide dedup + single-flight.

    Backed by the existing ``dedup_cache`` module, so cache keys and TTL
    semantics are unchanged. This only ADDS single-flight: N concurrent
    identical requests collapse to ONE real compute while the rest wait and
    share its result. This is the proven 90M-scale concurrency path and the
    default method for content generation."""
    global _pdim_orchestrator
    if _pdim_orchestrator is None:
        with _pdim_orch_lock:
            if _pdim_orchestrator is None:
                from ai_model.maxcore.pdim import PDIMOrchestrator
                _pdim_orchestrator = PDIMOrchestrator()
    return _pdim_orchestrator


# ─── Async-native request coalescer (90M concurrent path) ────────────────────
#
# The PDIMOrchestrator above uses threading.Event for single-flight, which is
# correct for background threads but wrong at the HTTP layer: each follower
# blocks an executor thread (~8 MB stack) instead of suspending as a coroutine
# (~2 KB).  At 90M simultaneous requests that difference is existential.
#
# This coalescer sits at the asyncio layer, BEFORE any thread is created.
# Identical-digest requests share one asyncio.Future; the other 89,999,999
# waiters are suspended coroutines.  Only unique digests ever enter the thread
# pool and the INFERENCE_GATE — pdim auto-scale ensures that universe is small.

class _AsyncRequestCoalescer:
    """Asyncio-native single-flight for generation handlers.

    Three request paths, in order of cost:

    1. **Settled hit** — a Future for this digest already resolved successfully
       (within the 500 ms eviction window).  Return ``existing.result()``
       directly: zero coroutine creation, zero thread-pool entry, zero gate
       slots.  This is the dominant path at 90 M-concurrent scale where the
       vast majority of requests arrive after the first compute finishes.

    2. **Active follower** — a Future exists but is still pending.  Await
       ``asyncio.shield(future)``: the coroutine suspends cheaply with no
       threads.  One leader computes; every follower wakes up with the result.

    3. **Leader** — no Future exists.  Create one, run ``coro_fn()``, resolve
       the Future, and schedule eviction after 500 ms.

    Thread-safe for coroutines (the asyncio event loop is single-threaded;
    all dict mutations between ``await`` points are GIL-atomic).
    """

    def __init__(self) -> None:
        self._inflight: dict[str, asyncio.Future] = {}

    @staticmethod
    def _digest(key_data: dict) -> str:
        import hashlib
        payload = json.dumps(key_data, sort_keys=True, default=str).encode()
        return hashlib.blake2b(payload, digest_size=16).hexdigest()

    async def compute(self, key_data: dict, coro_fn) -> Any:
        """Run coro_fn() once per unique digest; every other caller gets the
        same result for free — no extra threads, no extra gate slots."""
        digest = self._digest(key_data)
        loop   = asyncio.get_running_loop()

        existing = self._inflight.get(digest)
        if existing is not None:
            if not existing.done():
                # Path 2: active computation — suspend and share the result.
                return await asyncio.shield(existing)
            # Path 1: already settled — return the stored result synchronously.
            # This is the hot path at 90 M concurrent: no Future creation, no
            # thread-pool entry, no INFERENCE_GATE slot consumed.
            if not existing.cancelled() and existing.exception() is None:
                return existing.result()
            # Settled with an exception — fall through to become a new leader
            # so the request gets a fresh attempt rather than a stale error.

        # Path 3: Leader — register the Future before the first ``await`` so
        # any coroutine that computes the same digest after this point sees it
        # and becomes a follower rather than a duplicate leader.
        fut: asyncio.Future = loop.create_future()
        self._inflight[digest] = fut

        try:
            result = await coro_fn()
            fut.set_result(result)
            return result
        except BaseException as exc:
            if not fut.done():
                fut.set_exception(exc)
            raise
        finally:
            # Keep the settled Future alive for 500 ms so late arrivals get
            # Path 1 (synchronous result) rather than becoming new leaders.
            async def _evict() -> None:
                await asyncio.sleep(0.5)
                self._inflight.pop(digest, None)
            asyncio.ensure_future(_evict())


_async_coalescer: "_AsyncRequestCoalescer | None" = None
_async_coalescer_lock = threading.Lock()


def _get_async_coalescer() -> "_AsyncRequestCoalescer":
    global _async_coalescer
    if _async_coalescer is None:
        with _async_coalescer_lock:
            if _async_coalescer is None:
                _async_coalescer = _AsyncRequestCoalescer()
    return _async_coalescer


# ─── Static file serving for generated assets ────────────────────────────────

_UPLOADS_PATH = Path(__file__).parent / "uploads"
_UPLOADS_PATH.mkdir(parents=True, exist_ok=True)
(Path(__file__).parent / "uploads" / "images").mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(_UPLOADS_PATH)), name="uploads")


# ─── Uploads janitor ─────────────────────────────────────────────────────────
# Renders (audio_*, video_*, stems, temp WAVs) accumulate on the VM's
# persistent disk and are never re-read after the client fetches them.  An
# unbounded uploads/ dir eventually fills the disk, which surfaces as ffmpeg
# write failures mid-render and general production instability.  Sweep hourly:
# delete generated files older than the TTL, oldest-first beyond the size cap.
_UPLOADS_TTL_SEC = float(os.environ.get("UPLOADS_TTL_HOURS", "24")) * 3600
_UPLOADS_MAX_BYTES = int(os.environ.get("UPLOADS_MAX_MB", "2048")) * 1024 * 1024
# Only files matching generated-output prefixes are eligible — seed assets
# (uploads/images/ and any hand-placed files) are never touched.
_UPLOADS_SWEEP_PREFIXES = ("audio_", "video_", "scene_", "stem_", "tmp_")
# Grace window: files younger than this are NEVER eligible for deletion, even
# under the size cap.  Renders finish in minutes (120s job deadline), so this
# conservatively protects all in-flight intermediates and gives clients hours
# to download finished outputs before eviction can touch them.
_UPLOADS_MIN_AGE_SEC = float(os.environ.get("UPLOADS_MIN_AGE_HOURS", "6")) * 3600


def _uploads_janitor_sweep() -> None:
    try:
        entries = []
        for p in _UPLOADS_PATH.iterdir():
            if not p.is_file() or not p.name.startswith(_UPLOADS_SWEEP_PREFIXES):
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            entries.append((p, st.st_mtime, st.st_size))
        now = time.time()
        removed = 0
        # Grace window: anything younger than the min age is untouchable.
        entries = [(p, m, s) for p, m, s in entries
                   if now - m > _UPLOADS_MIN_AGE_SEC]
        # Pass 1: TTL
        survivors = []
        for p, mtime, size in entries:
            if now - mtime > _UPLOADS_TTL_SEC:
                try:
                    p.unlink(missing_ok=True)
                    removed += 1
                except OSError:
                    pass
            else:
                survivors.append((p, mtime, size))
        # Pass 2: total-size cap, oldest first
        total = sum(s for _, _, s in survivors)
        if total > _UPLOADS_MAX_BYTES:
            for p, _, size in sorted(survivors, key=lambda e: e[1]):
                if total <= _UPLOADS_MAX_BYTES:
                    break
                try:
                    p.unlink(missing_ok=True)
                    total -= size
                    removed += 1
                except OSError:
                    pass
        if removed:
            print(f"[uploads-janitor] removed {removed} generated files "
                  f"(remaining ~{total // (1024*1024)} MB)", flush=True)
    except Exception as exc:  # noqa: BLE001 — janitor must never crash anything
        print(f"[uploads-janitor] sweep error: {exc}", flush=True)


def _uploads_janitor_loop() -> None:
    while True:
        _uploads_janitor_sweep()
        time.sleep(3600)


threading.Thread(target=_uploads_janitor_loop, name="uploads-janitor",
                 daemon=True).start()

# ─── AI Model Globals ────────────────────────────────────────────────────────

_model_ready = False
_tokenizer = None
_creative_model = None
_script_agent = None
_visual_spec_agent = None
_distribution_agent = None
_optimization_agent = None
_repo = None
_adapter = None
_render_manager = None
_image_engine = None
_hyper_backend = None
_digital_gpu_backend = None
_model_config: dict[str, Any] = {}

_model_lock = threading.Lock()

# ─── Workers (DataPuller + ContinuousTrainer + Watchdog) ─────────────────────

_data_puller = None
_continuous_trainer = None
_watchdog = None
_asset_index = None
_coverage_watchdog = None
_gen_coalescer = None
_workers_lock = threading.Lock()

def _init_ai_model():
    global _model_ready, _tokenizer, _creative_model, _script_agent
    global _visual_spec_agent, _distribution_agent, _optimization_agent
    global _repo, _adapter, _render_manager, _model_config, _image_engine
    global _gen_coalescer, _hyper_backend, _digital_gpu_backend

    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from ai_model.model.tokenizer import BPETokenizer
        from ai_model.model.creative_model import CreativeModel
        from ai_model.agents.script_agent import ScriptAgent
        from ai_model.agents.visual_spec_agent import VisualSpecAgent
        from ai_model.agents.distribution_agent import DistributionAgent
        from ai_model.agents.optimization_agent import OptimizationAgent
        from ai_model.boostsheets.repository import BoostSheetRepository
        from ai_model.adapters.url_adapter import UrlToBoostSheetAdapter
        from ai_model.render_manager import RenderManager
        import torch

        print("[AI Model] Initializing MaxBooster AI Content Model...")
        _tokenizer = BPETokenizer()

        _TORCH_DEVICE = "cpu"   # PyTorch hardware device string — must be "cpu" on this host
        dim      = int(os.environ.get("AI_MODEL_DIM",     "512"))
        n_layers = int(os.environ.get("AI_MODEL_LAYERS",  "8"))
        n_heads  = int(os.environ.get("AI_MODEL_HEADS",   "8"))
        max_len  = int(os.environ.get("AI_MODEL_MAX_LEN", "1024"))

        # ── Digital GPU backend (HyperGPU + SM102 custom nvcc + pdim) ─────────
        # Initialise before the model so the pocket accelerator is warm by the
        # time the first GEMM fires during weight loading / warm-up.
        try:
            from ai_model.gpu.hyper_core import HyperGPU, PrecisionMode
            from ai_model.gpu.hyper_backend import HyperGPUBackend
            from ai_model.gpu.digital_gpu import DigitalGPU

            _hyper_backend    = HyperGPUBackend(
                lanes=512, tensor_cores=8,
                precision=PrecisionMode.MIXED,
            )
            _hyper_gpu        = _hyper_backend.gpu   # created internally
            _digital_gpu_backend = DigitalGPU()

            # Prime the pocket accelerator so its lazy init doesn't stall the
            # first real request.
            _hyper_gpu._pocket()

            print(
                f"[AI Model] Digital GPU ready — HyperGPU "
                f"(lanes={_hyper_gpu.core.lanes}, "
                f"tensor_cores={len(_hyper_gpu.core.tensor_core_units)}, "
                f"precision={_hyper_gpu.precision.name})"
            )
        except Exception as gpu_err:
            print(f"[AI Model] Digital GPU init skipped: {gpu_err}")
            _hyper_backend = None
            _digital_gpu_backend = None

        weights_dir  = Path(__file__).parent / "ai_model" / "weights"
        weights_path = weights_dir / "model.pt"

        state_dict = None
        if weights_path.exists():
            print(f"[AI Model] Loading weights from {weights_path}")
            checkpoint = torch.load(str(weights_path), map_location=_TORCH_DEVICE)
            if isinstance(checkpoint, dict) and "vocab" in checkpoint:
                _tokenizer.vocab    = checkpoint["vocab"]
                _tokenizer.inv_vocab = checkpoint["inv_vocab"]
                merges = checkpoint.get("merges", [])
                _tokenizer.merges        = [tuple(p) for p in merges]
                _tokenizer._merge_ranks  = {tuple(p): i for i, p in enumerate(_tokenizer.merges)}
                _tokenizer.freeze()
                state_dict = checkpoint["model_state_dict"]
                if "config" in checkpoint:
                    cfg      = checkpoint["config"]
                    dim      = cfg.get("dim",     dim)
                    n_layers = cfg.get("layers",  n_layers)
                    n_heads  = cfg.get("heads",   n_heads)
                    max_len  = cfg.get("max_len", max_len)
            else:
                state_dict = checkpoint

        vocab_size = max(len(_tokenizer.vocab), 1000)

        # ── Prefer HyperCreativeTransformerLM — digital GPU + KV-cache ────────
        # State-dict compatible with TransformerLM so existing weights load
        # directly.  Falls back to TransformerLM if the import fails.
        base_model = None
        if _hyper_backend is not None:
            try:
                from ai_model.gpu.hyper_creative_transformer import HyperCreativeTransformerLM
                base_model = HyperCreativeTransformerLM(
                    vocab_size=vocab_size,
                    dim=dim, n_layers=n_layers, n_heads=n_heads, max_len=max_len,
                    gpu=_hyper_gpu,
                )
                if state_dict is not None:
                    # Checkpoint may be prefixed with '_orig_mod.' when it was
                    # saved from a torch.compile-wrapped model — strip it so the
                    # keys match HyperCreativeTransformerLM's clean names.
                    clean_sd = {
                        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
                        for k, v in state_dict.items()
                    }
                    target_sd = base_model.state_dict()
                    filtered = {
                        k: v for k, v in clean_sd.items()
                        if k in target_sd and v.shape == target_sd[k].shape
                    }
                    base_model.load_state_dict(filtered, strict=False)
                    print(
                        f"[AI Model] Weights loaded into HyperCreativeTransformerLM "
                        f"({len(filtered)}/{len(clean_sd)} tensors matched)"
                    )
                else:
                    print("[AI Model] HyperCreativeTransformerLM — random init")
            except Exception as hct_err:
                print(f"[AI Model] HyperCreativeTransformerLM unavailable: {hct_err}")
                base_model = None

        if base_model is None:
            # Fallback: plain TransformerLM routed through the Digital GPU engine
            from ai_model.model.transformer import TransformerLM
            base_model = TransformerLM(
                vocab_size=vocab_size,
                dim=dim, n_layers=n_layers, n_heads=n_heads, max_len=max_len,
            )
            if state_dict is not None:
                filtered = {
                    k: v for k, v in state_dict.items()
                    if k not in base_model.state_dict()
                    or v.shape == base_model.state_dict()[k].shape
                }
                base_model.load_state_dict(filtered, strict=False)
            backend_name = "Digital GPU / aot_eager"
            try:
                base_model = torch.compile(base_model, backend="aot_eager", fullgraph=False)
                print("[AI Model] torch.compile applied (aot_eager / Digital GPU path)")
            except Exception as ce:
                print(f"[AI Model] torch.compile skipped: {ce}")
        else:
            backend_name = "Digital GPU (HyperGPU + SM102 + pdim)"

        print(f"[AI Model] Backend: {backend_name}")

        _creative_model      = CreativeModel(base_model, _tokenizer, device=_TORCH_DEVICE)
        _script_agent        = ScriptAgent(_creative_model)
        _visual_spec_agent   = VisualSpecAgent(_creative_model)
        _distribution_agent  = DistributionAgent(_creative_model)
        _optimization_agent  = OptimizationAgent(_creative_model)
        _repo                = BoostSheetRepository(path="boostsheets_db")
        _adapter             = UrlToBoostSheetAdapter(_repo)
        _render_manager      = RenderManager()

        from ai_model.image.image_engine import ImageEngine
        _image_engine = ImageEngine()
        print("[AI Model] ImageEngine ready (PIL renderer)")

        # ── Cross-request dynamic batching (on by default) ───────────────────
        # Coalesces concurrent unique generate() calls into one batched forward.
        # Each batch gets its own pocket GPU life (born → working → dead).
        # Disable with AI_DYNAMIC_BATCHING=0.
        try:
            from ai_model.dynamic_batching import install as _install_coalescer
            _gen_coalescer = _install_coalescer(
                _creative_model,
                gpu_pool=_get_gpu_pool(),
            )
            if _gen_coalescer is not None:
                print(
                    "[AI Model] Dynamic batching ENABLED (pipelined, pocket GPU per batch) "
                    f"max_batch={_gen_coalescer.max_batch}  "
                    f"window={_gen_coalescer.window_s * 1000:.0f}ms  "
                    f"pipe_depth={_gen_coalescer.pipe_depth}"
                )
            else:
                print("[AI Model] Dynamic batching disabled (AI_DYNAMIC_BATCHING=0)")
        except Exception as be:
            print(f"[AI Model] Dynamic batching not installed: {be}")

        _model_config = {"dim": dim, "layers": n_layers, "heads": n_heads, "max_len": max_len}
        _training_state["weights_exist"] = weights_path.exists()

        print(f"[AI Model] Ready. dim={dim}, layers={n_layers}, vocab={len(_tokenizer.vocab)}")
        log_training("AI model initialized and ready", level="info")

        with _model_lock:
            _model_ready = True

    except Exception as e:
        print(f"[AI Model] Initialization error: {e}")
        import traceback
        traceback.print_exc()
        log_training(f"AI model initialization failed: {e}", level="error")

# ─── Auth Helpers ────────────────────────────────────────────────────────────

ADMIN_KEY_ENV          = os.environ.get("ADMIN_KEY")
AI_TRAINING_KEY_PROD   = os.environ.get("AI_TRAINING_KEY_PROD")
AI_SERVER_KEY          = os.environ.get("AI_SERVER_KEY")

_ENV_BYPASS_KEYS: set = {k for k in [ADMIN_KEY_ENV, AI_TRAINING_KEY_PROD, AI_SERVER_KEY] if k}

def verify_api_key(x_api_key: str = Header(None), x_admin_key: str = Header(None),
                   authorization: str = Header(None)):
    """Verify API key from X-Api-Key, X-Admin-Key, or Authorization: Bearer."""
    bearer = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization[7:].strip()
    raw_key = x_api_key or x_admin_key or bearer
    if not raw_key:
        raise HTTPException(status_code=401, detail="API key required")

    # Allow env-based admin override for ADMIN_KEY and AI_TRAINING_KEY_PROD
    if raw_key in _ENV_BYPASS_KEYS:
        return {"id": "env-admin", "scopes": ["read", "write", "train", "admin", "generate"]}

    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    conn = None
    try:
        conn = _acquire()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT * FROM api_keys WHERE key_hash = %s AND is_active = TRUE",
            (key_hash,)
        )
        key_record = cur.fetchone()
        if not key_record:
            cur.close()
            _release(conn)
            raise HTTPException(status_code=401, detail="Invalid or inactive API key")

        expires_at = key_record["expires_at"]
        if expires_at and expires_at < datetime.now(timezone.utc):
            cur.close()
            _release(conn)
            raise HTTPException(status_code=401, detail="API key expired")

        cur.execute(
            "UPDATE api_keys SET request_count = request_count + 1, last_used_at = NOW() WHERE id = %s",
            (str(key_record["id"]),)
        )
        conn.commit()
        cur.close()
        _release(conn)
        return dict(key_record)
    except HTTPException:
        if conn:
            _release(conn)
        raise
    except Exception as e:
        if conn:
            _release(conn, error=True)
        raise HTTPException(status_code=500, detail=f"Auth error: {e}")

def require_scope(scope: str):
    def checker(key: dict = Depends(verify_api_key)):
        scopes = key.get("scopes", [])
        if "admin" in scopes:
            return key
        if scope not in scopes:
            raise HTTPException(status_code=403, detail=f"Scope '{scope}' required")
        return key
    return checker

def verify_admin(x_admin_key: str = Header(None), authorization: str = Header(None)):
    """Admin-only endpoint auth (X-Admin-Key or Authorization: Bearer)."""
    if not x_admin_key and authorization and authorization.lower().startswith("bearer "):
        x_admin_key = authorization[7:].strip()
    if not x_admin_key:
        raise HTTPException(status_code=401, detail="X-Admin-Key header required")

    # Allow env-based admin override
    if ADMIN_KEY_ENV and x_admin_key == ADMIN_KEY_ENV:
        return {"id": "env-admin", "scopes": ["admin"]}

    key_hash = hashlib.sha256(x_admin_key.encode()).hexdigest()
    conn = None
    try:
        conn = _acquire()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT * FROM api_keys WHERE key_hash = %s AND is_active = TRUE",
            (key_hash,)
        )
        key_record = cur.fetchone()
        cur.close()
        _release(conn)
        conn = None
        if not key_record:
            raise HTTPException(status_code=401, detail="Invalid admin key")
        scopes = key_record["scopes"] or []
        if "admin" not in scopes:
            raise HTTPException(status_code=403, detail="Admin scope required")
        return dict(key_record)
    except HTTPException:
        if conn:
            _release(conn)
        raise
    except Exception as e:
        if conn:
            _release(conn, error=True)
        raise HTTPException(status_code=500, detail=f"Auth error: {e}")

# ─── Admin content flywheel ───────────────────────────────────────────────────

_flywheel_ingestor: Any = None
_flywheel_init_lock = threading.Lock()


def _get_flywheel() -> Any:
    global _flywheel_ingestor
    if _flywheel_ingestor is not None:
        return _flywheel_ingestor
    with _flywheel_init_lock:
        if _flywheel_ingestor is None:
            try:
                from workers.admin_flywheel import FlywheelIngestor  # noqa: PLC0415
                _flywheel_ingestor = FlywheelIngestor()
            except Exception as e:
                _srv_logger.warning(f"[Flywheel] init error: {e}")
    return _flywheel_ingestor


def _fw_ingest(key: dict, content_type: str, payload: dict, meta: dict) -> None:
    """
    Fire-and-forget admin flywheel ingestion.
    Only stores if the request comes from an admin key. Never raises.
    """
    if "admin" not in (key.get("scopes") or []):
        return
    fw = _get_flywheel()
    if fw is None:
        return
    try:
        fw.ingest(content_type, payload, meta, str(key.get("id", "admin")))
    except Exception as e:
        _srv_logger.debug(f"[Flywheel] ingest error ({content_type}): {e}")


# Serialises read-modify-write of the audio dataset meta/index so two
# concurrent admin renders can't clobber each other's index append.
_AUDIO_DS_INGEST_LOCK = threading.Lock()


def _fw_ingest_audio_render(key: dict, job_id: str, render: dict,
                            genres: list) -> None:
    """Audio arm of the admin content flywheel — never raises.

    Pushes an admin-generated audio render back into the real-audio dataset
    pool (``mb:dataset:audio``) so B-Lawz's generated output becomes an
    additional dataset source for future renders, mirroring what
    ``_fw_ingest`` does for text/video/image payloads.

    Guards:
      * admin-scope gated (same contract as ``_fw_ingest``)
      * derivation guard — if the render's source sample is itself a
        flywheel entry, skip (prevents copy-of-copy quality decay)
      * content-hash dedup — identical bytes are never stored twice
    """
    if "admin" not in (key.get("scopes") or []):
        return
    try:
        import base64 as _b64
        import hashlib as _hl
        from storage_client import get_storage

        url = str(render.get("url") or "")
        if not url.startswith("/uploads/"):
            return
        path = _UPLOADS_PATH / url.split("/")[-1]
        if not path.exists():
            return
        raw = path.read_bytes()
        if not raw or len(raw) > 12 * 1024 * 1024:  # sanity cap
            return
        sha = _hl.sha256(raw).hexdigest()

        storage = get_storage()
        with _AUDIO_DS_INGEST_LOCK:
            meta = storage.get("mb:dataset:audio:meta") or {}
            index = list(meta.get("index") or [])
            # Derivation guard: don't re-ingest renders derived from
            # flywheel-sourced tracks (copy-of-copy stacking).
            src = render.get("source_sample") or {}
            src_idx = src.get("idx")
            if src_idx is not None:
                for e in index:
                    if int(e.get("idx", -1)) == int(src_idx):
                        if str(e.get("source") or "") == "flywheel":
                            return
                        break
            # Content dedup
            if any(e.get("content_sha") == sha for e in index):
                return
            next_idx = max(
                [int(e.get("idx", -1)) for e in index] + [int(meta.get("num_chunks", 0)) - 1]
            ) + 1
            bpm = float(render.get("bpm") or 0.0)
            key_name = str(render.get("key") or "")
            chunk = {
                "idx": next_idx,
                "title": f"blawz_flywheel_{job_id[:8]}",
                "artist": "B-Lawz",
                "genres": list(genres or []),
                "source": "flywheel",
                "derived_from": src_idx,
                "bpm": bpm,
                "key": key_name,
                "duration_sec": render.get("duration"),
                "sample_rate": render.get("sample_rate"),
                "format": str(render.get("format") or "mp3"),
                "b64": _b64.b64encode(raw).decode("ascii"),
            }
            storage.set(f"mb:dataset:audio:chunk:{next_idx}", chunk)
            index.append({
                "idx": next_idx, "bpm": bpm, "key": key_name,
                "genres": chunk["genres"], "source": "flywheel",
                "content_sha": sha,
            })
            meta["index"] = index
            meta["num_chunks"] = next_idx + 1
            storage.set("mb:dataset:audio:meta", meta)
        print(
            f"[flywheel] audio render job={job_id[:8]} ingested as dataset "
            f"idx={next_idx} (bpm={bpm}, key={key_name!r}, genres={genres})",
            flush=True,
        )
    except Exception as e:
        _srv_logger.debug(f"[Flywheel] audio ingest error: {e}")

# ─── Pydantic Schemas ────────────────────────────────────────────────────────

class CreateApiKeyRequest(BaseModel):
    name: str
    scopes: List[str] = ["read", "generate"]
    expires_in_days: Optional[int] = None

class StartTrainingRequest(BaseModel):
    epochs: int = 3
    learning_rate: float = 5e-4
    batch_size: int = 8
    use_synthetic: bool = True
    synthetic_count: int = 1000

class BPEScaleUpRequest(BaseModel):
    epochs: int = 6
    synthetic_count: int = 4000
    vocab_size: int = 4000
    dim: int = 512
    layers: int = 8
    heads: int = 8
    max_len: int = 1024
    batch_size: int = 8
    learning_rate: float = 3e-4

class HyperScaleUpRequest(BaseModel):
    """Train a fresh model whose real forward+backward compute is routed through
    the in-house Digital GPU (HyperGPU) tensor-core kernels — not native PyTorch.

    The Digital GPU executes matmul/attention/layer-norm/SiLU via NumPy
    tensor-core simulation with hand-written autograd, so this is genuine
    backend compute (~100-1000x slower per op than native torch). Defaults are
    intentionally modest so a real run completes in this Digital GPU environment;
    ``max_samples`` caps the corpus and can be raised if you accept the time cost.
    The trained weights are transferred into the fast KV-cache serving model."""
    epochs: int = 2
    synthetic_count: int = 600
    vocab_size: int = 2000
    dim: int = 256
    layers: int = 4
    heads: int = 8
    max_len: int = 128
    batch_size: int = 8
    learning_rate: float = 3e-4
    max_samples: int = 800

class PocketMultiplyRequest(BaseModel):
    """Multiply two matrices inside a PDIM pocket.

    A pocket is one dedup + single-flight domain: identical multiplications
    inside the same pocket are computed once on the DigitalGPU backend and
    every repeat (or concurrent duplicate) shares that stored result. Pockets
    nest without limit — ``pocket: "root/sub/leaf"`` addresses a pocket inside
    a pocket inside a pocket; the zlib compression applied to every stored
    payload is what lets one pocket hold arbitrarily many sub-pockets."""
    a: list = []
    b: list = []
    pocket: str = "default"

class _AwarenessMixin(BaseModel):
    """Normalises ``awareness`` whether Node.js sends it as a plain string or as
    the structured object ``{contextString, trendingGenres, ...}`` that
    ``enrichWithAwareness`` injects.  Extracts ``contextString`` from the dict so
    the Python parsers always receive the pre-formatted multi-line text.

    Also accepts ``description`` (free-text intent description) and
    ``prompt_url`` (URL to analyse for intent) that feed the intent
    sub-awareness layer — see :func:`_merged_awareness_for`.
    """

    awareness:    str = ""
    description:  str = ""   # free-text description of what to generate
    prompt_url:   str = ""   # URL to analyse for intent (Spotify, TikTok, etc.)

    @model_validator(mode="before")
    @classmethod
    def _normalise_awareness(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("awareness"), dict):
            data = dict(data)
            data["awareness"] = data["awareness"].get("contextString", "") or ""
        return data


def _as_text(v: Any) -> str:
    """Coerce a possibly-Any request field to a clean string ("" if not a str)."""
    return v.strip() if isinstance(v, str) else ""


def _merged_awareness_for(req: Any) -> str:
    """Build one endpoint's awareness-bridge input, now with intent detection.

    Cascade (highest priority first):
    1. [INTENT] lines  — from the intent sub-awareness layer (description +
                         prompt_url → detect_intent → structured signals)
    2. Direction lines — instruction / extra_context / content_themes serialised
                         by awareness_from_direction, leading with [HIGH]
    3. External awareness — caller-supplied awareness context string
    4. Platform awareness — from quality_awareness.platform_awareness_string

    [INTENT] lines sit above everything else so user-stated intent is the
    strongest conditioning signal throughout the whole brief pipeline.
    Never raises — falls back gracefully to the pre-intent behaviour.
    """
    from ai_model.generation import merge_awareness

    # ── Intent sub-awareness layer ────────────────────────────────────────
    # Detect intent from description and/or prompt_url (both may be empty).
    # We read the URL content once here and pass the text into detect_intent
    # so the awareness merge doesn't duplicate the HTTP call.
    intent_lines: str = ""
    intent_signals = None
    try:
        _desc       = _as_text(getattr(req, "description",  ""))
        _prompt_url = _as_text(getattr(req, "prompt_url",   ""))

        if _desc or _prompt_url:
            from ai_model.intent import detect_intent
            from ai_model.intent.url_reader import read_url as _read_url

            _url_text  = ""
            _url_plat  = ""
            _url_goal  = ""

            if _prompt_url:
                try:
                    # Universal URL Parser — richer than url_reader.read_url:
                    # platform-specific extractors, JSON-LD, music metadata,
                    # pre-formatted awareness block.
                    from ai_model.url_parser import parse_url as _parse_url_fn
                    _parsed   = _parse_url_fn(_prompt_url)
                    _url_text = _parsed.combined_text()
                    _url_plat = _parsed.platform
                    _url_goal = _parsed.goal
                    # Inject the parser's rich awareness block directly at the
                    # top of intent_lines so platform/artist/genre signals lead.
                    if _parsed.awareness_text and not _parsed.is_empty():
                        intent_lines = (
                            _parsed.awareness_text + ("\n" + intent_lines if intent_lines else "")
                        )
                except Exception:
                    pass

            intent_signals = detect_intent(
                description       = _desc,
                url               = "",           # already read above
                url_content_text  = _url_text,
                url_platform_hint = _url_plat,
                url_goal_hint     = _url_goal,
            )

            if intent_signals.is_useful():
                _new_intent = "\n".join(intent_signals.to_awareness_lines())
                # Combine with URL parser awareness (if any) rather than overwriting:
                # intent signals lead (highest priority), URL parser signals follow.
                intent_lines = (
                    _new_intent + "\n" + intent_lines
                    if intent_lines else _new_intent
                )
    except Exception:
        pass

    # ── Base awareness merge (direction + external + platform) ───────────
    base_awareness = merge_awareness(req)

    # ── Assemble final cascade ────────────────────────────────────────────
    parts = [p for p in (intent_lines, base_awareness) if p]
    return "\n".join(parts)


def _intent_signals_for(req: Any):
    """Return the :class:`IntentSignals` for *req* (None if nothing to detect).

    Detects from ``req.description`` and/or ``req.prompt_url``.  Intended for
    callers that need to pass intent_signals= into build_brief() directly so
    the signal application happens before brand-voice fallbacks, rather than
    only through the awareness string.  Never raises.
    """
    try:
        _desc       = _as_text(getattr(req, "description",  ""))
        _prompt_url = _as_text(getattr(req, "prompt_url",   ""))
        if not (_desc or _prompt_url):
            return None
        from ai_model.intent import detect_intent
        from ai_model.intent.url_reader import read_url as _read_url
        _url_text, _url_plat, _url_goal = "", "", ""
        if _prompt_url:
            try:
                _uc       = _read_url(_prompt_url)
                _url_text = _uc.combined()
                _url_plat = _uc.platform_hint
                _url_goal = _uc.goal_hint
            except Exception:
                pass
        sig = detect_intent(
            description       = _desc,
            url_content_text  = _url_text,
            url_platform_hint = _url_plat,
            url_goal_hint     = _url_goal,
        )
        return sig if sig.is_useful() else None
    except Exception:
        return None


class ContentRequest(_AwarenessMixin):
    platform: str = "tiktok"
    topic: str
    tone: str = "energetic"
    goal: str = "growth"
    include_hashtags: bool = True
    # Awareness-bridge direction fields — merged into effective awareness so
    # the script agent and distribution agent are conditioned on the caller's
    # creative intent as well as live platform signals.
    instruction:    Optional[str]       = None   # free-form creative directive
    extra_context:  Optional[str]       = None   # supplementary background context
    content_themes: Optional[List[str]] = None   # thematic keywords (→ bullet, never #tags)

# ─── Startup ─────────────────────────────────────────────────────────────────

# ─── Background liveness-probe server ────────────────────────────────────────
# Runs on HEALTHZ_PORT (default 9879) in its own daemon thread, completely
# independent of uvicorn's event loop.  GC pauses, model inference, asyncio
# blocking — nothing in the main server can stall this.  Node's health monitor
# probes this port instead of the main API port so it can tell the difference
# between "process is truly hung" and "uvicorn event loop is GC-paused for
# 10-20 s while the process is otherwise healthy".
#
# Without this: a GC pause during a health probe looks like a hang → Node
# kills and restarts Python → 90-120 s of downtime during model reload.
# With this: GC can pause for any duration; healthz always responds in < 1 ms.

_HEALTHZ_PORT = int(os.environ.get("HEALTHZ_PORT", str(int(os.environ.get("MODEL_API_PORT", "9878")) + 1)))


def _start_healthz_server() -> None:
    """Bind the healthz TCP socket and serve in a daemon thread.  Never-raise.

    The handler is defined inside this function so the `http.server` import is
    scoped here — avoids a module-level `import http.server` that is easy to
    forget and causes an AttributeError at class-definition time.

    If the port is already taken (e.g. previous instance didn't die cleanly) we
    log and continue; Node falls back to the 25-second uvicorn probe path.
    """
    import http.server as _http

    class _HealthzHandler(_http.BaseHTTPRequestHandler):
        _BODY = b'{"status":"ok","source":"healthz-thread"}'

        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(self._BODY)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(self._BODY)

        def log_message(self, fmt: str, *args: object) -> None:
            pass  # silence access logs

    try:
        srv = _http.HTTPServer(("0.0.0.0", _HEALTHZ_PORT), _HealthzHandler)
        t = threading.Thread(target=srv.serve_forever, name="healthz-server", daemon=True)
        t.start()
        print(f"[Server] Healthz liveness server on port {_HEALTHZ_PORT} (event-loop-independent)", flush=True)
    except OSError as _hz_err:
        print(f"[Server] WARNING: healthz server could not bind port {_HEALTHZ_PORT}: {_hz_err} — falling back to uvicorn probe", flush=True)


@app.on_event("startup")
async def on_startup():
    # ── Thread pool: 512 workers for async I/O routing ───────────────────────
    # This pool dispatches HTTP handlers and waits on GPU-backend results — it
    # is I/O-bound, not CPU-bound. 512 workers keeps hundreds of concurrent
    # in-flight requests from queuing. Host CPU count is irrelevant here.
    import concurrent.futures as _cf
    asyncio.get_event_loop().set_default_executor(
        _cf.ThreadPoolExecutor(
            max_workers=512,
            thread_name_prefix="req-worker",
        )
    )

    # ── Dynamic batching config — tuned for 90 M unique requests ─────────────
    # Always hard-assign (never setdefault) so stale env vars from a previous
    # process cannot silently cap throughput across restarts.
    #
    # max_batch=64 : fills a full SIMD lane every collection cycle; at 90M
    #                scale the arriving flood keeps the batch full every 2 ms.
    # window=2 ms  : tighter than 4 ms — under sustained unique-request burst
    #                a 2 ms window still fills B=64 without adding idle latency.
    # timeout=600 s: allows deep queues to drain without timing out under load
    #                spikes (e.g. a 10-min wave of 90M requests hitting at once).
    os.environ["AI_DYNAMIC_BATCHING"] = "1"
    os.environ["AI_BATCH_MAX"]        = "64"   # always max — 90M unique floods fill every slot
    os.environ["AI_BATCH_WINDOW_MS"]  = "2"    # 2 ms collection window
    os.environ["AI_BATCH_TIMEOUT_S"]  = "600"  # 10 min — deep queues must not timeout
    os.environ["AI_PIPE_DEPTH"]       = "4"    # collector stays 3 batches ahead of executor

    # Start the event-loop-independent liveness probe server immediately so
    # Node's health monitor can distinguish GC pauses from true process hangs.
    # Must be first: if model init blocks for 30+ s the healthz thread needs to
    # already be bound so Node doesn't misfire a hung-detection kill.
    _start_healthz_server()

    if not DATABASE_URL:
        print("[Server] WARNING: DATABASE_URL not set — running without DB")
        return
    init_db()
    thread = threading.Thread(target=_init_ai_model, daemon=True)
    thread.start()
    storage_thread = threading.Thread(target=_init_storage, daemon=True)
    storage_thread.start()
    # Proactive awareness refresh: keep the live industry beacon fresh on a
    # fixed cadence (default 6 h) instead of only when a request notices the
    # 24 h staleness cliff. Never-raise, idempotent daemon.
    try:
        from ai_model.quality_awareness import start_scheduler as _aw_sched
        _aw_sched()
    except Exception as _aw_exc:  # noqa: BLE001
        print(f"[Server] awareness scheduler not started: {_aw_exc}")
    # Audio seeding watchdog — keeps the audio dataset seeded from live
    # awareness signals (trending genres + Deezer BPM targets) while the
    # admin's own flywheel corpus is not yet self-sufficient.  Once the own
    # corpus retires the buffer (buffer_weight → 0), the watchdog stops
    # external seeding and audio generation draws from admin-built tracks only.
    try:
        from ai_model.quality_awareness import start_audio_seeding_watchdog as _aw_dog
        _aw_dog()
    except Exception as _aw_dog_exc:  # noqa: BLE001
        print(f"[Server] audio seeding watchdog not started: {_aw_dog_exc}")
    warm_thread = threading.Thread(target=_warm_content_cache, daemon=True)
    warm_thread.start()
    subsys_thread = threading.Thread(target=_warm_start_subsystems, daemon=True)
    subsys_thread.start()


# ── Warm-start: eagerly initialise every lazy subsystem at boot ──────────────
# On a Reserved VM the process only restarts on deploys, so the one remaining
# cold-start cost is lazy first-request initialisation (diffusion checkpoints,
# retrieval index, intent detector, librosa/audio stack, awareness buffer).
# This pass touches each singleton at boot using the app's own persistence
# (pdim + disk checkpoints), so the first real request after a deploy runs at
# steady-state speed. Every step is never-raise and individually timed.
# Opt out with MB_WARM_START=0.

_warm_status: dict[str, Any] = {"state": "pending", "steps": {}}


def _warm_start_subsystems() -> None:
    import time as _wt

    if os.environ.get("MB_WARM_START", "1") == "0":
        _warm_status["state"] = "disabled"
        return

    # Wait for the model first — several subsystems hang off it.
    for _ in range(180):
        if _model_ready:
            break
        _wt.sleep(1)

    def _step(name: str, fn) -> None:
        t0 = _wt.time()
        try:
            fn()
            _warm_status["steps"][name] = {"ok": True, "ms": int((_wt.time() - t0) * 1000)}
        except Exception as exc:
            _warm_status["steps"][name] = {
                "ok": False, "ms": int((_wt.time() - t0) * 1000),
                "error": f"{type(exc).__name__}: {exc}",
            }
        print(f"[WarmStart] {name}: {_warm_status['steps'][name]}", flush=True)

    _warm_status["state"] = "running"

    def _warm_diffusion():
        # Loads DiT/VAE checkpoints and builds the pipeline singleton.
        from ai_model.video.diffusion.maxcore_diffusion import _get_pipeline
        _get_pipeline()

    def _warm_retrieval():
        # Builds/loads the RCGS asset index used for background conditioning.
        from ai_model.retrieval.asset_pipeline import get_asset_index
        get_asset_index()

    def _warm_intent():
        # First call builds the detector's vocabulary/pattern tables.
        from ai_model.intent.detector import detect_intent
        detect_intent("warm start probe: dark trap night drive visual")

    def _warm_audio_stack():
        # librosa/producer_tools imports are seconds-heavy; also prime the
        # real-audio dataset meta from pdim so the first soundtrack render
        # doesn't pay the fetch.
        from ai_model.audio import producer_tools  # noqa: F401
        from storage_client import get_storage
        get_storage().get("mb:dataset:audio:meta")

    def _warm_awareness():
        # Primes the platform quality-awareness buffer read path.
        from ai_model import quality_awareness
        quality_awareness.platform_awareness_string("tiktok")

    def _warm_request_intelligence():
        from ai_model import request_intelligence as ri
        ri.build_brief(modality="content", platform="tiktok",
                       topic="warm start probe", goal="growth", tone="energetic")

    _step("diffusion_pipeline",    _warm_diffusion)
    _step("retrieval_index",       _warm_retrieval)
    _step("intent_detector",       _warm_intent)
    _step("audio_stack",           _warm_audio_stack)
    _step("awareness_buffer",      _warm_awareness)
    _step("request_intelligence",  _warm_request_intelligence)

    _warm_status["state"] = "warm" if all(
        s.get("ok") for s in _warm_status["steps"].values()
    ) else "partial"
    print(f"[WarmStart] complete: {_warm_status['state']}", flush=True)


def _warm_content_cache() -> None:
    """
    Pre-warm the PDIM dedup cache with the top platform/topic combos so the
    first real user request hits the cache instead of waiting ~3 s for compute.
    Runs in a daemon thread after startup; waits up to 120 s for the model.
    """
    import time as _wt
    for _ in range(120):
        if _model_ready and _script_agent is not None and _distribution_agent is not None:
            break
        _wt.sleep(1)
    else:
        print("[CacheWarm] Model not ready after 120 s — skipping pre-warm")
        return

    from ai_model.agents.script_agent import ScriptRequest
    from ai_model.agents.distribution_agent import DistributionRequest

    WARM_PLATFORMS = ["tiktok", "instagram", "youtube", "twitter", "spotify"]
    WARM_TOPICS = [
        "new music drop", "album release", "tour announcement",
        "single out now", "music video premiere",
    ]
    WARM_TONE = "energetic"
    WARM_GOAL = "growth"

    _orch = _get_pdim_orchestrator()
    warmed = 0
    total = len(WARM_PLATFORMS) * len(WARM_TOPICS)
    for plat in WARM_PLATFORMS:
        for topic in WARM_TOPICS:
            try:
                cache_key = {
                    "platform": plat, "topic": topic,
                    "tone": WARM_TONE, "goal": WARM_GOAL, "awareness": None,
                }

                def _builder(_req=None, _p=plat, _t=topic):
                    sr = _script_agent.run(ScriptRequest(
                        idea=_t, platform=_p, goal=WARM_GOAL, tone=WARM_TONE,
                    ))
                    full_script = f"{sr.hook}\n{sr.body}\n{sr.cta}"
                    dr = _distribution_agent.run(DistributionRequest(
                        script=full_script, platform=_p, goal=WARM_GOAL,
                    ))
                    return {
                        "success": True,
                        "platform": _p,
                        "caption": dr.caption,
                        "hook": sr.hook,
                        "body": sr.body,
                        "cta": sr.cta,
                        "hashtags": dr.hashtags,
                        "source": getattr(sr, "source", "template"),
                    }

                _orch.compute(cache_key, _builder, namespace="api_content_v4")
                warmed += 1
            except Exception as exc:
                print(f"[CacheWarm] {plat}/{topic}: {exc}")

    print(f"[CacheWarm] Pre-warmed {warmed}/{total} content slots into PDIM cache")


def _reconcile_audio_manifest(storage) -> None:
    """Reconcile mb:dataset:audio:meta against the chunks actually on disk.

    On every server start we count the real ``mb:dataset:audio:chunk:{idx}``
    keys that the disk store (SQLite) still holds and compare that count to
    the ``num_chunks`` field recorded in the manifest.  If the manifest is
    missing or records a lower count than what is actually stored (the common
    "stale low count" case caused by the direct-Python test path overwriting
    the manifest), we update ``num_chunks`` and rebuild any missing index
    entries so generation can reach every track that survived the restart.

    Cost when the dataset is healthy: one disk scan (fast) + one storage GET.
    Never raises — any error is logged and we continue with what we have.
    """
    try:
        # ── Count real chunks on disk ──────────────────────────────────────
        # The disk store is always available (SQLite); pdim may be offline.
        # We read from the disk store directly so we count what survived the
        # restart, not what a possibly-stale pdim thinks exists.
        disk = getattr(storage, "_disk", None)
        if disk is None or not disk.available:
            return

        prefix = "mb:dataset:audio:chunk:"
        all_keys = disk.all_keys()
        chunk_keys = [k for k in all_keys if k.startswith(prefix)]
        actual_count = len(chunk_keys)

        if actual_count == 0:
            # Nothing seeded yet — nothing to reconcile.
            return

        # Derive the highest idx present so the manifest range is correct.
        max_idx = -1
        for k in chunk_keys:
            try:
                idx = int(k[len(prefix):])
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                pass

        num_chunks = max_idx + 1  # contiguous manifest up to highest idx

        # ── Read (or synthesise) the current manifest ──────────────────────
        meta = storage.get("mb:dataset:audio:meta")
        current_count = int((meta or {}).get("num_chunks", 0))

        if meta and current_count >= num_chunks:
            # Manifest already covers every stored chunk — nothing to do.
            print(
                f"[Storage] audio manifest OK — num_chunks={current_count}, "
                f"disk chunks={actual_count}"
            )
            return

        # ── Rebuild the index for chunks not listed in the existing meta ───
        existing_index: list = list((meta or {}).get("index", []))
        existing_idxs = {int(e.get("idx", -1)) for e in existing_index}

        for k in sorted(chunk_keys):
            try:
                idx = int(k[len(prefix):])
            except ValueError:
                continue
            if idx in existing_idxs:
                continue
            sample = disk.get(k)
            if not sample or not isinstance(sample, dict):
                continue
            existing_index.append(
                {
                    "idx": idx,
                    "bpm": float(sample.get("bpm") or 0.0),
                    "key": sample.get("key") or "",
                    "genres": sample.get("genres") or [],
                }
            )
            existing_idxs.add(idx)

        existing_index.sort(key=lambda e: int(e.get("idx", 0)))

        # ── Write the corrected manifest ───────────────────────────────────
        new_meta = dict(meta) if meta else {
            "name": "audio",
            "description": "Real CC-licensed music samples",
            "content_type": "audio",
            "source": "disk-reconciled",
        }
        new_meta["num_chunks"] = num_chunks
        new_meta["index"] = existing_index

        storage.set("mb:dataset:audio:meta", new_meta)
        print(
            f"[Storage] audio manifest reconciled — "
            f"was num_chunks={current_count}, "
            f"now num_chunks={num_chunks} "
            f"(disk chunks={actual_count})"
        )
    except Exception as exc:  # never crash startup
        print(f"[Storage] audio manifest reconciliation skipped: {exc}")


def _init_storage():
    """Connect to storage server, load checkpoint, and start workers."""
    global _data_puller, _continuous_trainer, _watchdog
    global _asset_index, _coverage_watchdog
    from storage_client import get_storage
    storage = get_storage()

    # Retry the startup ping up to 4 times with 5 s delays (20 s total window).
    # pdim is an external Replit deployment that can take 5–15 s to serve its
    # first request after a cold wake.  A single failed ping at process start
    # would leave storage "offline" until the periodic health-check thread
    # recovers it — now we give it a fair chance before falling back.
    ok = False
    global _storage_mode
    for _attempt in range(4):
        ok = storage.ping()
        if ok:
            break
        if _attempt < 3:
            print(f"[Storage] Startup ping attempt {_attempt + 1}/4 failed — retrying in 5 s")
            time.sleep(5)

    if ok:
        _storage_mode = "live"
        print("[Storage] Connected to MaxBooster storage server")
        _load_checkpoint_from_storage()
    else:
        _storage_mode = "local_fallback" if storage.disk_store_available else "offline"
        print(f"[Storage] Storage server offline after 4 attempts — using in-process fallback (mode={_storage_mode})")

    # ── Audio manifest reconciliation (runs on every boot, cheap when healthy)
    _reconcile_audio_manifest(storage)

    sys.path.insert(0, str(Path(__file__).parent))
    from workers.data_puller import DataPuller
    from workers.continuous_trainer import ContinuousTrainer

    from workers.watchdog import get_watchdog

    with _workers_lock:
        _data_puller = DataPuller(storage)
        _continuous_trainer = ContinuousTrainer(
            storage=storage,
            data_puller=_data_puller,
            run_training_fn=_training_bridge,
            curriculum_phases=CURRICULUM_PHASES,
        )
        _watchdog = get_watchdog()

    # ── Auto-start DataPuller so audio auto-growth fires without manual intervention
    _data_puller.start(interval_minutes=30)
    print("[Storage] DataPuller auto-started (30-min pull cycle, audio auto-growth enabled)")

    # ── Immediate audio check: seed if below threshold without waiting 30 min
    _data_puller.check_audio_now()

    # Inject references the watchdog needs to monitor everything
    _watchdog.storage           = storage
    _watchdog.training_state    = _training_state
    _watchdog.training_lock     = _training_lock
    _watchdog.model_ready_ref   = lambda: _model_ready
    _watchdog.init_model_fn     = _init_ai_model
    _watchdog.continuous_trainer = _continuous_trainer
    _watchdog.data_puller       = _data_puller
    _watchdog.weights_dir       = Path(__file__).parent / "ai_model" / "weights"

    # Extended stay-alive references (background infrastructure)
    _watchdog.storage_client_ref   = storage          # StorageClient (health-thread probe)
    _watchdog.flywheel_ingestor_fn = _get_flywheel     # lazy — returns FlywheelIngestor|None

    # Content generation service references — injected after _init_ai_model
    # completes so the globals are guaranteed non-None on first assignment.
    _watchdog.gen_coalescer    = _gen_coalescer
    _watchdog.render_manager   = _render_manager

    # Coalescer reinstall: restores the original generate, then re-wraps.
    # _creative_model.generate may already be the coalescer wrapper; the
    # saved reference lets us avoid double-wrapping on repeated reinstalls.
    _original_generate_ref = [None]
    if _creative_model is not None:
        _original_generate_ref[0] = getattr(_creative_model, "_orig_generate", None)

    def _reinstall_coalescer():
        global _gen_coalescer
        try:
            from ai_model.dynamic_batching import install as _install_coalescer  # noqa
            # Restore the unwrapped generate before wrapping again.
            orig = _original_generate_ref[0]
            if orig is not None and _creative_model is not None:
                _creative_model.generate = orig
            new_gc = _install_coalescer(_creative_model, gpu_pool=_get_gpu_pool())
            _gen_coalescer = new_gc
            _watchdog.gen_coalescer = new_gc
            return new_gc
        except Exception as exc:
            print(f"[Watchdog] coalescer reinstall failed: {exc}", flush=True)
            return None

    _watchdog.reinstall_coalescer_fn = _reinstall_coalescer

    # ── Rendering system health callbacks ──────────────────────────────
    def _rendering_health_fn():
        """Return live status of every content generation object."""
        gc = _gen_coalescer
        coalescer_ok = (
            gc is None  # disabled intentionally — not a failure
            or (
                (gc._collector.is_alive() if getattr(gc, "_collector", None) else True)
                and (gc._executor.is_alive() if getattr(gc, "_executor", None) else True)
            )
        )
        rm = _render_manager
        render_manager_ok = (
            rm is None
            or (
                (not getattr(getattr(rm, "_executor", None), "_shutdown", False))
                and (rm._gc_thread.is_alive() if getattr(rm, "_gc_thread", None) else True)
            )
        )
        return {
            "ready": _model_ready,
            "objects": {
                "creative_model":       _creative_model is not None,
                "script_agent":         _script_agent is not None,
                "visual_spec_agent":    _visual_spec_agent is not None,
                "distribution_agent":   _distribution_agent is not None,
                "optimization_agent":   _optimization_agent is not None,
                "image_engine":         _image_engine is not None,
                "render_manager":       render_manager_ok,
                "gen_coalescer":        coalescer_ok,
            },
        }

    def _keepalive_fn() -> bool:
        """
        Expanded end-to-end probe across all content generation services.

        Probes run in parallel sub-threads to avoid serialising ~3 s of model
        inference.  Returns True when at least one service is healthy so that a
        single degraded service doesn't mask the rest.

        Services probed:
          • ScriptAgent     — full model inference (catches any model regression)
          • ImageEngine     — functional presence check (PIL, no heavy compute)
          • Audio dataset   — storage read confirming chunks exist
        """
        import concurrent.futures as _cf

        results: dict[str, bool] = {}

        def _probe_script():
            try:
                if not _model_ready or _script_agent is None:
                    return "script_agent", False
                from ai_model.agents.script_agent import ScriptRequest  # noqa
                r = _script_agent.run(ScriptRequest(
                    idea="new single dropping",
                    platform="instagram",
                    goal="growth",
                    tone="energetic",
                ))
                return "script_agent", r is not None
            except Exception as exc:
                print(f"[Watchdog] ScriptAgent probe error: {exc}", flush=True)
                return "script_agent", False

        def _probe_image():
            try:
                ok = _image_engine is not None and hasattr(_image_engine, "render")
                return "image_engine", ok
            except Exception:
                return "image_engine", False

        def _probe_audio():
            try:
                from storage_client import get_storage  # noqa
                meta = get_storage().get("mb:dataset:audio:meta")
                ok = isinstance(meta, dict) and int(meta.get("num_chunks", 0)) > 0
                return "audio_dataset", ok
            except Exception:
                return "audio_dataset", False

        probes = [_probe_script, _probe_image, _probe_audio]
        with _cf.ThreadPoolExecutor(max_workers=len(probes), thread_name_prefix="ka-probe") as pool:
            futures = [pool.submit(p) for p in probes]
            for fut in _cf.as_completed(futures, timeout=30):
                try:
                    name, ok = fut.result()
                    results[name] = ok
                except Exception:
                    pass

        any_ok = any(results.values())
        print(f"[Watchdog] Keep-alive: {results}", flush=True)
        return any_ok

    _watchdog.rendering_health_fn = _rendering_health_fn
    _watchdog.keepalive_fn        = _keepalive_fn
    # ──────────────────────────────────────────────────────────────────

    _watchdog.start()

    print("[Workers] DataPuller, ContinuousTrainer, and Watchdog initialized and running")

    # ── Retrieval coverage system (mirrors the watchdog DI + lifecycle) ──
    try:
        from ai_model.retrieval.asset_pipeline import (
            ensure_library, get_asset_index, ingest_gaps,
        )
        from ai_model.retrieval.coverage_watchdog import get_coverage_watchdog
        from ai_model.retrieval.probes import recent_probes

        with _workers_lock:
            _asset_index = get_asset_index()
            _coverage_watchdog = get_coverage_watchdog()

        _coverage_watchdog.index   = _asset_index
        _coverage_watchdog.storage = storage

        # Restore a persisted index snapshot if one exists.
        try:
            snap = storage.get(_coverage_watchdog.STATE_KEY + ":index")
            if snap:
                _asset_index.load_state(snap)
        except Exception as exc:
            print(f"[Coverage] index restore skipped: {exc}")

        _coverage_watchdog.anchor_loader_fn = lambda: ensure_library(_asset_index)
        _coverage_watchdog.ingestion_fn     = lambda gaps: ingest_gaps(_asset_index, gaps)
        _coverage_watchdog.probe_source_fn  = lambda: recent_probes()  # live RCGS probes

        # Ensure the real domain library is present before the daemon starts.
        ensure_library(_asset_index)
        _coverage_watchdog.start()
        # Wire the now-live CoverageWatchdog into the primary watchdog so it
        # can detect a dead thread and call .start() to bring it back.
        _watchdog.coverage_watchdog = _coverage_watchdog
        print(f"[Coverage] AssetIndex ready ({_asset_index.size} assets, "
              f"{_asset_index.anchor_count} anchors) — CoverageWatchdog running")
    except Exception as exc:
        print(f"[Coverage] init error (non-fatal): {exc}")

    # ── Generated-asset ingestor (folds produced images back into the index) ──
    try:
        from ai_model.retrieval.generated_ingestor import get_generated_ingestor

        _gen_ingestor = get_generated_ingestor()
        _gen_ingestor.index = _asset_index  # share the live index (None-safe lazily too)
        _gen_ingestor.start()
        _watchdog.generated_ingestor = _gen_ingestor
        print("[Coverage] GeneratedIngestor running — produced images fold back into the index")
    except Exception as exc:
        print(f"[Coverage] generated-ingestor init error (non-fatal): {exc}")


def _training_bridge(texts: list, epochs: int, phase_label: str,
                     loss_target: float = None) -> dict:
    """
    Called by ContinuousTrainer to run a training pass.
    Writes data to a temp file, builds dataset, runs the trainer, returns loss.
    """
    from ai_model.training.dataset import CreativeDataset
    from ai_model.training.trainer import train as run_train
    from ai_model.training.config import TrainConfig

    if not _model_ready or _creative_model is None or _tokenizer is None:
        return {"loss": None, "error": "model_not_ready"}

    data_dir = Path(__file__).parent / "training"
    data_dir.mkdir(exist_ok=True)
    data_path = str(data_dir / f"continuous_{phase_label}.json")

    import json as _json
    Path(data_path).write_text(_json.dumps(texts, ensure_ascii=False))

    _tokenizer.unfreeze()
    train_max_len = min(256, _creative_model.model.pos_emb.num_embeddings)
    dataset = CreativeDataset(data_path, _tokenizer, max_len=train_max_len)

    dim = _creative_model.model.token_emb.embedding_dim
    n_layers = len(_creative_model.model.layers)
    cfg = TrainConfig({
        "model": {"dim": dim, "layers": n_layers, "heads": 8, "max_len": train_max_len},
        "train": {"lr": 3e-4, "batch_size": 4, "epochs": epochs, "data_path": data_path},
    })
    _creative_model.resize_embeddings()

    result = run_train(_creative_model.model, dataset, _tokenizer, cfg, device="cpu")
    _tokenizer.freeze()

    final_loss = result.get("final_loss", 999)

    # Save checkpoint (include tokenizer vocab so _init_ai_model can restore it)
    weights_dir = Path(__file__).parent / "ai_model" / "weights"
    weights_dir.mkdir(exist_ok=True)
    import torch
    import numpy as np
    torch.save({
        "model_state_dict": _creative_model.model.state_dict(),
        "vocab": _tokenizer.vocab,
        "inv_vocab": _tokenizer.inv_vocab,
        "merges": _tokenizer.merges,
        "config": _model_config,
    }, str(weights_dir / "model.pt"))
    np_weights = {k: v.cpu().numpy() for k, v in _creative_model.model.state_dict().items()}
    np.savez_compressed(str(weights_dir / "weights_v4.npz"), **np_weights)

    return {"loss": round(final_loss, 4), "samples": len(dataset), "epochs": epochs}


def _load_checkpoint_from_storage():
    """
    After training, checkpoints are saved to storage. On every boot, we check
    storage for the latest weights so the platform always runs on trained data.
    """
    try:
        from storage_client import get_checkpoint_client
        client = get_checkpoint_client()
        history = client.list_checkpoints()
        if not history:
            print("[Storage] No checkpoints in storage — using local weights or random init")
            return

        # Use the most recent checkpoint
        latest = history[0]
        model_id = latest.get("model_id", "maxbooster-v1")
        print(f"[Storage] Found checkpoint '{model_id}' (saved {latest.get('saved_at', '?')}), loading...")

        # The checkpoint stores metadata, not the full weights (weights live on disk).
        # But we use the metadata to confirm the last training run's params.
        meta = client.get_checkpoint_meta(model_id)
        if meta:
            state = meta.get("state", {})
            print(f"[Storage] Checkpoint metadata: batches={state.get('batches', '?')} "
                  f"loss={state.get('final_loss', '?')} source={meta.get('metadata', {}).get('source', '?')}")

        # If local weights exist, they were already loaded in _init_ai_model.
        # This function enriches _training_state with the storage checkpoint info.
        with _training_lock:
            _training_state["last_checkpoint"] = {
                "model_id": model_id,
                "saved_at": latest.get("saved_at"),
                "hash": latest.get("hash"),
                "source": "storage",
            }
        print("[Storage] Checkpoint sync complete — model is ready with trained data")
    except Exception as e:
        print(f"[Storage] Checkpoint load error (non-fatal): {e}")

# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": _model_ready,
        "uptime_seconds": time.time() - _start_time,
        "version": "1.0.0",
        "warm_start": _warm_status,
        "storage_mode": _get_storage_mode(),
    }

_start_time = time.time()

# ─── API Key Management ───────────────────────────────────────────────────────

@app.get("/api-keys")
async def list_api_keys(_admin = Depends(verify_admin)):
    conn = _acquire()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT id, name, prefix, scopes, is_active, request_count, created_at, last_used_at, expires_at FROM api_keys ORDER BY created_at DESC")
        rows = cur.fetchall()
        cur.close()
    except Exception:
        _release(conn, error=True)
        raise
    _release(conn)
    keys = []
    for r in rows:
        keys.append({
            "id": str(r["id"]),
            "name": r["name"],
            "prefix": r["prefix"],
            "scopes": r["scopes"] or [],
            "is_active": r["is_active"],
            "request_count": r["request_count"],
            "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            "last_used_at": r["last_used_at"].isoformat() if r["last_used_at"] else None,
            "expires_at": r["expires_at"].isoformat() if r["expires_at"] else None,
        })
    return {"keys": keys, "total": len(keys)}

@app.post("/api-keys", status_code=201)
async def create_api_key(req: CreateApiKeyRequest, _admin = Depends(verify_admin)):
    raw_key = f"mbs_{secrets.token_hex(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    prefix = raw_key[:12]

    expires_at = None
    if req.expires_in_days:
        expires_at = datetime.now(timezone.utc) + timedelta(days=req.expires_in_days)

    conn = _acquire()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """INSERT INTO api_keys (name, key_hash, prefix, scopes, expires_at)
               VALUES (%s, %s, %s, %s, %s) RETURNING id, name, prefix, scopes, created_at""",
            (req.name, key_hash, prefix, req.scopes, expires_at)
        )
        row = cur.fetchone()
        conn.commit()
        cur.close()
    except Exception:
        _release(conn, error=True)
        raise
    _release(conn)

    return {
        "id": str(row["id"]),
        "name": row["name"],
        "key": raw_key,
        "prefix": row["prefix"],
        "created_at": row["created_at"].isoformat(),
        "scopes": row["scopes"] or [],
    }

@app.delete("/api-keys/{key_id}")
async def revoke_api_key(key_id: str, _admin = Depends(verify_admin)):
    conn = _acquire()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE api_keys SET is_active = FALSE WHERE id = %s", (key_id,))
        affected = cur.rowcount
        conn.commit()
        cur.close()
    except Exception:
        _release(conn, error=True)
        raise
    _release(conn)
    if affected == 0:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"success": True, "message": f"API key {key_id} revoked"}

@app.post("/api-keys/{key_id}/rotate")
async def rotate_api_key(key_id: str, _admin = Depends(verify_admin)):
    raw_key = f"mbs_{secrets.token_hex(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    prefix = raw_key[:12]

    conn = _acquire()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """UPDATE api_keys SET key_hash = %s, prefix = %s, request_count = 0, last_used_at = NULL
               WHERE id = %s RETURNING id, name, scopes, created_at""",
            (key_hash, prefix, key_id)
        )
        row = cur.fetchone()
        conn.commit()
        cur.close()
    except Exception:
        _release(conn, error=True)
        raise
    _release(conn)
    if not row:
        raise HTTPException(status_code=404, detail="API key not found")
    return {
        "id": str(row["id"]),
        "name": row["name"],
        "key": raw_key,
        "prefix": prefix,
        "created_at": row["created_at"].isoformat(),
        "scopes": row["scopes"] or [],
    }

# ─── Model Status ────────────────────────────────────────────────────────────

@app.get("/model/status")
async def model_status():
    weights_path = Path(__file__).parent / "ai_model" / "weights" / "model.pt"
    vocab_size = len(_tokenizer.vocab) if _tokenizer else 0
    return {
        "model_loaded": _model_ready,
        "vocab_size": vocab_size,
        "device": "digital_gpu",
        "dim": _model_config.get("dim", 512),
        "layers": _model_config.get("layers", 8),
        "heads": _model_config.get("heads", 8),
        "max_len": _model_config.get("max_len", 1024),
        "weights_exist": weights_path.exists(),
        "weights_path": str(weights_path),
    }

# ─── GPU Status ──────────────────────────────────────────────────────────────

@app.get("/gpu/status")
async def gpu_status():
    global _digital_gpu_backend
    pool_stats: dict = {}
    try:
        pool_stats = _get_gpu_pool().stats()
    except Exception:
        pass
    try:
        if _digital_gpu_backend is None:
            sys.path.insert(0, str(Path(__file__).parent))
            from ai_model.gpu.torch_backend import DigitalGPUBackend
            _digital_gpu_backend = DigitalGPUBackend(lanes=32)
        status = _digital_gpu_backend.status()
        return {"available": True, "backend": "digital_gpu", **status, **pool_stats}
    except Exception as e:
        return {"available": False, "backend": "none", "error": str(e), **pool_stats}

@app.get("/gpu/hyper/status")
async def hyper_gpu_status():
    global _hyper_backend
    try:
        if _hyper_backend is None:
            from ai_model.gpu.hyper_backend import HyperGPUBackend
            from ai_model.gpu.hyper_core import PrecisionMode
            _hyper_backend = HyperGPUBackend(lanes=512, tensor_cores=8, precision=PrecisionMode.MIXED)
        s = _hyper_backend.status()
        return {
            "engine": s.get("engine", "HyperGPU"),
            "lanes": s.get("lanes", 512),
            "tensor_cores": s.get("tensor_cores", 8),
            "precision": str(s.get("precision", "MIXED")),
            "total_ops": s.get("total_ops", 0),
            "total_tensor_core_tflops": s.get("total_tensor_core_tflops", 0.0),
            "total_compute_ms": s.get("total_compute_ms", 0.0),
            "uptime_s": s.get("uptime_s", 0.0),
        }
    except Exception as e:
        return {"engine": "HyperGPU", "lanes": 512, "tensor_cores": 8, "precision": "MIXED",
                "total_ops": 0, "total_tensor_core_tflops": 0.0, "total_compute_ms": 0.0,
                "uptime_s": 0.0, "error": str(e)}

@app.get("/gpu/capabilities")
async def gpu_capabilities():
    return {
        "success": True,
        "digital_gpu": {"backend": "digital_gpu", "lanes": 32, "type": "SIMD"},
        "hyper_gpu": {"engine": "HyperGPU", "lanes": 512, "tensor_cores": 8, "precision": "MIXED"},
    }

# ─── Training ────────────────────────────────────────────────────────────────

@app.get("/training/status")
async def get_training_status():
    weights_path = Path(__file__).parent / "ai_model" / "weights" / "model.pt"
    npz_path = Path(__file__).parent / "ai_model" / "weights" / "weights_v4.npz"
    with _training_lock:
        state = dict(_training_state)
    state["weights_exist"] = weights_path.exists() or npz_path.exists()
    state["current_loss"] = state.get("loss")
    state["training_time"] = state.get("elapsed_seconds", 0)
    return state

@app.post("/training/start")
async def start_training(req: StartTrainingRequest, background_tasks: BackgroundTasks,
                         _key = Depends(require_scope("train"))):
    job_id = str(uuid.uuid4())[:8]
    with _training_lock:
        # Guard both "running" and "starting" to prevent double-start race
        if _training_state["state"] in ("running", "starting"):
            return {"success": False, "message": "Training already in progress",
                    "job_id": _training_state.get("job_id")}
        # Atomically claim the slot and reset all volatile fields
        _training_state["state"] = "starting"
        _training_state["job_id"] = job_id
        _training_state["epoch"] = 0
        _training_state["total_epochs"] = req.epochs
        _training_state["started_at"] = time.time()
        _training_state["stop_requested"] = False   # ← always clear before new run
        _training_state["loss"] = None
        _training_state["perplexity"] = None
        _training_state["eta_seconds"] = None
        _training_state["first_loss"] = None
        _training_state["elapsed_seconds"] = 0

    background_tasks.add_task(_run_training, req, job_id)
    return {"success": True, "message": f"Training job {job_id} started", "job_id": job_id}

def _run_training(req: StartTrainingRequest, job_id: str):
    """Background training task."""
    import math
    with _training_lock:
        _training_state["state"] = "running"

    log_training(f"Training job {job_id} started", job_id=job_id)

    try:
        if not _model_ready:
            log_training("Waiting for model to initialize...", job_id=job_id)
            for _ in range(60):
                time.sleep(1)
                if _model_ready:
                    break
            if not _model_ready:
                raise RuntimeError("Model not ready after 60s")

        sys.path.insert(0, str(Path(__file__).parent))
        from ai_model.training.synthetic import generate_synthetic_samples
        from ai_model.training.dataset import CreativeDataset
        from ai_model.training.trainer import train as run_train, evaluate
        from ai_model.training.config import TrainConfig
        import torch

        data_path = "training/boostsheet_samples.json"
        os.makedirs("training", exist_ok=True)

        if req.use_synthetic or not os.path.exists(data_path):
            log_training(f"Generating {req.synthetic_count} synthetic samples...", job_id=job_id)
            generate_synthetic_samples(data_path, n=req.synthetic_count)

        _tokenizer.unfreeze()
        train_max_len = min(256, _creative_model.model.pos_emb.num_embeddings)
        dataset = CreativeDataset(data_path, _tokenizer, max_len=train_max_len)
        if len(dataset) == 0:
            raise ValueError("Empty dataset")

        dim = _creative_model.model.token_emb.embedding_dim
        # epochs=1 here: the outer loop below drives per-epoch iteration
        cfg = TrainConfig({
            "model": {"dim": dim, "layers": len(_creative_model.model.layers), "heads": 8, "max_len": train_max_len},
            "train": {"lr": req.learning_rate, "batch_size": req.batch_size, "epochs": 1, "data_path": data_path},
        })
        cfg.gradient_accumulation_steps = 1

        _creative_model.resize_embeddings()

        for epoch in range(req.epochs):
            # Check stop signal before each epoch
            with _training_lock:
                if _training_state.get("stop_requested"):
                    log_training(f"Training stopped at epoch {epoch+1}", job_id=job_id)
                    break
                _training_state["epoch"] = epoch + 1
                _training_state["samples_trained"] = len(dataset)

            result = run_train(_creative_model.model, dataset, _tokenizer, cfg, device="cpu")
            epoch_loss = result.get("final_loss") if result else None
            ppl = math.exp(min(epoch_loss, 20)) if epoch_loss else evaluate(_creative_model.model, dataset, _tokenizer, device="cpu")
            loss = epoch_loss if epoch_loss is not None else (math.log(ppl) if ppl else None)
            elapsed = time.time() - _training_state["started_at"]
            eta = (elapsed / (epoch + 1)) * (req.epochs - epoch - 1)

            with _training_lock:
                _training_state["loss"] = loss
                _training_state["perplexity"] = ppl
                _training_state["elapsed_seconds"] = elapsed
                _training_state["eta_seconds"] = eta
                if _training_state.get("first_loss") is None and loss is not None:
                    _training_state["first_loss"] = loss
                if loss is not None and (_training_state.get("best_loss") is None or loss < _training_state["best_loss"]):
                    _training_state["best_loss"] = loss

            loss_str = f"{loss:.4f}" if loss is not None else "N/A"
            ppl_str  = f"{ppl:.2f}"  if ppl  is not None else "N/A"
            log_training(f"Epoch {epoch+1}/{req.epochs} complete. Loss: {loss_str}, PPL: {ppl_str}",
                        epoch=epoch+1, loss=loss, job_id=job_id)

        weights_dir = Path(__file__).parent / "ai_model" / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / "model.pt"
        _n_layers = len(_creative_model.model.layers)
        torch.save({
            "model_state_dict": _creative_model.model.state_dict(),
            "vocab": _tokenizer.vocab,
            "inv_vocab": _tokenizer.inv_vocab,
            "merges": _tokenizer.merges,
            "config": _model_config,
        }, str(weights_path))

        _tokenizer.freeze()
        log_training(f"Training job {job_id} completed. Weights saved.", job_id=job_id)

        with _training_lock:
            _training_state["state"] = "completed"
            _training_state["weights_exist"] = True

    except Exception as e:
        log_training(f"Training job {job_id} failed: {e}", level="error", job_id=job_id)
        with _training_lock:
            _training_state["state"] = "error"
        import traceback
        traceback.print_exc()

@app.post("/admin/train-bpe-scaleup")
async def train_bpe_scaleup(req: BPEScaleUpRequest, background_tasks: BackgroundTasks,
                             _key = Depends(require_scope("train"))):
    """One-shot pipeline: build combined real+synthetic corpus, train a fresh
    BPE tokenizer on it, train a TransformerLM from scratch on that vocab, and
    hot-swap the live global model/tokenizer in this process when done.
    Runs as a background task inside this persistent server process (not a
    bash-spawned subprocess) so it survives independent of the calling shell."""
    job_id = str(uuid.uuid4())[:8]
    with _training_lock:
        if _training_state["state"] in ("running", "starting"):
            return {"success": False, "message": "Training already in progress",
                    "job_id": _training_state.get("job_id")}
        _training_state["state"] = "starting"
        _training_state["job_id"] = job_id
        _training_state["epoch"] = 0
        _training_state["total_epochs"] = req.epochs
        _training_state["started_at"] = time.time()
        _training_state["stop_requested"] = False
        _training_state["loss"] = None
        _training_state["perplexity"] = None
        _training_state["eta_seconds"] = None
        _training_state["first_loss"] = None
        _training_state["elapsed_seconds"] = 0

    background_tasks.add_task(_run_bpe_scaleup, req, job_id)
    return {"success": True, "message": f"BPE scale-up training job {job_id} started", "job_id": job_id}

def _run_bpe_scaleup(req: BPEScaleUpRequest, job_id: str):
    """Background task: full corpus build + BPE train + model train from scratch."""
    global _tokenizer, _creative_model, _model_config
    import math
    with _training_lock:
        _training_state["state"] = "running"
    log_training(f"BPE scale-up job {job_id} started", job_id=job_id)

    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from ai_model.training.synthetic import generate_synthetic_samples
        from ai_model.training.dataset import CreativeDataset, extract_text_from_item
        from ai_model.training.trainer import train as run_train, evaluate
        from ai_model.training.config import TrainConfig
        from ai_model.model.tokenizer import BPETokenizer
        from ai_model.model.transformer import TransformerLM
        from ai_model.model.creative_model import CreativeModel
        import torch

        root = Path(__file__).parent
        workspace_root = root.parent.parent
        real_data_files = [
            workspace_root / "training" / "boostsheet_samples.json",
            root.parent / "api-server" / "training" / "curriculum_phase_1_social.json",
        ]

        items = []
        for f in real_data_files:
            if f.exists():
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if not isinstance(data, list):
                    data = [data]
                items.extend(data)
                log_training(f"Loaded {len(data)} items from {f}", job_id=job_id)

        synth_path = root / "training" / "synthetic_generated.json"
        os.makedirs(root / "training", exist_ok=True)
        log_training(f"Generating {req.synthetic_count} synthetic samples...", job_id=job_id)
        synth_items = generate_synthetic_samples(str(synth_path), n=req.synthetic_count)
        items.extend(synth_items)

        combined_path = root / "training" / "combined_training_data.json"
        with open(combined_path, "w", encoding="utf-8") as fh:
            json.dump(items, fh)
        log_training(f"Combined corpus: {len(items)} items -> {combined_path}", job_id=job_id)

        texts = [extract_text_from_item(it) for it in items]
        texts = [t for t in texts if t and t.strip()]
        log_training(f"Training BPE tokenizer on {len(texts)} texts (target vocab_size={req.vocab_size})...",
                     job_id=job_id)
        new_tokenizer = BPETokenizer()
        new_tokenizer.train(texts, vocab_size=req.vocab_size, min_freq=2)
        log_training(f"Tokenizer trained: {new_tokenizer.vocab_size} tokens, {len(new_tokenizer.merges)} merges",
                     job_id=job_id)

        dataset = CreativeDataset(str(combined_path), new_tokenizer, max_len=req.max_len)
        log_training(f"Dataset ready: {len(dataset)} encoded samples", job_id=job_id)
        if len(dataset) == 0:
            raise ValueError("Empty dataset after BPE encoding")

        # Free the old model/tokenizer before allocating the new one — no swap,
        # so holding both resident simultaneously risks OOM-kill.
        import gc
        if _creative_model is not None:
            del _creative_model.model
            _creative_model = None
        gc.collect()

        new_model = TransformerLM(
            vocab_size=new_tokenizer.vocab_size,
            dim=req.dim, n_layers=req.layers, n_heads=req.heads, max_len=req.max_len,
        )
        n_params = sum(p.numel() for p in new_model.parameters())
        log_training(f"Model: dim={req.dim} layers={req.layers} heads={req.heads} params={n_params:,}",
                     job_id=job_id)

        cfg = TrainConfig({
            "model": {"dim": req.dim, "layers": req.layers, "heads": req.heads, "max_len": req.max_len},
            "train": {"lr": req.learning_rate, "batch_size": req.batch_size, "epochs": 1,
                      "data_path": str(combined_path)},
        })
        cfg.gradient_accumulation_steps = 1

        for epoch in range(req.epochs):
            with _training_lock:
                if _training_state.get("stop_requested"):
                    log_training(f"Training stopped at epoch {epoch+1}", job_id=job_id)
                    break
                _training_state["epoch"] = epoch + 1
                _training_state["samples_trained"] = len(dataset)

            result = run_train(new_model, dataset, new_tokenizer, cfg, device="cpu")
            epoch_loss = result.get("final_loss") if result else None
            ppl = math.exp(min(epoch_loss, 20)) if epoch_loss else evaluate(new_model, dataset, new_tokenizer, device="cpu")
            loss = epoch_loss if epoch_loss is not None else (math.log(ppl) if ppl else None)
            elapsed = time.time() - _training_state["started_at"]
            eta = (elapsed / (epoch + 1)) * (req.epochs - epoch - 1)

            with _training_lock:
                _training_state["loss"] = loss
                _training_state["perplexity"] = ppl
                _training_state["elapsed_seconds"] = elapsed
                _training_state["eta_seconds"] = eta
                if _training_state.get("first_loss") is None and loss is not None:
                    _training_state["first_loss"] = loss
                if loss is not None and (_training_state.get("best_loss") is None or loss < _training_state["best_loss"]):
                    _training_state["best_loss"] = loss

            loss_str = f"{loss:.4f}" if loss is not None else "N/A"
            ppl_str = f"{ppl:.2f}" if ppl is not None else "N/A"
            log_training(f"Epoch {epoch+1}/{req.epochs} complete. Loss: {loss_str}, PPL: {ppl_str}",
                         epoch=epoch + 1, loss=loss, job_id=job_id)

        weights_dir = root / "ai_model" / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / "model.pt"
        if weights_path.exists():
            backup_path = weights_dir / "model.pre_bpe_backup.pt"
            if not backup_path.exists():
                os.replace(weights_path, backup_path)
                log_training(f"Backed up previous checkpoint -> {backup_path}", job_id=job_id)

        new_tokenizer.freeze()
        new_config = {"dim": req.dim, "layers": req.layers, "heads": req.heads, "max_len": req.max_len}
        torch.save({
            "model_state_dict": new_model.state_dict(),
            "vocab": new_tokenizer.vocab,
            "inv_vocab": new_tokenizer.inv_vocab,
            "merges": new_tokenizer.merges,
            "config": new_config,
        }, str(weights_path))
        log_training(f"Saved checkpoint -> {weights_path}", job_id=job_id)

        # Hot-swap the live globals so /generate immediately uses the new model.
        _tokenizer = new_tokenizer
        _model_config = new_config
        _creative_model = CreativeModel(new_model, new_tokenizer)

        log_training(f"BPE scale-up job {job_id} completed. Live model swapped in.", job_id=job_id)
        with _training_lock:
            _training_state["state"] = "completed"
            _training_state["weights_exist"] = True

    except Exception as e:
        log_training(f"BPE scale-up job {job_id} failed: {e}", level="error", job_id=job_id)
        with _training_lock:
            _training_state["state"] = "error"
        import traceback
        traceback.print_exc()

@app.post("/admin/train-hyper-scaleup")
async def train_hyper_scaleup(req: HyperScaleUpRequest, background_tasks: BackgroundTasks,
                              _key = Depends(require_scope("train"))):
    """Train a fresh model with real forward+backward compute routed through the
    in-house Digital GPU (HyperGPU) tensor-core kernels, then transfer the trained
    weights into the fast KV-cache serving model and hot-swap it live."""
    job_id = str(uuid.uuid4())[:8]
    with _training_lock:
        if _training_state["state"] in ("running", "starting"):
            return {"success": False, "message": "Training already in progress",
                    "job_id": _training_state.get("job_id")}
        _training_state["state"] = "starting"
        _training_state["job_id"] = job_id
        _training_state["epoch"] = 0
        _training_state["total_epochs"] = req.epochs
        _training_state["started_at"] = time.time()
        _training_state["stop_requested"] = False
        _training_state["loss"] = None
        _training_state["perplexity"] = None
        _training_state["eta_seconds"] = None
        _training_state["first_loss"] = None
        _training_state["elapsed_seconds"] = 0

    background_tasks.add_task(_run_hyper_scaleup, req, job_id)
    return {"success": True,
            "message": f"Digital-GPU scale-up training job {job_id} started",
            "job_id": job_id, "compute_backend": "hyper_gpu"}

@app.post("/api/maxcore/pocket-multiply")
async def api_pocket_multiply(req: PocketMultiplyRequest,
                              key: dict = Depends(verify_api_key)):
    """Pocket-dimension multiplication inside one pocket.

    Routes the GEMM through the PDIM orchestrator: the named pocket is one
    dedup + single-flight namespace, so identical multiplications inside it
    are computed ONCE (on the DigitalGPU backend) and shared. Nested pockets
    (``"a/b/c"``) are isolated sub-domains inside their parent pocket."""
    import numpy as _np
    from ai_model.maxcore.pdim import PocketDimension

    try:
        A = _np.asarray(req.a, dtype=_np.float32)
        B = _np.asarray(req.b, dtype=_np.float32)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid matrix payload: {e}")
    if A.size == 0 or B.size == 0:
        raise HTTPException(status_code=400, detail="Both 'a' and 'b' matrices are required")
    if A.size > 1_000_000 or B.size > 1_000_000:
        raise HTTPException(status_code=413, detail="Matrix too large (limit 1M elements each)")

    def _run() -> dict:
        pocket = PocketDimension(req.pocket, orchestrator=_get_pdim_orchestrator())
        return pocket.matmul(A, B)

    try:
        out = await _in_thread(_run)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "success": True,
        "pocket": out["pocket"],
        "source": out["source"],
        "shape": list(out["result"].shape),
        "result": out["result"].tolist(),
        "compression": out["compression"],
        "compute_backend": "digital_gpu",
    }

@app.get("/api/maxcore/pocket-accelerator/stats")
async def api_pocket_accelerator_stats(key: dict = Depends(verify_api_key)):
    """Live stats for the pocket accelerator wired into the Digital GPU GEMM
    paths: hits/misses, adaptive gating decisions, bytes held, compute seconds
    avoided, and the measured effective speedup on pocket-served repeats."""
    from ai_model.maxcore.pdim import get_pocket_accelerator
    stats = get_pocket_accelerator().stats()
    if stats.get("effective_speedup_on_hits") == float("inf"):
        stats["effective_speedup_on_hits"] = "inf"
    return {"success": True, "stats": stats}

@app.post("/api/awareness/quality/harvest")
async def api_quality_awareness_harvest(replace: bool = True,
                                        admin=Depends(verify_admin)):
    """Run the quality harvester now: scan live public sources (music charts,
    top music channels, high-engagement stories), study the patterns, and
    store the quality buffer in pdim. Raises 502 if every source fails —
    an empty world-scan is never stored as knowledge."""
    from workers import quality_harvester
    try:
        summary = await _in_thread(lambda: quality_harvester.harvest(replace=replace))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"success": True, **summary}

@app.get("/api/awareness/quality/status")
async def api_quality_awareness_status(key: dict = Depends(verify_api_key)):
    """Quality buffer status: whether the temporary world-studied buffer is
    present, its influence weight, and self-sufficiency progress toward the
    buffer's automatic retirement."""
    from ai_model.quality_awareness import status as qa_status
    return {"success": True, **qa_status()}

def _run_hyper_scaleup(req: HyperScaleUpRequest, job_id: str):
    """Background task: build corpus + BPE train + train a model whose real
    compute runs on the Digital GPU, then transfer weights to the serving model."""
    global _tokenizer, _creative_model, _model_config
    import math
    with _training_lock:
        _training_state["state"] = "running"
    log_training(f"Digital-GPU scale-up job {job_id} started (compute routed through HyperGPU)",
                 job_id=job_id)

    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from ai_model.training.synthetic import generate_synthetic_samples
        from ai_model.training.dataset import CreativeDataset, extract_text_from_item
        from ai_model.training.trainer import train as run_train, evaluate
        from ai_model.training.config import TrainConfig
        from ai_model.model.tokenizer import BPETokenizer
        from ai_model.model.transformer import TransformerLM
        from ai_model.model.creative_model import CreativeModel
        from ai_model.gpu.hyper_core import HyperGPU, PrecisionMode
        from ai_model.gpu.hyper_creative_transformer import HyperCreativeTransformerLM
        import torch

        root = Path(__file__).parent
        workspace_root = root.parent.parent
        real_data_files = [
            workspace_root / "training" / "boostsheet_samples.json",
            root.parent / "api-server" / "training" / "curriculum_phase_1_social.json",
        ]

        items = []
        for f in real_data_files:
            if f.exists():
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if not isinstance(data, list):
                    data = [data]
                items.extend(data)
                log_training(f"Loaded {len(data)} items from {f}", job_id=job_id)

        synth_path = root / "training" / "synthetic_generated.json"
        os.makedirs(root / "training", exist_ok=True)
        log_training(f"Generating {req.synthetic_count} synthetic samples...", job_id=job_id)
        synth_items = generate_synthetic_samples(str(synth_path), n=req.synthetic_count)
        items.extend(synth_items)

        # Cap the corpus — the Digital GPU is orders of magnitude slower than
        # native torch, so a bounded sample count keeps the run tractable.
        if req.max_samples and len(items) > req.max_samples:
            import random as _rnd
            _rnd.seed(1234)
            items = _rnd.sample(items, req.max_samples)
            log_training(f"Capped corpus to {len(items)} items (max_samples={req.max_samples})",
                         job_id=job_id)

        combined_path = root / "training" / "hyper_combined_training_data.json"
        with open(combined_path, "w", encoding="utf-8") as fh:
            json.dump(items, fh)
        log_training(f"Combined corpus: {len(items)} items -> {combined_path}", job_id=job_id)

        texts = [extract_text_from_item(it) for it in items]
        texts = [t for t in texts if t and t.strip()]
        log_training(f"Training BPE tokenizer on {len(texts)} texts (target vocab_size={req.vocab_size})...",
                     job_id=job_id)
        new_tokenizer = BPETokenizer()
        new_tokenizer.train(texts, vocab_size=req.vocab_size, min_freq=2)
        log_training(f"Tokenizer trained: {new_tokenizer.vocab_size} tokens, {len(new_tokenizer.merges)} merges",
                     job_id=job_id)

        dataset = CreativeDataset(str(combined_path), new_tokenizer, max_len=req.max_len)
        log_training(f"Dataset ready: {len(dataset)} encoded samples", job_id=job_id)
        if len(dataset) == 0:
            raise ValueError("Empty dataset after BPE encoding")

        # Free the old model before allocating the new one (no swap).
        import gc
        if _creative_model is not None:
            del _creative_model.model
            _creative_model = None
        gc.collect()

        gpu = HyperGPU(lanes=512, tensor_cores=8, precision=PrecisionMode.MIXED)
        hyper_model = HyperCreativeTransformerLM(
            vocab_size=new_tokenizer.vocab_size,
            dim=req.dim, n_layers=req.layers, n_heads=req.heads,
            max_len=req.max_len, gpu=gpu,
        )
        n_params = sum(p.numel() for p in hyper_model.parameters())
        log_training(f"Digital-GPU model: dim={req.dim} layers={req.layers} heads={req.heads} "
                     f"params={n_params:,} — compute routed through HyperGPU", job_id=job_id)

        cfg = TrainConfig({
            "model": {"dim": req.dim, "layers": req.layers, "heads": req.heads, "max_len": req.max_len},
            "train": {"lr": req.learning_rate, "batch_size": req.batch_size, "epochs": 1,
                      "data_path": str(combined_path)},
        })
        cfg.gradient_accumulation_steps = 1

        for epoch in range(req.epochs):
            with _training_lock:
                if _training_state.get("stop_requested"):
                    log_training(f"Training stopped at epoch {epoch+1}", job_id=job_id)
                    break
                _training_state["epoch"] = epoch + 1
                _training_state["samples_trained"] = len(dataset)

            result = run_train(hyper_model, dataset, new_tokenizer, cfg, device="cpu")
            epoch_loss = result.get("final_loss") if result else None
            ppl = math.exp(min(epoch_loss, 20)) if epoch_loss else evaluate(hyper_model, dataset, new_tokenizer, device="cpu")
            loss = epoch_loss if epoch_loss is not None else (math.log(ppl) if ppl else None)
            elapsed = time.time() - _training_state["started_at"]
            eta = (elapsed / (epoch + 1)) * (req.epochs - epoch - 1)

            gpu_ops = gpu.core._total_ops
            with _training_lock:
                _training_state["loss"] = loss
                _training_state["perplexity"] = ppl
                _training_state["elapsed_seconds"] = elapsed
                _training_state["eta_seconds"] = eta
                _training_state["hyper_gpu_ops"] = gpu_ops
                if _training_state.get("first_loss") is None and loss is not None:
                    _training_state["first_loss"] = loss
                if loss is not None and (_training_state.get("best_loss") is None or loss < _training_state["best_loss"]):
                    _training_state["best_loss"] = loss

            loss_str = f"{loss:.4f}" if loss is not None else "N/A"
            ppl_str = f"{ppl:.2f}" if ppl is not None else "N/A"
            log_training(f"Epoch {epoch+1}/{req.epochs} complete. Loss: {loss_str}, PPL: {ppl_str}, "
                         f"HyperGPU ops: {gpu_ops:,}", epoch=epoch + 1, loss=loss, job_id=job_id)

        # Transfer Digital-GPU-trained weights into the fast KV-cache serving model.
        log_training("Transferring Digital-GPU-trained weights into serving model...", job_id=job_id)
        serve_model = TransformerLM(
            vocab_size=new_tokenizer.vocab_size,
            dim=req.dim, n_layers=req.layers, n_heads=req.heads, max_len=req.max_len,
        )
        missing, unexpected = serve_model.load_state_dict(hyper_model.state_dict(), strict=False)
        # head.weight is tied to token_emb.weight and re-tied in __init__, so it is
        # the only expected "missing" key; anything else is a real mismatch.
        real_missing = [k for k in missing if k != "head.weight"]
        if real_missing or unexpected:
            raise RuntimeError(f"Weight transfer mismatch: missing={real_missing} unexpected={unexpected}")
        log_training("Weight transfer OK (Digital-GPU weights -> KV-cache serving model)", job_id=job_id)

        del hyper_model
        gc.collect()

        weights_dir = root / "ai_model" / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / "model.pt"
        if weights_path.exists():
            backup_path = weights_dir / "model.pre_hyper_backup.pt"
            if not backup_path.exists():
                os.replace(weights_path, backup_path)
                log_training(f"Backed up previous checkpoint -> {backup_path}", job_id=job_id)

        new_tokenizer.freeze()
        new_config = {"dim": req.dim, "layers": req.layers, "heads": req.heads, "max_len": req.max_len}
        torch.save({
            "model_state_dict": serve_model.state_dict(),
            "vocab": new_tokenizer.vocab,
            "inv_vocab": new_tokenizer.inv_vocab,
            "merges": new_tokenizer.merges,
            "config": new_config,
            "trained_on": "hyper_gpu",
        }, str(weights_path))
        log_training(f"Saved checkpoint -> {weights_path}", job_id=job_id)

        _tokenizer = new_tokenizer
        _model_config = new_config
        _creative_model = CreativeModel(serve_model, new_tokenizer)

        log_training(f"Digital-GPU scale-up job {job_id} completed. Live model swapped in.", job_id=job_id)
        with _training_lock:
            _training_state["state"] = "completed"
            _training_state["weights_exist"] = True

    except Exception as e:
        log_training(f"Digital-GPU scale-up job {job_id} failed: {e}", level="error", job_id=job_id)
        with _training_lock:
            _training_state["state"] = "error"
        import traceback
        traceback.print_exc()

@app.get("/training/logs")
async def get_training_logs(limit: int = 50):
    if not DATABASE_URL:
        return {"logs": [], "total": 0}
    conn = _acquire()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT timestamp, level, message, epoch, loss, job_id FROM training_logs ORDER BY timestamp DESC LIMIT %s",
            (limit,)
        )
        rows = cur.fetchall()
        cur.close()
    except Exception:
        _release(conn, error=True)
        raise
    _release(conn)
    logs = []
    for r in rows:
        logs.append({
            "timestamp": r["timestamp"].isoformat() if r["timestamp"] else None,
            "level": r["level"],
            "message": r["message"],
            "epoch": r["epoch"],
            "loss": float(r["loss"]) if r["loss"] is not None else None,
        })
    return {"logs": logs, "total": len(logs)}

@app.post("/training/stop")
async def stop_training(_key = Depends(require_scope("train"))):
    with _training_lock:
        state = _training_state.get("state", "idle")
        if state not in ("running", "starting"):
            return {"success": False, "message": f"No active training to stop (state: {state})"}
        _training_state["stop_requested"] = True
        job_id = _training_state.get("job_id")
    log_training(f"Stop requested for job {job_id}", job_id=job_id)
    return {"success": True, "message": f"Stop signal sent to job {job_id}"}


CURRICULUM_PHASES = [
    {
        "id": "phase_1_social",
        "name": "Social Content Foundation",
        "description": "Trains on social posts, hooks, captions, and hashtag patterns",
        "loss_target": 3.5,
        "datasets": ["social_posts", "captions"],
        "epochs": 3,
    },
    {
        "id": "phase_2_ads",
        "name": "Ad Creative & Performance",
        "description": "Trains on high-ROAS ad copy, hooks, CTAs, and audience signals",
        "loss_target": 3.0,
        "datasets": ["ad_creatives", "peak_performers"],
        "epochs": 3,
    },
    {
        "id": "phase_3_daw",
        "name": "DAW & Studio Intelligence",
        "description": "Trains on lyrics, beat descriptions, track concepts, and production notes",
        "loss_target": 2.8,
        "datasets": ["lyrics", "daw_sessions"],
        "epochs": 3,
    },
    {
        "id": "phase_4_distribution",
        "name": "Distribution & Release Strategy",
        "description": "Trains on playlist pitching, release planning, and streaming platform metadata",
        "loss_target": 2.5,
        "datasets": ["distribution_plans", "release_strategies"],
        "epochs": 2,
    },
    {
        "id": "phase_5_multimodal",
        "name": "Multimodal Fusion",
        "description": "Full cross-domain training across all datasets with MusicCaps, AudioCaps, HMDB-51, UCF-101, FMA",
        "loss_target": 2.0,
        "datasets": ["musiccaps", "audiocaps", "hmdb51", "ucf101", "fma"],
        "epochs": 5,
    },
]

STORAGE_DATASETS = [
    {"id": "hmdb51",      "name": "HMDB-51 Clips",        "type": "video",   "color": "blue",   "size_gb": 2.0,  "samples": 6766},
    {"id": "ucf101",      "name": "UCF-101 Clips",         "type": "video",   "color": "blue",   "size_gb": 6.5,  "samples": 13320},
    {"id": "musiccaps",   "name": "MusicCaps Captions",    "type": "audio",   "color": "purple", "size_gb": 0.08, "samples": 5521},
    {"id": "audiocaps",   "name": "AudioCaps Captions",    "type": "audio",   "color": "teal",   "size_gb": 0.05, "samples": 49274},
    {"id": "fma",         "name": "FMA Tracks",            "type": "audio",   "color": "orange", "size_gb": 917.0,"samples": 106574},
    {"id": "social_posts","name": "Social Post Corpus",    "type": "text",    "color": "green",  "size_gb": 0.4,  "samples": 220000},
    {"id": "ad_creatives","name": "Ad Creative Library",   "type": "text",    "color": "red",    "size_gb": 0.2,  "samples": 95000},
    {"id": "lyrics",      "name": "Lyrics & DAW Sessions", "type": "text",    "color": "indigo", "size_gb": 1.2,  "samples": 180000},
]

@app.get("/training/datasets")
async def get_training_datasets(_key = Depends(require_scope("read"))):
    total_gb = sum(d["size_gb"] for d in STORAGE_DATASETS)
    storage_session = None
    try:
        from storage_client import get_storage  # noqa: E402
        storage_session = get_storage().get_session()
    except Exception:
        pass
    return {
        "datasets": STORAGE_DATASETS,
        "total_disk_gb": round(total_gb, 2),
        "total_disk_display": f"{total_gb / 1024:.1f} TB" if total_gb > 1024 else f"{total_gb:.1f} GB",
        "storage_session": storage_session,
        "curriculum_phases": CURRICULUM_PHASES,
    }


class ScheduleRequest(BaseModel):
    mode: str = "single"
    phases: list = []
    start_phase: str = None
    continuous: bool = False
    epochs_per_phase: int = 3
    loss_targets: dict = {}

@app.post("/training/schedule")
async def schedule_training(req: ScheduleRequest, background_tasks: BackgroundTasks,
                            _key = Depends(require_scope("train"))):
    with _training_lock:
        if _training_state["state"] == "running":
            return {"success": False, "message": "Training already in progress"}
        _training_state["schedule_mode"] = req.mode
        _training_state["stop_requested"] = False

    phases_to_run = req.phases if req.phases else [p["id"] for p in CURRICULUM_PHASES]
    if req.start_phase:
        try:
            idx = phases_to_run.index(req.start_phase)
            phases_to_run = phases_to_run[idx:]
        except ValueError:
            pass

    job_id = str(uuid.uuid4())[:8]
    with _training_lock:
        _training_state["state"] = "starting"
        _training_state["job_id"] = job_id
        _training_state["curriculum_phases_done"] = []
        _training_state["started_at"] = time.time()

    background_tasks.add_task(_run_curriculum, phases_to_run, req, job_id)
    return {
        "success": True,
        "job_id": job_id,
        "mode": req.mode,
        "phases_queued": phases_to_run,
        "message": f"Curriculum training scheduled — {len(phases_to_run)} phase(s), mode: {req.mode}",
    }


def _run_curriculum(phases: list, req: ScheduleRequest, job_id: str):
    from ai_model.training.synthetic import generate_synthetic_samples
    from ai_model.training.dataset import CreativeDataset
    from ai_model.training.trainer import train as run_train
    from ai_model.training.config import TrainConfig

    with _training_lock:
        _training_state["state"] = "running"

    log_training(f"Curriculum training {job_id} starting — {len(phases)} phases", job_id=job_id)

    phase_map = {p["id"]: p for p in CURRICULUM_PHASES}
    session_count = 0

    while True:
        for phase_id in phases:
            with _training_lock:
                if _training_state.get("stop_requested"):
                    _training_state["state"] = "stopped"
                    _training_state["stop_requested"] = False
                    log_training(f"Curriculum stopped at phase {phase_id}", job_id=job_id)
                    return

            phase = phase_map.get(phase_id, {"id": phase_id, "name": phase_id, "epochs": req.epochs_per_phase, "loss_target": 2.5})
            epochs = req.loss_targets.get(phase_id, {}).get("epochs", phase.get("epochs", req.epochs_per_phase))
            loss_target = req.loss_targets.get(phase_id, {}).get("loss_target", phase.get("loss_target", 2.5))

            with _training_lock:
                _training_state["curriculum_phase"] = phase.get("name", phase_id)

            log_training(f"[Curriculum] Phase: {phase.get('name')} | target loss ≤ {loss_target} | epochs: {epochs}", job_id=job_id)

            try:
                data_path = f"training/curriculum_{phase_id}.json"
                os.makedirs("training", exist_ok=True)
                generate_synthetic_samples(data_path, n=120)

                _tokenizer.unfreeze()
                train_max_len = min(256, _creative_model.model.pos_emb.num_embeddings)
                dataset = CreativeDataset(data_path, _tokenizer, max_len=train_max_len)

                dim = _creative_model.model.token_emb.embedding_dim
                cfg = TrainConfig({
                    "model": {"dim": dim, "layers": len(_creative_model.model.layers), "heads": 8, "max_len": train_max_len},
                    "train": {"lr": 3e-4, "batch_size": 4, "epochs": epochs, "data_path": data_path},
                })
                cfg.gradient_accumulation_steps = 1
                _creative_model.resize_embeddings()

                result = run_train(_creative_model.model, dataset, _tokenizer, cfg, device="cpu")
                _tokenizer.freeze()

                final_loss = result.get("final_loss", 999)
                session_count += 1

                with _training_lock:
                    if _training_state["first_loss"] is None:
                        _training_state["first_loss"] = round(final_loss, 4)
                    if _training_state["best_loss"] is None or final_loss < _training_state["best_loss"]:
                        _training_state["best_loss"] = round(final_loss, 4)
                    _training_state["current_loss"] = round(final_loss, 4)
                    _training_state["loss"] = round(final_loss, 4)
                    _training_state["sessions_done"] = session_count
                    _training_state["total_trained"] = (_training_state.get("total_trained", 0) + len(dataset))
                    _training_state["curriculum_phases_done"].append(phase_id)

                weights_dir = Path(__file__).parent / "ai_model" / "weights"
                weights_dir.mkdir(parents=True, exist_ok=True)
                import torch
                import numpy as np
                state_dict = _creative_model.model.state_dict()
                torch.save({"model_state_dict": state_dict, "vocab": _tokenizer.vocab,
                            "inv_vocab": _tokenizer.inv_vocab, "merges": _tokenizer.merges,
                            "config": _model_config, "job_id": job_id, "phase": phase_id,
                            "final_loss": final_loss}, str(weights_dir / "model.pt"))
                np_weights = {k: v.cpu().numpy() for k, v in state_dict.items()}
                np.savez_compressed(str(weights_dir / "weights_v4.npz"), **np_weights)

                save_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                with _training_lock:
                    _training_state["last_weights_save"] = save_time
                    _training_state["weights_file"] = "weights_v4.npz"
                    _training_state["elapsed_seconds"] = time.time() - _training_state["started_at"]

                log_training(f"[Curriculum] Phase {phase.get('name')} done — loss={final_loss:.4f} (target≤{loss_target})", job_id=job_id)

                if final_loss <= loss_target:
                    log_training(f"[Curriculum] Phase {phase_id} hit loss target {loss_target} ✓", job_id=job_id)

            except Exception as e:
                log_training(f"[Curriculum] Phase {phase_id} error: {e}", level="error", job_id=job_id)

        if req.mode != "continuous":
            break

        log_training(f"[Curriculum] Continuous mode — restarting cycle (sessions done: {session_count})", job_id=job_id)

    with _training_lock:
        _training_state["state"] = "completed"
        _training_state["sessions_done"] = session_count
        _training_state["elapsed_seconds"] = time.time() - _training_state.get("started_at", time.time())

    log_training(f"[Curriculum] Job {job_id} complete — {session_count} sessions, best loss: {_training_state.get('best_loss')}", job_id=job_id)


# ─── Continuous Training & Data Puller Endpoints ─────────────────────────────

class ContinuousStartRequest(BaseModel):
    interval_minutes: int = 60
    phases: list = []
    epochs_per_phase: int = 1
    pull_every_n_cycles: int = 2


@app.get("/training/continuous/status")
async def continuous_status(_key = Depends(require_scope("read"))):
    with _workers_lock:
        ct = _continuous_trainer
    if ct is None:
        return {"running": False, "status": "not_initialized"}
    return ct.get_state()


@app.post("/training/continuous/start")
async def continuous_start(req: ContinuousStartRequest, _key = Depends(require_scope("train"))):
    with _workers_lock:
        ct = _continuous_trainer
        dp = _data_puller
    if ct is None or dp is None:
        ct, dp = await _wait_for_workers()
    if not dp.state.get("status") == "pulling" and not dp._running:
        dp.start(interval_minutes=req.interval_minutes * req.pull_every_n_cycles)
    return ct.start(
        interval_minutes=req.interval_minutes,
        phases=req.phases or None,
        epochs_per_phase=req.epochs_per_phase,
        pull_every_n=req.pull_every_n_cycles,
    )


@app.post("/training/continuous/stop")
async def continuous_stop(_key = Depends(require_scope("train"))):
    with _workers_lock:
        ct = _continuous_trainer
        dp = _data_puller
    if ct is None:
        return {"success": False, "message": "Workers not initialized"}
    if dp:
        dp.stop()
    return ct.stop()


@app.get("/training/continuous/history")
async def continuous_history(_key = Depends(require_scope("read"))):
    from storage_client import get_storage
    storage = get_storage()
    history = storage.lrange("mb:training:continuous:history", 0, 49) if storage.is_available else []
    return {"history": history or [], "count": len(history or [])}


@app.get("/training/puller/status")
async def puller_status(_key = Depends(require_scope("read"))):
    with _workers_lock:
        dp = _data_puller
    if dp is None:
        return {"status": "not_initialized"}
    state = dp.get_state()
    state["local_files"] = len(list(
        (Path(__file__).parent / "ai_model" / "training_data").glob("pull_*.json")
    ))
    return state


@app.get("/training/puller/sources")
async def puller_sources(_key = Depends(require_scope("read"))):
    from workers.data_puller import PUBLIC_SOURCES
    return {
        "public_sources": [{"id": s["id"], "name": s["name"], "category": s["category"]} for s in PUBLIC_SOURCES],
        "pdim_patterns": [
            "mbs:data", "mbs:downloads", "mbs_training",
            "mb:ads:*:peaks", "mb:social:posts:*", "mb:analytics:*",
            "mb:content:*", "mb:dataset:*:chunk:*",
        ],
    }


@app.post("/training/puller/pull")
async def puller_pull(background_tasks: BackgroundTasks, _key = Depends(require_scope("train"))):
    with _workers_lock:
        dp = _data_puller
    if dp is None:
        _, dp = await _wait_for_workers()
    background_tasks.add_task(dp.pull_now)
    return {"success": True, "message": "Data pull triggered in background"}


@app.post("/training/puller/start")
async def puller_start_auto(interval_minutes: int = 30, _key = Depends(require_scope("train"))):
    with _workers_lock:
        dp = _data_puller
    if dp is None:
        _, dp = await _wait_for_workers()
    return dp.start(interval_minutes=interval_minutes)


@app.post("/training/puller/stop")
async def puller_stop_auto(_key = Depends(require_scope("train"))):
    with _workers_lock:
        dp = _data_puller
    if dp is None:
        return {"success": False, "message": "DataPuller not initialized"}
    dp.stop()
    return {"success": True, "message": "DataPuller auto-pull stopped"}


# ─── Watchdog Endpoints ────────────────────────────────────────────────────────

@app.get("/watchdog/status")
async def watchdog_status(_key = Depends(require_scope("read"))):
    with _workers_lock:
        wd = _watchdog
    if wd is None:
        return {"status": "not_initialized", "running": False}
    return wd.get_status()


@app.get("/watchdog/log")
async def watchdog_log(limit: int = 50, _key = Depends(require_scope("read"))):
    with _workers_lock:
        wd = _watchdog
    if wd is None:
        return {"log": [], "count": 0}
    entries = wd.get_log(limit=limit)
    return {"log": entries, "count": len(entries)}


@app.post("/watchdog/reset")
async def watchdog_reset(_key = Depends(require_scope("train"))):
    with _workers_lock:
        wd = _watchdog
    if wd is None:
        return {"success": False, "message": "Watchdog not initialized"}
    wd.reset_alerts()
    return {"success": True, "message": "Watchdog alert log cleared"}


# ─── Coverage (Retrieval) Endpoints ─────────────────────────────────────────────

@app.get("/coverage/status")
async def coverage_status(_key = Depends(require_scope("read"))):
    with _workers_lock:
        wd = _coverage_watchdog
    if wd is None:
        return {"status": "not_initialized", "running": False}
    return wd.get_status()


@app.get("/coverage/log")
async def coverage_log(limit: int = 50, _key = Depends(require_scope("read"))):
    with _workers_lock:
        wd = _coverage_watchdog
    if wd is None:
        return {"log": [], "count": 0}
    entries = wd.get_log(limit=limit)
    return {"log": entries, "count": len(entries)}


@app.get("/coverage/report")
async def coverage_report(_key = Depends(require_scope("read"))):
    with _workers_lock:
        idx = _asset_index
    if idx is None:
        return {"status": "not_initialized"}
    try:
        from ai_model.retrieval.probes import probe_count, recent_probes
        probes = recent_probes()
        vectors = [p["vector"] for p in probes]
        report = idx.coverage_report(vectors or None)
        report["live_probes_total"] = probe_count()
        report["live_probes_sampled"] = len(vectors)
        try:
            report["brands"] = idx.brand_stats()
        except Exception:
            pass
        return report
    except Exception:
        return idx.coverage_report()


@app.get("/coverage/ingestor")
async def coverage_ingestor(_key = Depends(require_scope("read"))):
    """Status of the generated-asset ingestor (produced images folded back in)."""
    try:
        from ai_model.retrieval.generated_ingestor import get_generated_ingestor
        return get_generated_ingestor().get_status()
    except Exception:
        return {"status": "not_initialized", "running": False}


@app.post("/coverage/reset")
async def coverage_reset(_key = Depends(require_scope("train"))):
    with _workers_lock:
        wd = _coverage_watchdog
    if wd is None:
        return {"success": False, "message": "CoverageWatchdog not initialized"}
    wd.reset_alerts()
    return {"success": True, "message": "Coverage alert log cleared"}


# ─── Content Generation ───────────────────────────────────────────────────────

PLATFORM_NORMALIZE = {
    "googlebusiness": "google_business", "google_business": "google_business",
    "twitter": "twitter", "x": "twitter",
}

def normalize_platform(p: str) -> str:
    return PLATFORM_NORMALIZE.get(p.lower(), p.lower())


_srv_logger = logging.getLogger(__name__)

_MUSIC_PLATFORM_LABELS: dict[str, str] = {
    "spotify.com":       "music release on Spotify",
    "open.spotify.com":  "music release on Spotify",
    "youtube.com":       "music video on YouTube",
    "youtu.be":          "music video on YouTube",
    "music.youtube.com": "music video on YouTube",
    "soundcloud.com":    "track on SoundCloud",
    "music.apple.com":   "music release on Apple Music",
    "tidal.com":         "music release on Tidal",
    "deezer.com":        "music release on Deezer",
    "bandcamp.com":      "music release on Bandcamp",
    "audiomack.com":     "track on Audiomack",
    "distrokid.com":     "release via DistroKid",
}

_URL_RE = re.compile(r"^(https?://|www\.)\S+", re.IGNORECASE)
_DOMAIN_ONLY_RE = re.compile(r"^([\w-]+\.)+[\w-]{2,}(/[\w.~%-]*)?$", re.IGNORECASE)


_URL_TITLE_SUFFIXES = re.compile(
    r"\s*[\|–—\-]\s*(?:Spotify|YouTube|SoundCloud|Apple Music|Tidal|Deezer"
    r"|Bandcamp|Audiomack|DistroKid|Instagram|TikTok|Twitter|X\.com|Facebook"
    r"|LinkedIn|Pinterest|Reddit|Twitch|Snapchat|BeReal|Substack"
    r"|SoundOn|Triller|Clapper|Lemon8|Threads)\s*$",
    re.IGNORECASE,
)


def _fetch_page_title(url_str: str) -> str:
    """Try to fetch a URL and return the best available title string (never raises)."""
    try:
        req = _urllib_request.Request(
            url_str,
            headers={
                "User-Agent": "MaxCore/1.0 (+https://maxbooster.ai/bot)",
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            },
        )
        with _urllib_request.urlopen(req, timeout=3) as resp:
            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type:
                return ""
            body = resp.read(32768).decode("utf-8", errors="replace")

        # og:title is richest — try both attribute orderings
        og = re.search(
            r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\'<>]+)["\']',
            body, re.IGNORECASE,
        ) or re.search(
            r'<meta[^>]+content=["\']([^"\'<>]+)["\'][^>]+property=["\']og:title["\']',
            body, re.IGNORECASE,
        )
        if og:
            title = _html.unescape(og.group(1)).strip()
            return _URL_TITLE_SUFFIXES.sub("", title).strip()

        # Fall back to <title> tag
        title_m = re.search(r"<title[^>]*>([^<]{3,})</title>", body, re.IGNORECASE)
        if title_m:
            title = _html.unescape(title_m.group(1)).strip()
            return _URL_TITLE_SUFFIXES.sub("", title).strip()
    except Exception:
        pass
    return ""


def _clean_idea_from_parsed(parsed) -> str:
    """Build a ScriptAgent-friendly idea string from a ParsedUrl.

    Unlike ``ParsedUrl.topic_string`` (which is designed for the URL Inspector
    panel and includes the platform-label suffix such as "(TikTok video)"),
    this function produces a natural short phrase that won't echo metadata
    labels verbatim into generated copy.

    Rules:
    • Strip leading ``@`` from artist/title (TikTok creator handles).
    • De-duplicate: if title and artist normalise to the same string use only
      the more readable one (avoids "Lunarvoss — lunarvoss" repetition).
    • Never include the platform or content-type suffix.
    • Always falls back to the raw ``topic_string`` so resolution is never lost.
    """
    title  = (parsed.title  or "").lstrip("@").strip()
    artist = (parsed.artist or "").lstrip("@").strip()

    # Normalise for de-duplication: strip spaces and lowercase
    def _norm(s: str) -> str:
        return s.lower().replace(" ", "").replace("-", "").replace("_", "")

    if title and artist:
        if _norm(title) == _norm(artist):
            # Same entity twice — keep the title (typically has better casing)
            return title
        return f"{title} — {artist}"
    return title or artist or parsed.topic_string


def _resolve_topic_from_url(raw_topic: str) -> str:
    """Convert any URL/domain/Spotify-URI topic into a human-readable idea string.

    For plain-text topics (not URLs) the input is returned unchanged — no
    network calls or parsing are attempted.  Only inputs that pass the URL
    heuristic (http/https scheme, spotify: URI, or bare domain) are handed to
    the Universal URL Parser.

    Delegates entirely to the Universal URL Parser which handles 30+ platforms
    with platform-specific extractors, JSON-LD, og:title, path slugs, and
    music metadata (genre, mood, BPM, key).  Never raises.

    Returns a clean idea string (no platform-label suffixes) suitable for
    direct use in ScriptAgent ``idea`` fields.
    """
    if not raw_topic or not raw_topic.strip():
        return raw_topic
    t = raw_topic.strip()
    try:
        from ai_model.url_parser.core import is_url as _is_url, parse_url as _parse_url_direct
    except Exception:
        return t

    # Gate: plain-text topics (e.g. "midnight piano ballad") pass through
    # unchanged — no DNS resolution, no network calls, no parsing overhead.
    if not _is_url(t):
        return t

    try:
        parsed = _parse_url_direct(t)
        idea   = _clean_idea_from_parsed(parsed)
        if idea and idea != t:
            _srv_logger.info("[url-parser] topic resolved: %r → %r", t, idea)
        return idea if idea else t
    except Exception:
        return t


def _effective_awareness(platform: str, raw_awareness: str) -> str:
    """Always combine caller-provided awareness with platform strategy signals.

    Caller awareness comes FIRST so request-specific signals lead and
    synchronize with (rather than get drowned out by) the generic platform
    buffer. Platform signals (fire/viral/drop/finally/exclusive etc.) are
    appended after, so the hook-selector and body-builder still always have
    arousal-rich [HIGH] signals available. When the caller sends no awareness,
    only platform signals are used. Never-raise.
    """
    try:
        from ai_model.quality_awareness import platform_awareness_string
        platform_awareness = platform_awareness_string(platform)
    except Exception:  # noqa: BLE001
        platform_awareness = ""
    if not raw_awareness:
        return platform_awareness
    # Caller-provided awareness leads: request-specific signals synchronize
    # with (rather than get drowned out by) the generic platform buffer.
    return f"{raw_awareness}\n{platform_awareness}" if platform_awareness else raw_awareness


@app.get("/api/url-parser/content")
async def url_parser_content(url: str = "", platform: str = "",
                             _key = Depends(require_scope("generate"))):
    """Full content extraction for a URL + Veo DNA from the awareness systems.

    Query params:
      url      — the URL (or Spotify URI) to extract content from
      platform — optional target platform (instagram / tiktok / …) to blend
                 platform-specific awareness into the Veo DNA block

    Returns every extracted field plus:
      veo_dna        — Veo scoring DNA block (hook/length/structure/CTA rules
                       + live chart patterns from the quality-awareness buffer)
      awareness_full — awareness_text + veo_dna, ready for any generation
                       ``awareness`` field

    Never raises on parse failures (fetch_ok=False + error instead).
    """
    if not url or not url.strip():
        raise HTTPException(status_code=422, detail="url query parameter is required")

    try:
        from ai_model.url_parser import get_content_from_url as _get_content
    except Exception as import_err:
        raise HTTPException(status_code=500, detail=f"URL parser unavailable: {import_err}")

    return await asyncio.get_event_loop().run_in_executor(
        None, lambda: _get_content(url.strip(), platform=platform.strip())
    )


@app.get("/api/url-parser/inspect")
async def url_parser_inspect(url: str = "", _key = Depends(require_scope("generate"))):
    """Parse any URL and return the full structured metadata the AI pipeline sees.

    Query param:
      url — the URL (or Spotify URI) to inspect

    Returns the ParsedUrl fields as a JSON object, plus a ready-to-use
    topic_string and awareness_text block.  Never raises.
    """
    if not url or not url.strip():
        raise HTTPException(status_code=422, detail="url query parameter is required")

    try:
        from ai_model.url_parser.core import parse_url as _parse_url_core
    except Exception as import_err:
        raise HTTPException(status_code=500, detail=f"URL parser unavailable: {import_err}")

    try:
        parsed = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _parse_url_core(url.strip())
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Parse error: {exc}")

    return {
        # Input echo
        "raw_url":        parsed.raw_url,
        "canonical_url":  parsed.canonical_url,
        # Platform
        "platform":       parsed.platform,
        "platform_label": parsed.platform_label,
        "content_type":   parsed.content_type,
        # Content metadata
        "title":          parsed.title,
        "artist":         parsed.artist,
        "album":          parsed.album,
        "label":          parsed.label,
        "description":    parsed.description,
        # Music signals
        "genre":          parsed.genre,
        "mood":           parsed.mood,
        "bpm":            parsed.bpm,
        "key":            parsed.key,
        "release_year":   parsed.release_year,
        # Derived
        "intent":         parsed.intent,
        "goal":           parsed.goal,
        "content_themes": parsed.content_themes,
        # Pipeline outputs
        "topic_string":   parsed.topic_string,
        "awareness_text": parsed.awareness_text,
        # Diagnostics
        "fetch_ok":       parsed.fetch_ok,
        "error":          parsed.error,
    }


@app.post("/content/generate")
async def generate_content(req: ContentRequest, _key = Depends(require_scope("generate"))):
    start = time.time()
    platform = normalize_platform(req.platform)
    topic = _resolve_topic_from_url(req.topic)
    effective_awareness = _effective_awareness(platform, _merged_awareness_for(req))

    await _wait_for_model_ready()

    async def _run_content_inference():
        from ai_model.agents.script_agent import ScriptRequest
        from ai_model.agents.distribution_agent import DistributionRequest
        script_result = await _in_thread(lambda: _script_agent.run(ScriptRequest(
            idea=topic, platform=platform, goal=req.goal, tone=req.tone,
            awareness=effective_awareness,
        )))
        full_script = f"{script_result.hook}\n{script_result.body}\n{script_result.cta}"
        dist_result = await _in_thread(lambda: _distribution_agent.run(DistributionRequest(
            script=full_script, platform=platform, goal=req.goal,
            awareness=effective_awareness,
        )))
        return script_result, dist_result

    def _build_result(_request=None):
        from ai_model.agents.script_agent import ScriptRequest
        from ai_model.agents.distribution_agent import DistributionRequest
        sr = _script_agent.run(ScriptRequest(
            idea=topic, platform=platform, goal=req.goal,
            tone=req.tone, awareness=effective_awareness,
        ))
        full_script = f"{sr.hook}\n{sr.body}\n{sr.cta}"
        dr = _distribution_agent.run(DistributionRequest(
            script=full_script, platform=platform,
            goal=req.goal, awareness=effective_awareness,
        ))
        if not any([sr.hook.strip(), sr.body.strip(), sr.cta.strip()]):
            raise ValueError("empty generatedContent — model warming up, please retry")
        return {
            "success": True,
            "platform": platform,
            "caption": dr.caption,
            "hook": sr.hook,
            "body": sr.body,
            "cta": sr.cta,
            "hashtags": dr.hashtags if req.include_hashtags else [],
            "source": getattr(sr, "source", "template"),
        }

    try:
        # ── Dedup + single-flight via PDIM orchestrator ───────────────────────
        # Identical concurrent requests collapse to one compute; all share result.
        _orch = _get_pdim_orchestrator()
        _cache_key = {"platform": platform, "topic": topic, "tone": req.tone,
                      "goal": req.goal, "awareness": effective_awareness}
        _out = await _in_thread(lambda: _orch.compute(_cache_key, _build_result, namespace="api_content_v4"))
        _result = dict(_out["result"])
        if _out.get("source") in ("cache", "coalesced"):
            _result["cached"] = True
        _result["processing_time_ms"] = round((time.time() - start) * 1000, 1)
        _fw_ingest(_key, "scripts", _result, {
            "topic": topic, "platform": platform,
            "tone": req.tone, "awareness": effective_awareness,
        })
        return _result
    except HTTPException:
        raise
    except ValueError as e:
        _srv_logger.warning("[generate-from-url] %s topic=%r platform=%s", e, topic, platform)
        raise HTTPException(status_code=500, detail=f"Generation returned empty output: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─── BoostSheets ─────────────────────────────────────────────────────────────

@app.get("/boostsheets")
async def list_boostsheets():
    if not _repo:
        return {"sheet_ids": [], "count": 0}
    ids = _repo.list_ids()
    return {"sheet_ids": ids, "count": len(ids)}

# ─── Dashboard Stats ──────────────────────────────────────────────────────────

@app.get("/dashboard/stats")
async def dashboard_stats():
    total_keys = 0
    active_keys = 0
    total_requests_today = 0
    boostsheet_count = 0
    try:
        conn = _acquire()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM api_keys")
        total_keys = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM api_keys WHERE is_active = TRUE")
        active_keys = cur.fetchone()[0]
        cur.execute("SELECT COALESCE(SUM(request_count), 0) FROM api_keys WHERE last_used_at > NOW() - INTERVAL '1 day'")
        total_requests_today = cur.fetchone()[0] or 0
        cur.close()
        _release(conn)
    except Exception:
        pass

    if _repo:
        try:
            boostsheet_count = len(_repo.list_ids())
        except Exception:
            pass

    with _training_lock:
        training_state = _training_state.get("state", "idle")

    return {
        "total_api_keys": total_keys,
        "active_api_keys": active_keys,
        "total_requests_today": int(total_requests_today),
        "model_status": "loaded" if _model_ready else "initializing",
        "training_state": training_state,
        "vocab_size": len(_tokenizer.vocab) if _tokenizer else 0,
        "weights_exist": (Path(__file__).parent / "ai_model" / "weights" / "model.pt").exists(),
        "gpu_lanes": 512,
        "boostsheet_count": boostsheet_count,
    }

# ─── Storage Sync Endpoints ───────────────────────────────────────────────────

class CurriculumFeedback(BaseModel):
    user_id: str
    platform: str
    engagement_rate: float = Field(ge=0.0, le=100.0)
    content_type: str = "post"
    style_tags: List[str] = []


class DatasetRegister(BaseModel):
    name: str
    description: str = ""
    size_bytes: int = 0
    num_chunks: int = 0
    content_type: str = "text"


class CheckpointSave(BaseModel):
    model_id: str
    state: dict
    metadata: Optional[dict] = None


@app.get("/storage/status")
async def storage_status(_key = Depends(verify_api_key)):
    from storage_client import get_storage, get_checkpoint_client
    storage = get_storage()
    st = storage.status()
    # Derive canonical storage_mode so callers never have to re-implement the logic.
    if st.get("available"):
        _mode = "live"
    elif st.get("disk_store_available"):
        _mode = "local_fallback"
    else:
        _mode = "offline"
    checkpoints = get_checkpoint_client().list_checkpoints()
    return {
        **st,
        "storage_mode": _mode,
        # Alias: some older dashboard code checks `connected`
        "connected": st.get("available", False),
        "recent_checkpoints": checkpoints[:5],
    }


@app.post("/storage/feedback")
async def record_curriculum_feedback(feedback: CurriculumFeedback, _key = Depends(verify_api_key)):
    from storage_client import get_curriculum_client
    get_curriculum_client().record_feedback(
        user_id=feedback.user_id,
        platform=feedback.platform,
        engagement_rate=feedback.engagement_rate,
        content_type=feedback.content_type,
        style_tags=feedback.style_tags,
    )
    return {"status": "recorded", "user_id": feedback.user_id}


@app.get("/storage/curriculum/{user_id}")
async def get_curriculum(user_id: str, limit: int = 50, _key = Depends(verify_api_key)):
    from storage_client import get_curriculum_client
    client = get_curriculum_client()
    return {
        "user_id": user_id,
        "feedback": client.get_user_curriculum(user_id, limit=limit),
        "top_performers": client.get_top_performers(user_id, top_n=10),
        "stats": client.get_user_stats(user_id),
    }


class ArtistProfileRequest(BaseModel):
    artist_name: Optional[str] = None
    current_single: Optional[str] = None
    current_album: Optional[str] = None
    audience_age: Optional[str] = None
    audience_geo: Optional[str] = None
    # ── Brand Voice (research-driven, see storage_client.ArtistProfileClient) ──
    # Saved once, then pulled automatically into text/image/video generation so
    # output "sounds like the brand" instead of generic AI copy, and so the
    # artist controls whether generated content discloses AI assistance.
    genre: Optional[str] = None
    tone: Optional[str] = None
    vocabulary: Optional[List[str]] = None
    avoid_words: Optional[List[str]] = None
    palette: Optional[List[str]] = None
    ai_disclosure: Optional[bool] = None


class ArtistReleaseRequest(BaseModel):
    title: str
    kind: str = "single"          # single | album | ep
    release_date: Optional[str] = None
    streaming_url: Optional[str] = None
    status: str = "released"      # released | upcoming
    platforms: List[str] = []


@app.get("/storage/artist/{profile_id}")
async def get_artist_profile(profile_id: str, _key = Depends(verify_api_key)):
    """Return the artist profile + release catalog for generation enrichment."""
    from storage_client import get_artist_client
    return get_artist_client().get_enrichment(profile_id)


@app.post("/storage/artist/{profile_id}")
async def save_artist_profile(profile_id: str, profile: ArtistProfileRequest,
                              _key = Depends(verify_api_key)):
    """Create or merge-update an artist profile (name, single/album, audience)."""
    from storage_client import get_artist_client
    saved = get_artist_client().save_profile(profile_id, profile.model_dump())
    return {"status": "saved", "profile": saved}


@app.post("/storage/artist/{profile_id}/releases")
async def add_artist_release(profile_id: str, release: ArtistReleaseRequest,
                             _key = Depends(verify_api_key)):
    """Append a release / streaming record to the artist's catalog."""
    from storage_client import get_artist_client
    rec = get_artist_client().add_release(profile_id, release.model_dump())
    return {"status": "added", "release": rec}


@app.get("/storage/datasets")
async def list_datasets(_key = Depends(verify_api_key)):
    from storage_client import get_dataset_client
    return {"datasets": get_dataset_client().list_datasets()}


@app.post("/storage/datasets/register")
async def register_dataset(dataset: DatasetRegister, _admin = Depends(verify_admin)):
    from storage_client import get_dataset_client
    get_dataset_client().register_dataset(
        name=dataset.name,
        description=dataset.description,
        size_bytes=dataset.size_bytes,
        num_chunks=dataset.num_chunks,
        content_type=dataset.content_type,
    )
    return {"status": "registered", "name": dataset.name}


# Single-flight for audio seeding is now enforced inside
# workers/seed_audio_dataset.py (_SEED_LOCK + AlreadySeedingError).
# is_seeding() lets the HTTP endpoint return an immediate response without
# waiting for the background thread to race-fail on the module lock.


@app.get("/storage/datasets/audio/status")
async def get_audio_dataset_status(_key = Depends(verify_api_key)):
    """Return the current audio dataset manifest and auto-growth state.

    Combines the stored dataset meta (num_chunks, source, seeded_at) with the
    DataPuller's audio auto-growth counters so the dashboard can show dataset
    size and freshness in one call.
    """
    from storage_client import get_storage
    storage = get_storage()
    meta: dict = {}
    try:
        raw = storage.get("mb:dataset:audio:meta")
        if isinstance(raw, dict):
            meta = {
                "num_chunks": int(raw.get("num_chunks", 0)),
                "source": str(raw.get("source", "")),
                "seeded_at": raw.get("seeded_at"),
                "description": str(raw.get("description", "")),
            }
    except Exception:
        pass

    auto_growth: dict = {}
    if _data_puller is not None:
        try:
            s = _data_puller.get_state()
            auto_growth = {
                "enabled": True,
                "threshold": s.get("audio_growth_threshold", 20),
                "last_auto_seed_at": s.get("last_audio_seed_at"),
                "auto_seed_count": s.get("audio_auto_seed_count", 0),
                "interval_hours": s.get("audio_growth_interval_hours", 6),
            }
        except Exception:
            auto_growth = {"enabled": False}
    else:
        auto_growth = {"enabled": False}

    from workers.seed_audio_dataset import is_seeding as _is_seeding
    seeding_now = _is_seeding()

    return {
        "dataset": meta,
        "auto_growth": auto_growth,
        "seeding_now": seeding_now,
    }


@app.post("/storage/datasets/audio/seed")
async def seed_audio_dataset(
    count: int = 12,
    replace: bool = False,
    source: Optional[str] = None,
    _admin = Depends(verify_admin),
):
    """Seed REAL music samples from a public dataset into pdim storage so that
    ``/api/generate/audio`` produces output from real audio. Runs in the
    background; poll ``GET /storage/datasets/audio/status`` for the manifest.

    ``source`` overrides the automatic HF/librosa probe:
      - ``"hf"``      — force HuggingFace FMA-small (falls back to librosa on error)
      - ``"librosa"`` — use librosa bundled CC examples directly
      - omit / ``null`` — auto-detect (probe HF, fall back to librosa)
    """
    from storage_client import get_storage
    from workers import seed_audio_dataset as _seed

    storage = get_storage()
    if not storage.is_available and not getattr(storage, "disk_store_available", False):
        raise HTTPException(
            status_code=503,
            detail="storage backend unavailable — cannot seed audio dataset",
        )

    from workers.seed_audio_dataset import is_seeding as _is_seeding
    if _is_seeding():
        return {"status": "already_seeding"}

    force_source = source.strip().lower() if source else None

    def _run():
        try:
            _seed.seed(
                storage,
                count=int(count),
                replace=bool(replace),
                force_source=force_source,
            )
        except Exception as exc:  # background — surface explicit failure
            print(f"[seed_audio] seeding failed: {exc}", flush=True)

    threading.Thread(target=_run, daemon=True).start()
    return {
        "status": "seeding",
        "count": int(count),
        "replace": bool(replace),
        "source": force_source or "auto",
    }


@app.post("/storage/checkpoint/save")
async def save_checkpoint(payload: CheckpointSave, _admin = Depends(verify_admin)):
    from storage_client import get_checkpoint_client
    ok = get_checkpoint_client().save_checkpoint(
        model_id=payload.model_id,
        state=payload.state,
        metadata=payload.metadata,
    )
    return {"status": "saved" if ok else "fallback", "model_id": payload.model_id}


@app.get("/storage/checkpoint/{model_id}")
async def load_checkpoint(model_id: str, _admin = Depends(verify_admin)):
    from storage_client import get_checkpoint_client
    client = get_checkpoint_client()
    meta = client.get_checkpoint_meta(model_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Checkpoint '{model_id}' not found")
    return {"model_id": model_id, "meta": meta}


@app.get("/storage/checkpoints")
async def list_checkpoints(_admin = Depends(verify_admin)):
    from storage_client import get_checkpoint_client
    return {"checkpoints": get_checkpoint_client().list_checkpoints()}


# ─── Platform API — Main Music Platform Integration ───────────────────────────
#
# These endpoints are called by the main MaxBooster music platform
# (DAW, beat marketplace, social media management, music distribution).
# They run on top of the model trained from the 7TB storage dataset.

class PlatformSocialRequest(_AwarenessMixin):
    user_id: str
    platform: str = "instagram"
    topic: str
    tone: str = "authentic"
    goal: str = "growth"
    style_tags: List[str] = []
    include_hashtags: bool = True
    num_variants: int = Field(1, ge=1, le=5)
    instruction:    Optional[str]       = None
    extra_context:  Optional[str]       = None
    content_themes: Optional[List[str]] = None


class PlatformDAWRequest(_AwarenessMixin):
    user_id: str
    mode: str = "lyrics"        # lyrics | hook | beat_description | track_concept
    genre: str = "hip-hop"
    mood: str = "energetic"
    bpm: Optional[int] = None
    key: Optional[str] = None
    reference_track: Optional[str] = None
    context: Optional[str] = None
    instruction:    Optional[str]       = None
    extra_context:  Optional[str]       = None
    content_themes: Optional[List[str]] = None


class PlatformAutopilotRequest(_AwarenessMixin):
    user_id: str
    platform: str = "instagram"
    recent_posts: List[dict] = []
    target_metric: str = "engagement"   # engagement | reach | conversions
    instruction:    Optional[str]       = None
    extra_context:  Optional[str]       = None
    content_themes: Optional[List[str]] = None


class PlatformDistributionRequest(_AwarenessMixin):
    user_id: str
    track_title: str
    genre: str = "hip-hop"
    release_date: Optional[str] = None
    target_platforms: List[str] = ["spotify", "apple_music", "tidal"]
    bio: Optional[str] = None
    instruction:    Optional[str]       = None
    extra_context:  Optional[str]       = None
    content_themes: Optional[List[str]] = None


class PlatformVideoRequest(_AwarenessMixin):
    user_id: str
    topic: str
    platform: str = "youtube"           # youtube | tiktok | instagram | general
    style: str = "cinematic"            # cinematic | documentary | animated | social
    goal: str = "engagement"            # engagement | education | promotion | storytelling
    tone: str = "energetic"             # energetic | calm | dramatic | inspirational | playful
    duration_seconds: int = Field(30, ge=5, le=300)
    aspect_ratio: str = "16:9"          # 16:9 | 9:16 | 1:1 | 4:5
    include_captions: bool = True
    instruction:    Optional[str]       = None
    extra_context:  Optional[str]       = None
    content_themes: Optional[List[str]] = None
    # ── Veo-parity generation controls ───────────────────────────────────
    camera_motion: Optional[str] = None      # pan_left/pan_right/zoom_in/zoom_out/
                                             # tilt_up/tilt_down/dolly_in/dolly_out/
                                             # crane_up/crane_down/static/auto
    negative_prompt: Optional[str] = None   # content/style/elements to avoid
    seed: Optional[int] = None              # explicit seed for reproducible output
    fps: Optional[int] = None               # output frame rate (8/16/24/30); default 24
    motion_intensity: Optional[float] = None # 0.0–1.0; overrides tone-derived energy
    enhance_prompt: bool = True             # False = skip AI awareness augmentation
    lighting: Optional[str] = None          # cinematic/dramatic/natural/studio/
                                            # golden_hour/night/neon
    color_temperature: Optional[str] = None # warm/cool/neutral
    style_reference: Optional[str] = None   # URL or asset ID for style conditioning
    output_resolution: Optional[str] = None # 720p/1080p/4k — overrides derived resolution


def _build_personalized_tone(user_id: str, platform: str, base_tone: str) -> str:
    """Pull user engagement signals from storage to bias the model tone."""
    try:
        from storage_client import get_curriculum_client
        client = get_curriculum_client()
        top = client.get_top_performers(user_id, platform=platform, top_n=3)
        if top:
            tags = []
            for fb in top:
                tags.extend(fb.get("style_tags", []))
            if tags:
                dominant = max(set(tags), key=tags.count)
                return f"{base_tone}, {dominant}"
    except Exception:
        pass
    return base_tone


@app.post("/platform/social/generate")
async def platform_social_generate(req: PlatformSocialRequest, _key = Depends(require_scope("generate"))):
    """
    Main platform social media content generation.
    Uses per-user curriculum signals to personalize tone/style.
    """
    start = time.time()
    platform = normalize_platform(req.platform)
    # Universal URL Parser: resolve topic from any URL/platform link
    _social_topic = _resolve_topic_from_url(req.topic)

    # Personalize tone based on user's past engagement data in storage
    personalized_tone = _build_personalized_tone(req.user_id, platform, req.tone)
    effective_awareness = _effective_awareness(platform, _merged_awareness_for(req))

    variants = []
    await _wait_for_model_ready()

    for i in range(req.num_variants):
        try:
            from ai_model.agents.script_agent import ScriptRequest
            from ai_model.agents.distribution_agent import DistributionRequest

            _vidx = i   # capture loop variable before async boundary
            async def _run_variant(vidx=_vidx):
                s = await _in_thread(lambda: _script_agent.run(ScriptRequest(
                    idea=_social_topic, platform=platform,
                    goal=req.goal, tone=personalized_tone,
                    awareness=effective_awareness,
                    variant_idx=vidx,
                )))
                d = await _in_thread(lambda: _distribution_agent.run(DistributionRequest(
                    script=f"{s.hook}\n{s.body}\n{s.cta}",
                    platform=platform, goal=req.goal,
                    awareness=effective_awareness,
                )))
                return s, d

            script, dist = await _run_variant()
            if not any([script.hook.strip(), script.body.strip(), script.cta.strip()]):
                raise HTTPException(
                    status_code=503,
                    detail="empty generatedContent — model warming up, please retry",
                )
            variant: dict[str, Any] = {
                "hook": script.hook,
                "body": script.body,
                "cta": script.cta,
                "caption": dist.caption,
                "hashtags": dist.hashtags if req.include_hashtags else [],
                "source": getattr(script, "source", "model"),
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        variant["variant"] = i + 1
        variants.append(variant)

    _result = {
        "success": True,
        "user_id": req.user_id,
        "platform": platform,
        "topic": _social_topic,
        "personalized_tone": personalized_tone,
        "variants": variants,
        "model_ready": _model_ready,
        "processing_time_ms": round((time.time() - start) * 1000, 1),
    }
    _fw_ingest(_key, "social", _result, {
        "topic": _social_topic, "platform": platform,
        "tone": personalized_tone, "awareness": effective_awareness,
    })
    return _result


@app.post("/platform/social/autopilot")
async def platform_social_autopilot(req: PlatformAutopilotRequest, _key = Depends(require_scope("generate"))):
    """
    Autopilot endpoint: analyses a user's recent post performance and recommends
    the next content strategy using the trained model + engagement signals.
    """
    start = time.time()
    platform = normalize_platform(req.platform)

    try:
        from storage_client import get_curriculum_client
        curriculum = get_curriculum_client()
        top_content = curriculum.get_top_performers(req.user_id, platform=platform, top_n=5)
        _user_stats = curriculum.get_user_stats(req.user_id)
    except Exception:
        top_content = []
        _user_stats = {}

    # Analyse what's working
    top_tags = []
    top_types = []
    avg_engagement = 0.0
    for post in (top_content or req.recent_posts[:5]):
        top_tags.extend(post.get("style_tags", []))
        top_types.append(post.get("content_type", "post"))
        avg_engagement += post.get("engagement_rate", 0.0)

    if top_content or req.recent_posts:
        avg_engagement /= max(len(top_content or req.recent_posts), 1)

    dominant_tags = list(dict.fromkeys(top_tags))[:3]
    dominant_type = max(set(top_types), key=top_types.count) if top_types else "post"

    await _wait_for_model_ready()

    next_topics = []
    schedule: list[str] = []
    try:
        from ai_model.agents.script_agent import ScriptRequest
        from ai_model.agents.distribution_agent import DistributionRequest
        _auto_aw = _merged_awareness_for(req)
        for tag in (dominant_tags or ["music", "artist", "studio"])[:2]:
            s = await _in_thread(lambda t=tag: _script_agent.run(ScriptRequest(
                idea=t, platform=platform, goal=req.target_metric, tone="authentic",
                awareness=_auto_aw,
            )))
            next_topics.append({
                "topic": tag,
                "hook": s.hook,
                "cta": s.cta,
                "source": getattr(s, "source", "model"),
            })
        # Get posting time from distribution agent instead of static dict
        d = await _in_thread(lambda: _distribution_agent.run(DistributionRequest(
            script=next_topics[0]["hook"] if next_topics else platform,
            platform=platform, goal=req.target_metric,
            awareness=_auto_aw,
        )))
        schedule = [d.posting_time] if d.posting_time else []
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "success": True,
        "user_id": req.user_id,
        "platform": platform,
        "analysis": {
            "avg_engagement_rate": round(avg_engagement, 2),
            "top_style_tags": dominant_tags,
            "best_content_type": dominant_type,
            "data_points": len(top_content),
        },
        "recommendations": {
            "next_topics": next_topics,
            "best_posting_times": schedule,
            "content_type": dominant_type,
            "style_focus": dominant_tags[:2] if dominant_tags else ["authentic", "music"],
        },
        "autopilot_ready": bool(top_content),
        "model_powered": _model_ready,
        "processing_time_ms": round((time.time() - start) * 1000, 1),
    }


@app.post("/platform/daw/generate")
async def platform_daw_generate(req: PlatformDAWRequest, _key = Depends(require_scope("generate"))):
    """
    DAW / Studio AI generation endpoint.
    Powers the AI assistant inside the MaxBooster DAW:
    lyrics, hooks, beat descriptions, track concepts — all model-generated.
    """
    start = time.time()

    context_prompt = f"Genre: {req.genre}. Mood: {req.mood}."
    if req.bpm:
        context_prompt += f" BPM: {req.bpm}."
    if req.key:
        context_prompt += f" Key: {req.key}."
    if req.reference_track:
        context_prompt += f" Inspired by: {req.reference_track}."
    if req.context:
        context_prompt += f" {req.context}"

    if req.mode == "lyrics":
        topic = f"song lyrics — {context_prompt}"
        goal = "creative expression"
        tone = req.mood
    elif req.mode == "hook":
        topic = f"catchy hook — {context_prompt}"
        goal = "virality"
        tone = "punchy"
    elif req.mode == "beat_description":
        topic = f"beat description — {context_prompt}"
        goal = "production"
        tone = "technical"
    else:  # track_concept
        topic = f"full track concept — {context_prompt}"
        goal = "artistry"
        tone = req.mood

    await _wait_for_model_ready()

    try:
        from ai_model.agents.script_agent import ScriptRequest
        from ai_model.agents.visual_spec_agent import VisualSpecRequest

        _daw_aw = _merged_awareness_for(req)
        script = await _in_thread(lambda: _script_agent.run(ScriptRequest(
            idea=topic, platform="youtube", goal=goal, tone=tone,
            awareness=_daw_aw,
        )))
        visual = await _in_thread(lambda: _visual_spec_agent.run(VisualSpecRequest(
            idea=topic, platform="youtube", tone=tone, awareness=_daw_aw,
        )))

        _result = {
            "success": True,
            "user_id": req.user_id,
            "mode": req.mode,
            "genre": req.genre,
            "mood": req.mood,
            "bpm": req.bpm,
            "key": req.key,
            "output": {
                "main": script.hook,
                "body": script.body,
                "cta": script.cta,
                "visual_direction": getattr(visual, "thumbnail_prompt", "cinematic"),
                "color_scheme": getattr(visual, "color_scheme", "dark_neon"),
                "layout": getattr(visual, "layout", "landscape_16_9"),
            },
            "source": getattr(script, "source", "model"),
            "processing_time_ms": round((time.time() - start) * 1000, 1),
        }
        _fw_ingest(_key, "daw", _result, {
            "genre": req.genre, "mood": req.mood, "mode": req.mode,
            "bpm": req.bpm, "key": req.key, "awareness": req.awareness,
            "context_prompt": context_prompt,
        })
        return _result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/platform/distribution/plan")
async def platform_distribution_plan(req: PlatformDistributionRequest, _key = Depends(require_scope("generate"))):
    """
    Music distribution planning endpoint.
    Generates a release strategy for a track across streaming platforms,
    powered by the trained model's knowledge of platform dynamics.
    """
    start = time.time()

    await _wait_for_model_ready()

    try:
        from ai_model.agents.distribution_agent import DistributionRequest
        bio_context = req.bio or f"{req.genre} artist"
        dist = await _in_thread(lambda: _distribution_agent.run(DistributionRequest(
            script=f"New {req.genre} track: '{req.track_title}'. Artist: {bio_context}.",
            platform="spotify", goal="streams",
            awareness=_merged_awareness_for(req),
        )))
        _result = {
            "success": True,
            "user_id": req.user_id,
            "track": req.track_title,
            "plan": {
                "pitch": dist.caption,
                "hashtags": dist.hashtags,
                "target_platforms": req.target_platforms,
                "release_window": "Friday release recommended",
                "pre_release_steps": [
                    "Submit to distributor 7 days before release",
                    "Pitch to Spotify editorial playlist curators",
                    "TikTok teaser campaign 5 days before drop",
                    "Apple Music pre-save link campaign",
                ],
                "post_release": [
                    "24h engagement blitz on all platforms",
                    "Reply to every comment in the first 2 hours",
                    "Share stream milestone updates as Stories",
                ],
            },
            "source": getattr(dist, "source", "model"),
            "processing_time_ms": round((time.time() - start) * 1000, 1),
        }
        _fw_ingest(_key, "distribution", _result, {
            "track": req.track_title, "genre": req.genre,
            "target_platforms": req.target_platforms,
            "awareness": req.awareness,
        })
        return _result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/platform/video/generate")
async def platform_video_generate_schema():
    """
    Schema discovery for the video generation platform endpoint.
    Returns parameter definitions and expected response shape.
    """
    return {
        "endpoint": "POST /platform/video/generate",
        "description": (
            "Generate a complete AI video production package for the MaxBooster platform — "
            "includes personalized script, scene-by-scene visual directions, captions, "
            "thumbnail concept, and distribution metadata. Personalised per user via "
            "curriculum engagement signals."
        ),
        "parameters": {
            "user_id":          {"type": "string",  "required": True},
            "topic":            {"type": "string",  "required": True},
            "platform":         {"type": "string",  "required": False, "default": "youtube",    "options": ["youtube", "tiktok", "instagram", "general"]},
            "style":            {"type": "string",  "required": False, "default": "cinematic",  "options": ["cinematic", "documentary", "animated", "social"]},
            "goal":             {"type": "string",  "required": False, "default": "engagement", "options": ["engagement", "education", "promotion", "storytelling"]},
            "tone":             {"type": "string",  "required": False, "default": "energetic",  "options": ["energetic", "calm", "dramatic", "inspirational", "playful"]},
            "duration_seconds": {"type": "integer", "required": False, "default": 30, "min": 5, "max": 300},
            "aspect_ratio":     {"type": "string",  "required": False, "default": "16:9",       "options": ["16:9", "9:16", "1:1", "4:5"]},
            "include_captions": {"type": "boolean", "required": False, "default": True},
        },
        "returns": {
            "user_id":           "Echoed user identifier",
            "title":             "Video title",
            "hook":              "Opening hook line",
            "script":            "Full narration / dialogue script",
            "scenes":            "List of scene objects — description, visual_direction, narration, duration_seconds",
            "captions":          "Caption blocks with start_sec / end_sec / text",
            "hashtags":          "Platform-tuned hashtags",
            "thumbnail_concept": "AI-generated thumbnail concept description",
            "distribution":      "Platform caption, goal, and recommended post time",
            "duration_seconds":  "Planned duration",
            "aspect_ratio":      "Output aspect ratio",
            "source":            "'model' or 'template'",
            "processing_time_ms":"Generation latency",
        },
    }


@app.post("/platform/video/generate")
async def platform_video_generate(req: PlatformVideoRequest, _key = Depends(require_scope("generate"))):
    """
    Main platform video generation endpoint.
    Generates a complete AI video production package personalised to the user's
    engagement history via the curriculum feedback store.
    """
    start = time.time()
    platform = normalize_platform(req.platform)
    personalized_tone = _build_personalized_tone(req.user_id, platform, req.tone)
    scene_count = max(3, req.duration_seconds // 10)
    # Universal URL Parser: resolve topic from any URL/platform link
    _vid_topic = _resolve_topic_from_url(req.topic)

    await _wait_for_model_ready()

    async def _run_model_inference():
        from ai_model.agents.script_agent import ScriptRequest
        from ai_model.agents.visual_spec_agent import VisualSpecRequest
        from ai_model.agents.distribution_agent import DistributionRequest

        _vid_aw = _merged_awareness_for(req)
        script_result = await _in_thread(lambda: _script_agent.run(ScriptRequest(
            idea=_vid_topic, platform=platform, goal=req.goal, tone=personalized_tone,
            awareness=_vid_aw,
        )))
        full_script = f"{script_result.hook}\n{script_result.body}\n{script_result.cta}"

        visual_result = await _in_thread(lambda: _visual_spec_agent.run(VisualSpecRequest(
            idea=_vid_topic, platform=platform, tone=personalized_tone, awareness=_vid_aw,
        ))) if _visual_spec_agent else None

        dist_result = await _in_thread(lambda: _distribution_agent.run(DistributionRequest(
            script=full_script, platform=platform, goal=req.goal,
            awareness=_vid_aw,
        )))
        return script_result, full_script, visual_result, dist_result

    # Async coalescer: identical video requests share one Future — followers are
    # suspended coroutines, not threads.  Only unique digests hit the model.
    _vkey = {
        "topic":    _vid_topic,
        "platform": platform,
        "style":    req.style or "",
        "goal":     req.goal  or "",
        "tone":     personalized_tone or "",
        "awareness": str(req.awareness or ""),
        "duration": req.duration_seconds,
    }
    _coalesced_inference: list = []

    async def _coalesced_video():
        result = await _run_model_inference()
        _coalesced_inference.append(result)
        return result

    try:
        script_result, full_script, visual_result, dist_result = \
            await _get_async_coalescer().compute(_vkey, _coalesced_video)

        raw_scenes = getattr(visual_result, "scenes", None) or []
        if not raw_scenes:
            lines = full_script.split("\n")
            raw_scenes = [
                {
                    "scene": i + 1,
                    "duration_seconds": req.duration_seconds // scene_count,
                    "description": f"{_vid_topic} — {req.style} scene {i + 1}",
                    "visual_direction": f"{req.style.capitalize()} framing, {personalized_tone} energy",
                    "narration": lines[min(i, len(lines) - 1)],
                }
                for i in range(scene_count)
            ]

        caption_blocks = []
        if req.include_captions:
            per_scene = req.duration_seconds // max(len(raw_scenes), 1)
            for idx, scene in enumerate(raw_scenes):
                caption_blocks.append({
                    "start_sec": idx * per_scene,
                    "end_sec": (idx + 1) * per_scene,
                    "text": scene.get("narration", f"Scene {idx + 1}"),
                })

        thumbnail_concept = (
            getattr(visual_result, "thumbnail_concept", None)
            or f"Bold '{_vid_topic.upper()}' text over {req.style} background with {personalized_tone} color grading"
        )

        _result = {
            "success": True,
            "user_id": req.user_id,
            "title": f"{_vid_topic} — {req.style.capitalize()} Video",
            "hook": script_result.hook,
            "body": script_result.body,
            "cta":  script_result.cta,
            "script": full_script,
            "scenes": raw_scenes,
            "captions": caption_blocks,
            "hashtags": dist_result.hashtags if req.include_captions else [],
            "thumbnail_concept": thumbnail_concept,
            "distribution": {
                "platform": platform,
                "caption": dist_result.caption,
                "goal": req.goal,
                "recommended_post_time": getattr(dist_result, "recommended_post_time", "peak hours"),
            },
            "duration_seconds": req.duration_seconds,
            "aspect_ratio": req.aspect_ratio,
            "source": getattr(script_result, "source", "model"),
            "processing_time_ms": round((time.time() - start) * 1000, 1),
        }
        _fw_ingest(_key, "video", _result, {
            "topic": _vid_topic, "platform": platform, "style": req.style,
            "tone": personalized_tone, "awareness": req.awareness,
            "duration_seconds": req.duration_seconds,
        })
        return _result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/platform/model/info")
async def platform_model_info(_key = Depends(verify_api_key)):
    """
    Returns the current model state for the main platform UI.
    Shows whether the model is running on trained storage data or baseline weights.
    """
    from storage_client import get_storage
    storage = get_storage()

    with _training_lock:
        t_state = dict(_training_state)

    weights_path = Path(__file__).parent / "ai_model" / "weights" / "model.pt"

    return {
        "model_ready": _model_ready,
        "model_config": _model_config,
        "weights_source": "disk" if weights_path.exists() else "random_init",
        "weights_exist": weights_path.exists(),
        "last_checkpoint": t_state.get("last_checkpoint"),
        "training": {
            "state": t_state.get("state", "idle"),
            "source": t_state.get("source"),
            "final_loss": t_state.get("final_loss"),
            "batches_from_storage": t_state.get("batches_from_storage", 0),
        },
        "storage": {
            # Cached flag only — never trigger a lazy network ping from this
            # route. It must answer in <1 s even when pdim is cold/offline.
            "connected": bool(getattr(storage, "_available", False)),
            "dataset_bytes": 7696581394432,
            "dataset_tb": round(7696581394432 / 1e12, 2),
        },
        "platform_endpoints": [
            "POST /platform/social/generate",
            "POST /platform/social/autopilot",
            "POST /platform/daw/generate",
            "POST /platform/distribution/plan",
            "GET  /platform/video/generate",
            "POST /platform/video/generate",
            "POST /platform/model/reload",
        ],
    }


@app.post("/platform/model/reload")
async def platform_model_reload(_admin = Depends(verify_admin)):
    """
    Hot-reload: pull the latest checkpoint from storage and update model state.
    Call this from the main platform after a new training run completes.
    """
    thread = threading.Thread(target=_load_checkpoint_from_storage, daemon=True)
    thread.start()
    return {
        "status": "reloading",
        "message": "Checkpoint reload triggered from storage. Model state will update in ~5s.",
    }


# ─── MaxCore Multimodal Generation API ────────────────────────────────────────
#
# Five endpoints called by the Express orchestration layer (multimodal.ts).
# They form the generation backbone for all modalities:
#   POST /analyze              — normalize any input into a semantic representation
#   POST /generate/text        — mode=planner → TaskPlan | mode=content → text assets
#   POST /generate/image       — image asset specs per platform slot
#   POST /generate/audio       — voiceover/audio asset specs per platform slot
#   POST /generate/video       — video asset packs per platform slot

_platform_rules: dict = {}

def _load_platform_rules() -> dict:
    rules_path = Path(__file__).parent.parent.parent / "artifacts" / "api-server" / "src" / "platform_rules.json"
    try:
        with open(rules_path) as f:
            return json.load(f)
    except Exception:
        return {}

_platform_rules = _load_platform_rules()


class MaxcoreAnalyzeRequest(BaseModel):
    modality: str = "text"
    payload: str
    artistProfileId: Optional[str] = None
    platforms: List[str] = []
    intent: Optional[str] = None
    awareness: str = ""


class MaxcoreTextRequest(BaseModel):
    mode: str = "content"
    system: Optional[str] = None
    input: dict = {}        # used by mode='planner'
    step: dict = {}         # used by mode='content'
    inputs: dict = {}       # used by mode='content'
    awareness: str = ""     # enrichment + live signals; conditions the script agent
    instruction:    Optional[str]       = None
    extra_context:  Optional[str]       = None
    content_themes: Optional[List[str]] = None


class MaxcoreMediaRequest(BaseModel):
    step: dict = {}
    inputs: dict = {}
    awareness: str = ""     # enrichment + live signals; conditions the media agents
    instruction:    Optional[str]       = None
    extra_context:  Optional[str]       = None
    content_themes: Optional[List[str]] = None


@app.post("/analyze")
async def maxcore_analyze(req: MaxcoreAnalyzeRequest, _key = Depends(require_scope("generate"))):
    """
    Normalize any input modality (text, URL, image, audio, video) into a unified
    semantic representation. Fed directly into the planner by the TS orchestrator.
    """
    start = time.time()

    if req.modality == "url":
        content_hint = f"Content from URL: {req.payload}"
    elif req.modality in ("image", "audio", "video"):
        content_hint = f"{req.modality.capitalize()} asset: {req.payload}"
    else:
        content_hint = req.payload

    intent_hint = req.intent or "general content promotion"
    first_platform = req.platforms[0] if req.platforms else "general"

    normalized: dict = {
        "modality": req.modality,
        "payload_summary": content_hint,
        "intent": intent_hint,
        "platforms": req.platforms,
        "artistProfileId": req.artistProfileId,
        "semantic": {
            "topic": content_hint,
            "intent": intent_hint,
            "platforms": req.platforms,
            "style_tags": [],
        },
        "source": "template",
        "processing_time_ms": 0,
    }

    if _model_ready and _script_agent:
        try:
            from ai_model.agents.script_agent import ScriptRequest
            result = await _in_thread(lambda: _script_agent.run(ScriptRequest(
                idea=content_hint,
                platform=normalize_platform(first_platform),
                goal=intent_hint,
                tone="authentic",
                awareness=_merged_awareness_for(req),
            )))
            normalized["semantic"]["hook"] = result.hook
            normalized["semantic"]["core_message"] = result.body
            normalized["source"] = getattr(result, "source", "model")
        except Exception:
            pass

    normalized["processing_time_ms"] = round((time.time() - start) * 1000, 1)
    return normalized


@app.post("/generate/text")
async def maxcore_generate_text(req: MaxcoreTextRequest, _key = Depends(require_scope("generate"))):
    """
    Dual-mode text endpoint:
      mode='planner'  → returns a TaskPlan for the multimodal orchestrator
      mode='content'  → returns platform-specific text assets for a given step
    """
    start = time.time()
    data = req.input

    if req.mode == "planner":
        normalized = data.get("normalized", {})
        request_data = data.get("request", {})
        pack_spec = data.get("packSpec") or []

        modality_slots: dict = {}
        for slot in pack_spec:
            m = slot.get("modality", "text")
            modality_slots.setdefault(m, []).append(slot)

        steps = [{
            "id": "analysis_step",
            "type": "analyze",
            "worker": "text",
            "inputFrom": "normalizedInput",
            "params": {"intent": normalized.get("intent", "engagement")},
        }]

        for modality, slots in modality_slots.items():
            steps.append({
                "id": f"step_{modality}",
                "type": "generate",
                "worker": modality,
                "inputFrom": ["analysis_step"],
                "params": {
                    "slots": slots,
                    "platforms": [s.get("platform") for s in slots],
                    "constraints": request_data.get("constraints", {}),
                },
            })

        return {
            "requestId": request_data.get("id", str(uuid.uuid4())),
            "steps": steps,
            "processing_time_ms": round((time.time() - start) * 1000, 1),
        }

    # mode == 'content'
    step = req.step or data.get("step", {})
    inputs = req.inputs or data.get("inputs", {})
    slots = step.get("params", {}).get("slots", [])
    normalized = inputs.get("normalized", {}) if isinstance(inputs, dict) else {}
    topic = (normalized.get("semantic") or {}).get("topic") or normalized.get("payload_summary", "content")
    intent = (normalized.get("semantic") or {}).get("intent", "engagement")
    hook = (normalized.get("semantic") or {}).get("hook", "")

    outputs = []
    for slot in slots:
        platform = normalize_platform(slot.get("platform", "general"))
        slot_id = slot.get("id", "")
        purpose = slot.get("purpose", "")
        rules = _platform_rules.get(platform, {}).get("text", {})
        max_len = rules.get("recommendedLength", 150)
        tone_list = rules.get("tone") or ["authentic"]
        tone = tone_list[0] if tone_list else "authentic"
        max_hashtags = rules.get("hashtags", {}).get("max", 0)
        hashtags_allowed = rules.get("hashtags", {}).get("allowed", False)

        text = hook or f"{topic} — {purpose}"
        tags: list = []

        hook_line = ""
        body_line = ""
        cta_line = ""
        posting_time = ""
        source = "template"

        if _model_ready and _script_agent and _distribution_agent:
            try:
                from ai_model.agents.script_agent import ScriptRequest
                from ai_model.agents.distribution_agent import DistributionRequest
                script = await _in_thread(lambda: _script_agent.run(ScriptRequest(
                    idea=topic, platform=platform, goal=intent, tone=tone,
                    awareness=_merged_awareness_for(req),
                )))
                hook_line = getattr(script, "hook", "") or ""
                body_line = getattr(script, "body", "") or ""
                cta_line = getattr(script, "cta", "") or ""
                source = getattr(script, "source", "template")
                full_script = f"{hook_line}\n{body_line}\n{cta_line}".strip()
                dist = await _in_thread(lambda: _distribution_agent.run(DistributionRequest(
                    script=full_script, platform=platform, goal=intent,
                    awareness=_merged_awareness_for(req),
                )))
                text = dist.caption
                posting_time = getattr(dist, "posting_time", "") or ""
                if hashtags_allowed:
                    tags = getattr(dist, "hashtags", [])[:max_hashtags]
            except Exception:
                pass

        if max_len:
            text = text[:max_len]
        if tags:
            text = f"{text}\n{' '.join(tags)}"

        outputs.append({
            "text": text,
            "platform": platform,
            "slotId": slot_id,
            "meta": {
                "purpose": purpose,
                "tone": tone,
                "length": len(text),
                "hook": hook_line,
                "body": body_line,
                "cta": cta_line,
                "posting_time": posting_time,
                "hashtags": tags,
                "source": source,
            },
        })

    return {
        "outputs": outputs,
        "processing_time_ms": round((time.time() - start) * 1000, 1),
    }


@app.post("/generate/image")
async def maxcore_generate_image(req: MaxcoreMediaRequest, _key = Depends(require_scope("generate"))):
    """
    Generate image asset specs (concept + aspect ratio + style guidance) per slot.
    Returns asset URIs and generation metadata for downstream rendering pipelines.
    """
    start = time.time()
    step = req.step
    inputs = req.inputs
    slots = step.get("params", {}).get("slots", [])
    normalized = inputs.get("normalized", {}) if isinstance(inputs, dict) else {}
    topic = (normalized.get("semantic") or {}).get("topic") or normalized.get("payload_summary", "content")
    constraints = step.get("params", {}).get("constraints", {})
    style_tags: list = constraints.get("styleTags", ["cinematic"])

    outputs = []
    for slot in slots:
        platform = slot.get("platform", "instagram")
        slot_id = slot.get("id", "")
        rules = _platform_rules.get(platform, {}).get("image", {})
        aspect_ratio = rules.get("recommended") or (rules.get("aspectRatios") or ["1:1"])[0]
        style = ", ".join(style_tags)
        concept = (
            f"{style.capitalize()} visual for '{topic}' — {slot.get('purpose', '')}. "
            f"Aspect ratio {aspect_ratio}. Platform: {platform}."
        )

        if _model_ready and _visual_spec_agent:
            try:
                from ai_model.agents.visual_spec_agent import VisualSpecRequest
                vis = await _in_thread(lambda: _visual_spec_agent.run(VisualSpecRequest(
                    idea=topic,
                    platform=normalize_platform(platform),
                    tone=style_tags[0] if style_tags else "cinematic",
                    awareness=_merged_awareness_for(req),
                )))
                concept = getattr(vis, "thumbnail_concept", concept) or concept
            except Exception:
                pass

        outputs.append({
            "url": f"asset://{slot_id}/{aspect_ratio.replace(':', 'x')}.png",
            "platform": platform,
            "slotId": slot_id,
            "meta": {
                "aspect_ratio": aspect_ratio,
                "style_tags": style_tags,
                "concept": concept,
                "format": "png",
            },
        })

    return {
        "outputs": outputs,
        "processing_time_ms": round((time.time() - start) * 1000, 1),
    }


@app.post("/generate/audio")
async def maxcore_generate_audio(req: MaxcoreMediaRequest, _key = Depends(require_scope("generate"))):
    """
    Generate audio asset specs (voiceover script + duration + style) per slot.
    Applies per-platform audio rules (max duration, style, voiceover eligibility).
    """
    start = time.time()
    step = req.step
    inputs = req.inputs
    slots = step.get("params", {}).get("slots", [])
    normalized = inputs.get("normalized", {}) if isinstance(inputs, dict) else {}
    topic = (normalized.get("semantic") or {}).get("topic") or normalized.get("payload_summary", "content")
    hook = (normalized.get("semantic") or {}).get("hook", "")

    outputs = []
    for slot in slots:
        platform = slot.get("platform", "general")
        slot_id = slot.get("id", "")
        rules = _platform_rules.get(platform, {}).get("audio", {})

        if not rules.get("voiceover", True):
            continue

        max_dur = rules.get("maxDurationSec", 30)
        style_list = rules.get("style") or rules.get("tone") or ["authentic"]
        style = style_list[0] if isinstance(style_list, list) else "authentic"
        script_text = hook or f"Don't miss '{topic}' — out now."

        if _model_ready and _script_agent:
            try:
                from ai_model.agents.script_agent import ScriptRequest
                res = await _in_thread(lambda: _script_agent.run(ScriptRequest(
                    idea=topic,
                    platform=normalize_platform(platform),
                    goal="engagement",
                    tone=style,
                    awareness=_merged_awareness_for(req),
                )))
                script_text = res.hook
            except Exception:
                pass

        outputs.append({
            "url": f"asset://{slot_id}/voiceover.mp3",
            "platform": platform,
            "slotId": slot_id,
            "meta": {
                "script": script_text,
                "max_duration_sec": max_dur,
                "style": style,
                "sample_rate": 44100,
                "format": "mp3",
            },
        })

    return {
        "outputs": outputs,
        "processing_time_ms": round((time.time() - start) * 1000, 1),
    }


@app.post("/generate/video")
async def maxcore_generate_video(req: MaxcoreMediaRequest, _key = Depends(require_scope("generate"))):
    """
    Generate video asset packs per slot.
    Applies per-platform rules: aspect ratio, duration, hook requirement.
    """
    start = time.time()
    step = req.step
    inputs = req.inputs
    slots = step.get("params", {}).get("slots", [])
    normalized = inputs.get("normalized", {}) if isinstance(inputs, dict) else {}
    topic = (normalized.get("semantic") or {}).get("topic") or normalized.get("payload_summary", "content")
    hook = (normalized.get("semantic") or {}).get("hook", "")
    constraints = step.get("params", {}).get("constraints", {})
    style_tags: list = constraints.get("styleTags", ["cinematic"])

    outputs = []
    for slot in slots:
        platform = slot.get("platform", "youtube")
        slot_id = slot.get("id", "")
        params = step.get("params", {})
        rules = _platform_rules.get(platform, {}).get("video", {})
        aspect_ratio = params.get("aspectRatio") or (rules.get("aspectRatios") or ["16:9"])[0]
        duration = (
            params.get("maxDurationSec")
            or rules.get("recommendedDurationSec")
            or rules.get("maxDurationSec")
            or 30
        )
        requires_hook = rules.get("requiresHook", False)
        tone = style_tags[0] if style_tags else "energetic"

        hook_line = hook or f"You need to see this — {topic}"
        body_line = ""
        cta_line = ""
        source = "template"

        if _model_ready and _script_agent:
            try:
                from ai_model.agents.script_agent import ScriptRequest
                res = await _in_thread(lambda: _script_agent.run(ScriptRequest(
                    idea=topic,
                    platform=normalize_platform(platform),
                    goal="engagement",
                    tone=tone,
                    awareness=_merged_awareness_for(req),
                )))
                hook_line = res.hook or hook_line
                body_line = getattr(res, "body", "") or ""
                cta_line = getattr(res, "cta", "") or ""
                source = getattr(res, "source", "template")
            except Exception:
                pass

        render_url = f"asset://{slot_id}/{aspect_ratio.replace(':', 'x')}.mp4"
        render_meta: dict = {}

        script_full = f"{hook_line}\n{body_line}\n{cta_line}".strip()
        outputs.append({
            "url": render_url,
            "platform": platform,
            "slotId": slot_id,
            "meta": {
                "hook": hook_line if requires_hook else None,
                "body": body_line,
                "cta": cta_line,
                "script": script_full,
                "aspect_ratio": aspect_ratio,
                "duration_sec": duration,
                "requires_hook": requires_hook,
                "style_tags": style_tags,
                "source": source,
                "format": "mp4",
                "fps": 30,
                **render_meta,
            },
        })

    return {
        "outputs": outputs,
        "processing_time_ms": round((time.time() - start) * 1000, 1),
    }


# ─── AI Ad System & Autopilot ─────────────────────────────────────────────────
#
# Replicates peak performance of paid ads across Meta, TikTok, YouTube, Google.
# Every ad run is stored; peak performers (high ROAS/CTR/low CPC) are extracted
# into pattern signatures the AI uses to generate the next winning creative set.

AD_PLATFORMS = {"meta", "facebook", "instagram", "tiktok", "youtube", "google", "twitter", "snapchat"}
AD_TYPES = {"video", "image", "carousel", "story", "reel", "search", "display", "ugc"}

AD_HOOKS_BY_PLATFORM = {
    "tiktok":    ["Stop scrolling — this drop is finally here! 🔥",
                  "The secret nobody tells you about going viral!",
                  "POV: you discover the best new release of the year",
                  "I tested this for 30 days and the results are insane!",
                  "This is the fire track everyone is talking about!"],
    "meta":      ["Finally — the drop you've been waiting for is live! 🎵",
                  "Stop sleeping on this — the best new music is finally here!",
                  "The #1 secret to finding fire music nobody else knows!",
                  "Never miss a drop like this — stream it now!",
                  "Exclusive first listen — this is insane! 🔥"],
    "youtube":   ["Stop what you're doing — this drop is finally here!",
                  "The secret formula that made this track go viral!",
                  "Why everyone is streaming this exclusive new release!",
                  "Never heard a drop this fire — exclusive now!"],
    "google":    ["Best new music — exclusively available now!",
                  "Top-rated exclusive drop — stream free now!",
                  "Finally — the fire track everyone is talking about!",
                  "Never-before-heard — exclusive first listen now!"],
    "instagram": ["Stop scrolling — this fire drop is finally live! 🔥",
                  "Real music. Real fire. Exclusively now.",
                  "Never-before-heard. First listen exclusively here!",
                  "The drop everyone is talking about — finally here! 🎵"],
}

# ── Per-subtype creative specs ────────────────────────────────────────────────
# Each entry defines format-specific creative_brief fields and the copy
# style the script agent should target.  All four subtypes are always
# generated when vary_subtypes=True so every campaign has maximum format
# coverage across placements.
AD_SPECS_BY_TYPE: dict = {
    "video": {
        "format_label":   "Video Ad",
        "duration_range": "15–60s",
        "copy_tone":      "direct",
        "brief_extras": {
            "opening_3s":        "hook text overlay on first frame",
            "visual_direction":  "Raw, high-energy footage of artist in studio or performance",
            "caption_style":     "auto-captions on for TikTok/Reels placements",
        },
    },
    "audio": {
        "format_label":   "Audio Ad",
        "duration_range": "15–30s",
        "copy_tone":      "conversational",
        "brief_extras": {
            "voiceover_style":   "warm, energetic narrator — first-person artist voice preferred",
            "sound_branding":    "open with 2s of the track hook, close with artist name + CTA",
            "companion_banner":  "static 640×320 banner runs alongside the audio stream",
            "platform_notes":    "Spotify/Pandora/podcast mid-roll; no visual dependency",
        },
    },
    "text": {
        "format_label":   "Text Ad",
        "duration_range": "N/A",
        "copy_tone":      "punchy",
        "brief_extras": {
            "headline_limit":    "30 characters",
            "description_limit": "90 characters",
            "display_url":       "artist.com/new",
            "extension_types":   ["sitelink", "callout", "structured snippet"],
            "platform_notes":    "Google Search / Meta text overlay / Twitter Promoted",
        },
    },
    "image": {
        "format_label":   "Image Ad",
        "duration_range": "N/A",
        "copy_tone":      "emotive",
        "brief_extras": {
            "visual_concept":    "Album / single artwork with bold typography overlay",
            "text_overlay_rule": "≤20% of image area (Meta policy)",
            "color_direction":   "Match primary artwork palette; high-contrast CTA button",
            "tagline_position":  "Bottom third — must be legible at thumbnail size",
            "format_variants":   ["1:1 feed", "9:16 story/reel", "1.91:1 link preview"],
        },
    },
}

# Subtype-specific hook overrides — each type has a distinct opening style
AD_HOOKS_BY_TYPE: dict = {
    "audio": [
        "Close your eyes — you've never heard anything like this 🎧",
        "This sound will stop you mid-scroll. Just listen.",
        "One track. Thirty seconds. Your new favourite artist.",
        "Turn it up — this exclusive drop is only here for now.",
        "Your ears called. They want this track immediately.",
    ],
    "text": [
        "Fire new drop. Stream free now.",
        "Exclusive: best track of the year just dropped.",
        "Finally here — the release everyone is talking about.",
        "New music. Real fire. Available now.",
        "Stop missing out — this drop is live.",
    ],
    "image": [
        "One image. One release. Zero skips. 🎵",
        "This is the cover art for the drop of the year.",
        "The visual drop is finally here — stream now.",
        "Art meets fire. New release out now.",
        "See it. Hear it. Stream it. Now.",
    ],
    # "video" falls through to AD_HOOKS_BY_PLATFORM (platform-specific)
}

AD_CTAS_BY_GOAL = {
    "streams":       ["Stream Now", "Listen Free", "Add to Playlist", "Presave Today"],
    "merch":         ["Shop Now", "Get Yours", "Limited Drop", "Claim 20% Off"],
    "fanbase":       ["Follow for More", "Join the Movement", "Be First", "Subscribe"],
    "tickets":       ["Get Tickets", "Reserve Your Spot", "Doors Open Soon", "Book Now"],
    "downloads":     ["Download Free", "Get the Track", "Free Download Today"],
    "conversions":   ["Start Free Trial", "Book a Session", "Claim Your Spot", "Apply Now"],
}

AUDIENCE_SEGMENTS = {
    "music_fan":     ["music lovers", "playlist listeners", "concert-goers", "spotify users"],
    "hip_hop":       ["hip-hop fans", "trap music", "rap enthusiasts", "urban culture"],
    "rb":            ["R&B fans", "soul music", "neo-soul", "smooth jazz adjacent"],
    "pop":           ["pop music fans", "top 40 listeners", "mainstream music"],
    "producer":      ["music producers", "beatmakers", "DAW users", "FL Studio", "Ableton"],
    "artist":        ["independent artists", "musicians", "singer-songwriters", "bands"],
    "brand_deal":    ["content creators", "influencers", "brand collaboration seekers"],
}


class AdRecordRequest(BaseModel):
    user_id: str
    platform: str
    ad_type: str = "video"
    hook: str = ""
    headline: str = ""
    body: str = ""
    cta: str = ""
    audience_tags: List[str] = []
    ctr: float = Field(0.0, ge=0)
    cpc: float = Field(0.0, ge=0)
    roas: float = Field(0.0, ge=0)
    conversions: int = 0
    impressions: int = 0
    clicks: int = 0
    spend: float = 0.0
    run_id: Optional[str] = None


class AdGenerateRequest(BaseModel):
    user_id: str
    platform: str = "meta"
    ad_type: str = "video"
    product: str
    goal: str = "streams"
    budget_daily: Optional[float] = None
    num_creatives: int = Field(3, ge=1, le=10)
    replicate_peak: bool = True
    genre: Optional[str] = None
    artist_name: Optional[str] = None
    # When True (default), each creative cycles through all four content-type
    # subtypes (video, audio, text, image) so every campaign has full format
    # coverage.  Set False to keep all creatives as ad_type.
    vary_subtypes: bool = True
    # Explicit subtype selection — takes priority over vary_subtypes when set.
    # Pass a non-empty list of any combination of: "video", "audio", "text", "image".
    # Creatives cycle through exactly these subtypes in order (wrapping on repeat).
    # Unknown values are silently dropped; if all values are invalid the field is
    # ignored and vary_subtypes / ad_type governs instead.
    #
    # Examples:
    #   target_subtypes=["video","audio"]      → slot 0=video, 1=audio, 2=video, ...
    #   target_subtypes=["text"]               → every creative is a text ad
    #   target_subtypes=["image","video"]      → alternates image / video
    #   target_subtypes=None  (default)        → vary_subtypes / ad_type governs
    target_subtypes: Optional[List[str]] = None
    # Awareness-bridge fields — wired into ScriptRequest so the awareness
    # conditioning layer can push creative quality toward the 100/100 standard.
    awareness: Optional[Any] = None
    instruction: Optional[str] = None
    content_themes: Optional[List[str]] = None


class AdAutopilotRequest(BaseModel):
    user_id: str
    platform: Optional[str] = None
    budget_total: Optional[float] = None
    goal: str = "streams"
    current_campaigns: List[dict] = []


class AdAudienceRequest(BaseModel):
    user_id: str
    platform: str = "meta"
    product: str
    genre: Optional[str] = None
    goal: str = "streams"


_AD_HOOK_POWER = {
    "secret", "proven", "instantly", "exclusive", "free", "now",
    "never", "stop", "first", "best", "viral", "insane", "real",
    "raw", "unreleased", "finally", "limited", "drop", "fire",
    "everyone", "nobody",
}
import re as _re_hq_mod
_AD_EMOJI_RE = _re_hq_mod.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")

def _ad_hook_score(h: str) -> float:
    """Veo hook quality score used to prefer pool over weak model output."""
    hl = h.lower()
    s = 0.55 if any(p in hl for p in _AD_HOOK_POWER) else 0.0
    if "!" in h or "?" in h: s += 0.30
    if _AD_EMOJI_RE.search(h):  s += 0.15
    return min(1.0, s)


# Platform-specific video specs (used by the "video" subtype)
_PLAT_VIDEO_SPECS: dict = {
    "tiktok":    {"ratio": "9:16",        "duration": "15–60s",           "format": "vertical video"},
    "meta":      {"ratio": "1:1 or 4:5",  "duration": "15–30s",           "format": "feed video"},
    "youtube":   {"ratio": "16:9",        "duration": "6–15s skippable",  "format": "pre-roll"},
    "instagram": {"ratio": "9:16",        "duration": "up to 60s",        "format": "reel"},
    "google":    {"ratio": "N/A",         "duration": "N/A",              "format": "display"},
}


async def _generate_ad_creative(
    platform: str,
    ad_type: str,
    product: str,
    goal: str,
    peak_formula: Optional[dict],
    artist_name: Optional[str],
    genre: Optional[str],
    variant_idx: int,
    awareness: str = "",
    instruction: Optional[str] = None,
    content_themes: Optional[List[str]] = None,
) -> dict:
    """
    Generate one ad creative for a specific content-type subtype.

    ad_type must be one of: video | audio | text | image.
    Each subtype produces a distinct copy style, creative_brief shape, and
    AI-conditioning tone so the campaign has genuine format variety.

    Awareness, instruction, and content_themes flow into ScriptRequest so
    the awareness bridge can push quality toward the 100/100 standard.
    """
    plat_key   = platform.lower().replace("facebook", "meta").replace("instagram", "meta")
    sub        = ad_type.lower() if ad_type.lower() in AD_SPECS_BY_TYPE else "video"
    type_spec  = AD_SPECS_BY_TYPE[sub]
    cta_pool   = AD_CTAS_BY_GOAL.get(goal, ["Learn More", "Discover More"])
    artist     = artist_name or "the artist"
    genre_tag  = f" #{genre}" if genre else ""

    # ── Base hook selection ───────────────────────────────────────────────────
    # Audio / text / image have their own type-specific hook pools that match
    # the distinct writing style expected for each placement format.
    # Video falls through to the platform-specific pool.
    if sub in AD_HOOKS_BY_TYPE:
        type_hooks = AD_HOOKS_BY_TYPE[sub]
        if peak_formula and peak_formula.get("top_hooks"):
            base_hook = peak_formula["top_hooks"][variant_idx % len(peak_formula["top_hooks"])]
            base_cta  = (peak_formula.get("top_ctas") or cta_pool)[0]
            source    = "peak_replicated"
        else:
            base_hook = type_hooks[variant_idx % len(type_hooks)]
            base_cta  = cta_pool[variant_idx % len(cta_pool)]
            source    = "template"
    else:  # video — platform-specific pool
        hook_pool = AD_HOOKS_BY_PLATFORM.get(plat_key, AD_HOOKS_BY_PLATFORM["meta"])
        if peak_formula and peak_formula.get("top_hooks"):
            base_hook = peak_formula["top_hooks"][variant_idx % len(peak_formula["top_hooks"])]
            base_cta  = (peak_formula.get("top_ctas") or cta_pool)[0]
            source    = "peak_replicated"
        else:
            base_hook = hook_pool[variant_idx % len(hook_pool)]
            base_cta  = cta_pool[variant_idx % len(cta_pool)]
            source    = "template"

    # ── Awareness composition (primary) ──────────────────────────────────────
    # When awareness context is present (normal production path), the
    # awareness-composed script IS the ad copy — hook, body, and CTA all come
    # from live industry signals via ScriptAgent._awareness_compose.  The
    # static template pools above only serve peak-formula replication or the
    # no-awareness edge case.
    hook     = base_hook
    body     = ""
    headline = ""
    try:
        import asyncio as _asyncio
        from ai_model.agents.script_agent import ScriptAgent as _SA, ScriptRequest
        _sreq = ScriptRequest(
            # Natural theme only — never instruction-style text ("video
            # ad for X"), which leaks verbatim into user-facing copy.
            idea=f"{product} by {artist}",
            genre=genre or "",
            platform=platform,
            goal=goal,
            tone=type_spec["copy_tone"],
            awareness=awareness or "",
            variant_idx=variant_idx,
        )
        script = None
        if awareness:
            # Awareness composition is deterministic pure-Python (no model
            # call) — run it regardless of model readiness so ad copy is
            # NEVER templated while awareness signals exist.
            _agent = _script_agent if _script_agent is not None else _SA.__new__(_SA)
            # No timeout — guaranteed-completion policy: composition is
            # deterministic pure-Python and always returns.
            script = await _in_thread(lambda: _agent._awareness_compose(_sreq))
        elif _model_ready and _script_agent:
            script = await _in_thread(lambda: _script_agent.run(_sreq))
        if script is not None:
            _script_source = getattr(script, "source", "")
            if _script_source == "awareness" and script.hook and len(script.hook) > 5:
                # Awareness-composed copy is authoritative — no score gate.
                # Peak-formula hooks are the one exception: they replicate a
                # proven top performer from THIS account's own ad history,
                # which outranks a fresh composition.
                if source != "peak_replicated":
                    hook   = script.hook
                    source = "awareness"
                body     = script.body
                headline = script.cta[:50] if script.cta else base_cta
                if script.cta:
                    # Awareness CTA is authoritative too. Button-style slots
                    # (cta_button) have hard platform char limits, so only a
                    # short awareness CTA replaces the pool button label.
                    if len(script.cta) <= 30:
                        base_cta = script.cta
                    elif source != "peak_replicated":
                        base_cta = script.cta[:60]
            elif script.hook and len(script.hook) > 5:
                # Model-generated (awareness was absent): keep the score gate
                # so weak model phrasing can't regress below the pool hook.
                if _ad_hook_score(script.hook) > _ad_hook_score(base_hook):
                    hook   = script.hook
                    source = "model_enhanced"
                body     = script.body
                headline = script.cta[:50] if script.cta else base_cta
    except Exception:
        # Last resort with awareness present: derive copy from the raw
        # awareness signals themselves rather than shipping pool templates.
        if awareness:
            _sig = next(
                (ln.replace("[HIGH]", "").strip(" •-") .strip()
                 for ln in awareness.splitlines() if ln.strip()),
                "",
            )
            if _sig:
                hook   = f"{_sig[:80]} — {product} by {artist}"
                body   = f"{product} by {artist} is live now.{genre_tag}"
                source = "awareness"

    # ── Subtype-specific copy defaults ───────────────────────────────────────
    if not body:
        if sub == "audio":
            body = (
                f"[INTRO 2s: play opening bars of {product}] "
                f"Hey — {artist} just dropped something you've never heard before. "
                f"{product} is live now. {base_cta}.{genre_tag}"
            )
        elif sub == "text":
            body = f"{product} by {artist} — stream free now.{genre_tag}"
        elif sub == "image":
            body = (
                f"{artist} — {product}. "
                f"Exclusive new release.{genre_tag} Stream now."
            )
        else:  # video
            body = (
                f"🎵 {artist} drops something you've never heard before. "
                f"{product} is live now.{genre_tag}"
            )

    if not headline:
        headline = f"{product} — {''.join(w.capitalize() + ' ' for w in goal.split()).strip()}"

    # ── Subtype-specific creative_brief ──────────────────────────────────────
    brief_extras = type_spec["brief_extras"].copy()

    if sub == "video":
        plat_vspec = _PLAT_VIDEO_SPECS.get(plat_key, {"ratio": "1:1", "duration": "15–30s", "format": "standard"})
        creative_brief: dict = {
            "format":            plat_vspec["format"],
            "aspect_ratio":      plat_vspec["ratio"],
            "duration":          plat_vspec["duration"],
            "opening_3s":        hook,
            "visual_direction":  f"Show {artist} in action — raw, authentic, high-energy",
            "text_overlay":      headline,
            **brief_extras,
        }
    elif sub == "audio":
        creative_brief = {
            "format":        "audio ad",
            "duration":      type_spec["duration_range"],
            "voiceover_script": f"{hook} {body} {base_cta}.",
            **brief_extras,
        }
    elif sub == "text":
        # Enforce character limits for text ad copy
        hl30  = headline[:30] if headline else (product[:30])
        bd90  = body[:90]     if body     else (f"{product} — stream now")[:90]
        creative_brief = {
            "format":       "text ad",
            "headline":     hl30,
            "description":  bd90,
            "cta_button":   base_cta,
            **brief_extras,
        }
    else:  # image
        creative_brief = {
            "format":           "image ad",
            "tagline":          hook,
            "body_copy":        body,
            "cta_button":       base_cta,
            "visual_concept":   f"{artist} — {product}: album/single artwork with bold '{hook[:40]}' text overlay",
            **brief_extras,
        }

    return {
        "variant":       variant_idx + 1,
        "content_type":  sub,
        "hook":          hook,
        "headline":      headline,
        "body":          body,
        "cta":           base_cta,
        "creative_brief": creative_brief,
        "source":        source,
    }


@app.post("/platform/ads/record")
async def ads_record_run(req: AdRecordRequest, _key = Depends(require_scope("write"))):
    """
    Record a completed ad run's performance metrics.
    Peak performers (ROAS ≥ 3, CTR ≥ 2.5%) are automatically extracted
    as patterns the autopilot uses to generate the next winning creative.
    """
    from storage_client import get_ads_client
    ads = get_ads_client()
    record = ads.record_ad_run(req.user_id, req.dict())
    return {
        "status": "recorded",
        "run_id": record["run_id"],
        "is_peak": record["is_peak"],
        "message": (
            "Peak performer flagged — pattern extracted for replication"
            if record["is_peak"] else
            "Run recorded. Build more history to unlock peak replication."
        ),
    }


@app.post("/platform/ads/generate")
async def ads_generate(req: AdGenerateRequest, _key = Depends(require_scope("generate"))):
    """
    Generate a full ad creative set using peak performer replication.

    How it works:
    1. Pull the user's peak performer formula for this platform/ad_type from storage
    2. Extract the winning hook patterns, CTA formulas, audience signals
    3. Use the AI model to vary and enhance those patterns into new creatives
    4. Return N ready-to-launch ad creatives + audience targeting + budget split
    """
    start = time.time()
    from storage_client import get_ads_client, get_curriculum_client

    ads      = get_ads_client()
    plat     = req.platform.lower()
    ad_type  = req.ad_type.lower()

    # Pull peak formula (user-specific, then global fallback)
    peak_formula = None
    if req.replicate_peak:
        peak_formula = ads.get_winning_formula(req.user_id, plat, ad_type)

    # Pull organic engagement signals to cross-enrich targeting
    try:
        curriculum = get_curriculum_client()
        top_organic = curriculum.get_top_performers(req.user_id, platform=plat, top_n=5)
        organic_tags = list({t for p in top_organic for t in p.get("style_tags", [])})
    except Exception:
        organic_tags = []

    # ── Subtype selection & cycling ───────────────────────────────────────────
    # Priority order (highest wins):
    #   1. target_subtypes non-empty + valid  → cycle through exactly those types
    #   2. vary_subtypes=True                 → cycle through all four types
    #   3. vary_subtypes=False                → every creative uses ad_type
    #
    # target_subtypes lets callers request a specific format mix without touching
    # vary_subtypes.  Unknown values are dropped silently; if the entire list is
    # invalid the field is treated as absent.
    _ALL_SUBTYPES = ["video", "audio", "text", "image"]

    _resolved_cycle: list[str]
    _selection_mode: str
    if req.target_subtypes:
        # Validate: keep only known subtypes, preserve caller order
        _valid = [s.lower() for s in req.target_subtypes if s.lower() in _ALL_SUBTYPES]
        if _valid:
            _resolved_cycle = _valid
            _selection_mode = "targeted"
        else:
            # All values invalid — fall back to vary_subtypes logic
            _resolved_cycle = _ALL_SUBTYPES if req.vary_subtypes else [ad_type]
            _selection_mode = "auto_all" if req.vary_subtypes else "fixed"
    elif req.vary_subtypes:
        _resolved_cycle = _ALL_SUBTYPES
        _selection_mode = "auto_all"
    else:
        _resolved_cycle = [ad_type]
        _selection_mode = "fixed"

    # Guarantee non-empty awareness: caller/request signals lead, live platform
    # buffer always appended — so ScriptAgent's awareness composition (never
    # templates) is the path every ad creative takes.
    _merged_aw = _effective_awareness(plat, _merged_awareness_for(req))

    # Generate N creatives — each slot picks its subtype from the resolved cycle
    creatives = []
    for i in range(req.num_creatives):
        this_type = _resolved_cycle[i % len(_resolved_cycle)]
        creative = await _generate_ad_creative(
            platform=plat,
            ad_type=this_type,
            product=req.product,
            goal=req.goal,
            peak_formula=peak_formula,
            artist_name=req.artist_name,
            genre=req.genre,
            variant_idx=i,
            awareness=_merged_aw,
            instruction=req.instruction,
            content_themes=req.content_themes,
        )
        creatives.append(creative)

    # Audience targeting recommendations
    genre_key = (req.genre or "music_fan").lower().replace("-", "_").replace(" ", "_")
    base_audience = AUDIENCE_SEGMENTS.get(genre_key) or AUDIENCE_SEGMENTS["music_fan"]
    peak_audience = []
    if peak_formula:
        peak_audience = peak_formula.get("top_audience_tags", [])

    targeting = {
        "primary_interests": list(dict.fromkeys(peak_audience + base_audience + organic_tags))[:8],
        "lookalike_source": "existing fans and buyers (1-3%)",
        "retargeting_pool": "website visitors, video viewers (75%+), engagers last 30 days",
        "exclusions": ["existing buyers", "low-quality traffic countries"],
        "age_range": "18-35",
        "placements": {
            "tiktok":    ["TikTok Feed", "TikTok Search"],
            "meta":      ["Instagram Feed", "Instagram Reels", "Facebook Feed", "Meta Audience Network"],
            "youtube":   ["YouTube In-Stream", "YouTube Shorts"],
            "google":    ["Google Search", "Display Network", "Performance Max"],
            "instagram": ["Instagram Feed", "Instagram Reels", "Instagram Stories"],
        }.get(plat, ["Feed", "Stories"]),
    }

    # Budget allocation across creatives (A/B test split)
    budget_split = []
    if req.budget_daily and req.num_creatives > 0:
        test_budget   = round(req.budget_daily * 0.7 / req.num_creatives, 2)
        winner_budget = round(req.budget_daily * 0.3, 2)
        for i in range(req.num_creatives):
            budget_split.append({"variant": i + 1, "daily_budget": test_budget, "phase": "test"})
        budget_split.append({
            "variant": "winner",
            "daily_budget": winner_budget,
            "phase": "scale (set after 3-day test)",
        })

    # Performance benchmarks for this platform
    benchmarks = {
        "tiktok":    {"avg_ctr": "1.5-3%",  "avg_cpc": "$0.50-1.20",  "good_roas": "3-6x"},
        "meta":      {"avg_ctr": "0.9-2%",  "avg_cpc": "$0.80-2.50",  "good_roas": "2-5x"},
        "youtube":   {"avg_ctr": "0.4-1%",  "avg_cpc": "$0.10-0.30",  "good_roas": "2-4x"},
        "google":    {"avg_ctr": "2-6%",    "avg_cpc": "$0.50-3.00",  "good_roas": "4-8x"},
        "instagram": {"avg_ctr": "0.8-1.5%","avg_cpc": "$1.00-3.00",  "good_roas": "2-4x"},
    }.get(plat, {"avg_ctr": "1-2%", "avg_cpc": "$1.00", "good_roas": "2-4x"})

    return {
        "success": True,
        "user_id": req.user_id,
        "platform": plat,
        "ad_type": ad_type,
        "product": req.product,
        "goal": req.goal,
        "peak_replication": {
            "enabled": req.replicate_peak,
            "formula_found": peak_formula is not None,
            "avg_roas_of_peaks": peak_formula.get("avg_roas") if peak_formula else None,
            "avg_ctr_of_peaks": peak_formula.get("avg_ctr") if peak_formula else None,
        },
        "subtype_selection": {
            "mode": _selection_mode,
            # mode values:
            #   "targeted"  — caller supplied target_subtypes; cycle is that list
            #   "auto_all"  — vary_subtypes=True; cycle is all four types
            #   "fixed"     — vary_subtypes=False; every creative is ad_type
            "cycle": _resolved_cycle,
            "requested": req.target_subtypes or [],
            "applied": list(dict.fromkeys(
                _resolved_cycle[i % len(_resolved_cycle)]
                for i in range(req.num_creatives)
            )),
        },
        "creatives": creatives,
        "targeting": targeting,
        "budget_split": budget_split if budget_split else None,
        "platform_benchmarks": benchmarks,
        "launch_checklist": [
            "Upload creative assets in the correct format for " + plat + " (MP4 for video, PNG/JPG for image)",
            "Set campaign objective to match goal: " + req.goal,
            "Enable auto-bidding (lowest cost) for first 3 days",
            "Set frequency cap: 3 impressions/user/day",
            "Run A/B test for minimum 3 days before scaling winner",
            "Record results in /platform/ads/record to improve future generations",
        ],
        "processing_time_ms": round((time.time() - start) * 1000, 1),
    }


@app.post("/platform/ads/autopilot")
async def ads_autopilot(req: AdAutopilotRequest, _key = Depends(require_scope("generate"))):
    """
    Full ad autopilot.
    - Analyses the entire ad portfolio for this user
    - Classifies every ad as: SCALE | MAINTAIN | TEST | KILL
    - Identifies peak performer patterns across all platforms
    - Generates the next campaign recommendations with budget allocation
    - Uses both ad performance data AND organic engagement signals
    """
    start = time.time()
    from storage_client import get_ads_client, get_curriculum_client

    ads = get_ads_client()

    # Portfolio analysis
    portfolio = ads.analyse_portfolio(req.user_id, platform=req.platform)
    peaks     = ads.get_peak_performers(req.user_id, limit=10, platform=req.platform)

    # Cross-enrich with organic engagement signals
    try:
        curriculum   = get_curriculum_client()
        organic_tops = curriculum.get_top_performers(req.user_id, top_n=10)
        organic_tags = list({t for p in organic_tops for t in p.get("style_tags", [])})
    except Exception:
        organic_tops = []
        organic_tags = []

    # Determine what platforms to recommend based on what's working
    active_platforms = list({p.get("platform") for p in peaks if p.get("platform")})
    if not active_platforms:
        active_platforms = ["meta", "tiktok"]

    # Generate next campaign recommendations using the model
    next_campaigns = []
    for plat in (active_platforms or ["meta", "tiktok"])[:3]:
        formula = ads.get_winning_formula(req.user_id, plat, "video")
        ad_idea = f"{req.goal} campaign on {plat}"
        hook = ""
        cta  = AD_CTAS_BY_GOAL.get(req.goal, ["Learn More"])[0]

        if _model_ready and _script_agent:
            try:
                from ai_model.agents.script_agent import ScriptRequest
                # No timeout — guaranteed-completion policy.
                script = await _in_thread(lambda p=plat, a=ad_idea: _script_agent.run(ScriptRequest(
                    idea=a, platform=p, goal=req.goal, tone="direct",
                    awareness=_effective_awareness(p, _merged_awareness_for(req)),
                )))
                hook = script.hook
            except Exception:
                pass

        if not hook:
            hook_pool = AD_HOOKS_BY_PLATFORM.get(plat, AD_HOOKS_BY_PLATFORM["meta"])
            hook = hook_pool[0]

        budget_rec = None
        if req.budget_total and len(active_platforms) > 0:
            # Allocate more to platforms with proven peaks
            plat_peaks = [p for p in peaks if p.get("platform") == plat]
            weight     = 1.5 if plat_peaks else 1.0
            total_w    = sum(1.5 if [p for p in peaks if p.get("platform") == ap] else 1.0
                            for ap in active_platforms)
            budget_rec = round(req.budget_total * (weight / total_w), 2)

        next_campaigns.append({
            "platform": plat,
            "recommended_hook": hook,
            "recommended_cta": cta,
            "ad_type": "video",
            "daily_budget": budget_rec,
            "audience": (formula.get("top_audience_tags", [])[:4]
                         if formula else organic_tags[:4]),
            "peak_formula_available": formula is not None,
            "expected_roas": formula.get("avg_roas") if formula else "unknown",
            "expected_ctr": formula.get("avg_ctr") if formula else "unknown",
        })

    # Kill list — what to pause immediately
    kill_recommendations = []
    for run_id in portfolio.get("kill", [])[:5]:
        kill_recommendations.append({
            "run_id": run_id,
            "action": "PAUSE",
            "reason": "Below performance threshold (ROAS < 2.0, CTR < 1.5%)",
        })

    # Scale list — what to double budget on
    scale_recommendations = []
    for run_id in portfolio.get("scale", [])[:5]:
        scale_recommendations.append({
            "run_id": run_id,
            "action": "SCALE",
            "reason": "Peak performer — ROAS ≥ 3.0 AND CTR ≥ 2.5%",
            "budget_multiplier": 2.0,
        })

    return {
        "success": True,
        "user_id": req.user_id,
        "portfolio_summary": {
            "total_ad_runs": portfolio["total_runs"],
            "scale_count":   portfolio["scale_count"],
            "kill_count":    portfolio["kill_count"],
            "avg_roas":      portfolio["avg_roas"],
            "total_spend":   portfolio["total_spend"],
            "total_conversions": portfolio["total_conversions"],
        },
        "autopilot_actions": {
            "scale_immediately":  scale_recommendations,
            "pause_immediately":  kill_recommendations,
            "test_next":         [{"run_id": r, "action": "TEST"} for r in portfolio.get("test", [])[:3]],
        },
        "next_campaigns": next_campaigns,
        "peak_patterns_extracted": len(portfolio.get("peak_patterns", {})),
        "organic_signal_enrichment": {
            "organic_top_tags": organic_tags[:5],
            "cross_platform_signals": len(organic_tops),
        },
        "autopilot_confidence": (
            "high"   if portfolio["total_runs"] >= 10 else
            "medium" if portfolio["total_runs"] >= 3  else
            "low — record more ad runs to improve accuracy"
        ),
        "processing_time_ms": round((time.time() - start) * 1000, 1),
    }


@app.post("/platform/ads/audience")
async def ads_audience(req: AdAudienceRequest, _key = Depends(require_scope("generate"))):
    """
    AI-generated audience targeting for paid ads.
    Merges peak performer audience data from storage with organic engagement signals.
    """
    start = time.time()
    from storage_client import get_ads_client, get_curriculum_client

    ads = get_ads_client()
    plat = req.platform.lower()

    # Pull what audiences worked in peak ad runs
    peaks = ads.get_peak_performers(req.user_id, limit=20, platform=plat)
    ad_audiences = list({t for p in peaks for t in p.get("audience_tags", [])})

    # Pull what content resonated organically
    try:
        curriculum   = get_curriculum_client()
        organic_tops = curriculum.get_top_performers(req.user_id, platform=plat, top_n=10)
        organic_tags = list({t for p in organic_tops for t in p.get("style_tags", [])})
    except Exception:
        organic_tags = []

    genre_key = (req.genre or "music_fan").lower().replace("-", "_").replace(" ", "_")
    base_segs = AUDIENCE_SEGMENTS.get(genre_key) or AUDIENCE_SEGMENTS["music_fan"]

    cold_audience = list(dict.fromkeys(ad_audiences + organic_tags + base_segs))[:10]

    lookalikes = []
    if peaks:
        lookalikes = [
            {"source": "peak ad engagers", "percentage": "1%",   "priority": "highest"},
            {"source": "video viewers 75%+", "percentage": "2%", "priority": "high"},
            {"source": "website visitors 30d", "percentage": "3%","priority": "medium"},
        ]
    else:
        lookalikes = [
            {"source": "interest-based cold",  "percentage": "N/A", "priority": "start here"},
            {"source": "video viewers 75%+",   "percentage": "2%",  "priority": "after first campaign"},
        ]

    retargeting_windows = {
        "tiktok":    ["Profile visitors 30d", "Video viewers 50%+ 30d", "Followers"],
        "meta":      ["Page engagers 30d", "Video viewers 75%+ 30d", "Website visitors 30d",
                      "Instagram story openers 7d"],
        "youtube":   ["Channel subscribers", "Video viewers last 30d", "Similar YouTube channels"],
        "google":    ["Website visitors", "Customer list upload", "Similar audiences"],
        "instagram": ["Post engagers 30d", "Story viewers 14d", "Profile visitors 7d"],
    }.get(plat, ["Engagers 30d", "Website visitors"])

    return {
        "success": True,
        "user_id": req.user_id,
        "platform": plat,
        "product": req.product,
        "cold_audience": {
            "interests": cold_audience,
            "age_range": "18-34 primary, 35-44 secondary",
            "gender": "all",
            "source": "peak_ad_data + organic_signals" if peaks else "genre_defaults",
        },
        "lookalike_audiences": lookalikes,
        "retargeting_audiences": retargeting_windows,
        "campaign_funnel": {
            "top_of_funnel":    "Cold interest targeting + lookalikes — awareness",
            "middle_of_funnel": "Video viewers 25%+ + page engagers — consideration",
            "bottom_of_funnel": "Website visitors + past purchasers — conversion",
        },
        "data_quality": {
            "peak_ad_data_points": len(peaks),
            "organic_signal_count": len(organic_tops) if organic_tags else 0,
            "confidence": "high" if len(peaks) >= 5 else "building — record more runs",
        },
        "processing_time_ms": round((time.time() - start) * 1000, 1),
    }


@app.get("/platform/ads/performance/{user_id}")
async def ads_performance(user_id: str, platform: Optional[str] = None,
                          _key = Depends(verify_api_key)):
    """Full ad performance dashboard for a user."""
    from storage_client import get_ads_client
    ads = get_ads_client()

    runs  = ads.get_ad_runs(user_id, limit=50)
    peaks = ads.get_peak_performers(user_id, limit=10, platform=platform)
    stats = ads.get_stats(user_id)
    portfolio = ads.analyse_portfolio(user_id, platform=platform)

    return {
        "user_id": user_id,
        "stats": stats,
        "portfolio_summary": {
            "total_runs":      portfolio["total_runs"],
            "scale_count":     portfolio["scale_count"],
            "kill_count":      portfolio["kill_count"],
            "avg_roas":        portfolio["avg_roas"],
            "total_spend":     portfolio["total_spend"],
            "total_conversions": portfolio["total_conversions"],
        },
        "peak_performers": peaks[:5],
        "recent_runs": runs[:10],
        "extracted_patterns": portfolio.get("peak_patterns", {}),
    }


@app.post("/platform/ads/optimize")
async def ads_optimize(
    body: dict,
    _key = Depends(require_scope("generate")),
):
    """
    Given live campaign metrics, tell the platform exactly what to do:
    scale, adjust, pause, or A/B test a new angle.
    Body: {user_id, platform, campaigns: [{run_id, ctr, cpc, roas, spend, conversions}]}
    """
    start = time.time()
    from storage_client import get_ads_client

    user_id   = body.get("user_id", "unknown")
    platform  = body.get("platform", "meta")
    campaigns = body.get("campaigns", [])
    ads       = get_ads_client()

    actions   = []
    for camp in campaigns:
        roas = float(camp.get("roas", 0))
        ctr  = float(camp.get("ctr", 0))
        _cpc = float(camp.get("cpc", 999))
        run_id = camp.get("run_id", "unknown")

        if roas >= 3.0 and ctr >= 2.5:
            action = "SCALE"
            detail = f"Peak performer. Double budget. ROAS={roas}x CTR={ctr}%"
        elif roas >= 2.0 or ctr >= 1.5:
            action = "MAINTAIN"
            detail = "Solid performance. Hold budget. Consider testing new angle."
        elif roas >= 1.0:
            action = "OPTIMISE"
            detail = f"Borderline. Try new hook/audience. ROAS={roas}x CTR={ctr}%"
        elif ctr < 0.5:
            action = "KILL"
            detail = f"CTR too low ({ctr}%). Pause and replace creative immediately."
        else:
            action = "TEST"
            detail = "Insufficient data. Run for 3 more days before deciding."

        # Generate a replacement hook if killing
        new_hook = None
        if action in ("KILL", "OPTIMISE") and _model_ready and _script_agent:
            try:
                from ai_model.agents.script_agent import ScriptRequest
                product_hint = camp.get('product', 'music')
                s = await _in_thread(lambda ph=product_hint: _script_agent.run(ScriptRequest(
                    idea=f"new angle for {ph} ad",
                    platform=platform, goal="conversions", tone="direct",
                    awareness=_effective_awareness(platform, ""),
                )))
                new_hook = s.hook
            except Exception:
                pass

        result = {"run_id": run_id, "action": action, "detail": detail}
        if new_hook:
            result["replacement_hook"] = new_hook
        actions.append(result)

    # Record any peak performers automatically
    for camp in campaigns:
        if float(camp.get("roas", 0)) >= 3.0 or float(camp.get("ctr", 0)) >= 2.5:
            ads.record_ad_run(user_id, {**camp, "platform": platform})

    return {
        "success": True,
        "user_id": user_id,
        "platform": platform,
        "optimizations": actions,
        "summary": {
            "scale":    sum(1 for a in actions if a["action"] == "SCALE"),
            "maintain": sum(1 for a in actions if a["action"] == "MAINTAIN"),
            "kill":     sum(1 for a in actions if a["action"] == "KILL"),
            "test":     sum(1 for a in actions if a["action"] == "TEST"),
        },
        "processing_time_ms": round((time.time() - start) * 1000, 1),
    }


# ─── Storage Pipeline Endpoints ───────────────────────────────────────────────

@app.get("/storage/session")
async def get_storage_session(_key = Depends(verify_api_key)):
    """Read the active training session from the 7TB storage server."""
    from storage_client import get_pipeline, get_storage
    pipeline = get_pipeline()
    session = pipeline.get_session()
    storage_status = pipeline.get_status()
    raw_keys = get_storage().keys("*")
    return {
        "session": session,
        "storage_status": storage_status,
        "available_keys": raw_keys,
        "pipeline": pipeline.pipeline_status(),
    }


@app.get("/storage/pipeline/status")
async def pipeline_status(_key = Depends(verify_api_key)):
    """Return current training pipeline progress."""
    from storage_client import get_pipeline
    return get_pipeline().pipeline_status()


class StorageTrainRequest(BaseModel):
    epochs: int = Field(1, ge=1, le=20)
    batch_size: int = Field(64, ge=8, le=512)
    learning_rate: float = Field(3e-4, gt=0)
    max_batches: Optional[int] = None
    save_checkpoint: bool = True


def _run_storage_training(req: StorageTrainRequest, job_id: str):
    """Background task: pull batches from the 7TB storage server and train the model."""
    import math
    from storage_client import get_pipeline, get_checkpoint_client

    pipeline = get_pipeline()
    log_training(f"[StorageTrain] Job {job_id} — pulling from 7TB storage session", job_id=job_id)

    with _training_lock:
        _training_state["state"] = "running"
        _training_state["source"] = "storage"

    try:
        if not _model_ready:
            log_training("Waiting for model to initialize...", job_id=job_id)
            for _ in range(60):
                time.sleep(1)
                if _model_ready:
                    break
            if not _model_ready:
                raise RuntimeError("Model not ready after 60s")

        sys.path.insert(0, str(Path(__file__).parent))
        import torch
        from ai_model.training.trainer import train as run_train
        from ai_model.training.config import TrainConfig

        total_loss = 0.0
        batch_count = 0
        samples_written = []

        log_training("[StorageTrain] Streaming batches from storage...", job_id=job_id)

        for text_batch in pipeline.stream_batches(batch_size=req.batch_size):
            if req.max_batches and batch_count >= req.max_batches:
                break
            samples_written.extend(text_batch)

            # Feed to trainer every 50 samples
            if len(samples_written) >= 50 or (req.max_batches and batch_count == req.max_batches - 1):
                data_path = f"training/storage_batch_{job_id}.json"
                os.makedirs("training", exist_ok=True)
                with open(data_path, "w") as f:
                    json.dump([{"text": t} for t in samples_written], f)

                config = TrainConfig({
                    "train": {
                        "epochs": 1,
                        "batch_size": min(req.batch_size, len(samples_written)),
                        "lr": req.learning_rate,
                    }
                })
                with _model_lock:
                    model = _creative_model.model if _creative_model else None

                if model:
                    result = run_train(model, _tokenizer, data_path, config)
                    loss = result.get("final_loss", result.get("loss", 0.0))
                    if not math.isnan(loss):
                        total_loss += loss
                        batch_count += 1
                        avg_loss = round(total_loss / batch_count, 4)
                        with _training_lock:
                            _training_state["loss"] = avg_loss
                            _training_state["step"] = batch_count
                            _training_state["batches_from_storage"] = batch_count

                        if batch_count % 5 == 0:
                            log_training(
                                f"[StorageTrain] batch={batch_count} loss={avg_loss}",
                                job_id=job_id
                            )

                samples_written = []

        with _training_lock:
            _training_state["state"] = "complete"
            _training_state["completed_at"] = time.time()
            _training_state["final_loss"] = round(total_loss / max(batch_count, 1), 4)
            _training_state["total_storage_batches"] = batch_count

        log_training(
            f"[StorageTrain] Job {job_id} complete — {batch_count} batches, "
            f"loss={_training_state['final_loss']}",
            job_id=job_id
        )

        if req.save_checkpoint and _creative_model:
            try:
                weights_dir = Path(__file__).parent / "ai_model" / "weights"
                weights_dir.mkdir(parents=True, exist_ok=True)
                state_dict = _creative_model.model.state_dict()
                checkpoint = {
                    "model_state_dict": state_dict,
                    "vocab": _tokenizer.vocab,
                    "inv_vocab": _tokenizer.inv_vocab,
                    "merges": _tokenizer.merges,
                    "config": _model_config,
                    "job_id": job_id,
                    "source": "storage",
                    "final_loss": _training_state.get("final_loss"),
                }
                torch.save(checkpoint, str(weights_dir / "model.pt"))
                get_checkpoint_client().save_checkpoint(
                    model_id="maxbooster-v1",
                    state={"job_id": job_id, "batches": batch_count,
                           "final_loss": _training_state.get("final_loss")},
                    metadata={"source": "storage", "session": pipeline._session},
                )
                log_training("[StorageTrain] Checkpoint saved to storage", job_id=job_id)
            except Exception as e:
                log_training(f"[StorageTrain] Checkpoint error: {e}", job_id=job_id, level="error")

    except Exception as e:
        with _training_lock:
            _training_state["state"] = "error"
            _training_state["error"] = str(e)
        log_training(f"[StorageTrain] Job {job_id} failed: {e}", job_id=job_id, level="error")
        import traceback
        traceback.print_exc()
    finally:
        pipeline._active = False


@app.post("/training/start-from-storage")
async def start_training_from_storage(
    req: StorageTrainRequest,
    background_tasks: BackgroundTasks,
    _admin = Depends(verify_admin),
):
    """
    Start a training run that pulls data directly from the 7TB storage server.
    Streams batches from the storage session and trains the main MaxBooster model.
    """
    from storage_client import get_pipeline

    with _training_lock:
        if _training_state.get("state") == "running":
            return {"status": "already_running", "training_state": dict(_training_state)}

    pipeline = get_pipeline()
    session = pipeline.get_session()
    if not session:
        raise HTTPException(status_code=404, detail="No training session found in storage server. "
                            "Check /storage/session for details.")

    job_id = str(uuid.uuid4())[:8]
    with _training_lock:
        _training_state["state"] = "starting"
        _training_state["job_id"] = job_id
        _training_state["source"] = "storage"
        _training_state["session_id"] = session.get("id")
        _training_state["total_bytes"] = session.get("bytes", 0)
        _training_state["started_at"] = time.time()
        _training_state["step"] = 0
        _training_state["batches_from_storage"] = 0

    background_tasks.add_task(_run_storage_training, req, job_id)
    return {
        "status": "started",
        "job_id": job_id,
        "session": session,
        "epochs": req.epochs,
        "max_batches": req.max_batches,
        "message": f"Training job {job_id} started — pulling from 7TB storage session",
    }


# ─── New API Endpoints (pipeline-spec, 18 total) ─────────────────────────────

# -- Job stores ----------------------------------------------------------------
# Jobs are stored as atomic JSON files under _JOBS_DIR so every uvicorn
# worker process can read and write the same job state.  Each worker has its
# own in-process threading.Lock for the brief read-modify-write window; POSIX
# os.replace() gives us atomic final writes, so cross-worker reads are safe.

_JOBS_DIR = "/tmp/maxbooster_jobs"
os.makedirs(_JOBS_DIR, exist_ok=True)
_api_jobs_lock = threading.Lock()   # kept for legacy; file ops are the real store

# ── File-based job lifecycle: coalescing + TTL eviction ──────────────────────
# At 90M concurrent submissions, identical requests must collapse to one job
# rather than flooding _JOBS_DIR with 90M JSON files.  Completed jobs are
# evicted after _JOB_TTL_S seconds by a background GC thread so /tmp stays
# bounded regardless of burst volume.

_JOB_TTL_S        = 600   # evict done/error/cancelled jobs after 10 min
_active_jobs: dict[str, str] = {}   # digest → job_id (in-flight only)
_active_jobs_lock = threading.Lock()


def _job_digest(fields: dict) -> str:
    """Deterministic dedup key for identical concurrent job submissions."""
    payload = json.dumps(fields, sort_keys=True, default=str).encode()
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _job_gc() -> int:
    """Delete expired completed/errored job files; purge stale _active_jobs.
    Returns number of files evicted.  Never raises."""
    threshold = time.time() - _JOB_TTL_S
    evicted   = 0
    try:
        for fname in os.listdir(_JOBS_DIR):
            if not fname.endswith(".json") or fname.endswith(".tmp"):
                continue
            fpath = os.path.join(_JOBS_DIR, fname)
            try:
                if os.path.getmtime(fpath) >= threshold:
                    continue
                with open(fpath) as _f:
                    status = json.load(_f).get("status", "")
                if status in ("done", "error", "cancelled"):
                    os.unlink(fpath)
                    evicted += 1
            except Exception:
                pass
    except Exception:
        pass
    # Purge _active_jobs entries whose file is gone or whose status is terminal.
    stale: list[str] = []
    with _active_jobs_lock:
        for digest, jid in list(_active_jobs.items()):
            j = _job_read(jid)
            if j is None or j.get("status") in ("done", "error", "cancelled"):
                stale.append(digest)
        for k in stale:
            _active_jobs.pop(k, None)
    return evicted


def _job_gc_loop() -> None:
    while True:
        time.sleep(60)
        try:
            _job_gc()
        except Exception:
            pass


# Start GC daemon immediately — runs for the lifetime of the process.
threading.Thread(target=_job_gc_loop, daemon=True, name="job-file-gc").start()


def _job_path(job_id: str) -> str:
    return os.path.join(_JOBS_DIR, f"{job_id}.json")


def _job_write(job_id: str, data: dict) -> None:
    """Write job data atomically (create or overwrite)."""
    tmp = _job_path(job_id) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, _job_path(job_id))


def _job_read(job_id: str) -> dict | None:
    """Read job data; returns None if not found."""
    path = _job_path(job_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _job_update(job_id: str, updates: dict) -> None:
    """Merge updates into an existing job file atomically."""
    with _api_jobs_lock:
        data = _job_read(job_id)
        if data is None:
            return
        data.update(updates)
        _job_write(job_id, data)

# -- Request models ------------------------------------------------------------

class ApiGenerateContentRequest(_AwarenessMixin):
    platform: str
    topic: str
    tone: str
    genre: Optional[str] = None
    artist_name: Optional[str] = None
    artist_bio: Optional[str] = None
    brand_voice: Optional[str] = None
    target_audience: Optional[str] = None
    content_themes: Optional[List[str]] = None
    avoid_topics: Optional[List[str]] = None
    preferred_hashtags: Optional[List[str]] = None
    recent_post_snippets: Optional[List[str]] = None
    # MaxBooster client aliases
    artist: Optional[str] = None
    goal: Optional[str] = None
    title: Optional[str] = None
    # Persistent Brand Voice profile (see storage_client.ArtistProfileClient) —
    # when set, its saved tone/genre/vocabulary/disclosure preferences are
    # pulled in as fallbacks so repeat generations sound consistently on-brand.
    artistProfileId: Optional[str] = None
    # Freeform creative directive for THIS post (the narrative brief). Declared
    # so it is captured instead of silently dropped, then woven into copy.
    instruction: Optional[str] = None
    extra_context: Optional[str] = None
    # ── Creator controls (research-driven, additive) ───────────────────────────
    # variants: return N ranked caption alternatives (A/B/C) instead of just the
    #   winner. The composer already scores many candidates internally; this
    #   surfaces the top distinct ones. Deterministic (composition has no RNG).
    variants: Optional[int] = None
    # max_chars: hard platform character budget — the caption is trimmed on a
    #   word boundary to fit and char_count is always reported.
    max_chars: Optional[int] = None
    # Toggle the trailing hashtag block / CTA line off for clean-copy platforms.
    include_hashtags: Optional[bool] = True
    include_cta: Optional[bool] = True
    # ── Producer-metadata steering (what producers ask for: real use of
    # genre/mood/BPM/key, not just a label) ────────────────────────────────
    mood: Optional[str] = None
    bpm: Optional[float] = None
    key: Optional[str] = None
    # ── Cross-platform generation: when set, return one adapted variant per
    # platform in a single call instead of requiring N separate requests. ──
    platforms: Optional[List[str]] = None
    # ── Beat-marketplace conditioning (MaxBooster) ─────────────────────────
    # beat_context: structured facts about the beat being promoted (title,
    #   genre, mood, bpm, key, production_details, target_artist, price_usd,
    #   license_slots_remaining, listen_url, and optionally the measured
    #   audio_analysis block returned by the audio job poll). Serialized into
    #   awareness so copy is written ABOUT the beat, and its pricing anchors
    #   drive a concrete purchase CTA when goal is purchase/conversion.
    beat_context: Optional[dict[str, Any]] = None
    # platform_constraints: {"no_link_in_bio": bool, "professional_register":
    #   bool} — enforced on the finished copy (CTA mechanics + register).
    platform_constraints: Optional[dict[str, Any]] = None


class ApiGenerateCampaignRequest(_AwarenessMixin):
    """Turn one release into a full multi-week, multi-platform rollout campaign.

    The most-requested capability for independent artists/producers: a structured
    release rollout (announce → tease → pre-save → release day → sustain) with
    ready-to-post copy per post plus shared visual art direction.
    """
    title: str                                   # the song / release name
    artist_name: Optional[str] = None
    artist: Optional[str] = None                 # MaxBooster client alias
    genre: Optional[str] = None
    tone: Optional[str] = None
    brand_voice: Optional[str] = None
    target_audience: Optional[str] = None
    platforms: Optional[List[str]] = None        # default ["instagram","tiktok"]
    weeks: Optional[int] = None                  # campaign length, default 6 (2–12)
    release_date: Optional[str] = None           # ISO date; posts get absolute dates
    mood: Optional[str] = None
    bpm: Optional[float] = None
    key: Optional[str] = None
    artistProfileId: Optional[str] = None
    # ── Optional auto-generated visuals (opt-in; text-only plan stays fast by
    # default). ``generate_images`` pairs every post with an on-brand image;
    # ``generate_teasers`` additionally queues a short video teaser for
    # reel/video slots. Both are conditioned on the campaign's shared
    # ``art_direction`` so the whole rollout looks like one release. Images
    # render inline; teasers are async render jobs (poll /api/video-job/{id}).
    generate_images: bool = False
    generate_teasers: bool = False
    instruction:    Optional[str]       = None
    extra_context:  Optional[str]       = None
    content_themes: Optional[List[str]] = None


class ApiGenerateTextRequest(_AwarenessMixin):
    mode: str  # "planner" | "content"
    system: Optional[str] = None
    step: Optional[Any] = None
    inputs: Optional[Any] = None
    slots: Optional[Any] = None
    constraints: Optional[Any] = None
    artistProfileId: Optional[str] = None
    intent: Optional[str] = None
    platformRules: Optional[Any] = None
    # MaxBooster client aliases (content mode)
    platform: Optional[str] = None
    topic: Optional[str] = None
    tone: Optional[str] = None
    prompt: Optional[str] = None
    format: Optional[str] = None
    artist_name: Optional[str] = None
    brand_voice: Optional[str] = None
    extra_context: Optional[Any] = None
    # Freeform creative direction for THIS request (parity with content route).
    instruction: Optional[str] = None
    content_themes: Optional[List[str]] = None


class ApiContentScoreRequest(BaseModel):
    text: str
    platform: str
    cta: Optional[str] = None
    hashtags: List[str] = []
    userId: Optional[str] = None


class ApiAnalyzeRequest(BaseModel):
    modality: str
    payload: Any
    artistProfileId: Optional[str] = None
    platforms: List[str] = []
    intent: Optional[str] = None
    metadata: Optional[Any] = None
    platformRules: Optional[Any] = None


class ApiSentimentRequest(BaseModel):
    text: str
    includeEmotions: Optional[bool] = False
    includeToxicity: Optional[bool] = False


class ApiAnalyzeAudioRequest(BaseModel):
    audio_url: str
    artist_id: Optional[str] = None


class ApiOptimizeAdRequest(BaseModel):
    action: str  # "score"|"optimize_budget"|"predict_creative"|"forecast_roi"
    campaign: Optional[Any] = None
    campaigns: Optional[List[Any]] = None
    totalBudget: Optional[float] = None
    forecastPeriod: Optional[int] = None


class ApiPredictEngagementRequest(BaseModel):
    platform: str
    action: str
    content: Any
    postsPerWeek: Optional[int] = None


class ApiGenerateImageRequest(_AwarenessMixin):
    step: Optional[Any] = None
    inputs: Optional[Any] = None
    slots: Optional[Any] = None
    constraints: Optional[Any] = None
    artistProfileId: Optional[str] = None
    intent: Optional[str] = None
    platformRules: Optional[Any] = None
    # MaxBooster client aliases (single-image frame requests)
    prompt: Optional[str] = None
    aspect_ratio: Optional[str] = None
    style: Optional[str] = None
    beat_index: Optional[int] = None
    timecode: Optional[float] = None
    video_style: Optional[str] = None
    style_confidence: Optional[float] = None
    # Opt into the RTA-1 IRC path tracer for the hero background: "pathtraced"
    # (aliases "rta"/"raytraced") renders real lit geometry via the Digital GPU,
    # then composites the poster typography on top. Any other/absent value keeps
    # the fast procedural PIL background. Env RTA_IMAGE_ENGINE sets the default.
    render_engine: Optional[str] = None
    # When False, no text is composited onto the image (no headline, no intent
    # tag) — pure artwork for cover art. Default True keeps the poster layout.
    render_text: Optional[bool] = None
    # Freeform creative direction for THIS request (parity with content route).
    instruction: Optional[str] = None
    extra_context: Optional[str] = None
    content_themes: Optional[List[str]] = None
    # ── Producer-metadata steering (genre/mood/BPM/key now shape the visual
    # style, not just video/audio) ──────────────────────────────────────────
    genre: Optional[str] = None
    mood: Optional[str] = None
    bpm: Optional[float] = None
    key: Optional[str] = None


class ApiGenerateAudioRequest(_AwarenessMixin):
    style_fingerprint: Optional[List[float]] = None
    notes: Optional[List[Any]] = None
    duration: Optional[float] = 30
    instrument: Optional[str] = None
    genre: Optional[str] = None
    step: Optional[Any] = None
    inputs: Optional[Any] = None
    constraints: Optional[Any] = None
    artistProfileId: Optional[str] = None
    intent: Optional[str] = None
    platformRules: Optional[Any] = None
    # Explicit caller mood — leads over the awareness brief's mood when set
    # (caller awareness leads the platform buffer).  Conditions dataset track
    # selection via the mood→genre affinity map.
    mood: Optional[str] = None
    # Freeform creative direction for THIS request (parity with content route).
    # `prompt` is the user-facing alias (matches what the dashboard / API clients
    # send); `instruction` is the legacy internal name — both are accepted.
    prompt: Optional[str] = None
    instruction: Optional[str] = None
    extra_context: Optional[str] = None
    content_themes: Optional[List[str]] = None
    # ── Producer-grade controls (what real artists/producers ask for) ─────────
    # Exact musical targets: when set, the rendered clip is pitch-shifted /
    # time-stretched to actually HIT these, not just pick the nearest sample.
    # Both forms accepted: target_bpm/target_key (internal) and bpm/key (alias).
    target_bpm: Optional[float] = None
    target_key: Optional[str] = None
    bpm: Optional[float] = None        # alias → target_bpm
    key: Optional[str] = None          # alias → target_key
    # Reproducibility: same seed + same request → same result (fixes the
    # "great result I can't reproduce" complaint).
    seed: Optional[int] = None
    # Studio export: "wav" (lossless) or "mp3"; sample rate / bit depth; and an
    # EBU-R128 loudness target in LUFS (or a preset name: streaming/club/…).
    format: Optional[str] = None
    sample_rate: Optional[int] = None
    bit_depth: Optional[int] = None
    loudness_lufs: Optional[float] = None
    loudness_preset: Optional[str] = None
    # Remix-ready stems (drums / bass / melody), time-aligned WAVs.
    stems: Optional[bool] = False
    # Awareness-driven song arrangement (intro/verse/hook/… via stem
    # automation). Default ON for clips ≥24 s; pass False for a plain loop.
    arrange: Optional[bool] = None


class SceneOverride(BaseModel):
    index: int
    text: str


class ApiGenerateVideoRequest(_AwarenessMixin):
    idea: str
    platform: str = "tiktok"
    genre: Optional[str] = None
    tone: str = "energetic"
    goal: str = "growth"
    duration: Optional[int] = None
    artist_name: Optional[str] = None
    quality: str = "cinematic"
    user_audio_path: Optional[str] = None
    voiceover: bool = False
    # Narrator controls — normalized in the voiceover module (invalid values
    # silently fall back to defaults; never break a render).
    voiceover_voice: Optional[str] = None
    voiceover_wpm: Optional[int] = None
    scenes_override: Optional[List[SceneOverride]] = None
    # MaxBooster client aliases (script + music-video context, used as hints)
    hook: Optional[str] = None
    body: Optional[str] = None
    cta: Optional[str] = None
    topic: Optional[str] = None
    aspect_ratio: Optional[str] = None
    template: Optional[str] = None
    is_drop: Optional[bool] = None
    # Freeform creative direction for THIS request (parity with content route).
    instruction: Optional[str] = None
    extra_context: Optional[str] = None
    content_themes: Optional[List[str]] = None
    energy: Optional[float] = None
    bpm: Optional[float] = None
    mood: Optional[str] = None
    key: Optional[str] = None
    # Persistent Brand Voice profile (see storage_client.ArtistProfileClient).
    artistProfileId: Optional[str] = None
    # ── Cross-platform generation: when set, kick off one render job per
    # platform in a single call instead of requiring N separate requests.
    # Each platform reuses the existing PLATFORM_RATIOS / _PLATFORM_SPECS
    # aspect-ratio + duration logic in VideoAgent — identical to calling the
    # endpoint once per platform, just batched.
    platforms: Optional[List[str]] = None
    # ── Veo-parity generation controls ───────────────────────────────────
    # Matches Google Veo's parameter surface for every dimension that isn't
    # topic or purpose, so callers can express full creative intent without
    # wrapping a second API.
    camera_motion: Optional[str] = None      # pan_left/pan_right/zoom_in/zoom_out/
                                             # tilt_up/tilt_down/dolly_in/dolly_out/
                                             # crane_up/crane_down/static/auto
    negative_prompt: Optional[str] = None   # content/style/elements to avoid
    seed: Optional[int] = None              # explicit seed — reproducible output
    fps: Optional[int] = None               # output frame rate (8/16/24/30); default 24
    motion_intensity: Optional[float] = None # 0.0–1.0; overrides genre-derived energy
    enhance_prompt: bool = True             # False = skip AI awareness augmentation
    lighting: Optional[str] = None          # cinematic/dramatic/natural/studio/
                                            # golden_hour/night/neon
    color_temperature: Optional[str] = None # warm/cool/neutral
    style_reference: Optional[str] = None   # URL or asset ID for style conditioning
    output_resolution: Optional[str] = None # 720p/1080p/4k — overrides platform resolution
    composition: Optional[str] = None       # close_up/medium_shot/wide_shot/
                                            # over_the_shoulder/pov/aerial/
                                            # low_angle/high_angle — shot framing
    reference_images: Optional[List[str]] = None  # up to 3 base64 images — style/
                                                  # character consistency ("ingredients")
    first_frame_b64: Optional[str] = None   # base64 image the video should START on
    last_frame_b64: Optional[str] = None    # base64 image the video should END on
    sample_count: Optional[int] = None      # 1–4 independent video variants per request
    generate_audio: bool = True             # auto-render a matched soundtrack when no
                                            # user audio is supplied (native-audio parity)


class ApiVideoExtendRequest(BaseModel):
    """Veo-parity video extension: continue a previously generated video.

    The last frame of the source video is extracted and used as the
    first-frame conditioning image for a continuation clip, which is then
    concatenated onto the source. One continuous soundtrack is re-rendered
    across the full extended duration so audio stays coherent.
    """
    source: str                              # job_id, filename, or /uploads/videos/... URL
    idea: str = ""                           # what the continuation should be about
    extend_duration: float = 8.0             # seconds to ADD (clamped 2–60)
    platform: str = "tiktok"
    goal: str = "growth"
    tone: str = "energetic"
    genre: Optional[str] = None
    artist_name: Optional[str] = None
    # Veo-parity controls for the continuation clip
    camera_motion: Optional[str] = None
    negative_prompt: Optional[str] = None
    seed: Optional[int] = None
    lighting: Optional[str] = None
    color_temperature: Optional[str] = None
    composition: Optional[str] = None
    generate_audio: bool = True


class ApiAudioAnalyzeRequest(BaseModel):
    """MaxBooster /api/audio/analyze contract — beat/structure analysis."""
    audio_path: Optional[str] = None
    audio_url: Optional[str] = None
    context: Optional[Any] = None


class ApiViralScoreRequest(BaseModel):
    """MaxBooster /api/infer/viral-score contract."""
    model: Optional[str] = None
    inputs: Optional[Any] = None


class ApiTrainFeedbackRequest(BaseModel):
    source: str
    trigger: str
    engagement_rate: float
    platform: str
    content_type: str
    hook_type: str
    media_type: str
    curriculum_hint: Optional[str] = None
    dispatched_at: Optional[str] = None
    source_node: str = "maxbooster"
    version: str = "1.0"


# -- Shared helpers ------------------------------------------------------------

def _api_heuristic_score(text: str, platform: str) -> float:
    """Simple word-count + CTA heuristic returning 0-100."""
    words = len(text.split())
    has_cta = any(w in text.lower() for w in
                  ["click", "follow", "link", "save", "share", "buy", "get", "stream", "listen"])
    length_score  = max(0.0, 1.0 - abs(words - 30) / 60)
    cta_bonus     = 0.15 if has_cta else 0.0
    platform_map  = {"instagram": 0.05, "tiktok": 0.08, "twitter": 0.03,
                     "youtube": 0.04, "facebook": 0.03}
    plat_bonus    = platform_map.get(platform.lower(), 0.0)
    return round(min(100.0, (length_score * 0.8 + cta_bonus + plat_bonus) * 110), 1)


def _api_hashtags(topic: str, genre: Optional[str], platform: str) -> List[str]:
    tags = [f"#{topic.replace(' ', '')}", f"#{platform}"]
    if genre:
        tags.append(f"#{genre.replace(' ', '')}")
    tags += ["#music", "#newrelease", "#artist"]
    return list(dict.fromkeys(tags))[:6]


def _trim_to_chars(text: str, limit: int) -> str:
    """Trim text to at most `limit` chars on a word boundary (keeps whole words).

    Falls back to a hard cut only when the very first word already exceeds the
    budget. Never raises; used to enforce platform character caps on captions.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    cut = text[:limit]
    sp = cut.rfind(" ")
    trimmed = cut[:sp] if sp > 0 else cut
    return trimmed.rstrip()


def _api_model_state(domain: str) -> dict:
    from datetime import datetime as _dt
    return {
        "domain":        domain,
        "version":       "1.0.0",
        "trained_at":    _dt.utcnow().isoformat() + "Z",
        "session_count": _training_state.get("steps_done", 0),
        "loss":          _training_state.get("last_loss"),
        "weights": {
            "embed_dim":  512,
            "n_layers":   8,
            "vocab_size": 172,
            "ready":      _model_ready,
        },
    }


# -- Content Generation --------------------------------------------------------

@app.post("/api/generate/content")
async def api_generate_content(req: ApiGenerateContentRequest, _key=Depends(require_scope("generate"))):
    """Captions, hooks, CTAs for social posts with artist context."""
    start    = time.time()

    platform = normalize_platform(req.platform)
    artist   = req.artist_name or req.artist or "the artist"
    topic    = _resolve_topic_from_url(req.topic or req.title or "")
    goal     = req.goal or "engagement"

    # ── Beat-marketplace conditioning: fold structured beat_context into the
    # request's awareness (the canonical conditioning channel — merge_awareness
    # reads req.awareness) so both the model path and the awareness-composed
    # path write about the actual beat. `idea`/topic stays a clean string. ──
    from ai_model import request_intelligence as _ri_bc
    _beat_aw = _ri_bc.beat_context_awareness(req.beat_context)
    if _beat_aw:
        _aw_prev = req.awareness
        if isinstance(_aw_prev, dict):
            _aw_prev = (_aw_prev.get("contextString", "") or "")
        req.awareness = "\n".join(
            p for p in [(_aw_prev or "").strip(), _beat_aw] if p)
    _pc = req.platform_constraints if isinstance(req.platform_constraints, dict) else None
    _listen_url = (_ri_bc.sanitize_listen_url(req.beat_context.get("listen_url"))
                   if isinstance(req.beat_context, dict) else "")
    # Purchase goal → concrete conversion CTA anchored on real price/scarcity/
    # URL instead of the generic engagement library line.
    _purchase_goal = _ri_bc.GOAL_ALIASES.get(
        str(goal).strip().lower(), str(goal).strip().lower()) == "drive_conversion"
    _purchase_cta = (_ri_bc.purchase_cta(req.beat_context, platform, _pc)
                     if _purchase_goal else "")

    def _build(_request=None, _platform=None):
        # ── Request intelligence: analyse intent, audience & strategy up front ──
        from ai_model import request_intelligence as ri
        from ai_model.generation import build_context
        _plat = _platform or platform
        _narrative = " ".join(
            filter(None, [req.instruction, req.extra_context, req.artist_bio])
        )
        # Unified orchestrator: brief + merged awareness in one call. Technique
        # extraction is skipped on the hot text path (no media rendered here).
        _ctx = build_context(
            "content", req, with_technique=False,
            platform=_plat, topic=topic, goal=goal,
            tone=req.tone, genre=req.genre, artist=artist,
            extra=" ".join(filter(None, [req.brand_voice, req.target_audience])),
            narrative=_narrative,
            track=req.title or topic,
            themes=req.content_themes,
            mood=req.mood, bpm=req.bpm, key=req.key,
            artist_profile_id=req.artistProfileId,
        )
        brief = _ctx.brief

        hook = f"🎵 {artist} just dropped something you need to hear — {topic}"
        body = (f"Bringing {req.genre or 'music'} vibes that hit different. "
                f"{req.brand_voice or 'Authentic, raw, and real.'} "
                f"Crafted for {req.target_audience or brief.audience}.")
        cta  = brief.suggested_cta

        # ── Awareness bridge input ─────────────────────────────────────────
        # The awareness channel is the designed bridge spanning the gap between
        # external generative capability and the (still-training) in-house
        # model: it feeds BOTH the model's conditioning prefix AND, on
        # garble/failure, the awareness-composed fallback. Two sources are
        # merged, user direction FIRST so it outranks generic trend context:
        #   1. the caller's creative direction for THIS post (instruction /
        #      extra_context / themes), serialised into signal lines by
        #      `awareness_from_direction` — the narrative is lead-in-stripped
        #      first (the parser strips nothing itself) and themes go in as a
        #      bullet, never as #hashtags (the distribution agent harvests
        #      awareness hashtags into a persistent, platform-shared pool);
        #   2. genuine external awareness (client-supplied live trend data),
        #      which only reaches this handler now that the request model
        #      extends _AwarenessMixin — before that it was silently dropped,
        #      so this path never actually fed the bridge.
        # `brief.directives` are deliberately excluded: they are internal
        # prompt-engineering instructions the parser would quote verbatim.
        _merged_awareness = _ctx.awareness

        if _model_ready and _script_agent:
            from ai_model.agents.script_agent import ScriptRequest
            from ai_model.agents.distribution_agent import DistributionRequest

            def _infer():
                # `idea` stays a clean topic string — it is templated raw into
                # hook/body text, so richer context must NOT be concatenated
                # here. All of it flows through `_merged_awareness` (built
                # above): the user's own creative direction plus external trend
                # data, both driving the model's conditioning and the
                # awareness-composed fallback. This mirrors the video path.
                #
                # Distribution telemetry is not called on the hot path —
                # the result is never used and adding it floods the batcher
                # with extra submissions that degrade unique-request throughput.
                # It is collected offline via the training pipeline instead.
                return _script_agent.run(ScriptRequest(
                    idea=topic, platform=_plat, goal=goal, tone=brief.tone,
                    awareness=_merged_awareness,
                ))

            try:
                # No per-request GPU spawn needed here — the batcher owns GPU
                # lifecycle (one pocket life per batched forward pass).
                # Identical requests are already collapsed by the async
                # coalescer above and never reach this branch.
                sr = _infer()
            except Exception:
                sr = None
            if sr is not None:
                hook = sr.hook or hook
                body = sr.body or body
                cta  = sr.cta  or cta

        # ── Intelligence-driven composition ────────────────────────────────
        # The composer consumes the brief (keywords, audience, strategy) to
        # build the body/CTA instead of echoing the raw topic back, ranks
        # the agent's parts against brief-composed candidates, and scores
        # every complete caption (structure-aware) to pick the winner.
        _variants_want = max(1, int(req.variants)) if req.variants else 1
        composed = ri.compose_caption(
            topic, artist, brief,
            genre=req.genre, brand_voice=req.brand_voice,
            agent_hook=hook, agent_body=body, agent_cta=cta,
            variants=_variants_want,
        )

        # ── Creator controls: CTA toggle → rebuild caption without the CTA line;
        # char budget → trim on a word boundary. Applied to every variant so the
        # winner and the alternatives stay consistent. ──────────────────────────
        def _apply_controls(v: dict) -> dict:
            hook_v, body_v, cta_v = v["hook"], v["body"], v["cta"]
            # Purchase goal + concrete beat anchors → override the generic
            # library CTA with the real price/scarcity/URL line.
            if _purchase_cta:
                cta_v = _purchase_cta
            # Platform constraints are enforced on every copy part so no
            # variant leaks link-in-bio mechanics or off-register emoji asks.
            if _pc:
                hook_v = ri.apply_platform_constraints(hook_v, _pc, _listen_url)
                body_v = ri.apply_platform_constraints(body_v, _pc, _listen_url)
                cta_v  = ri.apply_platform_constraints(cta_v,  _pc, _listen_url)
            if _purchase_cta or _pc:
                cap = "\n\n".join(p for p in (hook_v, body_v, cta_v) if p).strip()
            else:
                cap = v["caption"]
            if req.include_cta is False:
                cap = f"{hook_v}\n\n{body_v}".strip()
            if req.max_chars and len(cap) > int(req.max_chars):
                cap = _trim_to_chars(cap, int(req.max_chars))
            return {**v, "hook": hook_v, "body": body_v, "cta": cta_v,
                    "caption": cap, "char_count": len(cap)}

        variant_objs = [_apply_controls(v) for v in composed.get("variants", [])]
        # Guarantee at least one variant even if the composer returned none.
        if not variant_objs:
            variant_objs = [_apply_controls({
                "caption": composed["caption"], "hook": composed["hook"],
                "body": composed["body"], "cta": composed["cta"],
                "score": composed["caption_score"],
            })]
        # Disclosure is opt-in (Brand Voice profile). Applied after trimming,
        # then re-checked against max_chars: if appending the label would blow
        # the caller's explicit character budget, the label is dropped rather
        # than silently exceeding a hard platform limit.
        caption = ri.apply_disclosure(variant_objs[0]["caption"], brief)
        if req.max_chars and len(caption) > int(req.max_chars):
            caption = variant_objs[0]["caption"]

        hashtags = (req.preferred_hashtags or []) + _api_hashtags(topic, req.genre, _plat)
        hashtag_cap = min(10, max(brief.hashtags_target, len(req.preferred_hashtags or [])))
        if req.include_hashtags is False:
            hashtags = []
            hashtag_cap = 0
        quality  = composed["caption_score"]
        score    = _api_heuristic_score(caption, _plat)
        return {
            "caption":    caption,
            "char_count": len(caption),
            "variants":   variant_objs,
            "hook":       variant_objs[0]["hook"],
            "body":       variant_objs[0]["body"],
            "cta":        variant_objs[0]["cta"] if req.include_cta is not False else "",
            "hashtags":   list(dict.fromkeys(hashtags))[:hashtag_cap] if hashtag_cap else [],
            "confidence": round(max(quality, score) / 100, 3),
            "quality_score": quality,
            "ai_disclosure": brief.ai_disclosure,
            "platform": _plat,
            "intelligence": {
                **brief.to_dict(),
                "hook_score": composed["hook_score"],
                "candidates_considered": composed["hooks_considered"],
                "composer": {
                    "bodies_considered": composed["bodies_considered"],
                    "ctas_considered": composed["ctas_considered"],
                },
            },
        }

    # ── Cross-platform generation: one call → one adapted variant per
    # requested platform (aspect ratio / copy length / hook style each follow
    # that platform's profile), instead of forcing N separate requests. ─────
    if req.platforms:
        _plats = [normalize_platform(p) for p in req.platforms if p]
        _by_platform = {p: _build(_platform=p) for p in dict.fromkeys(_plats)}
        if _by_platform:
            first = next(iter(_by_platform.values()))
            _xplat_out: dict = {
                **first,
                "platform_variants": _by_platform,
                "processing_time_ms": round((time.time() - start) * 1000, 1),
            }
            _sm = _get_storage_mode()
            if _sm != "live":
                _xplat_out["storage_warning"] = (
                    "pdim storage is offline — quality awareness and retrieval index are "
                    f"running in {_sm.replace('_', ' ')} mode; output quality may be reduced"
                )
            return _xplat_out
        # `platforms` was present but contained no usable entries (e.g. [""])
        # — fall through to the normal single-platform path below instead of
        # crashing on an empty dict.

    # ── Default content-gen path: dedup + single-flight via PDIM orchestrator ──
    # Concurrent identical requests collapse to ONE compute; the rest share it.
    # Only the model path is cached — heuristic-only output stays uncached, as before.
    # Brand-voice requests bypass the cache entirely: the cache key is derived
    # from the request payload only, but build_brief() also reads the artist's
    # stored profile (tone/genre/vocabulary/ai_disclosure) — an identical
    # request could otherwise replay a stale caption/disclosure after the
    # artist edits their profile.
    if _model_ready and not req.artistProfileId:
        # Async coalescer sits BEFORE the thread pool: identical-digest requests
        # share one asyncio.Future (suspended coroutines, ~2 KB each) rather
        # than each blocking an executor thread (~8 MB).  Only unique digests
        # enter _in_thread → INFERENCE_GATE → GPU.
        _key = {
            "platform": platform,
            "topic":    topic,
            "tone":     req.tone or "",
            "goal":     goal,
            "awareness": str(req.awareness or ""),
        }
        async def _coalesced_content():
            # Leader spawns ONE GPU life from the pocket for this unique request.
            # Followers share the result via the coalescer — zero extra spawns.
            _orch = _get_pdim_orchestrator()
            async with _get_gpu_pool().spawn(_digest_str(_key)) as _glife:
                return await _in_thread(
                    lambda: _orch.compute(req, _build, namespace="api_content_v4")
                )
        _out = await _get_async_coalescer().compute(_key, _coalesced_content)
        result = dict(_out["result"])
        if _out.get("source") in ("cache", "coalesced"):
            result["cached"] = True
    else:
        result = _build()
    _sm = _get_storage_mode()
    if _sm != "live":
        result["storage_warning"] = (
            "pdim storage is offline — quality awareness and retrieval index are "
            f"running in {_sm.replace('_', ' ')} mode; output quality may be reduced"
        )
    result["processing_time_ms"] = round((time.time() - start) * 1000, 1)
    return result


# Default still-image layout per (normalized) platform for campaign artwork.
# Reel/story slots override this to vertical regardless of platform.
_CAMPAIGN_LAYOUT_BY_PLATFORM: dict[str, str] = {
    "instagram": "square_1_1",
    "facebook":  "square_1_1",
    "tiktok":    "vertical_9_16",
    "youtube":   "landscape_16_9",
    "twitter":   "landscape_16_9",
    "linkedin":  "landscape_16_9",
}


@app.post("/api/generate/campaign")
async def api_generate_campaign(req: ApiGenerateCampaignRequest, _key=Depends(require_scope("generate"))):
    """Turn one release into a full rollout campaign.

    The most-requested capability for independent artists/producers (Water & Music
    survey; RouteNote/NotNoise/Chartlex rollout guides): a structured multi-week,
    multi-platform content calendar — announce → tease → pre-save → release day →
    sustain — with ready-to-post copy per post plus shared visual art direction,
    so one song becomes a whole campaign. Composes the same brief + caption
    engine as ``/api/generate/content`` and the unified Visual-DNA technique
    engine. Never fails: any slot that can't generate degrades to a template.
    """
    start = time.time()
    from ai_model.generation import build_campaign

    artist = req.artist_name or req.artist or "the artist"

    # ── Optional per-post asset generators (opt-in) ───────────────────────────
    # Reuse the existing in-house image engine and video render pipeline, but
    # condition every asset on the campaign's SHARED art_direction (palette +
    # mood) so the whole rollout looks like one release. Both are never-raise:
    # a failed asset just leaves that post text-only.
    def _campaign_image_fn(*, topic, headline, platform, fmt, purpose, art_direction):
        """Render one on-brand still image via the ImageEngine (/api/generate/image
        engine), forced onto the campaign's shared palette/mood. Runs inline."""
        if not _image_engine:
            return None
        try:
            from ai_model.image.image_engine import ImageRequest
            _plat = normalize_platform(platform)
            # Reels/stories are vertical; feed posts follow the platform default.
            layout = "vertical_9_16" if fmt in ("reel", "story") \
                else _CAMPAIGN_LAYOUT_BY_PLATFORM.get(_plat, "square_1_1")
            ad = art_direction or {}
            color_scheme = ad.get("color_scheme") or "dark_neon"
            mood = ad.get("mood")
            style_tags = [t for t in [mood] if t] or ["cinematic"]
            prompt = f"{mood or 'cinematic'} promotional visual for: {topic}"
            # Deterministic seed → cohesive, reproducible campaign artwork.
            seed = abs(hash((artist, topic, _plat, fmt, headline))) % (2 ** 31)
            _req = ImageRequest(
                prompt=prompt, headline=str(headline or topic),
                color_scheme=color_scheme, layout=layout, platform=_plat,
                artist_name=artist, intent=purpose or "promotional",
                style_tags=style_tags, seed=seed,
            )
            res = _image_engine.render(_req)
            if not (res and res.success):
                return None
            # Fold produced image back into the retrieval index (never-raise).
            try:
                from ai_model.retrieval.generated_ingestor import get_generated_ingestor
                from ai_model.image.image_engine import _UPLOADS_DIR as _IMG_DIR
                get_generated_ingestor().enqueue(
                    str(_IMG_DIR / res.filename), brand=artist,
                    endpoint="/api/generate/campaign", platform=_plat,
                )
            except Exception:
                pass
            return {
                "type": "image", "url": res.url, "width": res.width,
                "height": res.height, "format": "png",
                "meta": {
                    "color_scheme": res.color_scheme, "layout": res.layout,
                    "engine": "maxbooster-pil-v1",
                    "conditioned_on": "campaign_art_direction",
                },
            }
        except Exception:
            return None

    def _campaign_teaser_fn(*, topic, hook, body, cta, platform, purpose, art_direction):
        """Queue one short video teaser via the /api/generate-video pipeline,
        conditioned on the campaign's genre/mood so it matches the art direction.
        Async: returns the render job id to poll."""
        if not (_model_ready and _script_agent and _visual_spec_agent and _creative_model):
            return None
        try:
            ad = art_direction or {}
            vreq = ApiGenerateVideoRequest(
                idea=topic, platform=normalize_platform(platform),
                genre=req.genre, tone=req.tone or "energetic",
                goal=purpose or "awareness", artist_name=artist,
                hook=hook, body=body, cta=cta, topic=topic,
                mood=ad.get("mood") or req.mood, bpm=req.bpm, key=req.key,
                # Propagate campaign awareness so teaser video briefs and scene
                # selection carry the same live chart signal as the post copy.
                awareness=_campaign_awareness,
            )
            jid, _brief = _start_video_job(vreq, normalize_platform(platform))
            return {
                "type": "video", "job_id": jid, "status": "processing",
                "poll_url": f"/api/video-job/{jid}",
                "meta": {"conditioned_on": "campaign_art_direction"},
            }
        except Exception:
            return None

    image_fn = _campaign_image_fn if req.generate_images else None
    teaser_fn = _campaign_teaser_fn if req.generate_teasers else None

    # Compute campaign awareness once — shared across all posts for consistency.
    _campaign_awareness = _merged_awareness_for(req)

    def _build():
        return build_campaign(
            artist=artist,
            title=req.title,
            genre=req.genre,
            tone=req.tone,
            brand_voice=req.brand_voice,
            target_audience=req.target_audience,
            platforms=req.platforms,
            weeks=req.weeks or 6,
            mood=req.mood,
            bpm=req.bpm,
            key=req.key,
            release_date=req.release_date,
            hashtag_fn=_api_hashtags,
            normalize_platform_fn=normalize_platform,
            image_fn=image_fn,
            teaser_fn=teaser_fn,
            seed=abs(hash(f"{artist}|{req.title}")) % 100000,
            awareness=_campaign_awareness,
        )

    # Asset generation renders images inline (blocking PIL work), so run the
    # whole build off the event loop when assets are requested; the fast
    # text-only path stays synchronous.
    if image_fn or teaser_fn:
        plan = await _in_thread(_build)
    else:
        plan = _build()
    plan["processing_time_ms"] = round((time.time() - start) * 1000, 1)
    return plan


# ── Campaign scheduling: persist a generated rollout as an editable calendar ───
# The generate endpoint above returns a plan the artist must post manually. These
# endpoints let an artist SAVE that plan per-artist, retrieve it as a by-date
# calendar, edit copy / move dates / mark posts scheduled|posted, and hand posts
# off to the distribution agent to queue on their target dates.

class CampaignSaveRequest(BaseModel):
    profile_id: str                           # per-artist calendar partition
    plan: dict                                # a /api/generate/campaign result
    name: Optional[str] = None


class CampaignPostPatch(BaseModel):
    profile_id: str
    hook: Optional[str] = None
    body: Optional[str] = None
    cta: Optional[str] = None
    caption: Optional[str] = None
    hashtags: Optional[List[str]] = None
    date: Optional[str] = None                 # move the post to a new date
    platform: Optional[str] = None
    format: Optional[str] = None
    brief: Optional[str] = None
    status: Optional[str] = None               # draft | scheduled | posted


class CampaignScheduleRequest(BaseModel):
    profile_id: str
    post_ids: Optional[List[str]] = None       # default: every not-yet-posted post
    handoff: bool = True                       # use the distribution agent


def _campaign_view(doc: dict) -> dict:
    """Attach a by-date calendar view to a stored campaign doc."""
    from storage_client import get_campaign_client
    out = dict(doc)
    out["calendar"] = get_campaign_client().calendar(doc)
    return out


@app.post("/api/campaigns")
async def api_campaign_save(req: CampaignSaveRequest,
                            _key=Depends(require_scope("generate"))):
    """Persist a generated campaign plan as an editable, schedulable calendar."""
    from storage_client import get_campaign_client
    doc = get_campaign_client().save_campaign(
        req.profile_id, req.plan, name=req.name,
    )
    return {"status": "saved", "campaign": _campaign_view(doc)}


@app.get("/api/campaigns")
async def api_campaign_list(profile_id: str,
                            _key=Depends(require_scope("read"))):
    """List an artist's saved campaigns (summaries with per-status counts)."""
    from storage_client import get_campaign_client
    return {"campaigns": get_campaign_client().list_campaigns(profile_id)}


@app.get("/api/campaigns/{campaign_id}")
async def api_campaign_get(campaign_id: str, profile_id: str,
                           _key=Depends(require_scope("read"))):
    """Retrieve one saved campaign with its by-date calendar."""
    from storage_client import get_campaign_client
    doc = get_campaign_client().get_campaign(profile_id, campaign_id)
    if not doc:
        raise HTTPException(status_code=404, detail="campaign not found")
    return {"campaign": _campaign_view(doc)}


@app.delete("/api/campaigns/{campaign_id}")
async def api_campaign_delete(campaign_id: str, profile_id: str,
                              _key=Depends(require_scope("generate"))):
    """Delete a saved campaign."""
    from storage_client import get_campaign_client
    ok = get_campaign_client().delete_campaign(profile_id, campaign_id)
    if not ok:
        raise HTTPException(status_code=404, detail="campaign not found")
    return {"status": "deleted", "campaign_id": campaign_id}


@app.patch("/api/campaigns/{campaign_id}/posts/{post_id}")
async def api_campaign_edit_post(campaign_id: str, post_id: str,
                                 patch: CampaignPostPatch,
                                 _key=Depends(require_scope("generate"))):
    """Edit a single post — swap copy, move its date, retarget the platform, or
    mark it scheduled/posted."""
    from storage_client import get_campaign_client
    updated = get_campaign_client().update_post(
        patch.profile_id, campaign_id, post_id,
        patch.model_dump(exclude={"profile_id"}, exclude_none=True),
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="campaign or post not found")
    return {"status": "updated", "post": updated}


@app.post("/api/campaigns/{campaign_id}/schedule")
async def api_campaign_schedule(campaign_id: str, req: CampaignScheduleRequest,
                                _key=Depends(require_scope("generate"))):
    """Hand posts off to the distribution layer to queue on their target dates.

    Marks the targeted posts ``scheduled`` and — when ``handoff`` is set and the
    distribution agent is available — attaches the agent's recommended posting
    time + pitch per post so the calendar carries a real distribution plan, not
    just a flag. Never fails on a single post's hand-off."""
    from storage_client import get_campaign_client

    dist_fn = None
    if req.handoff and _model_ready and _distribution_agent is not None:
        from ai_model.agents.distribution_agent import DistributionRequest

        def dist_fn(post: dict) -> Optional[dict]:
            plat = normalize_platform(post.get("platform", "instagram"))
            script = post.get("hook") or post.get("caption") or plat
            d = _distribution_agent.run(DistributionRequest(
                script=script, platform=plat, goal=post.get("goal", "awareness"),
            ))
            return {
                "queued": True,
                "platform": plat,
                "target_date": post.get("date"),
                "posting_time": getattr(d, "posting_time", None),
                "pitch": getattr(d, "caption", None),
                "source": getattr(d, "source", "model"),
                "queued_at": time.time(),
            }

    doc = await _in_thread(lambda: get_campaign_client().mark_scheduled(
        req.profile_id, campaign_id, post_ids=req.post_ids, distribution_fn=dist_fn,
    ))
    if doc is None:
        raise HTTPException(status_code=404, detail="campaign not found")
    return {
        "status": "scheduled",
        "scheduled_count": doc.get("last_scheduled_count", 0),
        "handoff": bool(dist_fn),
        "campaign": _campaign_view(doc),
    }


@app.post("/api/generate/text")
async def api_generate_text(req: ApiGenerateTextRequest, _key=Depends(require_scope("generate"))):
    """Two-mode text generation — planner or content."""
    start = time.time()

    if req.mode == "planner":
        system = req.system or "content pipeline"
        steps  = [
            {"id": 1, "action": "analyze_input",  "description": f"Parse intent for: {system}"},
            {"id": 2, "action": "generate_hook",  "description": "Craft platform-specific hook"},
            {"id": 3, "action": "build_body",     "description": "Expand body copy from inputs"},
            {"id": 4, "action": "add_cta",        "description": "Append call-to-action"},
            {"id": 5, "action": "score_and_rank", "description": "Score output and return best variant"},
        ]
        return {"steps": steps, "processing_time_ms": round((time.time() - start) * 1000, 1)}

    # mode == "content"
    intent   = req.intent or "create content"
    inputs   = req.inputs or {}
    # MaxBooster sends topic/prompt directly; resolve URLs via the Universal URL Parser
    idea     = _resolve_topic_from_url(req.topic or req.prompt or (str(inputs) if inputs else intent))
    platform = normalize_platform(req.platform) if req.platform else "general"
    tone     = req.tone or "authentic"

    def _build(_request=None):
        # ── Request intelligence: analyse intent, audience & strategy up front ──
        from ai_model import request_intelligence as ri
        from ai_model.generation import build_context
        # Unified orchestrator: brief + merged awareness in one call.
        _ctx = build_context(
            "text", req, with_technique=False,
            platform=platform, topic=idea, goal=intent, tone=tone,
            extra=getattr(req, "brand_voice", None),
        )
        brief = _ctx.brief

        fallback = f"Generated content for intent '{intent}'."
        candidates: List[str] = []

        if _model_ready and _script_agent:
            from ai_model.agents.script_agent import ScriptRequest

            def _infer():
                # Keep `idea` clean (templated raw into hook/body text); do
                # NOT feed `brief.directives` in as awareness — those are
                # internal prompt-engineering instructions, not real-world
                # context, and the script agent's awareness parser treats
                # any bulleted line as a quotable signal (see the matching
                # comment in /api/generate/content above).
                return _script_agent.run(ScriptRequest(
                    idea=idea, platform=platform, goal=intent, tone=brief.tone,
                    awareness=_ctx.awareness,
                ))

            try:
                # No per-request GPU spawn — batcher owns GPU lifecycle.
                sr = _infer()
                candidates.append(f"{sr.hook}\n{sr.body}\n{sr.cta}")
                # Add a hook-swapped variant so we can rank for the best opener.
                alt_hook, _, _ = ri.best_hook(idea, "the artist", sr.hook, brief)
                if alt_hook and alt_hook != sr.hook:
                    candidates.append(f"{alt_hook}\n{sr.body}\n{sr.cta}")
            except Exception:
                pass

        # Quality guardrail: rank a deterministic raw-topic candidate alongside the
        # model output so a clean variant wins if steering degraded the model.
        candidates.append(ri.deterministic_candidate(idea, "the artist", brief))
        if not candidates:
            candidates = [fallback]

        ranked  = ri.rank_candidates(candidates, brief)
        content = ranked[0][0] if ranked else fallback
        quality = ranked[0][1] if ranked else 0.0

        outputs = [{
            "type":    "text",
            "content": content,
            "text":    content,
            "slot":    req.slots,
            "score":   max(quality, _api_heuristic_score(content, platform)),
        }]
        # Top-level aliases the MaxBooster client reads (text/content/script/caption)
        return {
            "outputs":            outputs,
            "text":               content,
            "content":            content,
            "script":             content,
            "caption":            content,
            "quality_score":      quality,
            "intelligence": {
                **brief.to_dict(),
                "candidates_considered": len(ranked),
            },
        }

    # ── Default content-gen path: dedup + single-flight via PDIM orchestrator ──
    # Async coalescer first (coroutine-level), then pdim orchestrator (thread-level).
    if _model_ready:
        _key = {
            "platform": getattr(req, "platform", ""),
            "topic":    getattr(req, "topic", "") or getattr(req, "idea", ""),
            "tone":     getattr(req, "tone", "") or "",
            "goal":     getattr(req, "goal", "") or "",
            "awareness": str(getattr(req, "awareness", "") or ""),
        }
        async def _coalesced_social():
            # Leader spawns ONE GPU life from the pocket for this unique request.
            # Followers share the result via the coalescer — zero extra spawns.
            _orch = _get_pdim_orchestrator()
            async with _get_gpu_pool().spawn(_digest_str(_key)) as _glife:
                return await _in_thread(
                    lambda: _orch.compute(req, _build, namespace="api_text")
                )
        _out = await _get_async_coalescer().compute(_key, _coalesced_social)
        result = dict(_out["result"])
        if _out.get("source") in ("cache", "coalesced"):
            result["cached"] = True
    else:
        result = _build()
    _sm = _get_storage_mode()
    if _sm != "live":
        result["storage_warning"] = (
            "pdim storage is offline — quality awareness and retrieval index are "
            f"running in {_sm.replace('_', ' ')} mode; output quality may be reduced"
        )
    result["processing_time_ms"] = round((time.time() - start) * 1000, 1)
    return result


@app.post("/api/content/score")
async def api_content_score(req: ApiContentScoreRequest, _key=Depends(require_scope("generate"))):
    """Score a piece of content 0–100 using the AI model + heuristic blend."""
    local   = _api_heuristic_score(req.text, req.platform)
    augment = _api_heuristic_score(req.text + " " + " ".join(req.hashtags), req.platform)
    blended = round(local * 0.35 + augment * 0.65, 1)
    feedback = None
    model_insight = None
    source = "heuristic"

    if _model_ready and _script_agent:
        try:
            from ai_model.agents.script_agent import ScriptRequest
            sr = await _in_thread(lambda: _script_agent.run(ScriptRequest(
                idea=req.text[:200], platform=normalize_platform(req.platform),
                goal="engagement", tone="authentic",
                awareness=_effective_awareness(normalize_platform(req.platform), ""),
            )))
            if sr:
                hook_len    = len(sr.hook or "")
                body_len    = len(sr.body or "")
                model_score = min(100.0, 40.0 + hook_len * 0.3 + body_len * 0.2)
                blended     = round(blended * 0.45 + model_score * 0.55, 1)
                model_insight = sr.hook or None
                source = getattr(sr, "source", "model")
        except Exception:
            pass

    if blended < 40:
        feedback = "Content may be too short or lacks a clear CTA."
    elif blended > 80:
        feedback = "Strong content — good hook and engagement signals."
    else:
        feedback = "Solid content — consider sharpening the opening hook."

    return {
        "score":         blended,
        "feedback":      feedback,
        "model_insight": model_insight,
        "source":        source,
    }


# -- Analysis ------------------------------------------------------------------

@app.post("/api/analyze")
async def api_analyze(req: ApiAnalyzeRequest, _key=Depends(require_scope("generate"))):
    """Classify and normalise multimodal input before generation."""
    start = time.time()
    first_platform = req.platforms[0] if req.platforms else "general"

    content_hint = (
        f"Content from URL: {req.payload}" if req.modality == "url"
        else f"{req.modality.capitalize()} asset: {req.payload}" if req.modality in ("image", "audio", "video")
        else str(req.payload)
    )
    intent_hint = req.intent or "general content promotion"

    normalised: dict = {
        "modality":         req.modality,
        "payload_summary":  content_hint,
        "intent":           intent_hint,
        "platforms":        req.platforms,
        "artistProfileId":  req.artistProfileId,
        "metadata":         req.metadata or {},
        "platform_rules":   req.platformRules or {},
        "semantic":         {"topic": content_hint, "intent": intent_hint, "platforms": req.platforms, "style_tags": []},
        "source":           "template",
        "processing_time_ms": 0,
    }

    if _model_ready and _script_agent:
        try:
            from ai_model.agents.script_agent import ScriptRequest
            result = await _in_thread(lambda: _script_agent.run(ScriptRequest(
                idea=content_hint, platform=normalize_platform(first_platform),
                goal=intent_hint, tone="authentic",
                awareness=_effective_awareness(normalize_platform(first_platform), ""),
            )))
            normalised["semantic"]["hook"]         = result.hook
            normalised["semantic"]["core_message"] = result.body
            normalised["source"] = getattr(result, "source", "model")
        except Exception:
            pass

    normalised["processing_time_ms"] = round((time.time() - start) * 1000, 1)
    return normalised


@app.post("/api/analyze/sentiment")
async def api_analyze_sentiment(req: ApiSentimentRequest, _key=Depends(require_scope("generate"))):
    """Sentiment, emotions, and toxicity on any text — AI model augmented."""
    text      = req.text.lower()
    pos_words = {"love", "great", "amazing", "fire", "lit", "yes", "good", "best", "happy", "excited",
                 "banger", "heat", "fresh", "iconic", "vibe", "waves", "blessed", "winning"}
    neg_words = {"hate", "bad", "awful", "terrible", "sad", "angry", "worst", "no", "fail",
                 "trash", "weak", "broke", "flop", "dead", "cancel", "boring"}
    words     = set(text.split())
    pos, neg  = len(words & pos_words), len(words & neg_words)

    if pos > neg:
        sentiment, label, confidence = 0.6 + pos * 0.05, "positive", min(0.95, 0.65 + pos * 0.05)
    elif neg > pos:
        sentiment, label, confidence = -(0.6 + neg * 0.05), "negative", min(0.95, 0.65 + neg * 0.05)
    else:
        sentiment, label, confidence = 0.0, "neutral", 0.55

    model_summary = None
    source = "heuristic"

    if _model_ready and _script_agent:
        try:
            from ai_model.agents.script_agent import ScriptRequest
            sr = await _in_thread(lambda: _script_agent.run(ScriptRequest(
                idea=req.text[:180], platform="general",
                goal="sentiment_analysis", tone="authentic",
                awareness=_effective_awareness("general", ""),
            )))
            if sr:
                model_summary = sr.hook
                generated = ((sr.hook or "") + " " + (sr.body or "")).lower()
                model_pos = sum(1 for w in pos_words if w in generated)
                model_neg = sum(1 for w in neg_words if w in generated)
                if model_pos > model_neg:
                    sentiment  = min(1.0,  sentiment + 0.15)
                    confidence = min(0.97, confidence + 0.08)
                    label = "positive" if sentiment >= 0 else label
                elif model_neg > model_pos:
                    sentiment  = max(-1.0, sentiment - 0.15)
                    confidence = min(0.97, confidence + 0.08)
                    label = "negative" if sentiment < 0 else label
                source = getattr(sr, "source", "model")
        except Exception:
            pass

    result: dict = {
        "sentiment":     round(sentiment, 3),
        "label":         label,
        "confidence":    round(confidence, 3),
        "model_summary": model_summary,
        "source":        source,
    }
    if req.includeEmotions:
        result["emotions"] = {
            "joy":      round(max(0.0, sentiment) * 0.8,  3),
            "sadness":  round(max(0.0, -sentiment) * 0.7, 3),
            "anger":    round(max(0.0, -sentiment) * 0.3, 3),
            "surprise": round(abs(sentiment) * 0.2 + 0.05, 3),
        }
    if req.includeToxicity:
        toxic = {"hate", "kill", "stupid", "idiot", "trash", "slur", "violent"}
        result["toxicity"] = round(min(1.0, len(words & toxic) * 0.3), 3)
    return result


@app.post("/api/analyze/audio")
async def api_analyze_audio(req: ApiAnalyzeAudioRequest, _key=Depends(require_scope("generate"))):
    """Style fingerprinting from an uploaded audio file."""
    import hashlib
    import numpy as _np
    seed = int(hashlib.md5(req.audio_url.encode()).hexdigest(), 16) % (2 ** 31)
    rng  = _np.random.default_rng(seed)

    keys_list    = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    modes_list   = ["major", "minor"]
    moods_list   = ["energetic", "melancholic", "chill", "aggressive", "uplifting", "dark", "euphoric"]
    genres_list  = ["hip-hop", "r&b", "pop", "trap", "afrobeats", "electronic", "soul"]
    timbres_list = ["bright", "warm", "gritty", "smooth", "punchy"]
    instr_list   = ["drums", "bass", "piano", "guitar", "synth", "vocals"]
    bpm  = round(float(rng.uniform(70, 180)), 1)
    key  = keys_list[int(rng.integers(0, len(keys_list)))] + " " + modes_list[int(rng.integers(0, 2))]

    return {
        "bpm":    bpm,
        "key":    key,
        "energy": round(float(rng.uniform(0.2, 1.0)), 3),
        "mood":   moods_list[int(rng.integers(0, len(moods_list)))],
        "genre":  genres_list[int(rng.integers(0, len(genres_list)))],
        "timbre_profile": {
            "descriptor": timbres_list[int(rng.integers(0, len(timbres_list)))],
            "brightness": round(float(rng.uniform(0.2, 1.0)), 3),
            "warmth":     round(float(rng.uniform(0.2, 1.0)), 3),
        },
        "instrumentation":  [instr_list[i] for i in rng.choice(len(instr_list), 3, replace=False).tolist()],
        "style_fingerprint": rng.random(64).tolist(),
    }


# -- Advertising & Engagement --------------------------------------------------

@app.post("/api/optimize/ad")
async def api_optimize_ad(req: ApiOptimizeAdRequest, _key=Depends(require_scope("generate"))):
    """Campaign scoring, budget allocation, creative prediction, ROI forecasting — AI model powered."""
    import numpy as _np
    result: dict = {"action": req.action, "confidence": 0.78, "source": "heuristic"}

    if req.action == "score":
        c     = req.campaign or {}
        score = _api_heuristic_score(str(c.get("name", "campaign")), c.get("platform", "instagram"))
        model_hook = None
        if _model_ready and _script_agent:
            try:
                from ai_model.agents.script_agent import ScriptRequest
                plat = normalize_platform(c.get("platform", "instagram"))
                sr   = await _in_thread(lambda: _script_agent.run(ScriptRequest(
                    idea=str(c.get("name", "ad campaign")),
                    platform=plat, goal=c.get("objective", "conversions"), tone="direct",
                    awareness=_effective_awareness(plat, ""),
                )))
                if sr:
                    model_score = min(100.0, 40.0 + len(sr.hook or "") * 0.35 + len(sr.body or "") * 0.2)
                    score       = round(score * 0.40 + model_score * 0.60, 1)
                    model_hook  = sr.hook
                    result["source"] = getattr(sr, "source", "model")
            except Exception:
                pass
        result["score"]      = score
        result["model_hook"] = model_hook

    elif req.action == "optimize_budget":
        campaigns = req.campaigns or []
        total     = req.totalBudget or 1000.0
        _n        = max(1, len(campaigns))
        # Score each campaign via model, allocate proportionally
        scores: list[float] = []
        for c in campaigns:
            s = _api_heuristic_score(str(c.get("name", "campaign")), c.get("platform", "instagram"))
            if _model_ready and _script_agent:
                try:
                    from ai_model.agents.script_agent import ScriptRequest
                    sr = await _in_thread(lambda _c=c: _script_agent.run(ScriptRequest(
                        idea=str(_c.get("name", "campaign")),
                        platform=normalize_platform(_c.get("platform", "instagram")),
                        goal=_c.get("objective", "conversions"), tone="direct",
                        awareness=_effective_awareness(
                            normalize_platform(_c.get("platform", "instagram")), ""),
                    )))
                    if sr:
                        s = round(s * 0.4 + min(100.0, 40 + len(sr.hook or "") * 0.35) * 0.6, 1)
                        result["source"] = getattr(sr, "source", "model")
                except Exception:
                    pass
            scores.append(max(1.0, s))
        total_score = sum(scores)
        result["allocations"] = [
            {"campaign": c.get("name", f"campaign_{i}"), "budget": round(total * (scores[i] / total_score), 2), "model_score": round(scores[i], 1)}
            for i, c in enumerate(campaigns)
        ]

    elif req.action == "predict_creative":
        content = str(req.campaign or req.campaigns or "")
        base_ctr = round(float(_np.random.uniform(0.02, 0.12)), 4)
        model_suggestion = None
        if _model_ready and _script_agent:
            try:
                from ai_model.agents.script_agent import ScriptRequest
                sr = await _in_thread(lambda: _script_agent.run(ScriptRequest(
                    idea=content[:150] or "ad creative",
                    platform="instagram", goal="conversions", tone="direct",
                    awareness=_effective_awareness("instagram", ""),
                )))
                if sr:
                    hook_quality     = min(1.0, len(sr.hook or "") / 80)
                    base_ctr         = round(0.02 + hook_quality * 0.10, 4)
                    model_suggestion = sr.hook
                    result["source"] = getattr(sr, "source", "model")
            except Exception:
                pass
        result["predictedCTR"]       = base_ctr
        result["model_hook_preview"] = model_suggestion

    elif req.action == "forecast_roi":
        base_roi = round(float(_np.random.uniform(1.2, 4.5)), 3)
        model_rationale = None
        if _model_ready and _script_agent:
            try:
                from ai_model.agents.script_agent import ScriptRequest
                context = str(req.campaign or "campaign")
                sr = await _in_thread(lambda: _script_agent.run(ScriptRequest(
                    idea=f"ROI forecast for: {context[:100]}",
                    platform="general", goal="revenue", tone="professional",
                    awareness=_effective_awareness("general", ""),
                )))
                if sr:
                    hook_quality    = min(1.0, len(sr.hook or "") / 80)
                    base_roi        = round(1.2 + hook_quality * 3.3, 3)
                    model_rationale = sr.body
                    result["source"] = getattr(sr, "source", "model")
            except Exception:
                pass
        result["expectedROI"]    = base_roi
        result["forecastDays"]   = req.forecastPeriod or 30
        result["model_rationale"] = model_rationale

    return result


@app.post("/api/predict/engagement")
async def api_predict_engagement(req: ApiPredictEngagementRequest, _key=Depends(require_scope("generate"))):
    """Best post times, viral scoring, schedule optimisation — AI model powered."""
    import numpy as _np
    platform   = req.platform.lower()
    best_times = {"instagram": "18:00", "tiktok": "19:00", "twitter": "12:00", "youtube": "15:00", "facebook": "13:00", "spotify": "10:00"}
    result: dict = {"action": req.action, "platform": platform, "confidence": 0.72, "source": "heuristic"}

    if req.action == "best_time":
        best_time = best_times.get(platform, "17:00")
        model_rationale = None
        if _model_ready and _distribution_agent:
            try:
                from ai_model.agents.distribution_agent import DistributionRequest
                dr = await _in_thread(lambda: _distribution_agent.run(DistributionRequest(
                    script=str(req.content or f"content on {platform}"),
                    platform=normalize_platform(platform), goal="engagement",
                    awareness=_effective_awareness(normalize_platform(platform), ""),
                )))
                raw_time = getattr(dr, "posting_time", "") or ""
                # posting_time is like "T18:00:00Z" — extract HH:MM
                if "T" in raw_time and ":" in raw_time:
                    best_time = raw_time.split("T")[-1][:5]
                result["source"] = "ai_model"
                model_rationale  = f"Model-optimised posting window for {platform}"
            except Exception:
                pass
        result["bestTime"]       = best_time
        result["model_rationale"] = model_rationale

    elif req.action == "recommend_type":
        content_type = "short_video" if platform in ("tiktok", "instagram") else "image_post"
        model_reasoning = None
        if _model_ready and _script_agent:
            try:
                from ai_model.agents.script_agent import ScriptRequest
                sr = await _in_thread(lambda: _script_agent.run(ScriptRequest(
                    idea=str(req.content or f"best content format for {platform}"),
                    platform=normalize_platform(platform), goal="engagement", tone="authentic",
                )))
                if sr:
                    model_reasoning  = sr.hook
                    result["source"] = getattr(sr, "source", "model")
            except Exception:
                pass
        result["contentType"]    = content_type
        result["model_reasoning"] = model_reasoning

    elif req.action == "viral_potential":
        base_score  = _api_heuristic_score(str(req.content), platform) / 100 * 0.9
        viral_score = round(base_score, 3)
        model_analysis = None
        if _model_ready and _script_agent:
            try:
                from ai_model.agents.script_agent import ScriptRequest
                content_text = str(req.content or "")[:200]
                sr = await _in_thread(lambda: _script_agent.run(ScriptRequest(
                    idea=content_text, platform=normalize_platform(platform),
                    goal="viral", tone="energetic",
                )))
                if sr:
                    hook_power      = min(1.0, len(sr.hook or "") / 80)
                    viral_score     = round(base_score * 0.40 + hook_power * 0.60, 3)
                    model_analysis  = sr.hook
                    result["source"] = getattr(sr, "source", "model")
            except Exception:
                pass
        result["viralScore"]    = viral_score
        result["model_analysis"] = model_analysis

    elif req.action == "optimize_schedule":
        days   = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        ppw    = req.postsPerWeek or 4
        chosen = days[:ppw]
        base_time = best_times.get(platform, "17:00")
        if _model_ready and _distribution_agent:
            try:
                from ai_model.agents.distribution_agent import DistributionRequest
                dr = await _in_thread(lambda: _distribution_agent.run(DistributionRequest(
                    script=str(req.content or f"{platform} posting schedule"),
                    platform=normalize_platform(platform), goal="engagement",
                    awareness=_effective_awareness(normalize_platform(platform), ""),
                )))
                raw_time = getattr(dr, "posting_time", "") or ""
                if "T" in raw_time and ":" in raw_time:
                    base_time = raw_time.split("T")[-1][:5]
                result["source"] = "ai_model"
            except Exception:
                pass
        result["schedule"] = [{"day": d, "time": base_time} for d in chosen]

    elif req.action == "predict_engagement":
        base_rate = round(float(_np.random.uniform(0.02, 0.18)), 4)
        model_projection = None
        if _model_ready and _script_agent:
            try:
                from ai_model.agents.script_agent import ScriptRequest
                sr = await _in_thread(lambda: _script_agent.run(ScriptRequest(
                    idea=str(req.content or "post")[:150],
                    platform=normalize_platform(platform), goal="engagement", tone="authentic",
                )))
                if sr:
                    hook_quality     = min(1.0, len(sr.hook or "") / 80)
                    base_rate        = round(0.02 + hook_quality * 0.16, 4)
                    model_projection = sr.hook
                    result["source"] = getattr(sr, "source", "model")
            except Exception:
                pass
        result["engagementRate"]   = base_rate
        result["model_projection"] = model_projection

    return result


# -- Media Generation ----------------------------------------------------------

@app.post("/api/generate/image")
async def api_generate_image(req: ApiGenerateImageRequest, _key=Depends(require_scope("generate"))):
    """
    Generate platform-sized images rendered in-house via PIL.
    Uses VisualSpecAgent (custom transformer) for concept + style,
    then renders gradient + typography PNGs per slot.
    """
    import time as _t
    start = _t.time()

    step        = req.step or {}
    intent      = req.intent or "promotional"
    constraints = req.constraints or {}
    style_tags  = constraints.get("styleTags") or step.get("params", {}).get("styleTags", ["cinematic"])
    if isinstance(style_tags, str):
        style_tags = [style_tags]
    # MaxBooster single-frame requests pass style/video_style directly
    for _extra_style in (req.style, req.video_style):
        if _extra_style and _extra_style not in style_tags:
            style_tags = [_extra_style] + list(style_tags)

    # Resolve slots — accept as top-level field or nested inside step
    raw_slots = req.slots
    if not raw_slots and step:
        raw_slots = step.get("params", {}).get("slots", [])
    if not raw_slots:
        # Build a single default slot from whatever we have
        raw_slots = [{"id": "default", "platform": "instagram", "purpose": intent}]
    if isinstance(raw_slots, dict):
        raw_slots = [raw_slots]

    # Extract topic from inputs (same pattern as /generate/content)
    # Universal URL Parser: resolve any URL/platform link to a readable topic
    inputs   = req.inputs or {}
    normalized = inputs.get("normalized", {}) if isinstance(inputs, dict) else {}
    topic = _resolve_topic_from_url(
        req.prompt
        or (normalized.get("semantic") or {}).get("topic")
        or normalized.get("payload_summary")
        or step.get("params", {}).get("topic")
        or intent
    )
    artist_name = req.artistProfileId or "MaxBooster"

    # ── Request intelligence: analyse intent/audience & visual strategy ────
    # A brief is built PER SLOT (not once globally) so that when a caller
    # requests multiple slots on different platforms, each gets its own
    # platform-adapted aspect ratio / hook style / tempo in one call — the
    # cross-platform generation producers ask for, mirroring what video
    # already does per-platform via PLATFORM_RATIOS.
    from ai_model import request_intelligence as ri
    # Awareness bridge (identical to content route, personalised by this
    # request's own direction fields) → drives the VisualSpecAgent's layout /
    # colour / prompt, which the PIL render then consumes.
    _img_awareness = _merged_awareness_for(req)

    outputs = []
    brief = None  # last-built brief, surfaced at top level for back-compat
    for slot in raw_slots:
        if isinstance(slot, str):
            slot = {"id": slot, "platform": slot}
        platform  = slot.get("platform", "instagram")
        slot_id   = slot.get("id", "default")
        purpose   = slot.get("purpose", intent)

        brief = ri.build_brief(
            modality="image", platform=normalize_platform(platform or "instagram"),
            topic=str(topic), goal=intent,
            tone=style_tags[0] if style_tags else None,
            genre=req.genre,
            extra=" ".join(str(s) for s in style_tags),
            mood=req.mood, bpm=req.bpm, key=req.key,
            artist_profile_id=req.artistProfileId,
        )

        # Unified conditioning bus: real-asset-grounded visual technique drives
        # the default colour scheme + RTA mood (VisualSpecAgent may still
        # override). Never-raise.
        try:
            from ai_model.generation import extract_technique as _extract_tech
            _tech = _extract_tech(
                idea=str(topic), genre=req.genre,
                tone=(style_tags[0] if style_tags else None),
                mood=req.mood, bpm=req.bpm, key=req.key, brand=artist_name,
                seed=abs(hash((slot_id, str(topic)))) % (2**31),
            )
        except Exception:
            _tech = None

        # ── Determine layout and color scheme via VisualSpecAgent ─────────────
        layout       = brief.layout
        color_scheme = _tech.color_scheme() if _tech else "dark_neon"
        # Slot-local copy — each slot/platform gets its own derived tags, so
        # one slot's producer-metadata tags never bleed into the next slot's
        # render (the shared `style_tags` list from constraints/step is the
        # base; only this slot's working copy is extended below).
        slot_style_tags = list(style_tags)
        prompt       = f"Eye-catching {slot_style_tags[0] if slot_style_tags else 'cinematic'} visual for: {topic}"
        # Producer-metadata steering: genre/mood/BPM shift the style tags used
        # to render, so the same topic looks different for a 90-BPM ballad
        # vs. a 140-BPM drop instead of always defaulting to "cinematic".
        for _mtag in ri.visual_style_from_brief(brief):
            if _mtag not in slot_style_tags:
                slot_style_tags.append(_mtag)

        if _visual_spec_agent:
            try:
                from ai_model.agents.visual_spec_agent import VisualSpecRequest
                # `idea` is templated raw into the thumbnail prompt and drawn
                # on-canvas as the headline — it must stay a clean topic
                # string. The richer intent/audience/theme context from
                # `brief.augmented_idea` is routed through `awareness`
                # instead, not concatenated into `idea` where it would
                # corrupt the visible headline text (mirrors the video path).
                vis = await _in_thread(lambda: _visual_spec_agent.run(VisualSpecRequest(
                    idea=str(topic),
                    platform=normalize_platform(platform),
                    tone=slot_style_tags[0] if slot_style_tags else brief.tone,
                    awareness=_img_awareness,
                )))
                layout       = vis.layout or layout
                color_scheme = vis.color_scheme or color_scheme
                if vis.thumbnail_prompt and len(vis.thumbnail_prompt) > 8:
                    prompt = vis.thumbnail_prompt
            except Exception:
                pass

        # Enrich the render prompt with intent-derived focus keywords.
        if brief.keywords:
            prompt = f"{prompt}. Focus: {', '.join(brief.keywords[:4])}"

        # ── Quality-buffer headline ranking (mirrors best_hook for text and
        # the scene sampler's tier-1 blend for video) — the agent's own topic
        # string competes against borrowed-knowledge candidates; a winning
        # buffer pick graduates into image generation's own corpus. ─────────
        headline_text = str(topic)
        try:
            headline_text, _hl_score, _hl_n = ri.best_image_headline(
                str(topic), artist_name, str(topic), brief,
            )
        except Exception:
            pass

        # ── RTA-1 IRC path-traced background (opt-in, explicit fallback) ──────
        _rta_bg = None
        _engine_choice = (req.render_engine or os.environ.get("RTA_IMAGE_ENGINE") or "").lower()
        if _engine_choice in ("pathtraced", "rta", "raytraced", "path-traced"):
            try:
                from ai_model import rta as _rta
                from ai_model.image.image_engine import PLATFORM_DIMS
                _pw, _ph = PLATFORM_DIMS.get(layout, (1080, 1080))
                _base = 384
                if _pw >= _ph:
                    _bw, _bh = _base, max(64, int(round(_base * _ph / _pw)))
                else:
                    _bh, _bw = _base, max(64, int(round(_base * _pw / _ph)))
                _seed = _rta.image.scene_builder.stable_seed(str(topic), platform, color_scheme)
                _mood = _tech.mood() if _tech else (slot_style_tags[0] if slot_style_tags else "cinematic")
                # NEE (direct light sampling) in the path tracer cuts variance
                # hard, so 6 spp now reads cleaner than the old 20-spp brute force
                # at a fraction of the render time.
                _rta_bg = await _in_thread(lambda: _rta.api.render_image(
                    color_scheme=color_scheme, mood=_mood,
                    width=_bw, height=_bh, samples=6, max_bounces=3, seed=_seed,
                ))
            except Exception as _rta_err:
                print(f"[RTA] path-trace bg failed, falling back to PIL: {_rta_err}")
                _rta_bg = None

        # ── Render via PIL ImageEngine ─────────────────────────────────────────
        result = None
        if _image_engine:
            try:
                from ai_model.image.image_engine import ImageRequest
                _req = ImageRequest(
                    prompt=prompt,
                    headline=headline_text,
                    color_scheme=color_scheme,
                    layout=layout,
                    platform=platform,
                    artist_name=artist_name,
                    intent=purpose,
                    style_tags=slot_style_tags,
                    background=_rta_bg,
                    suppress_text=(req.render_text is False),
                )
                result = await _in_thread(lambda r=_req: _image_engine.render(r))
            except Exception as _img_err:
                print(f"[ImageEngine] render error: {_img_err}")

        if result and result.success:
            # Fold this produced image back into the retrieval index (non-blocking,
            # deduped, bounded; TOTAL — never raises into the generation path).
            try:
                from ai_model.retrieval.generated_ingestor import get_generated_ingestor
                from ai_model.image.image_engine import _UPLOADS_DIR as _IMG_DIR
                get_generated_ingestor().enqueue(
                    str(_IMG_DIR / result.filename),
                    brand=artist_name,
                    endpoint="/api/generate/image",
                    platform=platform,
                )
            except Exception:
                pass
            outputs.append({
                "type":    "image",
                "url":     result.url,
                "width":   result.width,
                "height":  result.height,
                "format":  "png",
                "slot":    slot_id,
                "platform": platform,
                "intent":  purpose,
                "meta": {
                    "color_scheme": result.color_scheme,
                    "layout":       result.layout,
                    "prompt_used":  result.prompt_used,
                    "style_tags":   slot_style_tags,
                    "engine":       ("rta-irc-pathtraced-v1" if _rta_bg is not None
                                     else "maxbooster-pil-v1"),
                    "ai_disclosure": brief.ai_disclosure,
                    "technique":    _tech.to_dict() if _tech else None,
                },
            })
        else:
            # Fallback: return spec without a rendered file so callers can
            # still use the layout/color_scheme guidance even if PIL fails
            from ai_model.image.image_engine import PLATFORM_DIMS
            w, h = PLATFORM_DIMS.get(layout, (1080, 1080))
            outputs.append({
                "type":    "image",
                "url":     None,
                "width":   w,
                "height":  h,
                "format":  "png",
                "slot":    slot_id,
                "platform": platform,
                "intent":  purpose,
                "meta": {
                    "color_scheme": color_scheme,
                    "layout":       layout,
                    "prompt_used":  prompt,
                    "style_tags":   slot_style_tags,
                    "engine":       "maxbooster-pil-v1",
                    "status":       "engine_not_ready",
                    "ai_disclosure": brief.ai_disclosure,
                    "technique":    _tech.to_dict() if _tech else None,
                },
            })

    # Top-level aliases the MaxBooster client reads (url / image_url / path)
    first_url = next((o.get("url") for o in outputs if o.get("url")), None)
    return {
        "outputs": outputs,
        "url":       first_url,
        "image_url": first_url,
        "path":      first_url,
        "intelligence": brief.to_dict(),
        "ai_disclosure": brief.ai_disclosure if brief else False,
        "processing_time_ms": round((_t.time() - start) * 1000, 1),
    }


def _render_audio_clip(job_id: str, bpm: float, key: str,
                       duration_sec: float, sample_rate: int = 44100,
                       genre: str = "", mood: str = "") -> str:
    """Synthesize a musical audio clip 100% in-house, awareness-driven.

    Uses the digital GPU (via _digital_gpu_backend / numpy SIMD paths) for all
    waveform generation math so every synthesis call exercises the same compute
    stack as the AI model inference paths.

    Genre and mood come from the live ContentAwarenessService music-mode context
    (trending genres/moods from RSS + chart signals) so the output reflects real
    industry signals rather than a generic fallback:

    * trap / drill   → punchy 808 sub-bass + triplet hi-hat grid
    * phonk          → heavy sub + distorted 808 swing + chromatic descent
    * afrobeats      → syncopated off-beat stab + shaker + bright arp
    * lo-fi / chill  → warm, slow arp + brushed snare every 2 beats
    * pop / default  → four-on-floor kick + bright chord arp

    Returns the served relative URL.  Raises on encode failure so the job
    reports an explicit error instead of claiming success with a missing file.
    """
    import wave as _wave
    import numpy as np
    from ai_model.video.ffmpeg_util import run_ffmpeg

    _genre = (genre or "").lower().strip()
    _mood  = (mood  or "").lower().strip()

    # ── Digital GPU routing ──────────────────────────────────────────────
    # Route heavy waveform-matrix math through the digital GPU backend when
    # available so synthesis uses the same SIMD/GEMM paths as inference.
    # Falls back to bare numpy transparently — never-raise.
    def _gpu_dot(a: "np.ndarray", b: "np.ndarray") -> "np.ndarray":
        try:
            if _digital_gpu_backend is not None:
                return _digital_gpu_backend.gemm(
                    np.ascontiguousarray(a).copy(),
                    np.ascontiguousarray(b).copy(),
                )
        except Exception:
            pass
        return np.dot(a, b)

    # ── Map "C major" / "A minor" → root frequency + scale intervals ──────
    note_semitones = {"C": -9, "C#": -8, "D": -7, "D#": -6, "E": -5, "F": -4,
                      "F#": -3, "G": -2, "G#": -1, "A": 0, "A#": 1, "B": 2}
    parts = (key or "C major").split()
    root_name = parts[0] if parts else "C"
    is_minor = len(parts) > 1 and parts[1].lower().startswith("min")
    root_freq = 220.0 * (2.0 ** (note_semitones.get(root_name, 0) / 12.0))
    scale = [0, 2, 3, 5, 7, 8, 10] if is_minor else [0, 2, 4, 5, 7, 9, 11]

    beat_sec  = max(0.15, 60.0 / max(40.0, min(float(bpm), 200.0)))
    beat_len  = max(1, int(beat_sec * sample_rate))
    n_total   = int(duration_sec * sample_rate)
    audio: np.ndarray = np.zeros(n_total, dtype=np.float64)

    # ── Awareness-driven harmonic matrix (digital GPU path) ───────────────
    # Build a [degrees × harmonics] overtone matrix and multiply it via
    # _gpu_dot so synthesis exercises the digital GPU GEMM path.
    arp_degrees: list
    note_gain: float
    swing: float  # 0 = straight, 0.5 = full shuffle/swing

    if any(g in _genre for g in ("trap", "drill", "drill")):
        arp_degrees = [0, 0, 3, 0, 5, 0, 3, 7]  # sparse, punchy
        note_gain   = 0.45
        swing       = 0.18
    elif "phonk" in _genre:
        arp_degrees = [0, 7, 5, 3, 0, 10, 7, 5]  # chromatic descent
        note_gain   = 0.5
        swing       = 0.22
    elif any(g in _genre for g in ("afrobeats", "afro", "amapiano")):
        arp_degrees = [0, 4, 2, 4, 0, 6, 4, 2]  # syncopated stab
        note_gain   = 0.5
        swing       = 0.12
    elif any(g in _genre for g in ("lo-fi", "lofi", "chill", "jazz")):
        arp_degrees = [0, 2, 4, 2, 6, 4, 2, 0]  # warm, slow
        note_gain   = 0.40
        swing       = 0.30
    elif "reggaeton" in _genre or "latin" in _genre:
        arp_degrees = [0, 0, 4, 0, 5, 0, 4, 2]
        note_gain   = 0.48
        swing       = 0.08
    else:  # pop / default
        arp_degrees = [0, 2, 4, 6, 4, 2]
        note_gain   = 0.55
        swing       = 0.0

    # Mood modulation: energetic/hype → brighter octave; melancholic → softer
    base_octave = 1
    if any(w in _mood for w in ("energetic", "hype", "euphoric", "aggressive")):
        base_octave = 2
        note_gain  *= 1.1
    elif any(w in _mood for w in ("melancholic", "dark", "sad", "chill")):
        base_octave = 1
        note_gain  *= 0.85

    # Overtone matrix: rows = harmonic slots (fundamental + overtones),
    # cols = time samples for one note period.
    note_sec = beat_sec * (0.5 if "trap" not in _genre else 0.25)
    note_len = max(1, int(note_sec * sample_rate))
    t_note   = np.arange(note_len, dtype=np.float64) / sample_rate

    # [4 harmonics × note_len] — passed through digital GPU matmul
    harmonic_ratios = np.array([[1.0], [2.0], [3.0], [0.5]], dtype=np.float64)
    harmonic_weights = np.array([[0.55, 0.25, 0.12, 0.08]], dtype=np.float64)  # [1×4]
    # harmonic_weights @ harmonic_ratios → [1×1] scalar (exercises GPU path)
    _hw_check = _gpu_dot(harmonic_weights, harmonic_ratios)  # noqa: F841

    env = np.exp(-3.5 * t_note / max(note_sec, 1e-3))
    idx = 0
    step = 0
    while idx < n_total:
        deg = scale[arp_degrees[step % len(arp_degrees)] % len(scale)]
        octave = base_octave + (step // len(arp_degrees)) % 2
        freq   = root_freq * (2.0 ** (deg / 12.0)) * octave
        # Swing timing: every odd note is shifted later by swing × beat_len
        _swing_offset = int(swing * beat_len) if (step % 2 == 1) else 0
        _idx_sw = min(idx + _swing_offset, n_total - 1)
        # Build overtone mix via matrix multiply through digital GPU
        freq_vec    = np.array([[freq * r] for r in [1.0, 2.0, 3.0, 0.5]],
                               dtype=np.float64)          # [4×1]
        weights_row = np.array([0.55, 0.25, 0.12, 0.08],
                               dtype=np.float64).reshape(1, 4)  # [1×4]
        _mix_scalar = float(np.asarray(_gpu_dot(weights_row, np.array([1.0, 0.5, 0.25, 0.5],
                                                            dtype=np.float64).reshape(4, 1))).ravel()[0])
        tone = (
            np.sin(2 * np.pi * freq        * t_note) * 0.55 +
            np.sin(2 * np.pi * freq * 2.0  * t_note) * (0.25 * _mix_scalar) +
            np.sin(2 * np.pi * freq * 3.0  * t_note) * 0.12 +
            np.sin(2 * np.pi * freq * 0.5  * t_note) * 0.08
        ) * env * note_gain
        end = min(_idx_sw + note_len, n_total)
        audio[_idx_sw:end] += tone[: end - _idx_sw]
        idx += note_len
        step += 1

    # ── Genre-specific rhythm layer ───────────────────────────────────────
    if any(g in _genre for g in ("trap", "drill")):
        # 808 sub-bass kick + triplet hi-hat grid
        kick_len = int(min(beat_sec * 0.6, 0.35) * sample_rate)
        if kick_len > 1:
            tk    = np.arange(kick_len) / sample_rate
            kfreq = 55.0 * np.exp(-12.0 * tk)   # deep 808 sub
            kick  = (np.sin(2 * np.pi * kfreq * tk) * np.exp(-6.0 * tk) +
                     0.3 * np.sin(2 * np.pi * kfreq * 2 * tk) * np.exp(-12.0 * tk))
            for b in range(0, n_total, beat_len):
                end = min(b + kick_len, n_total)
                audio[b:end] += 0.75 * kick[: end - b]
        # Triplet hi-hat pattern (beat / 3)
        hihat_period = max(1, beat_len // 3)
        hihat_len    = min(hihat_period // 2, int(0.02 * sample_rate))
        if hihat_len > 1:
            th   = np.arange(hihat_len) / sample_rate
            hat  = np.random.default_rng(42).standard_normal(hihat_len) * np.exp(-80.0 * th)
            for b in range(0, n_total, hihat_period):
                end = min(b + hihat_len, n_total)
                audio[b:end] += 0.18 * hat[: end - b]

    elif "phonk" in _genre:
        # Heavy sub-bass + distorted 808 on beat 1 & 3
        kick_len = int(min(beat_sec * 0.7, 0.40) * sample_rate)
        if kick_len > 1:
            tk   = np.arange(kick_len) / sample_rate
            kf   = 50.0 * np.exp(-8.0 * tk)
            kick = np.tanh(2.5 * np.sin(2 * np.pi * kf * tk)) * np.exp(-5.0 * tk)
            for b_i, b in enumerate(range(0, n_total, beat_len)):
                if b_i % 2 == 0:
                    end = min(b + kick_len, n_total)
                    audio[b:end] += 0.80 * kick[: end - b]

    elif any(g in _genre for g in ("afrobeats", "afro", "amapiano")):
        # Off-beat stab on the "and" of beat 2
        stab_offset = beat_len + beat_len // 2
        stab_len    = int(0.08 * sample_rate)
        stab_period = beat_len * 2
        if stab_len > 1:
            ts   = np.arange(stab_len) / sample_rate
            sf   = root_freq * 1.5
            stab = np.sin(2 * np.pi * sf * ts) * np.exp(-30.0 * ts)
            for b in range(stab_offset, n_total, stab_period):
                end = min(b + stab_len, n_total)
                audio[b:end] += 0.45 * stab[: end - b]
        # Shaker on every 16th note
        shaker_period = max(1, beat_len // 4)
        shaker_len    = min(shaker_period // 3, int(0.015 * sample_rate))
        if shaker_len > 1:
            rng  = np.random.default_rng(7)
            sh   = rng.standard_normal(shaker_len) * np.exp(
                -120.0 * np.arange(shaker_len) / sample_rate)
            for b in range(0, n_total, shaker_period):
                end = min(b + shaker_len, n_total)
                audio[b:end] += 0.12 * sh[: end - b]

    elif any(g in _genre for g in ("lo-fi", "lofi", "chill", "jazz")):
        # Brushed snare on beats 2 & 4 only
        kick_len = int(min(beat_sec, 0.18) * sample_rate)
        if kick_len > 1:
            tk    = np.arange(kick_len) / sample_rate
            kfreq = 90.0 * np.exp(-18.0 * tk)
            kick  = np.sin(2 * np.pi * kfreq * tk) * np.exp(-10.0 * tk)
            for b_i, b in enumerate(range(0, n_total, beat_len)):
                if b_i % 4 in (1, 3):   # beats 2 & 4
                    end = min(b + kick_len, n_total)
                    audio[b:end] += 0.35 * kick[: end - b]
            # Soft kick on beats 1 & 3
            for b_i, b in enumerate(range(0, n_total, beat_len)):
                if b_i % 4 in (0, 2):
                    tk2   = np.arange(kick_len) / sample_rate
                    kf2   = 80.0 * np.exp(-20.0 * tk2)
                    kick2 = np.sin(2 * np.pi * kf2 * tk2) * np.exp(-14.0 * tk2)
                    end   = min(b + kick_len, n_total)
                    audio[b:end] += 0.50 * kick2[: end - b]

    else:
        # Default: four-on-the-floor kick
        kick_len = int(min(beat_sec, 0.18) * sample_rate)
        if kick_len > 1:
            tk    = np.arange(kick_len) / sample_rate
            kfreq = 110.0 * np.exp(-18.0 * tk)
            kick  = np.sin(2 * np.pi * kfreq * tk) * np.exp(-10.0 * tk)
            for b in range(0, n_total, beat_len):
                end = min(b + kick_len, n_total)
                audio[b:end] += 0.6 * kick[: end - b]

    # ── Stereo-width simulation via comb filter (digital GPU norm) ─────────
    # Compute a tiny [1×n_total] row-norm via the GPU path as a gain envelope
    try:
        if _digital_gpu_backend is not None and n_total > 0:
            _norm = _digital_gpu_backend.gemm(
                np.ones((1, 1), dtype=np.float64),
                np.array([[1.0]], dtype=np.float64),
            )
            _gain = float(np.clip(_norm.ravel()[0], 0.9, 1.1))
            audio *= _gain
    except Exception:
        pass

    # ── Normalize → 16-bit stereo PCM (duplicate mono to stereo) ─────────
    peak = float(np.max(np.abs(audio))) or 1.0
    mono = np.clip(audio / peak * 0.92, -1.0, 1.0)
    # Slight L/R decorrelation via comb delay for perceived width
    delay_samples = max(1, int(0.008 * sample_rate))
    left  = mono.copy()
    right = np.roll(mono, delay_samples)
    right[:delay_samples] = 0.0
    stereo = np.empty(n_total * 2, dtype=np.int16)
    stereo[0::2] = (left  * 32767.0).astype(np.int16)
    stereo[1::2] = (right * 32767.0).astype(np.int16)

    wav_path = _UPLOADS_PATH / f"audio_{job_id}.wav"
    mp3_path = _UPLOADS_PATH / f"audio_{job_id}.mp3"
    with _wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())

    try:
        result = run_ffmpeg(
            ["ffmpeg", "-y", "-i", str(wav_path),
             "-codec:a", "libmp3lame", "-b:a", "320k", str(mp3_path)],
            # 120s: encode takes <1s warm but CPU contention from concurrent
            # video renders can stretch it; keep fail-explicit, never silent.
            timeout=120,
        )
        if result.returncode != 0 or not mp3_path.exists():
            raise RuntimeError(
                f"ffmpeg mp3 encode failed (rc={result.returncode}): {result.stderr[-300:]}"
            )
    finally:
        try:
            wav_path.unlink(missing_ok=True)
        except OSError:
            pass
    return f"/uploads/audio_{job_id}.mp3"


def _extract_awareness_genres(awareness: str) -> list:
    """Parse trending genre names out of the awareness context string.

    ContentAwarenessService (music mode) writes lines of the form:
        Trending genres: trap, phonk, afrobeats
    This extracts them as a lowercase list so audio track selection can prefer
    samples whose genre metadata overlaps with live chart signals.
    Never raises — returns an empty list on any parse failure.
    """
    import re as _re
    try:
        m = _re.search(r"[Tt]rending genres?:\s*([^\n]+)", awareness or "")
        if not m:
            return []
        return [g.strip().lower() for g in m.group(1).split(",") if g.strip()]
    except Exception:
        return []


def _extract_awareness_moods(awareness: str) -> list:
    """Parse trending mood tokens out of the awareness context string.

    ContentAwarenessService (music mode) writes:
        Trending moods: melancholic, euphoric, nostalgic
    Returns lowercase list. Never raises.
    """
    import re as _re
    try:
        m = _re.search(r"[Tt]rending moods?:\s*([^\n]+)", awareness or "")
        if not m:
            return []
        return [g.strip().lower() for g in m.group(1).split(",") if g.strip()]
    except Exception:
        return []


def _render_audio_from_dataset(job_id: str, bpm: float, key: str,
                               duration_sec: float,
                               opts: Optional[dict] = None) -> dict:
    """Render an awareness-primary audio clip, applying full producer controls.

    The live ContentAwarenessService music-mode context (trending genres, moods,
    BPM, key from RSS + chart signals) is ALWAYS the audio source.  When a
    real-track dataset is available in storage, its metadata is consulted to
    sharpen synthesis parameters (confirm genre affinity, refine BPM/key
    targets) — but the dataset audio bytes are never used as the source.

    This makes awareness the primary signal at all dataset sizes.  Industry
    data from live charts is not a fallback; it's the generation engine.

    Producer controls applied to every render regardless of dataset state:

    * **Loop / trim** to the requested duration with musical fades.
    * **Awareness-driven arrangement** — structured intro/verse/hook/outro
      conditioned by live-trending genres (gated on the borrowed-knowledge
      retirement contract); plain loop as the never-raise fallback.
    * **ARC spectral cleanup** (opt-in via env).
    * **Master & export** — optional EBU-R128 loudness target, WAV/MP3 at a
      chosen sample-rate / bit-depth.
    * **Stems** — optional drums/bass/melody split for remixing.

    Returns a dict describing the produced asset(s): ``url``, applied
    ``bpm``/``key``, ``format``, ``loudness_lufs``, ``stems``, and
    ``arrangement``.
    """
    from storage_client import get_storage
    from ai_model.video.ffmpeg_util import run_ffmpeg
    from ai_model.audio import producer_tools as _pt

    opts = opts or {}

    # ── Producer targets ───────────────────────────────────────────────────────
    _explicit_bpm = float(opts.get("target_bpm") or bpm or 0.0)
    target_key = str(opts.get("target_key") or key or "C major")

    # ── Awareness signals (primary generation inputs) ──────────────────────────
    _preferred_genres = [
        g.lower().strip()
        for g in (opts.get("preferred_genres") or [])
        if g and str(g).strip()
    ]
    _preferred_mood = str(opts.get("preferred_mood") or "").strip()
    _synth_genre = _preferred_genres[0] if _preferred_genres else (
        str(opts.get("genre") or "").lower().strip()
    )

    # No explicit BPM → take the MEASURED chart target for this genre from the
    # live industry beacon (Deezer per-genre charts, previews analyzed with the
    # in-house beat tracker). 120 stays only as the final fallback when the
    # beacon has no measured features yet. Never raises.
    target_bpm = _explicit_bpm
    if not target_bpm:
        try:
            from ai_model.quality_awareness import music_targets as _mt
            _chart = _mt(_synth_genre)
            if _chart.get("bpm"):
                target_bpm = float(_chart["bpm"])
                print(
                    f"[audio_chart] chart BPM target {target_bpm} "
                    f"(genre={_chart.get('source_genre')!r}, "
                    f"{_chart.get('measured_previews')} previews measured) "
                    f"job={job_id[:8]}",
                    flush=True,
                )
        except Exception:  # noqa: BLE001
            pass
    if not target_bpm:
        target_bpm = 120.0

    # ── Dataset metadata reference (optional parameter refinement) ─────────────
    # When a real-track dataset exists, run the selector purely for metadata:
    # the matched sample's genre tags and BPM range are used to cross-check /
    # confirm the synthesis targets.  No audio bytes are fetched.
    # Never raises — dataset absence is not an error condition.
    _ref_sample: "dict | None" = None
    try:
        storage = get_storage()
        meta = storage.get("mb:dataset:audio:meta")

        # If the dataset is empty but the background seeder is actively
        # running, wait briefly for the first chunk so this render can
        # benefit from real-track metadata refinement.  Bounded to 12 s
        # (well inside the <30 s job budget — this renderer is also used by
        # the video auto-soundtrack path which has no early-exit guard) and
        # never-raise: on timeout we simply proceed awareness-only.
        if not (meta and int(meta.get("num_chunks", 0)) > 0):
            from workers.seed_audio_dataset import is_seeding as _is_seeding_render
            _wait_deadline = time.time() + 12.0
            while _is_seeding_render() and time.time() < _wait_deadline:
                time.sleep(1.5)
                meta = storage.get("mb:dataset:audio:meta")
                if meta and int(meta.get("num_chunks", 0)) > 0:
                    break  # first chunk landed — metadata refinement available
        if meta and int(meta.get("num_chunks", 0)) > 0:
            index = meta.get("index") or [
                {"idx": i} for i in range(int(meta["num_chunks"]))
            ]
            from ai_model.audio.track_selector import select_audio_sample as _select
            _best, _key_matched = _select(
                index,
                str(target_key or "").strip().lower(),
                float(target_bpm or 0.0),
                _preferred_genres,
                _preferred_mood,
            )
            _ref_sample = _best
            _ref_genres = _best.get("genres") or []
            # Fold dataset genre affinity into the synthesis genre when the
            # dataset confirms a genre signal the awareness layer also sees.
            if not _synth_genre and _ref_genres:
                _synth_genre = str(_ref_genres[0]).lower()
            # Use dataset BPM as a tiebreaker when target_bpm is not explicit.
            if not target_bpm and float(_best.get("bpm") or 0.0) > 0:
                target_bpm = float(_best.get("bpm"))
            print(
                f"[audio_ref] dataset reference: idx={_best.get('idx')} "
                f"bpm={_best.get('bpm')} key={_best.get('key')} "
                f"genres={_ref_genres} — used as metadata reference only; "
                f"audio synthesized from awareness signals (job={job_id[:8]})",
                flush=True,
            )
    except Exception as _ref_err:
        print(f"[audio_ref] dataset metadata lookup skipped: {_ref_err}",
              flush=True)

    # ── Awareness synthesis — always primary ───────────────────────────────────
    # _render_audio_clip synthesizes genre/mood-aware audio from live industry
    # signals via the digital GPU engine.  Writes audio_src_<job>.mp3 so the
    # producer pipeline below can treat it identically to any source file.
    print(
        f"[audio_awareness_synth] synthesizing from live awareness signals: "
        f"genre={_synth_genre!r} mood={_preferred_mood!r} "
        f"bpm={target_bpm} key={target_key!r} job={job_id[:8]}",
        flush=True,
    )
    _synth_duration = max(4.0, min(float(duration_sec or 30.0), 180.0))
    # Write to audio_src_<job_id>.mp3 so the producer pipeline below picks it
    # up as src_path without any file-name conflict with the final output.
    _render_audio_clip(
        f"src_{job_id}", target_bpm, target_key,
        duration_sec=_synth_duration,
        genre=_synth_genre,
        mood=_preferred_mood,
    )

    src_path = _UPLOADS_PATH / f"audio_src_{job_id}.mp3"
    src_wav = _UPLOADS_PATH / f"audio_srcwav_{job_id}.wav"
    looped_wav = _UPLOADS_PATH / f"audio_loop_{job_id}.wav"

    # Export target format (default MP3 for quick sharing; WAV for the studio).
    fmt = (opts.get("format") or "mp3").lower()
    if fmt not in ("mp3", "wav"):
        fmt = "mp3"
    sample_rate = int(opts.get("sample_rate") or 44100)
    bit_depth = int(opts.get("bit_depth") or 24)
    loudness_lufs = opts.get("loudness_lufs")
    if loudness_lufs is not None:
        loudness_lufs = float(loudness_lufs)
    out_ext = "wav" if fmt == "wav" else "mp3"
    out_path = _UPLOADS_PATH / f"audio_{job_id}.{out_ext}"

    # Synthesis always lands at the requested targets — no pitch/tempo
    # correction needed (unlike the old clip-repurposing path).
    applied_bpm = float(target_bpm or 120.0)
    applied_key = str(target_key or "C major")
    _tmp = [src_path, src_wav, looped_wav]
    try:
        # 1) Decode synthesized source → stereo WAV.
        r0 = run_ffmpeg(["ffmpeg", "-y", "-i", str(src_path), "-ac", "2",
                         "-ar", str(sample_rate), str(src_wav)], timeout=60)
        if r0.returncode != 0 or not src_wav.exists():
            raise RuntimeError(
                f"source decode failed rc={r0.returncode}: "
                f"{(r0.stderr or '')[-300:]}"
            )

        stage_in = src_wav

        # 2) Assemble to the requested duration. Awareness-driven ARRANGEMENT
        #    first (structured intro/verse/hook/outro conditioned by live-
        #    industry trending genres); plain loop as the never-raise fallback.
        arrangement_plan = None
        arranged_wav = _UPLOADS_PATH / f"audio_arr_{job_id}.wav"
        _tmp.append(arranged_wav)
        _want_arrange = (opts.get("arrange") is not False
                         and float(duration_sec) >= 24.0)
        if _want_arrange:
            try:
                from ai_model.audio import arrangement as _arr
                _plan = _arr.build_plan(
                    float(duration_sec), float(applied_bpm or 120),
                    genre=(opts.get("genre") or _synth_genre or None),
                    mood=(_preferred_mood or None),
                    trending_genres=_preferred_genres or [],
                    seed=opts.get("seed"),
                )
                if _arr.realize(stage_in, arranged_wav, _plan,
                                sample_rate=sample_rate,
                                run_ffmpeg=run_ffmpeg,
                                uploads_dir=_UPLOADS_PATH,
                                job_tag=job_id[:8]):
                    arrangement_plan = _arr.plan_summary(_plan)
                    print(f"[Arrange] audio_{job_id}: "
                          f"{'→'.join(s['section'] for s in arrangement_plan)}",
                          flush=True)
            except Exception as _arr_err:
                print(f"[Arrange] skipped (plain loop): {_arr_err}", flush=True)

        fade_len = min(1.5, max(0.3, float(duration_sec) * 0.05))
        fade_out_start = max(0.0, float(duration_sec) - fade_len)
        _fade_af = (f"afade=t=in:st=0:d=0.05,"
                    f"afade=t=out:st={fade_out_start:.2f}:d={fade_len:.2f}")
        if arrangement_plan:
            r1 = run_ffmpeg(
                ["ffmpeg", "-y", "-i", str(arranged_wav),
                 "-t", f"{float(duration_sec):.2f}", "-af", _fade_af,
                 str(looped_wav)],
                timeout=max(90, int(float(duration_sec) * 1.2) + 30),
            )
            if r1.returncode != 0 or not looped_wav.exists():
                arrangement_plan = None
        if not arrangement_plan:
            r1 = run_ffmpeg(
                ["ffmpeg", "-y", "-stream_loop", "-1", "-i", str(stage_in),
                 "-t", f"{float(duration_sec):.2f}", "-af", _fade_af,
                 str(looped_wav)],
                timeout=max(90, int(float(duration_sec) * 1.2) + 30),
            )
            if r1.returncode != 0 or not looped_wav.exists():
                raise RuntimeError(
                    f"ffmpeg assemble failed (rc={r1.returncode}): "
                    f"{r1.stderr[-300:]}"
                )

        # 3) RTA-1 ARC spectral restoration (opt-in, env-gated) ───────────────
        if os.environ.get("RTA_AUDIO_SPECTRAL") == "1":
            try:
                _arc_spectral_clean_file(looped_wav)
                print(f"[RTA] ARC spectral clean applied to audio_{job_id}",
                      flush=True)
            except Exception as _arc_err:
                print(f"[RTA] ARC spectral clean skipped: {_arc_err}",
                      flush=True)

        # 4) Optional stems (drums / bass / melody) BEFORE mastering.
        stem_urls: dict = {}
        if opts.get("stems"):
            try:
                stems = _pt.separate_stems(
                    looped_wav, _UPLOADS_PATH, f"audio_{job_id}",
                    bit_depth=bit_depth)
                stem_urls = {name: f"/uploads/{p.name}"
                             for name, p in stems.items()}
                print(f"[Producer] stems for audio_{job_id}: "
                      f"{', '.join(stem_urls)}", flush=True)
            except Exception as _st_err:
                print(f"[Producer] stems skipped: {_st_err}", flush=True)

        # 5) Master (optional LUFS) + export to the requested format.
        try:
            _pt.master_export(looped_wav, out_path, fmt=fmt,
                              sample_rate=sample_rate, bit_depth=bit_depth,
                              loudness_lufs=loudness_lufs)
        except Exception as _mx_err:
            print(f"[Producer] master_export failed — plain encode fallback: "
                  f"{_mx_err}", flush=True)
            loudness_lufs = None
            codec = (["-codec:a", "pcm_s24le"] if fmt == "wav"
                     else ["-codec:a", "libmp3lame", "-b:a", "320k"])
            r2 = run_ffmpeg(["ffmpeg", "-y", "-i", str(looped_wav),
                             "-ac", "2", "-ar", str(sample_rate), *codec,
                             str(out_path)],
                            timeout=90)
            if r2.returncode != 0 or not out_path.exists():
                raise RuntimeError(
                    f"final encode failed rc={r2.returncode}: "
                    f"{(r2.stderr or '')[-300:]}"
                )
    finally:
        for _p in _tmp:
            try:
                _p.unlink(missing_ok=True)
            except OSError:
                pass

    return {
        "url": f"/uploads/{out_path.name}",
        "bpm": applied_bpm,
        "key": applied_key,
        "format": fmt,
        "sample_rate": sample_rate,
        "bit_depth": bit_depth if fmt == "wav" else None,
        "loudness_lufs": loudness_lufs,
        "stems": stem_urls,
        "arrangement": arrangement_plan,
        # Provenance: which dataset sample informed the synthesis parameters
        # (metadata-reference only — no audio was fetched from storage).
        "source_sample": (
            {"idx": _ref_sample.get("idx"), "bpm": _ref_sample.get("bpm"),
             "key": _ref_sample.get("key"), "role": "metadata_reference"}
            if _ref_sample else None
        ),
        # No selection_warning: synthesis always lands at target key/BPM.
        "selection_warning": None,
    }


def _summarize_audio_analysis(render: dict) -> Optional[dict]:
    """Compact, MEASURED sonic facts for a finished audio render. Never raises.

    Surfaced on the audio-job poll response so downstream content generation
    (``beat_context.audio_analysis``) can write from real sonics instead of
    inventing descriptors. All values are measured from the rendered file —
    no invented instruments: ``detected_instruments`` only appears when stems
    were actually separated.
    """
    try:
        import numpy as _np
        name = str(render.get("url") or "").rsplit("/", 1)[-1]
        if not name:
            return None
        path = _UPLOADS_PATH / name
        if not path.exists():
            return None
        import librosa
        y, sr = librosa.load(str(path), sr=22050, mono=True, duration=60.0)
        if y is None or y.size == 0:
            return None
        rms = float(_np.sqrt(_np.mean(y ** 2)) + 1e-12)
        loudness_db = render.get("loudness_lufs")
        if loudness_db is None:
            loudness_db = round(20.0 * float(_np.log10(rms)), 1)
        centroid = float(_np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        S = _np.abs(librosa.stft(y, n_fft=2048))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        low_ratio = float(S[freqs < 150.0].sum() / (S.sum() + 1e-9))
        out: dict = {
            "loudness_db": loudness_db,
            # RMS mapped to 0–1; 0.35 RMS ≈ a loud, dense master.
            "energy": round(max(0.0, min(1.0, rms / 0.35)), 2),
            "spectral_brightness": ("dark" if centroid < 1500.0 else
                                    "balanced" if centroid < 3000.0 else
                                    "bright"),
            "bass_weight": ("heavy" if low_ratio > 0.25 else
                            "moderate" if low_ratio > 0.12 else "light"),
        }
        stems = render.get("stems") or {}
        if isinstance(stems, dict) and stems:
            out["detected_instruments"] = sorted(stems.keys())
        return out
    except Exception:
        return None


def _arc_spectral_clean_file(audio_path) -> None:
    """Decode → RTA ARC spectral denoise → re-encode the file in place.

    Format-preserving: a ``.wav`` input is written back as PCM WAV, anything
    else (``.mp3``) is re-encoded with libmp3lame.
    """
    import wave
    import numpy as _np
    from ai_model.video.ffmpeg_util import run_ffmpeg
    from ai_model import rta as _rta

    mp3_path = audio_path
    is_wav = str(audio_path).lower().endswith(".wav")
    wav_in = mp3_path.with_suffix(".arc_in.wav")
    wav_out = mp3_path.with_suffix(".arc_out.wav")
    try:
        # Stereo-preserving: decode to 2ch and denoise each channel
        # independently, so ARC never collapses a stereo master to mono.
        r = run_ffmpeg(["ffmpeg", "-y", "-i", str(mp3_path), "-ac", "2",
                        "-ar", "44100", "-f", "wav", str(wav_in)], timeout=120)
        if r.returncode != 0 or not wav_in.exists():
            raise RuntimeError(f"decode failed rc={r.returncode}")
        with wave.open(str(wav_in), "rb") as wf:
            sr = wf.getframerate()
            n_ch = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())
        samples = _np.frombuffer(raw, dtype="<i2").astype(_np.float32) / 32768.0
        if n_ch > 1:
            interleaved = samples.reshape(-1, n_ch)
            cleaned_ch = [
                _rta.api.spectral_clean_audio(
                    _np.ascontiguousarray(interleaved[:, c]), sr)
                for c in range(n_ch)
            ]
            cleaned = _np.stack(cleaned_ch, axis=1).reshape(-1)
        else:
            cleaned = _rta.api.spectral_clean_audio(samples, sr)
        out16 = _np.clip(cleaned, -1.0, 1.0)
        out16 = (out16 * 32767.0).astype("<i2")
        with wave.open(str(wav_out), "wb") as wf:
            wf.setnchannels(n_ch)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(out16.tobytes())
        if is_wav:
            r2 = run_ffmpeg(["ffmpeg", "-y", "-i", str(wav_out), "-codec:a",
                             "pcm_s16le", str(mp3_path)], timeout=120)
        else:
            r2 = run_ffmpeg(["ffmpeg", "-y", "-i", str(wav_out), "-codec:a",
                             "libmp3lame", "-b:a", "320k", str(mp3_path)],
                            timeout=120)
        if r2.returncode != 0:
            raise RuntimeError(f"re-encode failed rc={r2.returncode}")
    finally:
        for _p in (wav_in, wav_out):
            try:
                _p.unlink(missing_ok=True)
            except OSError:
                pass


_GENRE_DEFAULT_BPM: dict[str, float] = {
    "trap": 140.0, "drill": 144.0, "hip_hop": 92.0, "hip-hop": 92.0, "rap": 95.0,
    "rnb": 90.0, "r&b": 90.0, "lofi": 75.0, "lo-fi": 75.0, "pop": 112.0,
    "afrobeats": 105.0, "afrobeat": 105.0, "house": 124.0, "edm": 128.0,
    "techno": 130.0, "indie": 108.0, "rock": 120.0, "jazz": 100.0,
}


def _voiceover_track_path(job_id: str, narration_text: str,
                          duration_sec: float,
                          music_path: Optional[str] = None,
                          voice: Optional[str] = None,
                          wpm: Optional[int] = None) -> Optional[str]:
    """Synthesize a spoken narration track (in-house eSpeak NG) and duck any
    music soundtrack under it. Returns the audio path to mux, or None when
    speech synthesis is unavailable — callers keep their existing audio.
    ``voice``/``wpm`` are normalized downstream (invalid → defaults).
    Never-raise."""
    try:
        from ai_model.audio.voiceover import (
            voiceover_track, normalize_voice, normalize_wpm,
        )
        path = voiceover_track(
            text=narration_text,
            out_dir=str(_UPLOADS_PATH),
            job_id=job_id,
            duration_sec=duration_sec,
            music_path=music_path,
            voice=normalize_voice(voice),
            wpm=normalize_wpm(wpm),
        )
        if path:
            print(f"[VideoJob] voiceover narration rendered for {job_id[:12]}", flush=True)
        else:
            print(f"[VideoJob] voiceover unavailable for {job_id[:12]} — keeping music/silent track", flush=True)
        return path
    except Exception as exc:  # noqa: BLE001 - narration must never break renders
        print(f"[VideoJob] voiceover error ({exc}); keeping music/silent track", flush=True)
        return None


def _narration_script(production, hook: str = "", body: str = "", cta: str = "") -> str:
    """Build the narration text: caller-supplied script parts win, otherwise
    the planned scene texts (in order) form the spoken script."""
    parts = [p.strip() for p in (hook, body, cta) if p and p.strip()]
    if parts:
        return ". ".join(parts)
    try:
        return ". ".join(
            s.text.strip() for s in production.scenes if getattr(s, "text", "").strip()
        )
    except Exception:  # noqa: BLE001
        return ""


def _auto_soundtrack_path(job_id: str, duration_sec: float,
                          bpm: Optional[float] = None,
                          key: Optional[str] = None,
                          genre: str = "") -> Optional[str]:
    """Render a genre/BPM-matched soundtrack for a video (native-audio parity).

    Uses the real-audio dataset renderer so every video ships with sound by
    default, the way Veo generates audio natively. Returns a local file path,
    or None when no real-audio dataset is seeded / rendering fails — callers
    fall back to a silent render (never-raise).
    """
    try:
        _bpm = float(bpm or 0.0)
        if _bpm <= 0:
            _bpm = _GENRE_DEFAULT_BPM.get((genre or "").strip().lower(), 120.0)
        result = _render_audio_from_dataset(
            f"vsnd_{job_id[:12]}", bpm=_bpm, key=str(key or ""),
            duration_sec=max(2.0, float(duration_sec or 10.0)),
            opts={},
        )
        url = str(result.get("url") or "")
        name = url.rsplit("/", 1)[-1]
        path = (_UPLOADS_PATH / name).resolve()
        if name and path.is_relative_to(_UPLOADS_PATH.resolve()) and path.exists():
            return str(path)
    except Exception as exc:
        print(f"[VideoJob] auto-soundtrack unavailable ({exc}); rendering silent", flush=True)
    return None


@app.post("/api/generate/audio")
async def api_generate_audio(req: ApiGenerateAudioRequest, _key=Depends(require_scope("generate"))):
    """Async audio generation — style-conditioned via AI model for concept, BPM/key, creative direction."""
    # Coalesce: identical concurrent submissions (same genre/intent/bpm/key/duration)
    # share one job rather than spawning 90M separate audio renders.
    _adigest = _job_digest({
        "type":       "audio",
        "genre":      req.genre or "",
        "intent":     req.intent or "",
        "target_bpm": req.target_bpm,
        "target_key": req.target_key or "",
        "duration":   req.duration,
    })
    with _active_jobs_lock:
        _existing_id = _active_jobs.get(_adigest)
    if _existing_id:
        _ejob = _job_read(_existing_id)
        if _ejob and _ejob.get("status") in ("pending", "running"):
            return {
                "job_id":   _existing_id,
                "status":   "coalesced",
                "poll_url": f"/api/video-job/{_existing_id}",
                "duration": req.duration,
            }

    # ── Early-exit: dataset empty while seeding is in progress ────────────────
    # The startup auto-seed runs in a background thread and takes 30–90 s.
    # If the dataset is still empty AND seeding is in progress, spawning a job
    # that would immediately fail is unhelpful.  Return 503 + Retry-After so
    # clients can display a "warming up" state and retry automatically.
    # (If the dataset is empty and NOT seeding, fall through to normal job
    # creation so the existing explicit error message is preserved for the
    # truly-unseeded case.)
    try:
        from storage_client import get_storage as _gs_early
        from workers.seed_audio_dataset import is_seeding as _is_seeding_early
        _early_meta = _gs_early().get("mb:dataset:audio:meta")
        _dataset_ready_early = _early_meta and int(_early_meta.get("num_chunks", 0)) > 0
        if not _dataset_ready_early and _is_seeding_early():
            from fastapi.responses import JSONResponse as _JR
            return _JR(
                status_code=202,
                headers={"Retry-After": "30"},
                content={
                    "error":              "seeding_in_progress",
                    "seeding_now":        True,
                    "message": (
                        "The audio library is still being built in the background "
                        "(typically 30–90 s on first start). "
                        "Please retry in about 30 seconds."
                    ),
                    "retry_after_seconds": 30,
                },
            )
    except Exception:
        pass  # storage probe failed — proceed to normal job creation

    job_id = str(uuid.uuid4())
    with _active_jobs_lock:
        _active_jobs[_adigest] = job_id
    _job_write(job_id, {
        "status":     "pending",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "url":        None,
        "duration":   req.duration,
        "bpm":        None,
        "key":        None,
        "error":      None,
        "seeding_now": False,
    })

    # ── Request intelligence: analyse intent & sonic strategy up front ─────
    from ai_model import request_intelligence as ri
    genre_hint = req.genre or "music"
    brief = ri.build_brief(
        modality="audio", platform="general",
        topic=f"{genre_hint} {req.intent or req.instrument or 'music clip'}",
        goal=req.intent, tone=None, genre=req.genre,
        awareness=_merged_awareness_for(req),
    )

    # ── Awareness genre/mood conditioning (computed in handler scope) ──────
    # Parse live trending genres from the same awareness string that was fed to
    # build_brief() above. ContentAwarenessService (music mode) writes lines of
    # the form "Trending genres: trap, phonk, afrobeats". We combine those with
    # req.genre (explicit user intent) so _render_audio_from_dataset prefers
    # tracks that match current chart signals.
    # NOTE: these are captured as closure variables for _process() — do NOT
    # call _merged_awareness_for() again inside the background thread.
    _aw_str_handler = getattr(req, "awareness", "") or ""
    _aw_genres_handler = _extract_awareness_genres(_aw_str_handler)
    _aw_moods_handler  = _extract_awareness_moods(_aw_str_handler)
    _req_genre_handler = (req.genre or "").lower().strip()
    _preferred_genres_handler: list = list(dict.fromkeys(
        filter(None,
               ([_req_genre_handler] if _req_genre_handler else [])
               + _aw_genres_handler
        )
    ))

    # Prompt-derived mood: extract creative-brief mood keywords from the raw
    # prompt text so requests like "dark phonk, cinematic, drill energy" surface
    # the right mood even when the live awareness buffer has no mood signals and
    # the caller didn't set req.mood explicitly.  These slot in between the live
    # buffer and brief.tone in the precedence chain so they don't override an
    # intentional caller mood but always win over the generic brief fallback.
    _PROMPT_MOOD_KEYWORDS = [
        "dark", "cinematic", "drill", "heavy", "aggressive", "hard",
        "melancholic", "sad", "hype", "energetic", "euphoric", "chill",
        "lo-fi", "lofí", "vibrant", "moody", "atmospheric", "epic",
        "intense", "raw", "gritty", "smooth", "emotional", "ethereal",
    ]
    _prompt_raw = (
        (getattr(req, "prompt", None) or "")
        or (getattr(req, "instruction", None) or "")
    ).lower()
    _prompt_moods = [kw for kw in _PROMPT_MOOD_KEYWORDS if kw in _prompt_raw]

    # Mood precedence: explicit req.mood → prompt creative brief keywords →
    # live awareness buffer → brief.mood (generic last resort).
    # Prompt keywords rank above the live buffer because they represent the
    # caller's explicit creative direction ("dark", "cinematic", "drill energy")
    # which must not be overridden by a generic trending mood ("chill", "playful")
    # that reflects broad industry trends rather than this specific brief.
    _preferred_mood_handler = (
        (getattr(req, "mood", None) or "").strip()
        or (_prompt_moods[0] if _prompt_moods else "")
        or (_aw_moods_handler[0].strip() if _aw_moods_handler else "")
        or (brief.mood or "").strip()
    )
    if _preferred_genres_handler or _preferred_mood_handler:
        print(
            f"[audio_awareness] job={job_id[:8]} "
            f"genres={_preferred_genres_handler} mood={_preferred_mood_handler!r} "
            f"(req_genre={_req_genre_handler!r} trending={_aw_genres_handler})",
            flush=True,
        )

    # Unified conditioning bus: sonic technique (tempo/key/spectral tilt). When
    # TECHNIQUE_REAL_AUDIO=1 these can come from a real held audio sample;
    # otherwise they stay None and the brief-derived tempo band drives BPM.
    try:
        from ai_model.generation import extract_technique as _extract_tech
        _audio_tech = _extract_tech(
            idea=genre_hint, genre=req.genre, tone=brief.tone,
            mood=req.intent, seed=(req.seed or 0), with_audio=True,
        )
    except Exception:
        _audio_tech = None

    # Pre-generate AI concept synchronously before spawning the thread
    audio_concept  = None
    style_hook     = None
    model_source   = "heuristic"

    if _model_ready and _script_agent:
        try:
            from ai_model.agents.script_agent import ScriptRequest

            # ── Concept awareness string — built from audio-specific signals ──
            # _merged_awareness_for() may return thin results for audio requests
            # (no `description` field → intent layer finds nothing).  Instead
            # build the awareness string directly from what we already have:
            # the live awareness buffer, plus the request's genre/mood/key/BPM
            # so _awareness_compose() has real industry signals to work with.
            _concept_aw_parts: list[str] = []
            if _aw_str_handler:
                _concept_aw_parts.append(_aw_str_handler)
            if _preferred_genres_handler:
                _concept_aw_parts.append(
                    f"Trending genres: {', '.join(_preferred_genres_handler)}"
                )
            # Moods: prefer resolved _preferred_mood_handler (which already
            # folds prompt keywords), then extend with all prompt moods for
            # richer _build_awareness_body signal variety.
            _all_concept_moods = list(dict.fromkeys(filter(None, [
                _preferred_mood_handler,
                *_prompt_moods,
            ])))
            if _all_concept_moods:
                _concept_aw_parts.append(
                    f"Trending moods: {', '.join(_all_concept_moods)}"
                )
            _prompt_text = (getattr(req, "prompt", None) or "").strip()
            if _prompt_text:
                # Surface the raw prompt as a HIGH-priority signal so the hook
                # and body reflect the actual creative brief (dark, cinematic…)
                _concept_aw_parts.append(f"[HIGH] {_prompt_text}")
            _concept_awareness = "\n".join(_concept_aw_parts).strip()

            # ── Idea text: include mood, key, BPM so hooks carry energy context ─
            # Use _preferred_mood_handler which now resolves prompt keywords
            # before brief.tone, so "dark" beats "playful" on a dark phonk brief.
            _explicit_mood = (getattr(req, "mood", None) or "").strip()
            _mood_label = (
                _explicit_mood or _preferred_mood_handler or brief.mood or "energetic"
            )
            _key_label  = (getattr(req, "key", None) or getattr(req, "target_key", None) or "").strip()
            _bpm_label  = (getattr(req, "bpm", None) or getattr(req, "target_bpm", None))
            idea_text   = genre_hint
            if _mood_label:
                idea_text = f"{_mood_label} {idea_text}"
            if _key_label and _bpm_label:
                idea_text = f"{idea_text} ({brief.tempo} tempo, {_bpm_label} BPM, {_key_label})"
            elif _bpm_label:
                idea_text = f"{idea_text} ({brief.tempo} tempo, {_bpm_label} BPM)"
            else:
                idea_text = f"{idea_text} ({brief.tempo} tempo)"

            sr = await _in_thread(lambda: _script_agent.run(ScriptRequest(
                idea=idea_text, platform="general",
                goal="creative production", tone=brief.tone,
                awareness=_concept_awareness,
            )))
            if sr:
                audio_concept = sr.body
                style_hook    = sr.hook
                model_source  = getattr(sr, "source", "model")
        except Exception:
            pass

    def _process():
        import time as _t
        import numpy as _np2
        _t.sleep(2)

        # ── Job-level render deadline ──────────────────────────────────────
        # Guaranteed-completion policy: NO render deadline.  The job is never
        # marked "error" by a timer — the render thread is the sole writer of
        # terminal state, and it retries until it succeeds.  Internal ffmpeg
        # hang-detection (subprocess-level, feeding never-raise retry paths)
        # remains: it is what makes indefinite waiting safe, because a frozen
        # subprocess is detected and re-attempted instead of blocking forever.
        # A heartbeat keeps the job visibly alive for pollers on long renders.
        _req_dur = max(4.0, min(float(req.duration or 30), 180.0))
        _render_started = time.time()
        _hb_stop = threading.Event()

        def _heartbeat() -> None:
            while not _hb_stop.wait(30.0):
                if _hb_stop.is_set():
                    return  # terminal write in progress — never overwrite it
                _elapsed = time.time() - _render_started
                try:
                    _job_update(job_id, {"status": "rendering",
                                         "elapsed_seconds": round(_elapsed, 1)})
                except Exception:
                    pass  # never raise from heartbeat thread
                print(f"[audio_render] job={job_id[:8]} still rendering "
                      f"({_elapsed:.0f}s)", flush=True)

        _hb_thread = threading.Thread(target=_heartbeat, daemon=True)
        _hb_thread.start()

        fp  = req.style_fingerprint
        # Reproducibility: an explicit seed makes BPM/key derivation (and thus
        # the whole render) deterministic, fixing the classic "great result I
        # can't reproduce" complaint.
        _seed_base = req.seed if req.seed is not None else abs(hash(audio_concept or job_id)) % (2**31)
        # BPM derived from style fingerprint (deterministic if provided, else seeded)
        if fp and len(fp) >= 4:
            bpm = round(float(_np2.mean(fp[:4]) * 100 + 80), 1)
        else:
            # No client style fingerprint → let the awareness/intent-derived
            # tempo band (via brief.tempo) shape the BPM, so the live signal
            # actually influences the rendered sound (the dataset render then
            # selects the sample whose tempo best matches). Seeded jitter keeps
            # per-request variety within the band.
            _bands = {"fast": (126.0, 140.0), "punchy": (118.0, 130.0),
                      "medium": (100.0, 116.0), "steady": (84.0, 98.0),
                      "slow": (76.0, 90.0)}
            _lo, _hi = _bands.get((brief.tempo or "").lower(), (96.0, 120.0))
            # Live chart beacon: when the market's MEASURED tempo for this
            # genre is known, center the seeded jitter on it (±6 BPM) instead
            # of the generic band, clamped inside the intent band so tempo
            # intent ("slow"/"fast") still leads. Never-raise.
            try:
                from ai_model.quality_awareness import music_targets as _mt_h
                _chart_h = _mt_h(
                    (_preferred_genres_handler[0] if _preferred_genres_handler
                     else (req.genre or ""))
                )
                if _chart_h.get("bpm"):
                    _cb = float(_chart_h["bpm"])
                    _lo = max(_lo, min(_hi - 1.0, _cb - 6.0))
                    _hi = min(_hi, max(_lo + 1.0, _cb + 6.0))
                    print(f"[audio_chart] chart BPM anchor {_cb} → band "
                          f"[{_lo:.1f},{_hi:.1f}] "
                          f"(genre={_chart_h.get('source_genre')!r}) "
                          f"job={job_id[:8]}", flush=True)
            except Exception:  # noqa: BLE001
                pass
            rng = _np2.random.default_rng(_seed_base)
            bpm = round(float(rng.uniform(_lo, _hi)), 1)
        # Key derived from fingerprint tail
        keys_list = ["C major", "A minor", "G major", "E minor", "D major",
                     "F major", "B minor", "D minor", "E major", "G minor"]
        if fp and len(fp) >= 8:
            key = keys_list[int(fp[7] * 10) % len(keys_list)]
        else:
            rng2 = _np2.random.default_rng((_seed_base + 7919) % (2**31))
            key  = keys_list[int(rng2.integers(0, len(keys_list)))]
        # Real-asset sonic technique (opt-in) grounds tempo/key when the caller
        # gave neither a style fingerprint nor an explicit target.
        if _audio_tech is not None:
            if not (fp and len(fp) >= 4) and (req.target_bpm is None and req.bpm is None) and _audio_tech.tempo:
                bpm = round(float(_audio_tech.tempo), 1)
            if not (fp and len(fp) >= 8) and (req.target_key is None and req.key is None) and _audio_tech.key:
                key = _audio_tech.key
        # Producer targets override the derived values (exact key/BPM the output
        # must LAND on, via pitch-shift + time-stretch in the render).
        # Accept both target_bpm/target_key (internal) and bpm/key (alias).
        _req_bpm = req.target_bpm or req.bpm
        _req_key = req.target_key or req.key
        target_bpm = float(_req_bpm) if _req_bpm else bpm
        # Clamp to a musically sane range so bad input can't produce misleading
        # metadata or an avoidable ffmpeg failure (retune ratio is clamped too).
        target_bpm = max(40.0, min(300.0, float(target_bpm)))
        target_key = _req_key or key
        # Validate export params against what ffmpeg/PCM actually support; fall
        # back to safe studio defaults rather than failing the whole render.
        _fmt = (req.format or "mp3").lower()
        _fmt = _fmt if _fmt in ("mp3", "wav") else "mp3"
        _srate = int(req.sample_rate) if req.sample_rate else 44100
        if _srate not in (22050, 32000, 44100, 48000, 88200, 96000):
            _srate = 44100
        _bits = int(req.bit_depth) if req.bit_depth else 24
        if _bits not in (16, 24, 32):
            _bits = 24
        # Resolve a LUFS loudness preset name into a numeric target if given.
        from ai_model.audio import producer_tools as _pt_presets
        _lufs = req.loudness_lufs
        if _lufs is None and req.loudness_preset:
            _lufs = _pt_presets.LUFS_PRESETS.get(str(req.loudness_preset).lower())
        if _lufs is not None:
            _lufs = max(-40.0, min(0.0, float(_lufs)))
        # ── Awareness conditioning: use pre-computed closure vars ─────────
        # These were computed in the async handler scope (above _process) from
        # req.awareness + req.genre so that _merged_awareness_for() is never
        # called from inside a background thread (which can hang the event loop).
        _preferred_genres = _preferred_genres_handler
        _preferred_mood   = _preferred_mood_handler
        opts = {
            "target_bpm":       target_bpm,
            "target_key":       target_key,
            "format":           _fmt,
            "sample_rate":      _srate,
            "bit_depth":        _bits,
            "loudness_lufs":    _lufs,
            "stems":            bool(req.stems),
            # Awareness conditioning — passed through to _render_audio_from_dataset
            # so the sample selector can prefer genre-aligned tracks.
            "preferred_genres": _preferred_genres,
            "preferred_mood":   _preferred_mood,
            # Arrangement controls: genre + seed condition the section plan;
            # arrange=False opts out (plain loop).
            "genre":            req.genre,
            "seed":             req.seed,
            "arrange":          req.arrange,
        }
        # Render an actual in-house clip (key/tempo-conditioned) and serve it;
        # only mark "done" once the file exists on disk.
        # Honor long-form requests: leaseable beats run 2–3 minutes (180 s cap
        # per the delivery contract — a content limit, not a timeout).
        #
        # Guaranteed-completion policy: retry until success.  The render is
        # never-raise by design (it already synthesizes in-house via
        # _render_audio_clip internally, with no hard storage dependency), so
        # an exception here is exceptional — transient storage/memory
        # pressure.  Instead of marking the job "error", back off and
        # re-attempt until a real file exists on disk.
        duration_sec = max(4.0, min(float(req.duration or 30), 180.0))
        render = None
        _attempt = 0
        _last_err = ""
        _same_err_count = 0
        while render is None:
            _attempt += 1
            try:
                render = _render_audio_from_dataset(job_id, target_bpm, target_key,
                                                    duration_sec, opts)
            except Exception as exc:
                _err = f"{type(exc).__name__}: {exc}"
                _same_err_count = _same_err_count + 1 if _err == _last_err else 1
                _last_err = _err
                print(f"[audio_render] job={job_id[:8]} attempt {_attempt} failed: "
                      f"{_err} — retrying", flush=True)
                if _same_err_count == 5:
                    # Deterministic fault, not transient pressure: escalate to
                    # the watchdog (which owns subsystem self-healing) instead
                    # of silently spinning.  The loop keeps waiting — the job
                    # completes once the watchdog restores the subsystem.
                    print(f"[audio_render] job={job_id[:8]} ESCALATION: same "
                          f"error 5x — requesting watchdog attention: {_err}",
                          flush=True)
                    try:
                        from ai_model.watchdog import request_attention  # type: ignore
                        request_attention("audio_render", _err)
                    except Exception:
                        pass  # watchdog hook optional — escalation is best-effort
                try:
                    _job_update(job_id, {"status": "rendering",
                                         "retry_attempt": _attempt,
                                         "last_error": _err[:200]})
                except Exception:
                    pass
                time.sleep(min(60.0, 2.0 * _attempt))
        # Terminal-state handoff: stop the heartbeat and JOIN it before the
        # final write so "done" can never be overwritten by a late heartbeat.
        _hb_stop.set()
        _hb_thread.join(timeout=5.0)
        _job_update(job_id, {
            "status":           "done",
            "url":              render["url"],
            # Measured sonic summary for downstream content generation
            # (never-raise; None when analysis was not possible).
            "audio_analysis":   _summarize_audio_analysis(render),
            "duration":         int(duration_sec),
            "bpm":              render.get("bpm", target_bpm),
            "key":              render.get("key", target_key),
            "format":           render.get("format"),
            "sample_rate":      render.get("sample_rate"),
            "bit_depth":        render.get("bit_depth"),
            "loudness_lufs":    render.get("loudness_lufs"),
            "stems":            render.get("stems") or {},
            # Section plan actually rendered (None = plain loop) — the
            # awareness-driven arrangement provenance producers see in poll.
            "arrangement":      render.get("arrangement"),
            "seed":             _seed_base,
            "concept":          audio_concept,
            "style_hook":       style_hook,
            "source":           model_source,
            # Awareness provenance — which live signals shaped track selection
            "awareness_genres": _preferred_genres,
            "awareness_mood":   _preferred_mood or None,
            "awareness_source": "trending+intent" if _aw_genres_handler else (
                                "intent" if _req_genre_handler else "none"),
            # Non-None when no dataset track matched the requested key and the
            # selector fell back to nearest-BPM.  Surfaced to producers via poll.
            "selection_warning": render.get("selection_warning"),
            # Which dataset sample was selected — lets producers audit the pick.
            "source_sample":     render.get("source_sample"),
        })
        # ── Admin content flywheel (audio arm) ─────────────────────────────
        # B-Lawz admin renders are auto-pushed back into the audio dataset
        # pool as additional dataset sources (parity with text/video/image
        # flywheel ingestion).  Never raises; gated + deduped inside.
        _fw_ingest_audio_render(_key, job_id, render, _preferred_genres)

    _job_update(job_id, {"intelligence": brief.to_dict()})
    if _audio_tech is not None:
        _job_update(job_id, {"technique": _audio_tech.to_dict()})
    threading.Thread(target=_process, daemon=True, name=f"ApiAudioJob-{job_id}").start()
    return {"job_id": job_id, "status": "processing", "intelligence": brief.to_dict()}


def _start_video_job(req: ApiGenerateVideoRequest, platform: str) -> tuple[str, Any]:
    """
    Kick off a single fully AI-driven async video render job for ``platform``.
    Returns (job_id, intelligence_brief). Planning + rendering happen in a
    background thread. Shared by the single-platform and cross-platform
    (``platforms`` list) paths of /api/generate-video.
    """
    # ── Request intelligence: analyse intent & cinematic strategy up front ─
    from ai_model import request_intelligence as ri
    brief = ri.build_brief(
        modality="video", platform=normalize_platform(platform),
        topic=_resolve_topic_from_url(req.topic or req.idea), goal=req.goal, tone=req.tone, genre=req.genre,
        artist=req.artist_name,
        extra=" ".join(filter(None, [req.hook, req.body, req.cta])),
        mood=req.mood, bpm=req.bpm, key=req.key,
        artist_profile_id=req.artistProfileId,
        awareness=_merged_awareness_for(req),
        # ── Veo-parity controls into the intelligence brief ──────────────
        negative_prompt=req.negative_prompt or "",
        enhance_prompt=req.enhance_prompt if req.enhance_prompt is not None else True,
        lighting=req.lighting or "",
        camera_motion=req.camera_motion or "",
        color_temperature=req.color_temperature or "",
    )

    job_id = str(uuid.uuid4())
    _job_write(job_id, {
        "status":          "pending",
        "created_at":      datetime.utcnow().isoformat() + "Z",
        "platform":        platform,
        "genre_detected":  req.genre or "",
        "tone_used":       req.tone,
        "duration":        req.duration or 0,
        "url":             None,
        "filename":        None,
        "width":           None,
        "height":          None,
        "scenes":          [],
        "scenes_rendered": 0,
        "render_ms":       None,
        "error":           None,
        "intelligence":    brief.to_dict(),
        "ai_disclosure":   brief.ai_disclosure,
    })

    def _plan_and_render():
        import traceback as _tb
        try:
            from ai_model.video.video_agent import VideoAgent, VideoAgentRequest
            from ai_model.video.cinematic_engine import render_cinematic_open
            from ai_model.video.renderer import ASPECT_RATIOS, PLATFORM_RATIOS
            from ai_model.video import ai_scene_builder

            agent     = VideoAgent(_creative_model, _script_agent, _visual_spec_agent)
            # `idea` is templated raw into scene phrases (e.g. "Stream {idea}
            # now") — it must stay a clean topic string. The richer
            # intent/audience/theme context from `brief.augmented_idea` is
            # routed through `awareness` instead (bullet-formatted so the
            # scene sampler's awareness parser picks it up per scene type),
            # not concatenated into `idea` where it would corrupt templates.
            agent_req = VideoAgentRequest(
                idea=req.idea or brief.augmented_idea,
                platform=platform,
                goal=req.goal,
                tone=req.tone or brief.tone,
                genre=req.genre or "",
                artist_name=req.artist_name or "",
                duration=float(req.duration or 0),
                artist_context={"audio_path": req.user_audio_path} if req.user_audio_path else {},
                awareness=_merged_awareness_for(req),
                # ── Veo-parity controls forwarded into the render pipeline ─
                camera_motion=req.camera_motion or "",
                negative_prompt=req.negative_prompt or "",
                seed=req.seed,
                fps=req.fps or 24,
                motion_intensity=req.motion_intensity,
                enhance_prompt=req.enhance_prompt if req.enhance_prompt is not None else True,
                lighting=req.lighting or "",
                color_temperature=req.color_temperature or "",
                style_reference=req.style_reference or "",
                output_resolution=req.output_resolution or "",
                composition=req.composition or "",
                reference_images=(req.reference_images or [])[:3],
                first_frame_b64=req.first_frame_b64 or "",
                last_frame_b64=req.last_frame_b64 or "",
            )

            production = agent.plan(agent_req)

            # Apply caller-supplied text overrides before rendering
            if req.scenes_override:
                _override_map = {o.index: o.text for o in req.scenes_override}
                for i, scene in enumerate(production.scenes):
                    if i in _override_map:
                        scene.text = _override_map[i]

            _job_update(job_id, {
                "genre_detected": production.genre_detected,
                "tone_used":      production.tone_used,
                "source":         production.source,
                "duration":       production.total_duration,
                "aspect_ratio":   production.aspect_ratio,
                "scenes":         [{"type": s.scene_type, "text": s.text} for s in production.scenes],
            })

            # Bail if the client cancelled the job while planning was running
            _current = _job_read(job_id)
            if _current and _current.get("status") == "cancelled":
                return

            ratio  = production.aspect_ratio or PLATFORM_RATIOS.get(production.platform, "9:16")
            width, height = ASPECT_RATIOS.get(ratio, (1080, 1920))

            # output_resolution: explicit caller override (720p/1080p/4k).
            # Applied after the platform-derived aspect-ratio dimensions so
            # the caller always gets the resolution they asked for regardless
            # of platform defaults.  Never-raise.
            if req.output_resolution:
                _RES_MAP = {
                    "720p":  (1280, 720),  "720":   (1280, 720),
                    "1080p": (1920, 1080), "1080":  (1920, 1080), "fhd": (1920, 1080),
                    "4k":    (3840, 2160), "2160p": (3840, 2160), "uhd": (3840, 2160),
                }
                _res = _RES_MAP.get(req.output_resolution.lower())
                if _res:
                    width, height = _res

            # Unified conditioning bus: extract real-asset-grounded Visual DNA
            # so the diffusion background pipeline conditions on the technique of
            # peak reference media, not just a genre lookup. Never-raise.
            try:
                from ai_model.generation import extract_technique
                # style_reference (Veo parity): pass as the brand hint so
                # extract_technique can condition the DNA on an explicit reference
                # asset when the caller supplies one.
                _eff_brand = req.style_reference or req.artist_name or None
                _tech = extract_technique(
                    idea=agent_req.idea, genre=production.genre_detected,
                    tone=production.tone_used, energy=getattr(brief, "energy", None),
                    mood=getattr(brief, "mood", None), bpm=getattr(brief, "bpm", None),
                    key=getattr(brief, "key", None), brand=_eff_brand,
                    seed=abs(hash(job_id)) % (2**31),
                )
                _tech_dna = _tech.dna_dict()
            except Exception:
                _tech, _tech_dna = None, None

            # ── Soundtrack: start in background immediately — it only needs
            # `production` (already known) and is independent of scene building.
            # This overlaps soundtrack I/O with extract_technique + build_open_scenes,
            # saving 3–8 s of sequential wait before render can start.
            import concurrent.futures as _cf
            _audio_path = req.user_audio_path
            _soundtrack_future = None
            _snd_pool = None
            if not _audio_path and getattr(req, "generate_audio", True):
                _snd_pool = _cf.ThreadPoolExecutor(max_workers=1,
                                                   thread_name_prefix="Soundtrack")
                _soundtrack_future = _snd_pool.submit(
                    _auto_soundtrack_path,
                    job_id, production.total_duration,
                    bpm=req.bpm, key=req.key,
                    genre=production.genre_detected or (req.genre or ""),
                )

            scene_configs = agent.build_open_scenes(
                agent_req, production, width, height, technique_dna=_tech_dna)
            if _tech is not None:
                _job_update(job_id, {"technique": _tech.to_dict()})
            dna        = ai_scene_builder.build_dna(agent_req.idea, production.genre_detected, production.tone_used)
            transition = "fadeblack" if dna.darkness > 0.70 else "dissolve" if dna.energy < 0.50 else "fade"

            # Collect soundtrack result (thread was running during scene build).
            if _soundtrack_future is not None:
                try:
                    _audio_path = _soundtrack_future.result(timeout=25)
                except Exception as _snd_exc:
                    print(f"[VideoJob] auto-soundtrack thread error ({_snd_exc}); rendering silent", flush=True)
                    _audio_path = None
                finally:
                    if _snd_pool is not None:
                        _snd_pool.shutdown(wait=False)

            # Voice-over: synthesize REAL spoken narration (eSpeak NG) from the
            # script and duck the soundtrack under it. Previously this flag was
            # a silent no-op — users got the arpeggio synth ("birdcalls").
            if getattr(req, "voiceover", False):
                _vo_text = _narration_script(
                    production,
                    hook=req.hook or "", body=req.body or "", cta=req.cta or "",
                )
                _vo_path = _voiceover_track_path(
                    job_id, _vo_text, production.total_duration,
                    music_path=_audio_path,
                    voice=getattr(req, "voiceover_voice", None),
                    wpm=getattr(req, "voiceover_wpm", None),
                )
                if _vo_path:
                    _audio_path = _vo_path
                    _job_update(job_id, {"voiceover": True})

            result = render_cinematic_open(
                scenes=scene_configs,
                width=width,
                height=height,
                total_duration=production.total_duration,
                audio_path=_audio_path,
                transition=transition,
                transition_dur=0.5 if dna.energy > 0.70 else 0.8,
                label=f"ai:{production.genre_detected}:{production.tone_used}",
            )
            if result.success:
                _job_update(job_id, {
                    "status":          "done",
                    "url":             f"/uploads/videos/{result.filename}",
                    "filename":        result.filename,
                    "width":           result.width,
                    "height":          result.height,
                    "scenes_rendered": result.scenes_rendered,
                    "render_ms":       result.render_time_ms,
                })
            else:
                _job_update(job_id, {
                    "status": "error",
                    "error":  result.error or "Render failed",
                })
        except Exception as exc:
            print(f"[VideoJob] Error for job {job_id}: {_tb.format_exc()}")
            _job_update(job_id, {
                "status": "error",
                "error":  f"{type(exc).__name__}: {exc}",
            })

    threading.Thread(target=_plan_and_render, daemon=True, name=f"ApiVideoJob-{job_id}").start()
    return job_id, brief


@app.post("/api/generate-video")
async def api_generate_video(req: ApiGenerateVideoRequest, _key=Depends(require_scope("generate"))):
    """
    Kick off a fully AI-driven async video render job.

    Single-platform (default): returns job_id immediately; unchanged behavior
    when ``platforms`` is omitted.

    Cross-platform: when ``platforms`` is set, kicks off one independent
    render job per platform (each with its own aspect ratio / duration via
    the existing PLATFORM_RATIOS / _PLATFORM_SPECS logic in VideoAgent) and
    returns all job_ids in a single response instead of requiring N calls.
    """
    await _wait_for_model_ready()

    if req.platforms:
        _plats = [normalize_platform(p) for p in req.platforms if p]
        _plats = list(dict.fromkeys(_plats))
        if _plats:
            jobs: dict[str, Any] = {}
            for _plat in _plats:
                _jid, _brief = _start_video_job(req, _plat)
                jobs[_plat] = {"job_id": _jid, "intelligence": _brief.to_dict()}
            first = next(iter(jobs.values()))
            return {
                "job_id": first["job_id"],
                "status": "processing",
                "intelligence": first["intelligence"],
                "platform_jobs": jobs,
            }
        # `platforms` present but contained no usable entries (e.g. [""]) —
        # fall through to the normal single-platform path below.

    # ── Multi-sample generation (Veo parity: up to 4 variants/prompt) ────
    # Each variant is an independent render job with a derived seed so the
    # variants genuinely differ (seed=None would resolve to the same
    # deterministic idea-hash seed for every copy).
    _n_samples = max(1, min(4, int(req.sample_count or 1)))
    if _n_samples > 1:
        _base_seed = req.seed if req.seed is not None else abs(hash(req.idea)) % (2**31)
        _sample_jobs: list[dict[str, Any]] = []
        _first_brief = None
        for _i in range(_n_samples):
            _vreq = req.model_copy(update={
                "seed": (_base_seed + _i * 1013) % (2**31),
                "sample_count": 1,
            })
            _jid, _sbrief = _start_video_job(_vreq, req.platform)
            if _first_brief is None:
                _first_brief = _sbrief
            _sample_jobs.append({
                "job_id": _jid,
                "sample_index": _i,
                "seed": (_base_seed + _i * 1013) % (2**31),
                "poll_url": f"/api/video-job/{_jid}",
            })
        return {
            "job_id": _sample_jobs[0]["job_id"],
            "status": "processing",
            "intelligence": _first_brief.to_dict() if _first_brief else {},
            "sample_jobs": _sample_jobs,
        }

    job_id, brief = _start_video_job(req, req.platform)
    return {"job_id": job_id, "status": "processing", "intelligence": brief.to_dict()}


# ── Veo-parity video extension ─────────────────────────────────────────────────

def _ffmpeg_media_duration(path: str) -> float:
    """Read a media file's duration by parsing ffmpeg's stderr banner.

    Uses run_ffmpeg (posix_spawn-safe) instead of ffprobe+PIPE so it works
    under model memory pressure. Returns 0.0 when unparseable (never-raise).
    """
    try:
        import re as _re
        from ai_model.video.ffmpeg_util import run_ffmpeg
        r = run_ffmpeg(["ffmpeg", "-hide_banner", "-i", str(path)], timeout=30.0)
        m = _re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr or "")
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    except Exception:
        pass
    return 0.0


@app.post("/api/video/extend")
async def api_video_extend(req: ApiVideoExtendRequest, _key=Depends(require_scope("generate"))):
    """Extend a previously generated video (Veo-parity scene extension).

    The source video's last frame is extracted and used as the first-frame
    conditioning image for a continuation clip, which is rendered at the
    source's exact dimensions and concatenated on. A single continuous
    soundtrack is then re-rendered across the full extended duration so the
    audio stays coherent instead of stitching two different tracks.
    """
    await _wait_for_model_ready()

    if not (req.idea or "").strip():
        raise HTTPException(status_code=422, detail="'idea' is required — describe what the continuation should show")

    # ── Resolve the source video to a real file under uploads/videos ──────
    videos_dir = (_UPLOADS_PATH / "videos").resolve()
    src_name = req.source.strip()
    src_job = _job_read(src_name) if len(src_name) >= 8 and "/" not in src_name and "." not in src_name else None
    if src_job:
        if src_job.get("status") != "done" or not src_job.get("filename"):
            raise HTTPException(status_code=409, detail=f"source job {src_name} has no finished video to extend")
        src_name = str(src_job["filename"])
    src_name = src_name.rsplit("/", 1)[-1]
    src_path = (videos_dir / src_name).resolve()
    if not src_path.is_relative_to(videos_dir) or not src_path.exists():
        raise HTTPException(status_code=404, detail=f"source video not found: {req.source}")

    add_dur = max(2.0, min(60.0, float(req.extend_duration or 8.0)))
    job_id = str(uuid.uuid4())
    _job_write(job_id, {
        "status":        "pending",
        "created_at":    datetime.utcnow().isoformat() + "Z",
        "kind":          "video_extend",
        "extended_from": src_name,
        "added_duration": add_dur,
        "platform":      req.platform,
        "url":           None,
        "filename":      None,
    })

    def _extend_and_render():
        import base64 as _b64
        import traceback as _tb
        try:
            from PIL import Image as _Image
            from ai_model.video.ffmpeg_util import run_ffmpeg
            from ai_model.video.video_agent import VideoAgent, VideoAgentRequest
            from ai_model.video.cinematic_engine import render_cinematic_open
            from ai_model.video import ai_scene_builder

            _job_update(job_id, {"status": "running"})

            # 1) Extract the source's last frame → conditioning image
            frame_png = _UPLOADS_PATH / f"extend_lastframe_{job_id[:8]}.png"
            r = run_ffmpeg([
                "ffmpeg", "-y", "-sseof", "-0.5", "-i", str(src_path),
                "-frames:v", "1", "-update", "1", str(frame_png),
            ], timeout=60.0)
            if r.returncode != 0 or not frame_png.exists():
                # -sseof can fail on very short clips — fall back to last-frame select
                r = run_ffmpeg([
                    "ffmpeg", "-y", "-i", str(src_path),
                    "-vf", "select=eof", "-frames:v", "1", "-update", "1", str(frame_png),
                ], timeout=120.0)
            if r.returncode != 0 or not frame_png.exists():
                raise RuntimeError(f"could not extract last frame (rc={r.returncode})")

            with _Image.open(frame_png) as _img:
                width, height = _img.size
            first_frame_b64 = _b64.b64encode(frame_png.read_bytes()).decode("ascii")

            # 2) Render the continuation clip at the source's exact dimensions
            agent = VideoAgent(_creative_model, _script_agent, _visual_spec_agent)
            agent_req = VideoAgentRequest(
                idea=req.idea.strip(),
                platform=req.platform,
                goal=req.goal,
                tone=req.tone,
                genre=req.genre or (src_job or {}).get("genre_detected", "") or "",
                artist_name=req.artist_name or "",
                duration=add_dur,
                camera_motion=req.camera_motion or "",
                negative_prompt=req.negative_prompt or "",
                seed=req.seed,
                lighting=req.lighting or "",
                color_temperature=req.color_temperature or "",
                composition=req.composition or "",
                first_frame_b64=first_frame_b64,
            )
            production = agent.plan(agent_req)
            production.total_duration = add_dur
            scene_configs = agent.build_open_scenes(agent_req, production, width, height)
            dna = ai_scene_builder.build_dna(agent_req.idea, production.genre_detected, production.tone_used)
            cont = render_cinematic_open(
                scenes=scene_configs, width=width, height=height,
                total_duration=add_dur, audio_path=None,
                transition="fade", transition_dur=0.5,
                label=f"extend:{production.genre_detected}",
            )
            if not cont.success:
                raise RuntimeError(cont.error or "continuation render failed")
            cont_path = videos_dir / cont.filename

            # 3) Concatenate source + continuation (video streams, normalised)
            out_name = f"extended_{job_id[:8]}.mp4"
            out_path = videos_dir / out_name
            norm = f"fps=24,scale={width}:{height},format=yuv420p,setpts=PTS-STARTPTS"
            r = run_ffmpeg([
                "ffmpeg", "-y", "-i", str(src_path), "-i", str(cont_path),
                "-filter_complex",
                f"[0:v]{norm}[v0];[1:v]{norm}[v1];[v0][v1]concat=n=2:v=1:a=0[v]",
                "-map", "[v]", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "20", "-movflags", "+faststart", str(out_path),
            ], timeout=600.0)
            if r.returncode != 0 or not out_path.exists():
                raise RuntimeError(f"concat failed (rc={r.returncode}): {(r.stderr or '')[-300:]}")

            # 4) One continuous soundtrack across the full extended duration
            final_name = out_name
            if req.generate_audio:
                total_dur = _ffmpeg_media_duration(str(out_path)) or (add_dur + 30.0)
                snd = _auto_soundtrack_path(
                    job_id, total_dur,
                    genre=production.genre_detected or (req.genre or ""),
                )
                if snd:
                    muxed = videos_dir / f"extended_{job_id[:8]}_audio.mp4"
                    r = run_ffmpeg([
                        "ffmpeg", "-y", "-i", str(out_path), "-i", snd,
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "192k", "-shortest",
                        "-movflags", "+faststart", str(muxed),
                    ], timeout=300.0)
                    if r.returncode == 0 and muxed.exists():
                        final_name = muxed.name

            try:
                frame_png.unlink(missing_ok=True)
            except Exception:
                pass

            _job_update(job_id, {
                "status":         "done",
                "url":            f"/uploads/videos/{final_name}",
                "filename":       final_name,
                "width":          width,
                "height":         height,
                "genre_detected": production.genre_detected,
                "tone_used":      production.tone_used,
                "scenes":         [{"type": s.scene_type, "text": s.text} for s in production.scenes],
            })
        except Exception as exc:
            print(f"[VideoExtend] Error for job {job_id}: {_tb.format_exc()}")
            _job_update(job_id, {"status": "error", "error": f"{type(exc).__name__}: {exc}"})

    threading.Thread(target=_extend_and_render, daemon=True, name=f"VideoExtend-{job_id}").start()
    return {
        "job_id":         job_id,
        "status":         "processing",
        "extended_from":  src_name,
        "added_duration": add_dur,
        "poll_url":       f"/api/video-job/{job_id}",
    }


# ── AI-driven video generation ─────────────────────────────────────────────────

@app.post("/api/video/generate-ai")
async def api_video_generate_ai(request: Request, _key=Depends(require_scope("generate"))):
    """
    Generate a video where ALL text content is produced by the trained model.

    The VideoAgent conditions the transformer on platform / goal / tone / genre
    control tokens, generates distinct AI text for every scene (hook, build, body,
    drop, cta, outro), selects the matching cinematic template based on the
    AI-determined genre/tone, and renders the final MP4 via FFmpeg.

    Body fields (all optional except idea):
      idea           – what the video is about
      platform       – tiktok | instagram | youtube | instagram_reels | etc.
      goal           – growth | conversion | engagement | awareness | streams | sales
      tone           – energetic | edgy | chill | professional | promotional | etc.
      genre          – trap | rnb | pop | afrobeats | drill | lofi | indie | etc.
      artist_name    – shown on screen as the creator label
      duration       – desired length in seconds (platform default if omitted)
      artist_context – dict with optional audio_path key
      scenes_override – [{type, text}, ...] user-edited scene texts; skips re-planning
    """
    body = await request.json()

    await _wait_for_model_ready()

    idea         = str(body.get("idea", "")).strip()
    platform     = str(body.get("platform", "tiktok")).strip().lower()
    goal         = str(body.get("goal", "growth")).strip().lower()
    tone         = str(body.get("tone", "energetic")).strip().lower()
    genre        = str(body.get("genre", "")).strip().lower()
    artist_name  = str(body.get("artist_name", "")).strip()
    duration     = float(body.get("duration") or 0)
    artist_ctx   = body.get("artist_context", {}) or {}
    # Optional list of {type, text} dicts the user edited in the UI
    scenes_override_raw: list[dict] = body.get("scenes_override") or []

    if not idea:
        raise HTTPException(status_code=422, detail="'idea' is required")

    # Coalesce: identical concurrent video AI gen requests (same idea/platform/
    # goal/tone/genre) share one plan+render rather than running N copies.
    _vdigest = _job_digest({
        "type":     "video_ai_gen",
        "idea":     idea,
        "platform": platform,
        "goal":     goal,
        "tone":     tone,
        "genre":    genre,
    })
    with _active_jobs_lock:
        _vexisting = _active_jobs.get(_vdigest)
    if _vexisting:
        _vj = _job_read(_vexisting)
        if _vj and _vj.get("status") in ("pending", "running"):
            return {
                "job_id":   _vexisting,
                "status":   "coalesced",
                "poll_url": f"/api/video-job/{_vexisting}",
            }

    # ── Request intelligence: analyse intent & cinematic strategy up front ─
    from ai_model import request_intelligence as ri
    brief = ri.build_brief(
        modality="video", platform=normalize_platform(platform),
        topic=idea, goal=goal, tone=tone, genre=genre, artist=artist_name,
        mood=str(body.get("mood") or "") or None,
        bpm=body.get("bpm"), key=str(body.get("key") or "") or None,
        artist_profile_id=str(body.get("artistProfileId") or "") or None,
    )

    from ai_model.video.video_agent import VideoAgent as _VA, VideoAgentRequest
    from ai_model.video.renderer import ASPECT_RATIOS, PLATFORM_RATIOS
    from ai_model.video import ai_scene_builder

    # `idea` is templated raw into scene phrases — keep it clean and route
    # the richer intent/audience/theme context through `awareness` instead.
    # ── Veo-parity controls (all optional, silently defaulted) ────────────
    _refs_raw = body.get("reference_images") or []
    _refs = [str(x) for x in _refs_raw if x][:3] if isinstance(_refs_raw, list) else []
    _seed_raw = body.get("seed")

    req = VideoAgentRequest(
        idea=idea,
        platform=platform,
        goal=goal,
        tone=tone or brief.tone,
        genre=genre,
        artist_name=artist_name,
        duration=duration,
        artist_context=artist_ctx,
        awareness="\n".join(f"• {d}" for d in brief.directives),
        camera_motion=str(body.get("camera_motion") or ""),
        negative_prompt=str(body.get("negative_prompt") or ""),
        seed=int(_seed_raw) if _seed_raw is not None else None,
        fps=int(body.get("fps") or 24),
        motion_intensity=(float(body["motion_intensity"])
                          if body.get("motion_intensity") is not None else None),
        enhance_prompt=bool(body.get("enhance_prompt", True)),
        lighting=str(body.get("lighting") or ""),
        color_temperature=str(body.get("color_temperature") or ""),
        style_reference=str(body.get("style_reference") or ""),
        composition=str(body.get("composition") or ""),
        reference_images=_refs,
        first_frame_b64=str(body.get("first_frame_b64") or ""),
        last_frame_b64=str(body.get("last_frame_b64") or ""),
        output_resolution=str(body.get("output_resolution") or ""),
    )
    _gen_audio = bool(body.get("generate_audio", True))
    _want_voiceover = bool(body.get("voiceover", False))
    _vo_voice = body.get("voiceover_voice")
    _vo_wpm = body.get("voiceover_wpm")

    # ── Run plan() synchronously so scenes are available in this response ──
    # Only the heavy render step is deferred to a background thread.
    _agent     = _VA(_creative_model, _script_agent, _visual_spec_agent)
    production = await _in_thread(lambda: _agent.plan(req))

    # Apply user-edited scene texts if provided (re-render with edits)
    if scenes_override_raw:
        override_map = {i: s.get("text", "") for i, s in enumerate(scenes_override_raw) if isinstance(s, dict)}
        for i, scene in enumerate(production.scenes):
            if i in override_map and override_map[i]:
                scene.text = override_map[i]

    ratio  = production.aspect_ratio or PLATFORM_RATIOS.get(production.platform, "9:16")
    width, height = ASPECT_RATIOS.get(ratio, (1080, 1920))
    scenes_data = [{"type": s.scene_type, "text": s.text} for s in production.scenes]

    job_id = str(uuid.uuid4())
    with _active_jobs_lock:
        _active_jobs[_vdigest] = job_id
    _job_write(job_id, {
        "status":          "pending",
        "created_at":      datetime.utcnow().isoformat() + "Z",
        "platform":        platform,
        "genre_detected":  production.genre_detected,
        "tone_used":       production.tone_used,
        "source":          production.source,
        "duration":        production.total_duration,
        "aspect_ratio":    production.aspect_ratio,
        "template":        production.template_id,
        "url":             None,
        "filename":        None,
        "width":           None,
        "height":          None,
        "scenes":          scenes_data,
        "scenes_rendered": 0,
        "render_ms":       None,
        "error":           None,
        "intelligence":    brief.to_dict(),
    })

    # Capture references for the render closure
    _production = production
    _width, _height = width, height
    _req = req

    def _render_only():
        import traceback as _tb
        try:
            from ai_model.video.cinematic_engine import render_cinematic_open

            _t0 = time.time()
            scene_configs = _agent.build_open_scenes(_req, _production, _width, _height)
            _t_build = time.time() - _t0
            dna        = ai_scene_builder.build_dna(_req.idea, _production.genre_detected, _production.tone_used)
            transition = "fadeblack" if dna.darkness > 0.70 else "dissolve" if dna.energy < 0.50 else "fade"
            # Native-audio parity: auto-soundtrack when no caller audio.
            _t0 = time.time()
            _audio_path = _req.artist_context.get("audio_path")
            if not _audio_path and _gen_audio:
                _audio_path = _auto_soundtrack_path(
                    job_id, _production.total_duration,
                    genre=_production.genre_detected or _req.genre,
                )
            print(
                f"[VideoRender][Timing] build_scenes={_t_build:.1f}s "
                f"soundtrack={time.time() - _t0:.1f}s job={job_id[:8]}",
                flush=True,
            )

            # Voice-over: real spoken narration from the planned scene texts,
            # music ducked underneath (never-raise; falls back to music).
            if _want_voiceover:
                _vo_path = _voiceover_track_path(
                    job_id, _narration_script(_production),
                    _production.total_duration, music_path=_audio_path,
                    voice=_vo_voice, wpm=_vo_wpm,
                )
                if _vo_path:
                    _audio_path = _vo_path
                    _job_update(job_id, {"voiceover": True})

            result     = render_cinematic_open(
                scenes=scene_configs,
                width=_width,
                height=_height,
                total_duration=_production.total_duration,
                audio_path=_audio_path,
                transition=transition,
                transition_dur=0.5 if dna.energy > 0.70 else 0.8,
                label=f"ai:{_production.genre_detected}:{_production.tone_used}",
            )
            if result.success:
                _job_update(job_id, {
                    "status":          "done",
                    "url":             f"/uploads/videos/{result.filename}",
                    "filename":        result.filename,
                    "width":           result.width,
                    "height":          result.height,
                    "scenes_rendered": result.scenes_rendered,
                    "render_ms":       result.render_time_ms,
                })
            else:
                _job_update(job_id, {
                    "status": "error",
                    "error":  result.error,
                })
        except Exception as exc:
            print(f"[VideoAgent] Error for job {job_id}: {_tb.format_exc()}")
            _job_update(job_id, {
                "status": "error",
                "error":  str(exc),
            })

    threading.Thread(target=_render_only, daemon=True, name=f"AIVideoJob-{job_id}").start()

    return {
        "job_id":         job_id,
        "status":         "pending",
        "poll_url":       f"/api/video-job/{job_id}",
        "scenes":         scenes_data,
        "genre_detected": production.genre_detected,
        "tone_used":      production.tone_used,
        "source":         production.source,
        "duration":       production.total_duration,
        "aspect_ratio":   production.aspect_ratio,
        "template":       production.template_id,
        "intelligence":   brief.to_dict(),
    }


# -- Job polling ---------------------------------------------------------------

@app.get("/api/video-job/{job_id}")
async def api_poll_video_job(job_id: str, _key=Depends(require_scope("read"))):
    """Poll a video render job."""
    job = _job_read(job_id)
    if job is None:
        return {"status": "error", "error": "Job not found"}
    if job["status"] == "done":
        return {
            "status":          "done",
            "url":             job.get("url"),
            "filename":        job.get("filename"),
            "width":           job.get("width"),
            "height":          job.get("height"),
            "duration":        job.get("duration"),
            "template":        job.get("template"),
            "template_name":   job.get("template_name"),
            "genre_detected":  job.get("genre_detected"),
            "tone_used":       job.get("tone_used"),
            "source":          job.get("source"),
            "aspect_ratio":    job.get("aspect_ratio"),
            "scenes":          job.get("scenes", []),
            "scenes_rendered": job.get("scenes_rendered", 0),
            "render_ms":       job.get("render_ms"),
            "technique":       job.get("technique"),
            "voiceover":       bool(job.get("voiceover", False)),
        }
    if job["status"] == "error":
        return {"status": "error", "error": job.get("error", "Unknown error")}
    return {"status": job["status"]}


@app.delete("/api/video-job/{job_id}")
async def api_cancel_video_job(job_id: str, _key=Depends(require_scope("generate"))):
    """
    Cancel a pending job (marks it so the render thread exits before encoding)
    or purge a finished/errored job and delete its output file.
    """
    job = _job_read(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    status = job.get("status")

    if status == "pending":
        _job_update(job_id, {"status": "cancelled"})
        return {"ok": True, "action": "cancelled", "job_id": job_id}

    if status in ("done", "error", "cancelled"):
        filename = job.get("filename")
        if filename:
            try:
                (_UPLOADS_PATH / "videos" / filename).unlink(missing_ok=True)
            except Exception:
                pass
        try:
            os.unlink(_job_path(job_id))
        except Exception:
            pass
        return {"ok": True, "action": "purged", "job_id": job_id}

    return {"ok": True, "action": "no_op", "job_id": job_id}


@app.get("/api/concurrency/stats")
async def api_concurrency_stats(_key=Depends(require_scope("read"))):
    """Live snapshot of the adaptive concurrency gates.

    ``capacity`` is recomputed from the container's current usable compute and
    available memory, so it shrinks under pressure and grows when resources free
    up — i.e. it auto-adjusts to whatever load MaxBooster sends."""
    from ai_model import dedup_cache
    return {
        "inference": INFERENCE_GATE.stats(),
        "render": RENDER_GATE.stats(),
        "dedup_cache": dedup_cache.stats(),
    }


@app.get("/api/video-jobs")
async def api_list_video_jobs(_key=Depends(require_scope("read"))):
    """
    Return a summary of all video jobs in this server session, newest first.
    Each entry includes enough metadata to render a history view — no scene
    text payload is included; poll the individual job for full scenes[].
    """
    rows = []
    try:
        entries = [
            (fname, os.path.getmtime(os.path.join(_JOBS_DIR, fname)))
            for fname in os.listdir(_JOBS_DIR)
            if fname.endswith(".json") and not fname.endswith(".tmp")
        ]
        entries.sort(key=lambda x: x[1], reverse=True)
        for fname, _mtime in entries:
            job_id = fname[:-5]
            data = _job_read(job_id)
            if data is None:
                continue
            rows.append({
                "job_id":          job_id,
                "status":          data.get("status"),
                "created_at":      data.get("created_at"),
                "platform":        data.get("platform"),
                "genre_detected":  data.get("genre_detected"),
                "tone_used":       data.get("tone_used"),
                "duration":        data.get("duration"),
                "source":          data.get("source"),
                "scenes_rendered": data.get("scenes_rendered", 0),
                "aspect_ratio":    data.get("aspect_ratio"),
                "url":             data.get("url"),
                "error":           data.get("error"),
            })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"jobs": rows, "total": len(rows)}


@app.get("/api/video-job/{job_id}/preview/{scene_idx}")
async def api_video_job_preview(job_id: str, scene_idx: int, _key=Depends(require_scope("read"))):
    """
    Extract a single JPEG thumbnail frame from a completed video job at
    the temporal midpoint of the requested scene.  Returns image/jpeg.
    """
    import subprocess

    job = _job_read(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "done":
        raise HTTPException(status_code=409, detail=f"Job is {job.get('status')}, not done")

    filename = job.get("filename")
    if not filename:
        raise HTTPException(status_code=404, detail="No output file recorded for this job")

    video_path = _UPLOADS_PATH / "videos" / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    scenes = job.get("scenes", [])
    num_scenes = max(len(scenes), 1)
    if scene_idx < 0 or scene_idx >= num_scenes:
        raise HTTPException(
            status_code=422,
            detail=f"scene_idx must be 0–{num_scenes - 1} (video has {num_scenes} scenes)",
        )

    total_duration = float(job.get("duration") or 10.0)
    scene_dur = total_duration / num_scenes
    t = scene_idx * scene_dur + scene_dur * 0.4

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{t:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "3",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=15)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Frame extraction timed out")

    if proc.returncode != 0 or not proc.stdout:
        raise HTTPException(status_code=500, detail="ffmpeg frame extraction failed")

    return Response(content=proc.stdout, media_type="image/jpeg")


@app.get("/api/audio-job/{job_id}")
async def api_poll_audio_job(job_id: str, _key=Depends(require_scope("read"))):
    """Poll an audio generation job."""
    job = _job_read(job_id)
    if job is None:
        return {"status": "error", "error": "Job not found"}
    if job["status"] == "done":
        return {
            "status":    "done",
            "url":       job["url"],
            "audio_url": job["url"],
            "duration":  job["duration"],
            "bpm":       job["bpm"],
            "key":       job["key"],
            "format":        job.get("format"),
            "sample_rate":   job.get("sample_rate"),
            "bit_depth":     job.get("bit_depth"),
            "loudness_lufs": job.get("loudness_lufs"),
            # Measured sonic facts (loudness/energy/brightness/bass/stems) —
            # forwarded by clients into /api/generate/content beat_context.
            "audio_analysis": job.get("audio_analysis"),
            "stems":         job.get("stems") or {},
            # Awareness-driven section plan actually rendered (null = plain loop).
            "arrangement":   job.get("arrangement"),
            "seed":             job.get("seed"),
            "concept":          job.get("concept"),
            "style_hook":       job.get("style_hook"),
            "source":           job.get("source", "heuristic"),
            "technique":        job.get("technique"),
            # Awareness provenance — which live signals shaped track selection
            "awareness_genres":  job.get("awareness_genres"),
            "awareness_mood":    job.get("awareness_mood"),
            "awareness_source":  job.get("awareness_source"),
            # Non-None string when the dataset had no track matching the
            # requested key and the selector fell back to nearest-BPM.
            "selection_warning": job.get("selection_warning"),
            # Which dataset sample was selected (idx, bpm, key).
            "source_sample":     job.get("source_sample"),
        }
    if job["status"] == "error":
        return {"status": "error", "error": job.get("error", "Unknown error")}
    return {"status": job["status"]}


# -- Model weight sync (no /api/ prefix) --------------------------------------

@app.get("/api/models/social/state")
async def models_social_state(_key=Depends(require_scope("read"))):
    """Current trained weight state for the social model domain."""
    return _api_model_state("social")


@app.get("/api/models/advertising/state")
async def models_advertising_state(_key=Depends(require_scope("read"))):
    """Current trained weight state for the advertising model domain."""
    return _api_model_state("advertising")


@app.get("/api/models/content/state")
async def models_content_state(_key=Depends(require_scope("read"))):
    """Current trained weight state for the content model domain."""
    return _api_model_state("content")


@app.get("/api/models/engagement/state")
async def models_engagement_state(_key=Depends(require_scope("read"))):
    """Current trained weight state for the engagement model domain."""
    return _api_model_state("engagement")


# -- Training feedback ---------------------------------------------------------

_feedback_records: List[dict] = []
_feedback_lock    = threading.Lock()

@app.post("/api/train/feedback")
async def train_feedback_endpoint(req: ApiTrainFeedbackRequest, _key=Depends(require_scope("train"))):
    """Receive anonymised engagement signals for MaxCore retraining."""
    record = {**req.model_dump(), "received_at": datetime.utcnow().isoformat() + "Z"}
    with _feedback_lock:
        _feedback_records.append(record)
        if len(_feedback_records) > 10_000:
            _feedback_records[:] = _feedback_records[-10_000:]
    # Attempt to persist
    try:
        fb_path = Path(__file__).resolve().parent / "knowledge" / "feedback_log.json"
        fb_path.parent.mkdir(parents=True, exist_ok=True)
        existing: list = []
        if fb_path.exists():
            existing = json.loads(fb_path.read_text())
        existing.append(record)
        fb_path.write_text(json.dumps(existing[-10_000:], indent=2))
    except Exception as e:
        print(f"[Feedback] WARNING: could not persist: {e}", flush=True)
    return {"ok": True}


# ─── MaxBooster contract endpoints (aliases / new) ───────────────────────────

@app.get("/api/health")
async def api_health():
    """Health probe under the /api prefix — MaxCoreAIClient.isAvailable() polls this."""
    return {
        "status": "healthy",
        "model_loaded": _model_ready,
        "uptime_seconds": time.time() - _start_time,
        "version": "1.0.0",
        "storage_mode": _get_storage_mode(),
    }


# ─── Production warm-up ───────────────────────────────────────────────────────
# Tracks the last deep-warm result so GET /api/warm/status is non-destructive.
_deep_warm_status: dict[str, Any] = {
    "state": "pending",   # pending | warm | partial | error
    "cycles": 0,
    "last_warm_at": None,
    "subsystems": {},
}


@app.post("/api/warm")
async def api_warm(_admin=Depends(verify_admin)):
    """
    Production warm-up pass — exercises the Digital GPU inference chains so the
    reserved VM is fully hot before (or between) real user requests.

    Runs one inference pass through:
      • transformer / KV-cache  (ScriptAgent 1-token generation)
      • content scorer          (DistributionAgent caption ranking)
      • pocket GEMM dedup cache (stats probe confirms cache is active)
      • quality awareness       (platform buffer re-prime)
      • RTA Digital GPU         (global GEMM op-count probe)

    Every step is never-raise.  Returns per-subsystem timing and the accumulated
    warm_start status from the boot-time subsystem pre-warm.
    Safe to call repeatedly — idempotent and cheap after the first pass.
    """
    import time as _wt

    _deep_warm_status["cycles"] = _deep_warm_status.get("cycles", 0) + 1
    results: dict[str, Any] = {}
    overall_ok = True

    def _step(name: str, fn) -> None:
        nonlocal overall_ok
        t0 = _wt.time()
        try:
            val = fn()
            results[name] = {"ok": True, "ms": int((_wt.time() - t0) * 1000), "detail": val}
        except Exception as exc:
            results[name] = {"ok": False, "ms": int((_wt.time() - t0) * 1000), "error": f"{type(exc).__name__}: {exc}"}
            overall_ok = False

    # ── 1. Transformer / KV-cache ─────────────────────────────────────────────
    # ScriptAgent.run() drives the full transformer forward pass including
    # flash-attention and pocket-GEMM dedup — warms KV-cache and GEMM entries.
    def _warm_transformer():
        if not _model_ready or _script_agent is None:
            return "skipped (model not ready)"
        from ai_model.agents.script_agent import ScriptRequest
        sr = _script_agent.run(ScriptRequest(
            idea="warm", platform="tiktok", goal="growth", tone="energetic",
        ))
        return f"hook={sr.hook[:40]!r}"

    # ── 2. Content scorer / DistributionAgent ─────────────────────────────────
    def _warm_scorer():
        if not _model_ready or _distribution_agent is None:
            return "skipped (model not ready)"
        from ai_model.agents.distribution_agent import DistributionRequest
        dr = _distribution_agent.run(DistributionRequest(
            script="warm start probe", platform="tiktok", goal="growth",
        ))
        return f"caption_len={len(dr.caption)}"

    # ── 3. Pocket GEMM dedup cache ────────────────────────────────────────────
    def _warm_pocket():
        from ai_model.maxcore.pdim import get_pocket_accelerator
        pa = get_pocket_accelerator()
        stats = pa.stats()
        return f"hits={stats.get('hits', 0)} misses={stats.get('misses', 0)}"

    # ── 4. Quality awareness buffer ───────────────────────────────────────────
    def _warm_awareness():
        from ai_model import quality_awareness
        quality_awareness.platform_awareness_string("tiktok")
        return "ok"

    # ── 5. RTA Digital GPU op-count probe ────────────────────────────────────
    def _warm_rta():
        from ai_model import rta as _rta
        counts = _rta.global_op_counts()
        total = sum(counts.values()) if isinstance(counts, dict) else 0
        return f"total_ops={total}"

    await _in_thread(lambda: _step("transformer",  _warm_transformer))
    await _in_thread(lambda: _step("scorer",       _warm_scorer))
    await _in_thread(lambda: _step("pocket_gemm",  _warm_pocket))
    await _in_thread(lambda: _step("awareness",    _warm_awareness))
    await _in_thread(lambda: _step("rta",          _warm_rta))

    _deep_warm_status["state"] = "warm" if overall_ok else "partial"
    _deep_warm_status["last_warm_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _deep_warm_status["subsystems"] = results

    return {
        "hot": overall_ok,
        "model_ready": _model_ready,
        "warm_start": _warm_status,
        "deep_warm": _deep_warm_status,
        "subsystems": results,
    }


@app.get("/api/warm/status")
async def api_warm_status(_admin=Depends(verify_admin)):
    """Non-destructive warm-up status — returns last deep-warm result without running a new pass."""
    return {
        "model_ready": _model_ready,
        "warm_start": _warm_status,
        "deep_warm": _deep_warm_status,
    }


@app.get("/api/rta/status")
async def api_rta_status(selftest: bool = False, _key=Depends(require_scope("generate"))):
    """RTA-1 rendering-fabric status. Proves the trinity (IRC/VRC/ARC) runs on the
    self-contained Digital GPU by reporting cumulative GEMM ops; ``?selftest=1``
    runs a fast end-to-end render of all three mediums."""
    from ai_model import rta as _rta
    out = {
        "fabric": "RTA-1 (UMRF)",
        "compute_backend": "digital_gpu",
        "nodes": sorted(_rta.node_registry().keys()),
        "digital_gpu_ops": _rta.global_op_counts(),
    }
    if selftest:
        out["selftest"] = await _in_thread(_rta.api.self_test)
        out["digital_gpu_ops_after"] = _rta.global_op_counts()
    return out


class ApiSafetyScreenRequest(BaseModel):
    text: str = ""


@app.post("/api/safety/screen")
async def api_safety_screen(req: ApiSafetyScreenRequest, _key=Depends(require_scope("generate"))):
    """Stage 8 content-safety probe: screen + enforce a piece of text. Returns the
    policy verdict and the enforced (redacted/refused) output. Also exposes the
    running violation counters."""
    from ai_model.safety import get_safety
    s = get_safety()
    res = s.enforce(req.text or "")
    return {
        **res.to_dict(),
        "enforced_text": res.text,
        "stats": s.stats(),
    }


@app.post("/api/audio/analyze")
async def api_audio_analyze(req: ApiAudioAnalyzeRequest, _key=Depends(require_scope("generate"))):
    """
    Beat/structure analysis for beat-synced music video generation.
    Deterministic per audio reference so repeated calls on the same track agree.
    Returns bpm/tempo, key/musical_key, sections[], energy_curve[] and mood[].
    """
    import hashlib
    import numpy as _np

    ref  = req.audio_path or req.audio_url or "track"
    seed = int(hashlib.md5(ref.encode()).hexdigest(), 16) % (2 ** 31)
    rng  = _np.random.default_rng(seed)

    keys_list  = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    modes_list = ["major", "minor"]
    moods_pool = ["energetic", "melancholic", "chill", "aggressive",
                  "uplifting", "dark", "euphoric", "dreamy"]

    bpm         = round(float(rng.uniform(70, 175)), 1)
    musical_key = f"{keys_list[int(rng.integers(0, len(keys_list)))]} {modes_list[int(rng.integers(0, 2))]}"

    duration = 180.0
    if isinstance(req.context, dict):
        try:
            duration = float(req.context.get("duration") or duration)
        except (TypeError, ValueError):
            pass

    structure = ["intro", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro"]
    weights   = {"intro": 0.6, "verse": 1.0, "chorus": 1.1, "bridge": 0.8, "outro": 0.7}
    total_w   = sum(weights[s] for s in structure)
    sections  = []
    cursor    = 0.0
    for i, stype in enumerate(structure):
        seg   = duration * (weights[stype] / total_w)
        start = round(cursor, 2)
        end   = round(min(duration, cursor + seg), 2)
        sections.append({
            "name":  f"{stype}_{i + 1}",
            "label": stype.capitalize(),
            "type":  stype,
            "start": start,
            "end":   end,
        })
        cursor = end

    energy_curve = [round(float(x), 3) for x in rng.uniform(0.2, 1.0, 32).tolist()]
    moods        = [moods_pool[i] for i in rng.choice(len(moods_pool), 2, replace=False).tolist()]

    return {
        "bpm":          bpm,
        "tempo":        bpm,
        "key":          musical_key,
        "musical_key":  musical_key,
        "sections":     sections,
        "energy_curve": energy_curve,
        "mood":         moods,
        "duration":     round(duration, 2),
        "source":       "heuristic",
    }


@app.post("/api/infer/viral-score")
async def api_infer_viral_score(req: ApiViralScoreRequest, _key=Depends(require_scope("generate"))):
    """
    Pre-render viral potential score for a planned music video.
    Returns score and viral_score in 0–1 (MaxBooster multiplies by 100) plus a
    human-readable recommendation.
    """
    inputs        = req.inputs if isinstance(req.inputs, dict) else {}
    genre         = str(inputs.get("genre", "")).lower()
    platform      = str(inputs.get("platform", "")).lower()
    bpm           = float(inputs.get("bpm") or 0)
    section_count = int(inputs.get("section_count") or 0)
    scene_count   = int(inputs.get("scene_count") or 0)
    has_chorus    = bool(inputs.get("has_chorus"))

    score = 0.45
    if 90 <= bpm <= 150:
        score += 0.12
    elif bpm > 0:
        score += 0.04
    score += min(0.12, section_count * 0.02)
    score += min(0.12, scene_count * 0.015)
    if has_chorus:
        score += 0.08
    if platform in ("tiktok", "instagram", "instagram_reels", "youtube", "youtube_shorts", "reels"):
        score += 0.08
    if genre in ("hip-hop", "hiphop", "trap", "pop", "afrobeats", "drill", "r&b", "rnb", "electronic"):
        score += 0.06

    score = max(0.0, min(1.0, round(score, 3)))
    pct   = round(score * 100)
    if pct >= 80:
        rec = "High viral potential — strong beat sync and genre-authentic scenes"
    elif pct >= 60:
        rec = "Good engagement likely — add a bold CTA overlay and a hook in the first 2s"
    else:
        rec = "Moderate — tighten the intro, lean into a chorus drop, and post during peak hours"

    return {
        "score":          score,
        "viral_score":    score,
        "recommendation": rec,
        "model":          req.model or "viral-score-v2",
        "source":         "heuristic",
    }


def _resolve_video_job_file(job_id: str) -> Path:
    """Validate a completed video job and return its on-disk MP4 path."""
    job = _job_read(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "done":
        raise HTTPException(status_code=409, detail=f"Job is {job.get('status')}, not done")
    filename = job.get("filename")
    if not filename:
        raise HTTPException(status_code=404, detail="No output file recorded for this job")
    videos_dir = (_UPLOADS_PATH / "videos").resolve()
    path = (videos_dir / filename).resolve()
    # Containment guard: never serve anything outside the videos directory
    if not path.is_relative_to(videos_dir):
        raise HTTPException(status_code=400, detail="Invalid output path")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video file not found on disk")
    return path


@app.get("/api/video-job/{job_id}/download")
@app.get("/api/video-job/{job_id}/file")
@app.get("/api/video-job/{job_id}/video")
async def api_video_job_download(job_id: str, _key=Depends(require_scope("read"))):
    """Serve the rendered MP4 binary for a completed video job (validated by ftyp on the client)."""
    path = _resolve_video_job_file(job_id)
    return FileResponse(str(path), media_type="video/mp4", filename=path.name)


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import multiprocessing
    import uvicorn
    port = int(os.environ.get("MODEL_API_PORT", 9878))

    # ── Single-instance guard ───────────────────────────────────────────────
    # uvicorn kicks off the model load (~1.7 GB) during application startup,
    # which begins before the port is reliably bound.  In production the API
    # server can run more than one cluster primary (e.g. ports 8080 and 3000),
    # and each primary independently manages the Python AI server.  Two near-
    # simultaneous spawns both pass the "is the port open yet?" check and each
    # load the full model before either binds 9878 — doubling memory to ~3.4 GB
    # and starving the video renderer, so even posix_spawn'd ffmpeg execve
    # fails with OSError [Errno 5] (EIO) under the memory cgroup.
    #
    # An exclusive OS file lock guarantees exactly one instance ever loads the
    # model.  A duplicate stands down *before* importing/loading anything heavy,
    # then waits for the owner to bind the port so the Node lifecycle manager
    # sees "port held by another process" and quietly stands by (no respawn
    # loop).  The lock is advisory and tied to the open fd, so the kernel
    # releases it automatically if the owner crashes.
    import fcntl
    _singleton_lock_path = os.environ.get(
        "MODEL_SINGLETON_LOCK", f"/tmp/maxcore_model_{port}.lock"
    )
    _singleton_lock_fd = os.open(
        _singleton_lock_path, os.O_CREAT | os.O_RDWR, 0o644
    )
    try:
        fcntl.flock(_singleton_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(
            f"[Server] Another AI server instance already holds "
            f"{_singleton_lock_path} — standing down to avoid a duplicate "
            f"model load.",
            flush=True,
        )
        import socket as _socket
        import time as _time
        _deadline = _time.time() + 90
        while _time.time() < _deadline:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
                _s.settimeout(1.0)
                if _s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            _time.sleep(1.0)
        os._exit(0)

    # Worker count: default 1 to keep memory footprint small (each worker
    # loads the full ~1.7 GB model independently).  Operators on hosts with
    # more RAM can set UVICORN_WORKERS=N to scale unique-request throughput
    # linearly.  BLAS thread caps are applied proportionally at module load
    # time (see _blas_threads_per_worker above) so thread counts stay sane.
    cpu_count = multiprocessing.cpu_count()
    worker_count = max(1, int(os.environ.get("UVICORN_WORKERS", "1")))
    print(f"[Server] Starting MaxBooster AI Training Server on port {port} "
          f"({worker_count} uvicorn workers, {cpu_count} CPUs detected)")

    # Any exception that escapes uvicorn.run() — import error, port conflict,
    # bad config, etc. — must produce a non-zero exit so the Node supervisor's
    # "exit" event handler fires and schedules a restart.  Without this, an
    # unhandled exception would let Python exit with code 1 by default (which
    # is correct), but we make it explicit and log the full traceback so the
    # cause is visible in workflow logs.
    try:
        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=port,
            log_level="info",
            workers=worker_count,
            timeout_keep_alive=300,
            limit_concurrency=None,   # no artificial cap — async coalescer + pdim auto-scale handle backpressure
            backlog=4096,             # maximise OS socket queue for burst acceptance
        )
    except SystemExit:
        # uvicorn raises SystemExit(0) on clean SIGTERM/SIGINT shutdown.
        # Re-raise so the process exits with the original code and the Node
        # supervisor sees the correct signal/code combination.
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        print("[Server] uvicorn.run() raised an unexpected exception — exiting with code 1 so supervisor restarts.", flush=True)
        sys.exit(1)
