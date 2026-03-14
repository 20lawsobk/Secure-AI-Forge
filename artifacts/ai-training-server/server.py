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
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException, Depends, Header, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ─── DB Setup ────────────────────────────────────────────────────────────────

DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialize database tables."""
    conn = psycopg2.connect(DATABASE_URL)
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
        # Write admin key to a file for first-time access
        key_file = Path(__file__).parent / "admin_key.txt"
        with open(key_file, "w") as f:
            f.write(f"ADMIN API KEY (save this - shown only once):\n{admin_key}\n")
        print(f"[Server] Default admin key created. Key saved to admin_key.txt")
        print(f"[Server] Admin key prefix: {prefix}...")

    conn.commit()
    cur.close()
    conn.close()

def log_training(message: str, level: str = "info", epoch: int = None,
                 loss: float = None, job_id: str = None):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO training_logs (level, message, epoch, loss, job_id) VALUES (%s, %s, %s, %s, %s)",
            (level, message, epoch, loss, job_id)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[Server] Failed to log training: {e}")

# ─── Training State ──────────────────────────────────────────────────────────

_training_state = {
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
_hyper_backend = None
_model_config = {}

_model_lock = threading.Lock()

def _init_ai_model():
    global _model_ready, _tokenizer, _creative_model, _script_agent
    global _visual_spec_agent, _distribution_agent, _optimization_agent
    global _repo, _adapter, _render_manager, _model_config

    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from ai_model.model.tokenizer import SimpleTokenizer
        from ai_model.model.transformer import TransformerLM
        from ai_model.model.creative_model import CreativeModel
        from ai_model.agents.script_agent import ScriptAgent, ScriptRequest
        from ai_model.agents.visual_spec_agent import VisualSpecAgent, VisualSpecRequest
        from ai_model.agents.distribution_agent import DistributionAgent, DistributionRequest
        from ai_model.agents.optimization_agent import OptimizationAgent, OptimizationRequest
        from ai_model.boostsheets.repository import BoostSheetRepository
        from ai_model.adapters.url_adapter import UrlToBoostSheetAdapter
        from ai_model.render_manager import RenderManager
        import torch

        print("[AI Model] Initializing MaxBooster AI Content Model...")
        _tokenizer = SimpleTokenizer()

        DEVICE = "cpu"
        dim = int(os.environ.get("AI_MODEL_DIM", "512"))
        n_layers = int(os.environ.get("AI_MODEL_LAYERS", "8"))
        n_heads = int(os.environ.get("AI_MODEL_HEADS", "8"))
        max_len = int(os.environ.get("AI_MODEL_MAX_LEN", "1024"))

        weights_dir = Path(__file__).parent / "ai_model" / "weights"
        weights_path = weights_dir / "model.pt"

        if weights_path.exists():
            print(f"[AI Model] Loading weights from {weights_path}")
            checkpoint = torch.load(str(weights_path), map_location=DEVICE)
            if isinstance(checkpoint, dict) and "vocab" in checkpoint:
                _tokenizer.vocab = checkpoint["vocab"]
                _tokenizer.inv_vocab = checkpoint["inv_vocab"]
                _tokenizer.next_id = checkpoint["next_id"]
                state_dict = checkpoint["model_state_dict"]
                if "config" in checkpoint:
                    cfg = checkpoint["config"]
                    dim = cfg.get("dim", dim)
                    n_layers = cfg.get("layers", n_layers)
                    n_heads = cfg.get("heads", n_heads)
                    max_len = cfg.get("max_len", max_len)
            else:
                state_dict = checkpoint

            base_model = TransformerLM(
                vocab_size=max(len(_tokenizer.vocab), 1000),
                dim=dim, n_layers=n_layers, n_heads=n_heads, max_len=max_len,
            )
            base_model.load_state_dict(state_dict, strict=False)
        else:
            print("[AI Model] No pre-trained weights, using random init")
            base_model = TransformerLM(
                vocab_size=max(len(_tokenizer.vocab), 1000),
                dim=dim, n_layers=n_layers, n_heads=n_heads, max_len=max_len,
            )

        _creative_model = CreativeModel(base_model, _tokenizer, device=DEVICE)
        _script_agent = ScriptAgent(_creative_model)
        _visual_spec_agent = VisualSpecAgent(_creative_model)
        _distribution_agent = DistributionAgent(_creative_model)
        _optimization_agent = OptimizationAgent(_creative_model)
        _repo = BoostSheetRepository(path="boostsheets_db")
        _adapter = UrlToBoostSheetAdapter(_repo)
        _render_manager = RenderManager()

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

ADMIN_KEY_ENV = os.environ.get("ADMIN_KEY")

def verify_api_key(x_api_key: str = Header(None), x_admin_key: str = Header(None)):
    """Verify API key from X-Api-Key or X-Admin-Key header."""
    raw_key = x_api_key or x_admin_key
    if not raw_key:
        raise HTTPException(status_code=401, detail="API key required")

    # Allow env-based admin override
    if ADMIN_KEY_ENV and raw_key == ADMIN_KEY_ENV:
        return {"id": "env-admin", "scopes": ["read", "write", "train", "admin", "generate"]}

    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT * FROM api_keys WHERE key_hash = %s AND is_active = TRUE",
            (key_hash,)
        )
        key_record = cur.fetchone()
        if not key_record:
            cur.close(); conn.close()
            raise HTTPException(status_code=401, detail="Invalid or inactive API key")

        expires_at = key_record["expires_at"]
        if expires_at and expires_at < datetime.now(timezone.utc):
            cur.close(); conn.close()
            raise HTTPException(status_code=401, detail="API key expired")

        cur.execute(
            "UPDATE api_keys SET request_count = request_count + 1, last_used_at = NOW() WHERE id = %s",
            (str(key_record["id"]),)
        )
        conn.commit()
        cur.close()
        conn.close()
        return dict(key_record)
    except HTTPException:
        raise
    except Exception as e:
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

def verify_admin(x_admin_key: str = Header(None)):
    """Admin-only endpoint auth."""
    if not x_admin_key:
        raise HTTPException(status_code=401, detail="X-Admin-Key header required")

    # Allow env-based admin override
    if ADMIN_KEY_ENV and x_admin_key == ADMIN_KEY_ENV:
        return {"id": "env-admin", "scopes": ["admin"]}

    key_hash = hashlib.sha256(x_admin_key.encode()).hexdigest()
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT * FROM api_keys WHERE key_hash = %s AND is_active = TRUE",
            (key_hash,)
        )
        key_record = cur.fetchone()
        cur.close(); conn.close()
        if not key_record:
            raise HTTPException(status_code=401, detail="Invalid admin key")
        scopes = key_record["scopes"] or []
        if "admin" not in scopes:
            raise HTTPException(status_code=403, detail="Admin scope required")
        return dict(key_record)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auth error: {e}")

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

class ContentRequest(BaseModel):
    platform: str = "tiktok"
    topic: str
    tone: str = "energetic"
    goal: str = "growth"
    include_hashtags: bool = True

# ─── Startup ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    if not DATABASE_URL:
        print("[Server] WARNING: DATABASE_URL not set — running without DB")
        return
    init_db()
    thread = threading.Thread(target=_init_ai_model, daemon=True)
    thread.start()
    storage_thread = threading.Thread(target=_init_storage, daemon=True)
    storage_thread.start()


def _init_storage():
    from storage_client import get_storage
    storage = get_storage()
    ok = storage.ping()
    if ok:
        print("[Storage] Connected to MaxBooster storage server")
    else:
        print("[Storage] Storage server offline — using in-process fallback")

# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": _model_ready,
        "uptime_seconds": time.time() - _start_time,
        "version": "1.0.0",
    }

_start_time = time.time()

# ─── API Key Management ───────────────────────────────────────────────────────

@app.get("/api-keys")
async def list_api_keys(_admin = Depends(verify_admin)):
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT id, name, prefix, scopes, is_active, request_count, created_at, last_used_at, expires_at FROM api_keys ORDER BY created_at DESC")
    rows = cur.fetchall()
    cur.close(); conn.close()
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

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        """INSERT INTO api_keys (name, key_hash, prefix, scopes, expires_at)
           VALUES (%s, %s, %s, %s, %s) RETURNING id, name, prefix, scopes, created_at""",
        (req.name, key_hash, prefix, req.scopes, expires_at)
    )
    row = cur.fetchone()
    conn.commit()
    cur.close(); conn.close()

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
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("UPDATE api_keys SET is_active = FALSE WHERE id = %s", (key_id,))
    affected = cur.rowcount
    conn.commit()
    cur.close(); conn.close()
    if affected == 0:
        raise HTTPException(status_code=404, detail="API key not found")
    return {"success": True, "message": f"API key {key_id} revoked"}

@app.post("/api-keys/{key_id}/rotate")
async def rotate_api_key(key_id: str, _admin = Depends(verify_admin)):
    raw_key = f"mbs_{secrets.token_hex(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    prefix = raw_key[:12]

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        """UPDATE api_keys SET key_hash = %s, prefix = %s, request_count = 0, last_used_at = NULL
           WHERE id = %s RETURNING id, name, scopes, created_at""",
        (key_hash, prefix, key_id)
    )
    row = cur.fetchone()
    conn.commit()
    cur.close(); conn.close()
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
        "device": "cpu",
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
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from ai_model.gpu.torch_backend import DigitalGPUBackend
        backend = DigitalGPUBackend(lanes=32)
        status = backend.status()
        return {"available": True, "backend": "digital_gpu", **status}
    except Exception as e:
        return {"available": False, "backend": "none", "error": str(e)}

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
    with _training_lock:
        state = dict(_training_state)
    state["weights_exist"] = weights_path.exists()
    return state

@app.post("/training/start")
async def start_training(req: StartTrainingRequest, background_tasks: BackgroundTasks,
                         _key = Depends(require_scope("train"))):
    with _training_lock:
        if _training_state["state"] == "running":
            return {"success": False, "message": "Training already in progress", "job_id": _training_state.get("job_id")}

    job_id = str(uuid.uuid4())[:8]
    with _training_lock:
        _training_state["state"] = "starting"
        _training_state["job_id"] = job_id
        _training_state["epoch"] = 0
        _training_state["total_epochs"] = req.epochs
        _training_state["started_at"] = time.time()

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
        dataset = CreativeDataset(data_path, _tokenizer, max_len=_creative_model.model.pos_emb.num_embeddings)
        if len(dataset) == 0:
            raise ValueError("Empty dataset")

        dim = _creative_model.model.token_emb.embedding_dim
        cfg = TrainConfig({
            "model": {"dim": dim, "layers": len(_creative_model.model.layers), "heads": 8, "max_len": _model_config.get("max_len", 1024)},
            "train": {"lr": req.learning_rate, "batch_size": req.batch_size, "epochs": req.epochs, "data_path": data_path},
        })

        _creative_model.resize_embeddings()

        for epoch in range(req.epochs):
            start = time.time()
            with _training_lock:
                _training_state["epoch"] = epoch + 1
                _training_state["samples_trained"] = len(dataset)

            run_train(_creative_model.model, dataset, _tokenizer, cfg, device="cpu")
            ppl = evaluate(_creative_model.model, dataset, _tokenizer, device="cpu")
            elapsed = time.time() - _training_state["started_at"]
            eta = (elapsed / (epoch + 1)) * (req.epochs - epoch - 1)
            loss = math.log(ppl) if ppl else None

            with _training_lock:
                _training_state["loss"] = loss
                _training_state["perplexity"] = ppl
                _training_state["elapsed_seconds"] = elapsed
                _training_state["eta_seconds"] = eta

            log_training(f"Epoch {epoch+1}/{req.epochs} complete. Loss: {loss:.4f}, PPL: {ppl:.2f}",
                        epoch=epoch+1, loss=loss, job_id=job_id)

        weights_dir = Path(__file__).parent / "ai_model" / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / "model.pt"
        n_layers = len(_creative_model.model.layers)
        torch.save({
            "model_state_dict": _creative_model.model.state_dict(),
            "vocab": _tokenizer.vocab,
            "inv_vocab": _tokenizer.inv_vocab,
            "next_id": _tokenizer.next_id,
            "config": {"dim": dim, "layers": n_layers, "heads": cfg.heads, "max_len": cfg.max_len},
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

@app.get("/training/logs")
async def get_training_logs(limit: int = 50):
    if not DATABASE_URL:
        return {"logs": [], "total": 0}
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        "SELECT timestamp, level, message, epoch, loss, job_id FROM training_logs ORDER BY timestamp DESC LIMIT %s",
        (limit,)
    )
    rows = cur.fetchall()
    cur.close(); conn.close()
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

# ─── Content Generation ───────────────────────────────────────────────────────

PLATFORM_NORMALIZE = {
    "googlebusiness": "google_business", "google_business": "google_business",
    "twitter": "twitter", "x": "twitter",
}

def normalize_platform(p: str) -> str:
    return PLATFORM_NORMALIZE.get(p.lower(), p.lower())

@app.post("/content/generate")
async def generate_content(req: ContentRequest, _key = Depends(require_scope("generate"))):
    start = time.time()
    platform = normalize_platform(req.platform)

    if not _model_ready or _script_agent is None:
        # Fallback templates when model not ready
        return {
            "success": True,
            "platform": platform,
            "caption": f"Check out this {req.topic} content!",
            "hook": f"🔥 {req.topic} is going viral!",
            "body": f"Here's everything you need to know about {req.topic}.",
            "cta": "Follow for more!",
            "hashtags": ["#content", f"#{req.topic.replace(' ', '')}"],
            "source": "template",
            "processing_time_ms": (time.time() - start) * 1000,
        }

    try:
        from ai_model.agents.script_agent import ScriptRequest
        from ai_model.agents.distribution_agent import DistributionRequest

        script_result = _script_agent.run(ScriptRequest(
            idea=req.topic, platform=platform, goal=req.goal, tone=req.tone,
        ))
        full_script = f"{script_result.hook}\n{script_result.body}\n{script_result.cta}"
        dist_result = _distribution_agent.run(DistributionRequest(
            script=full_script, platform=platform, goal=req.goal,
        ))

        return {
            "success": True,
            "platform": platform,
            "caption": dist_result.caption,
            "hook": script_result.hook,
            "body": script_result.body,
            "cta": script_result.cta,
            "hashtags": dist_result.hashtags if req.include_hashtags else [],
            "source": getattr(script_result, "source", "template"),
            "processing_time_ms": (time.time() - start) * 1000,
        }
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
    total_keys = 0; active_keys = 0; total_requests_today = 0; boostsheet_count = 0
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM api_keys")
        total_keys = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM api_keys WHERE is_active = TRUE")
        active_keys = cur.fetchone()[0]
        cur.execute("SELECT COALESCE(SUM(request_count), 0) FROM api_keys WHERE last_used_at > NOW() - INTERVAL '1 day'")
        total_requests_today = cur.fetchone()[0] or 0
        cur.close(); conn.close()
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
    checkpoints = get_checkpoint_client().list_checkpoints()
    return {
        **storage.status(),
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
        from ai_model.training.dataset import CreativeDataset
        from ai_model.training.trainer import train as run_train, evaluate
        from ai_model.training.config import TrainConfig

        total_loss = 0.0
        batch_count = 0
        samples_written = []

        log_training(f"[StorageTrain] Streaming batches from storage...", job_id=job_id)

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

                config = TrainConfig(
                    epochs=1,
                    batch_size=min(req.batch_size, len(samples_written)),
                    lr=req.learning_rate,
                    grad_clip=1.0,
                )
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
                    "next_id": _tokenizer.next_id,
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
                log_training(f"[StorageTrain] Checkpoint saved to storage", job_id=job_id)
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
        raise HTTPException(status_code=503, detail="No training session found in storage server. "
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


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("MODEL_API_PORT", 9878))
    print(f"[Server] Starting MaxBooster AI Training Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
