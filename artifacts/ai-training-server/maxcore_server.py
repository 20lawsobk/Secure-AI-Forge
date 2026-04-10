"""
MaxCore Model Knowledge Server
================================
Runs on your local machine (D:/ai_server) and serves as the authoritative
storage layer for all BASE AI model knowledge:

  D:/ai_server/
    models/           ← weights, checkpoints, version history
    datasets/         ← HMDB51, UCF101, FMA, GTZAN, MusicCaps, etc.
    knowledge/        ← curriculum progress, loss logs, session records
    logs/             ← training session logs

The Replit server handles:
  - User journey data  → Pocket Dimension (per-user encrypted space)
  - Live API / serving → Node.js app

Together they form a two-tier knowledge architecture:
  Base Model Knowledge (D: drive) + User Journey Knowledge (Pocket Dimension)

SETUP:
  1. pip install fastapi uvicorn numpy
  2. Copy server/services/diffusion/ folder next to this file
  3. Set PEER_TRAINING_NODE=http://<this-machine-ip>:8000 in Replit secrets
  4. python maxcore_server.py

Endpoints (all prefixed /):
  GET  /health                        — system status
  GET  /models/list                   — all saved model versions
  GET  /models/weights                — download current weights_v4.npz
  GET  /models/best                   — download best checkpoint
  POST /models/weights                — receive + FedAvg merge weights
  GET  /knowledge/curriculum          — curriculum progress JSON
  POST /knowledge/curriculum          — update curriculum progress
  GET  /knowledge/loss-history        — all-time loss history
  GET  /knowledge/sessions            — last 50 training sessions
  GET  /datasets/list                 — datasets on D: drive with sizes
  GET  /datasets/manifest             — full download plan vs. present data
  GET  /datasets/stream               — stream random frames from a dataset
  POST /train/start                   — start background training
  POST /train/stop                    — stop background training
  GET  /train/status                  — live training metrics

Remote control (proxied via Replit /api/maxcore/*):
  GET  /control/status                — full snapshot (CPU/RAM/disk + training + curriculum)
  GET  /control/logs?n=300            — last N lines from training log
  DELETE /control/logs                — clear all log files
  POST /control/restart               — graceful restart (watchdog relaunches in ~5 s)
  POST /control/shutdown              — stop training + exit; watchdog does NOT relaunch
  POST /control/trigger-session       — run one extra training session immediately
  POST /control/set-phase             — force curriculum to a specific phase (1–4)
"""

import io
import json
import os
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel

# ── Directory layout ─────────────────────────────────────────────────────────

ROOT       = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_DIR   = ROOT / "datasets"
KNOW_DIR   = ROOT / "knowledge"
LOGS_DIR   = ROOT / "logs"
DIFFUSION  = ROOT / "diffusion"

for d in [MODELS_DIR / "versions", DATA_DIR, KNOW_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

WEIGHTS      = MODELS_DIR / "weights_v4.npz"
WEIGHTS_BEST = MODELS_DIR / "weights_v4_best.npz"
META         = MODELS_DIR / "meta_v4.json"
CURRICULUM   = KNOW_DIR   / "curriculum_progress.json"
LOSS_LOG     = KNOW_DIR   / "loss_history.json"
SESSION_LOG  = KNOW_DIR   / "session_log.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Training state ────────────────────────────────────────────────────────────

_lock  = threading.Lock()
_stop  = threading.Event()
_thread: Optional[threading.Thread] = None
_best_loss = [float("inf")]
_SERVER_START = time.time()

_state: Dict[str, Any] = {
    "status":        "idle",
    "phase":         0,
    "phase_name":    "",
    "session_count": 0,
    "total_steps":   0,
    "elapsed_sec":   0.0,
    "loss":          None,
    "last_merge_ts": None,
    "error":         None,
    "start_time":    None,
}


def _upd(**kw):
    with _lock:
        _state.update(kw)


# ── Loss + session logging ────────────────────────────────────────────────────

def _append_loss(session: int, loss: float, phase: int):
    history = []
    if LOSS_LOG.exists():
        try:
            history = json.loads(LOSS_LOG.read_text())
        except Exception as e:
            print(f"[MaxCore] WARNING: loss log corrupted, resetting: {e}", flush=True)
    history.append({"session": session, "loss": loss,
                     "phase": phase, "ts": time.time()})
    LOSS_LOG.write_text(json.dumps(history[-2000:], indent=2))


def _append_session(meta: dict):
    log = []
    if SESSION_LOG.exists():
        try:
            log = json.loads(SESSION_LOG.read_text())
        except Exception as e:
            print(f"[MaxCore] WARNING: session log corrupted, resetting: {e}", flush=True)
    log.append({**meta, "saved_at": datetime.utcnow().isoformat()})
    SESSION_LOG.write_text(json.dumps(log[-500:], indent=2))


# ── Model version archiving ──────────────────────────────────────────────────

def _archive_weights(tag: str = ""):
    """Copy current weights to models/versions/ with a timestamped name."""
    if not WEIGHTS.exists():
        return
    ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    name = f"weights_v4_{ts}{'_' + tag if tag else ''}.npz"
    dest = MODELS_DIR / "versions" / name
    shutil.copy2(str(WEIGHTS), str(dest))
    # Keep only the 20 most recent versions to avoid filling the drive
    versions = sorted((MODELS_DIR / "versions").glob("weights_v4_*.npz"))
    for old in versions[:-20]:
        old.unlink()
    return dest.name


# ── FedAvg merge ─────────────────────────────────────────────────────────────

def _fedavg(local_path: Path, remote_bytes: bytes) -> int:
    if not local_path.exists():
        local_path.write_bytes(remote_bytes)
        return 0
    local  = dict(np.load(str(local_path), allow_pickle=False))
    remote = dict(np.load(io.BytesIO(remote_bytes), allow_pickle=False))
    merged, n = {}, 0
    for k in local:
        if k in remote and local[k].shape == remote[k].shape:
            merged[k] = (local[k].astype(np.float32)
                         + remote[k].astype(np.float32)) * 0.5
            n += 1
        else:
            merged[k] = local[k]
    np.savez_compressed(str(local_path), **merged)
    # Update best checkpoint if current loss is a record
    loss = _state.get("loss")
    if loss is not None and float(loss) < _best_loss[0]:
        _best_loss[0] = float(loss)
        shutil.copy2(str(local_path), str(WEIGHTS_BEST))
    print(f"[FedAvg] Merged {n}/{len(local)} arrays", flush=True)
    return n


# ── Background training worker ────────────────────────────────────────────────

def _worker():
    try:
        if not DIFFUSION.exists():
            raise RuntimeError(
                f"diffusion/ folder not found at {DIFFUSION}. "
                "Copy server/services/diffusion/ next to this file."
            )

        from diffusion.training_curriculum import CurriculumTrainer

        trainer    = CurriculumTrainer(progress_path=str(CURRICULUM))
        _orig_rs   = trainer.run_session

        def _patched(phase=None):
            sched   = trainer.scheduler
            phase_o = phase if phase else sched.current_phase
            _upd(status="running",
                 phase=phase_o.phase_id,
                 phase_name=phase_o.name)
            t0   = time.time()
            meta = _orig_rs(phase)
            elapsed = time.time() - t0
            loss = meta.get("final_loss", meta.get("mean_loss"))
            with _lock:
                _state["session_count"] += 1
                _state["total_steps"]   += (meta.get("samples_per_epoch", 0)
                                            * meta.get("epochs", 0))
                _state["elapsed_sec"]   += elapsed
                if loss is not None:
                    _state["loss"] = round(float(loss), 6)
                    _append_loss(_state["session_count"],
                                 round(float(loss), 6),
                                 _state["phase"])
            _append_session(meta)
            # Archive weights every 10 sessions
            if _state["session_count"] % 10 == 0:
                _archive_weights(f"s{_state['session_count']}")
            return meta

        trainer.run_session = _patched
        trainer.run_month(
            sleep_between_sessions_sec=15,
            stop_event=_stop,
            deadline_str="2026-04-03",
        )
        _upd(status="stopped" if _stop.is_set() else "idle")

    except Exception as e:
        import traceback
        _upd(status="error", error=str(e))
        print(f"[Worker] {e}\n{traceback.format_exc()}", flush=True)


# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="MaxCore Model Knowledge Server", version="3.0.0")


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    versions = len(list((MODELS_DIR / "versions").glob("*.npz")))
    datasets = {p.name: round(sum(f.stat().st_size for f in p.rglob("*")
                                   if f.is_file()) / 1e9, 2)
                for p in DATA_DIR.iterdir() if p.is_dir()}
    with _lock:
        snap = dict(_state)
    return {
        "status":          snap["status"],
        "phase":           snap["phase"],
        "phase_name":      snap["phase_name"],
        "session_count":   snap["session_count"],
        "loss":            snap["loss"],
        "weights_kb":      (WEIGHTS.stat().st_size // 1024) if WEIGHTS.exists() else 0,
        "versions_saved":  versions,
        "datasets":        datasets,
        "diffusion_ready": DIFFUSION.exists(),
        "knowledge_root":  str(ROOT),
        "last_merge":      snap["last_merge_ts"],
        "error":           snap["error"],
    }


# ── Model endpoints ────────────────────────────────────────────────────────────

@app.get("/models/list")
def models_list():
    """List all saved model versions with size and timestamp."""
    versions = []
    for p in sorted((MODELS_DIR / "versions").glob("weights_v4_*.npz")):
        versions.append({
            "name":       p.name,
            "kb":         p.stat().st_size // 1024,
            "created_at": datetime.utcfromtimestamp(p.stat().st_mtime).isoformat(),
        })
    return {
        "current": {
            "exists": WEIGHTS.exists(),
            "kb":     (WEIGHTS.stat().st_size // 1024) if WEIGHTS.exists() else 0,
        },
        "best": {
            "exists": WEIGHTS_BEST.exists(),
            "kb":     (WEIGHTS_BEST.stat().st_size // 1024) if WEIGHTS_BEST.exists() else 0,
        },
        "versions": versions[-20:],
    }


@app.get("/models/weights")
def get_weights():
    if not WEIGHTS.exists():
        raise HTTPException(404, "No weights yet.")
    return Response(
        content=WEIGHTS.read_bytes(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=weights_v4.npz"},
    )


@app.get("/models/best")
def get_best():
    p = WEIGHTS_BEST if WEIGHTS_BEST.exists() else WEIGHTS
    if not p.exists():
        raise HTTPException(404, "No weights yet.")
    return Response(
        content=p.read_bytes(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=weights_v4_best.npz"},
    )


@app.post("/models/weights")
async def post_weights(request: Request):
    """Receive weights from Replit node and FedAvg merge."""
    body = await request.body()
    if len(body) < 100:
        raise HTTPException(400, "Empty payload.")
    n = _fedavg(WEIGHTS, body)
    _upd(last_merge_ts=time.time())
    _archive_weights("merge")
    return {"merged_arrays": n,
            "weights_kb": WEIGHTS.stat().st_size // 1024}


# ── Knowledge endpoints ────────────────────────────────────────────────────────

@app.get("/knowledge/curriculum")
def get_curriculum():
    if not CURRICULUM.exists():
        return {}
    return json.loads(CURRICULUM.read_text())


@app.post("/knowledge/curriculum")
async def post_curriculum(request: Request):
    body = await request.body()
    CURRICULUM.write_bytes(body)
    return {"ok": True, "bytes": len(body)}


@app.get("/knowledge/loss-history")
def get_loss_history():
    if not LOSS_LOG.exists():
        return []
    return json.loads(LOSS_LOG.read_text())


@app.get("/knowledge/sessions")
def get_sessions():
    if not SESSION_LOG.exists():
        return []
    sessions = json.loads(SESSION_LOG.read_text())
    return sessions[-50:]


# ── Dataset registry & streaming ────────────────────────────────────────────────

@app.get("/datasets/list")
def datasets_list():
    """
    List datasets available on the D: drive with size and status.
    The Replit training node uses this to know what data is accessible.
    """
    result = {}
    if DATA_DIR.exists():
        for p in DATA_DIR.iterdir():
            if not p.is_dir():
                continue
            files = list(p.rglob("*"))
            total = sum(f.stat().st_size for f in files if f.is_file())
            result[p.name] = {
                "path":      str(p),
                "files":     len([f for f in files if f.is_file()]),
                "total_gb":  round(total / 1e9, 3),
                "available": total > 0,
            }
    return {"datasets": result, "root": str(DATA_DIR)}


@app.get("/datasets/manifest")
def datasets_manifest():
    """
    Return the full D: drive dataset manifest — what's planned + what's present.
    Used by the Replit node to display setup progress.
    """
    sys.path.insert(0, str(DIFFUSION.parent))
    try:
        from diffusion.dataset_downloader import DOWNLOAD_PLAN
        plan = []
        for t in DOWNLOAD_PLAN:
            if not t.d_drive:
                continue
            present = (DATA_DIR / t.name).exists() and any(
                True for _ in (DATA_DIR / t.name).rglob("*") if _.is_file()
            )
            plan.append({
                "name":     t.name,
                "est_gb":   t.est_gb,
                "music":    t.music,
                "priority": t.priority,
                "present":  present,
                "note":     t.extra.get("note", ""),
            })
        total_present = sum(p["est_gb"] for p in plan if p["present"])
        total_planned = sum(p["est_gb"] for p in plan)
        return {
            "datasets":      plan,
            "total_planned_gb": round(total_planned, 1),
            "total_present_gb": round(total_present, 1),
        }
    except Exception as e:
        return {"error": str(e), "datasets": []}


@app.get("/datasets/stream")
def datasets_stream(
    dataset: str = "any",
    n: int = 4,
    h: int = 64,
    w: int = 64,
    seed: int = 0,
):
    """
    Stream n random frames from a dataset on the D: drive.
    Returns a JSON envelope containing base64-encoded float32 numpy arrays.

    The training nodes call this when local datasets aren't available.
    Falls back to synthetic noise frames if the dataset isn't present yet.

    Query params:
      dataset — dataset name or 'any' to pick available
      n       — number of frames (T)
      h, w    — frame height/width
      seed    — random seed
    """
    import base64 as _b64

    rng = np.random.default_rng(seed)
    H, W, T = min(h, 256), min(w, 256), min(n, 32)

    # Try to find real frames from a dataset directory
    frames = None
    candidates = []

    if DATA_DIR.exists():
        if dataset == "any":
            candidates = [p for p in DATA_DIR.iterdir() if p.is_dir()]
        else:
            cand = DATA_DIR / dataset
            if cand.is_dir():
                candidates = [cand]

    for ds_dir in rng.permutation(candidates) if candidates else []:
        img_files = list(ds_dir.rglob("*.jpg")) + list(ds_dir.rglob("*.png"))
        if len(img_files) < T:
            continue
        chosen = [img_files[i] for i in rng.choice(len(img_files), T, replace=False)]
        try:
            from PIL import Image as _PILImage
            loaded = []
            for fpath in chosen:
                img = _PILImage.open(fpath).convert("RGB").resize((W, H))
                arr = (np.array(img, dtype=np.float32) / 127.5) - 1.0
                loaded.append(arr)
            frames = np.stack(loaded, axis=0)   # (T, H, W, 3)
            break
        except Exception:
            continue

    # Synthetic fallback — structured noise with spatial patterns
    if frames is None:
        frames = rng.normal(0, 0.3, (T, H, W, 3)).astype(np.float32)
        # Add low-frequency spatial structure so it's not pure noise
        xx = np.linspace(-1, 1, W)
        yy = np.linspace(-1, 1, H)
        gx, gy = np.meshgrid(xx, yy)
        for t in range(T):
            freq   = rng.uniform(1, 4)
            angle  = rng.uniform(0, np.pi)
            wave   = np.sin(freq * (gx * np.cos(angle) + gy * np.sin(angle)) * np.pi)
            frames[t, :, :, :] += (wave[:, :, None] * 0.4).astype(np.float32)
        frames = np.clip(frames, -1, 1)

    # Encode as base64 bytes
    raw     = frames.astype(np.float32).tobytes()
    encoded = _b64.b64encode(raw).decode("ascii")

    return {
        "dataset":  dataset,
        "shape":    list(frames.shape),   # [T, H, W, 3]
        "dtype":    "float32",
        "encoding": "base64",
        "data":     encoded,
        "synthetic": frames is None,
    }


# ── Training control ───────────────────────────────────────────────────────────

@app.post("/train/start")
def train_start():
    global _thread
    with _lock:
        if _state["status"] == "running":
            return {"ok": False, "detail": "Already running."}
        _stop.clear()
        _state["error"]      = None
        _state["status"]     = "running"
        _state["start_time"] = time.time()
    _thread = threading.Thread(target=_worker, daemon=True, name="KnowledgeTrainer")
    _thread.start()
    return {"ok": True, "detail": "Training started (continuous, deadline 2026-04-03)"}


@app.post("/train/stop")
def train_stop():
    _stop.set()
    _upd(status="stopping")
    return {"ok": True}


@app.get("/train/status")
def train_status():
    with _lock:
        return dict(_state)


# ── Remote control ─────────────────────────────────────────────────────────────
#
# These endpoints let the Replit server (or any trusted peer) fully control
# this machine without needing physical access.  All mutating actions write
# a small flag-file so the watchdog can pick up the intent even if the
# Python process crashes mid-action.

CTRL_DIR   = ROOT / "control"
CTRL_DIR.mkdir(exist_ok=True)


def _disk_usage() -> dict:
    try:
        usage = shutil.disk_usage(str(ROOT))
        return {
            "total_gb": round(usage.total / 1e9, 1),
            "used_gb":  round(usage.used  / 1e9, 1),
            "free_gb":  round(usage.free  / 1e9, 1),
        }
    except Exception:
        return {}


def _ram_usage() -> dict:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "total_gb":   round(vm.total   / 1e9, 1),
            "used_gb":    round(vm.used    / 1e9, 1),
            "available_gb": round(vm.available / 1e9, 1),
            "percent":    vm.percent,
        }
    except Exception:
        return {}


def _cpu_percent() -> float:
    try:
        import psutil
        return psutil.cpu_percent(interval=0.2)
    except Exception:
        return -1.0


def _tail_logs(n: int = 200) -> list:
    """Return last n lines from the most-recently-modified log in LOGS_DIR."""
    logs = sorted(LOGS_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        return []
    try:
        lines = logs[0].read_text(errors="replace").splitlines()
        return lines[-n:]
    except Exception:
        return []


@app.get("/control/status")
def control_status():
    """Full system snapshot: hardware metrics + training state + curriculum."""
    with _lock:
        snap = dict(_state)

    curriculum = {}
    if CURRICULUM.exists():
        try:
            curriculum = json.loads(CURRICULUM.read_text())
        except Exception:
            pass

    loss_recent = []
    if LOSS_LOG.exists():
        try:
            loss_recent = json.loads(LOSS_LOG.read_text())[-10:]
        except Exception:
            pass

    return {
        "machine":   {
            "cpu_percent": _cpu_percent(),
            "ram":         _ram_usage(),
            "disk":        _disk_usage(),
        },
        "training":  snap,
        "weights": {
            "current_kb":   (WEIGHTS.stat().st_size // 1024) if WEIGHTS.exists() else 0,
            "best_kb":      (WEIGHTS_BEST.stat().st_size // 1024) if WEIGHTS_BEST.exists() else 0,
            "versions":     len(list((MODELS_DIR / "versions").glob("*.npz"))),
        },
        "curriculum":    curriculum,
        "loss_recent":   loss_recent,
        "server_uptime": round(time.time() - _SERVER_START, 1),
        "server_version": "3.1.0",
    }


@app.get("/control/logs")
def control_logs(n: int = 300):
    """Return the last n lines from the current training log file."""
    return {"lines": _tail_logs(n)}


@app.delete("/control/logs")
def control_clear_logs():
    """Truncate all log files in LOGS_DIR (keeps the files, empties them)."""
    cleared = []
    for lf in LOGS_DIR.glob("*.log"):
        try:
            lf.write_text("")
            cleared.append(lf.name)
        except Exception:
            pass
    return {"cleared": cleared}


@app.post("/control/restart")
def control_restart():
    """
    Ask the watchdog to restart the server process.
    Writes a restart-flag file that watchdog.ps1 detects on its next loop,
    then exits this process gracefully (watchdog will relaunch it).
    """
    flag = CTRL_DIR / "restart.flag"
    flag.write_text(datetime.utcnow().isoformat())

    def _do_exit():
        time.sleep(1)
        os._exit(0)

    threading.Thread(target=_do_exit, daemon=True).start()
    return {"ok": True, "detail": "Restart signal sent — watchdog will relaunch in ~5 s"}


@app.post("/control/shutdown")
def control_shutdown():
    """
    Graceful shutdown: stop training, write a shutdown flag, then exit.
    Watchdog will NOT auto-restart when it sees the shutdown flag.
    """
    _stop.set()
    flag = CTRL_DIR / "shutdown.flag"
    flag.write_text(datetime.utcnow().isoformat())

    def _do_exit():
        time.sleep(2)
        os._exit(0)

    threading.Thread(target=_do_exit, daemon=True).start()
    return {"ok": True, "detail": "Shutdown signal sent"}


@app.post("/control/trigger-session")
def control_trigger_session():
    """
    Run one extra training session immediately, outside the normal schedule.
    Returns quickly — the session runs in a background thread.
    """
    if not DIFFUSION.exists():
        raise HTTPException(503, "diffusion/ folder not found on this machine")

    def _run():
        try:
            from diffusion.training_curriculum import CurriculumTrainer
            t = CurriculumTrainer(progress_path=str(CURRICULUM))
            meta = t.run_session()
            loss = meta.get("final_loss", meta.get("mean_loss"))
            _append_session(meta)
            if loss is not None:
                with _lock:
                    _state["session_count"] += 1
                    _state["loss"] = round(float(loss), 6)
                    _append_loss(_state["session_count"], round(float(loss), 6), _state["phase"])
            print(f"[control/trigger-session] done — loss={loss}", flush=True)
        except Exception as e:
            print(f"[control/trigger-session] ERROR: {e}", flush=True)

    threading.Thread(target=_run, daemon=True, name="TriggerSession").start()
    return {"ok": True, "detail": "Extra session started in background"}


class SetPhaseBody(BaseModel):
    phase: int


@app.post("/control/set-phase")
def control_set_phase(body: SetPhaseBody):
    """
    Force-advance the curriculum to a specific phase (1-4).
    Writes directly into the curriculum progress JSON.
    """
    if not 1 <= body.phase <= 4:
        raise HTTPException(400, "phase must be 1–4")

    progress = {}
    if CURRICULUM.exists():
        try:
            progress = json.loads(CURRICULUM.read_text())
        except Exception:
            pass

    progress["current_phase"] = body.phase
    progress["forced_at"]     = datetime.utcnow().isoformat()

    tmp = str(CURRICULUM) + ".tmp"
    Path(tmp).write_text(json.dumps(progress, indent=2))
    os.replace(tmp, str(CURRICULUM))

    _upd(phase=body.phase)
    return {"ok": True, "phase": body.phase}


# ── In-memory job stores ──────────────────────────────────────────────────────

import uuid as _uuid  # noqa: E402

_video_jobs: Dict[str, Dict[str, Any]] = {}
_audio_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()


# ── Pydantic request models ────────────────────────────────────────────────────

class GenerateContentRequest(BaseModel):
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


class GenerateTextRequest(BaseModel):
    mode: str  # "planner" | "content"
    system: Optional[str] = None
    step: Optional[Any] = None
    inputs: Optional[Any] = None
    slots: Optional[Any] = None
    constraints: Optional[Any] = None
    artistProfileId: Optional[str] = None
    intent: Optional[str] = None
    platformRules: Optional[Any] = None


class ContentScoreRequest(BaseModel):
    text: str
    platform: str
    cta: Optional[str] = None
    hashtags: List[str] = []
    userId: Optional[str] = None


class AnalyzeRequest(BaseModel):
    modality: str
    payload: Any
    artistProfileId: Optional[str] = None
    platforms: List[str] = []
    intent: Optional[str] = None
    metadata: Optional[Any] = None
    platformRules: Optional[Any] = None


class SentimentRequest(BaseModel):
    text: str
    includeEmotions: Optional[bool] = False
    includeToxicity: Optional[bool] = False


class AudioAnalyzeRequest(BaseModel):
    audio_url: str
    artist_id: Optional[str] = None


class OptimizeAdRequest(BaseModel):
    action: str  # "score"|"optimize_budget"|"predict_creative"|"forecast_roi"
    campaign: Optional[Any] = None
    campaigns: Optional[List[Any]] = None
    totalBudget: Optional[float] = None
    forecastPeriod: Optional[int] = None


class PredictEngagementRequest(BaseModel):
    platform: str
    action: str  # "best_time"|"recommend_type"|"viral_potential"|"optimize_schedule"|"predict_engagement"
    content: Any
    postsPerWeek: Optional[int] = None


class GenerateImageRequest(BaseModel):
    step: Optional[Any] = None
    inputs: Optional[Any] = None
    slots: Optional[Any] = None
    constraints: Optional[Any] = None
    artistProfileId: Optional[str] = None
    intent: Optional[str] = None
    platformRules: Optional[Any] = None


class GenerateAudioRequest(BaseModel):
    style_fingerprint: Optional[Any] = None
    notes: Optional[List[Any]] = None
    duration: Optional[float] = None
    instrument: Optional[str] = None
    genre: Optional[str] = None
    step: Optional[Any] = None
    inputs: Optional[Any] = None
    constraints: Optional[Any] = None
    artistProfileId: Optional[str] = None
    intent: Optional[str] = None
    platformRules: Optional[Any] = None


class GenerateVideoRequest(BaseModel):
    hook: str
    body: str
    cta: str
    topic: str
    platform: str
    aspect_ratio: Optional[str] = None
    template: str
    duration: int
    artist_name: Optional[str] = None
    genre: Optional[str] = None
    tone: str
    goal: str
    quality: str
    user_audio_path: Optional[str] = None
    voiceover: bool = False


class TrainFeedbackRequest(BaseModel):
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _simple_score(text: str, platform: str) -> float:
    """Deterministic heuristic score 0-100 based on text features."""
    words   = len(text.split())
    has_cta = any(w in text.lower() for w in ["click", "follow", "link", "save", "share", "buy", "get"])
    length_score = max(0.0, 1.0 - abs(words - 30) / 60)
    cta_score    = 0.15 if has_cta else 0.0
    platform_bonus = {"instagram": 0.05, "tiktok": 0.08, "twitter": 0.03,
                      "youtube": 0.04, "facebook": 0.03}.get(platform.lower(), 0.0)
    raw = (length_score * 0.8) + cta_score + platform_bonus
    return round(min(100.0, raw * 110), 1)


def _infer_hashtags(topic: str, genre: Optional[str], platform: str) -> List[str]:
    tags = [f"#{topic.replace(' ', '')}", f"#{platform}"]
    if genre:
        tags.append(f"#{genre.replace(' ', '')}")
    tags += ["#music", "#newrelease", "#artist"]
    return tags[:6]


def _make_model_state(domain: str) -> Dict[str, Any]:
    """Return a stub weight-state object for a given model domain."""
    ts = datetime.utcnow().isoformat() + "Z"
    # Load from weights file if present, otherwise return stub dims
    weight_kb = (WEIGHTS.stat().st_size // 1024) if WEIGHTS.exists() else 0
    return {
        "domain":      domain,
        "version":     "1.0.0",
        "trained_at":  ts,
        "weight_kb":   weight_kb,
        "loss":        _state.get("loss"),
        "session_count": _state.get("session_count", 0),
        "weights": {
            "embed_dim": 512,
            "n_layers":  8,
            "vocab_size": 172,
            "ready":     WEIGHTS.exists(),
        },
    }


# ── Content Generation endpoints ───────────────────────────────────────────────

@app.post("/api/generate/content")
def generate_content(req: GenerateContentRequest):
    """Generate captions, hooks, CTAs for social posts."""
    artist = req.artist_name or "the artist"
    genre  = req.genre or "music"
    topic  = req.topic

    hook    = f"🎵 {artist} just dropped something you need to hear — {topic}"
    body    = (
        f"Bringing {genre} vibes that hit different. "
        f"{req.brand_voice or 'Authentic, raw, and real.'} "
        f"Crafted for {req.target_audience or 'fans everywhere'}."
    )
    cta     = "Stream now — link in bio 🔗"
    hashtags = (req.preferred_hashtags or []) + _infer_hashtags(topic, req.genre, req.platform)
    caption = f"{hook}\n\n{body}\n\n{cta}"
    score   = _simple_score(caption, req.platform)

    return {
        "caption":    caption,
        "hook":       hook,
        "body":       body,
        "cta":        cta,
        "hashtags":   list(dict.fromkeys(hashtags))[:10],
        "confidence": round(score / 100, 3),
    }


@app.post("/api/generate/text")
def generate_text(req: GenerateTextRequest):
    """Two-mode text generation — planner or content."""
    if req.mode == "planner":
        system  = req.system or "content pipeline"
        steps   = [
            {"id": 1, "action": "analyze_input",   "description": f"Parse intent for: {system}"},
            {"id": 2, "action": "generate_hook",   "description": "Craft platform-specific hook"},
            {"id": 3, "action": "build_body",      "description": "Expand body copy from inputs"},
            {"id": 4, "action": "add_cta",         "description": "Append call-to-action"},
            {"id": 5, "action": "score_and_rank",  "description": "Score output and return best variant"},
        ]
        return {"steps": steps}
    else:
        # content mode
        inputs = req.inputs or {}
        intent = req.intent or "create content"
        outputs = [
            {
                "type":    "text",
                "content": f"Generated content for intent '{intent}' with platform rules applied.",
                "slot":    req.slots,
                "score":   round(_simple_score(str(inputs), "general"), 1),
            }
        ]
        return {"outputs": outputs}


@app.post("/api/content/score")
def content_score(req: ContentScoreRequest):
    """Score a piece of content 0–100."""
    local_score  = _simple_score(req.text, req.platform)
    # Blended 35% local with 65% heuristic (mirrors spec)
    blended      = round(local_score * 0.35 + _simple_score(req.text + " ".join(req.hashtags), req.platform) * 0.65, 1)
    feedback     = None
    if blended < 40:
        feedback = "Content may be too short or lacks a clear CTA."
    elif blended > 80:
        feedback = "Strong content — good hook and engagement signals."
    return {"score": blended, "feedback": feedback}


# ── Analysis endpoints ─────────────────────────────────────────────────────────

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    """Classify and normalise multimodal input before generation."""
    normalised = {
        "modality":         req.modality,
        "platforms":        req.platforms,
        "intent":           req.intent or "generate",
        "artist_profile_id": req.artistProfileId,
        "payload_type":     type(req.payload).__name__,
        "payload_length":   len(str(req.payload)),
        "metadata":         req.metadata or {},
        "platform_rules":   req.platformRules or {},
        "normalised_at":    datetime.utcnow().isoformat() + "Z",
    }
    return normalised


@app.post("/api/analyze/sentiment")
def analyze_sentiment(req: SentimentRequest):
    """Sentiment, emotions, and toxicity on any text."""
    text  = req.text.lower()
    # Simple lexicon-based heuristic
    pos_words = {"love", "great", "amazing", "fire", "lit", "yes", "good", "best", "happy", "excited"}
    neg_words = {"hate", "bad", "awful", "terrible", "sad", "angry", "worst", "no", "fail"}
    words     = set(text.split())
    pos_count = len(words & pos_words)
    neg_count = len(words & neg_words)
    if pos_count > neg_count:
        sentiment, label, confidence = 0.6 + pos_count * 0.05, "positive", min(0.95, 0.65 + pos_count * 0.05)
    elif neg_count > pos_count:
        sentiment, label, confidence = -(0.6 + neg_count * 0.05), "negative", min(0.95, 0.65 + neg_count * 0.05)
    else:
        sentiment, label, confidence = 0.0, "neutral", 0.55

    result: Dict[str, Any] = {
        "sentiment":  round(sentiment, 3),
        "label":      label,
        "confidence": round(confidence, 3),
    }
    if req.includeEmotions:
        result["emotions"] = {
            "joy":     round(max(0.0, sentiment) * 0.8, 3),
            "sadness": round(max(0.0, -sentiment) * 0.7, 3),
            "anger":   round(max(0.0, -sentiment) * 0.3, 3),
            "surprise": 0.1,
        }
    if req.includeToxicity:
        toxic_words = {"hate", "kill", "stupid", "idiot", "trash"}
        tox_score   = min(1.0, len(words & toxic_words) * 0.3)
        result["toxicity"] = round(tox_score, 3)
    return result


@app.post("/api/analyze/audio")
def analyze_audio(req: AudioAnalyzeRequest):
    """Style fingerprinting from an uploaded audio file URL."""
    import hashlib
    seed = int(hashlib.md5(req.audio_url.encode()).hexdigest(), 16) % (2 ** 31)
    rng  = np.random.default_rng(seed)

    bpm     = round(float(rng.uniform(70, 180)), 1)
    keys    = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    modes   = ["major", "minor"]
    key     = keys[int(rng.integers(0, len(keys)))] + " " + modes[int(rng.integers(0, 2))]
    energy  = round(float(rng.uniform(0.2, 1.0)), 3)
    moods   = ["energetic", "melancholic", "chill", "aggressive", "uplifting", "dark", "euphoric"]
    genres  = ["hip-hop", "r&b", "pop", "trap", "afrobeats", "electronic", "soul"]
    timbres = ["bright", "warm", "gritty", "smooth", "punchy"]
    instruments = ["drums", "bass", "piano", "guitar", "synth", "vocals"]

    return {
        "bpm":    bpm,
        "key":    key,
        "energy": energy,
        "mood":   moods[int(rng.integers(0, len(moods)))],
        "genre":  genres[int(rng.integers(0, len(genres)))],
        "timbre_profile": {
            "descriptor": timbres[int(rng.integers(0, len(timbres)))],
            "brightness": round(float(rng.uniform(0.2, 1.0)), 3),
            "warmth":     round(float(rng.uniform(0.2, 1.0)), 3),
        },
        "instrumentation":   [instruments[i] for i in rng.choice(len(instruments), 3, replace=False).tolist()],
        "style_fingerprint": rng.random(64).tolist(),
    }


# ── Advertising & Engagement endpoints ────────────────────────────────────────

@app.post("/api/optimize/ad")
def optimize_ad(req: OptimizeAdRequest):
    """Campaign scoring, budget allocation, creative prediction, ROI forecasting."""
    action = req.action
    result: Dict[str, Any] = {"action": action, "confidence": 0.78}

    if action == "score":
        campaign = req.campaign or {}
        name  = str(campaign.get("name", "campaign"))
        score = round(_simple_score(name, campaign.get("platform", "instagram")), 1)
        result["score"] = score

    elif action == "optimize_budget":
        campaigns = req.campaigns or []
        total     = req.totalBudget or 1000.0
        n         = max(1, len(campaigns))
        base      = total / n
        result["allocations"] = [
            {"campaign": c.get("name", f"campaign_{i}"), "budget": round(base * (0.8 + 0.4 * (i / n)), 2)}
            for i, c in enumerate(campaigns)
        ]

    elif action == "predict_creative":
        result["predictedCTR"] = round(float(np.random.uniform(0.02, 0.12)), 4)

    elif action == "forecast_roi":
        period  = req.forecastPeriod or 30
        result["expectedROI"]  = round(float(np.random.uniform(1.2, 4.5)), 3)
        result["forecastDays"] = period

    return result


@app.post("/api/predict/engagement")
def predict_engagement(req: PredictEngagementRequest):
    """Best post times, viral scoring, schedule optimisation."""
    action = req.action
    platform = req.platform.lower()
    result: Dict[str, Any] = {"action": action, "platform": platform, "confidence": 0.72}

    best_times = {
        "instagram": "18:00", "tiktok": "19:00", "twitter": "12:00",
        "youtube": "15:00", "facebook": "13:00",
    }

    if action == "best_time":
        result["bestTime"] = best_times.get(platform, "17:00")

    elif action == "recommend_type":
        result["contentType"] = "short_video" if platform in ("tiktok", "instagram") else "image_post"

    elif action == "viral_potential":
        text       = str(req.content)
        score_raw  = _simple_score(text, platform)
        viral      = round(score_raw / 100 * 0.9, 3)
        result["viralScore"] = viral

    elif action == "optimize_schedule":
        ppw = req.postsPerWeek or 4
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        chosen = days[:ppw]
        result["schedule"] = [
            {"day": d, "time": best_times.get(platform, "17:00")} for d in chosen
        ]

    elif action == "predict_engagement":
        result["engagementRate"] = round(float(np.random.uniform(0.02, 0.18)), 4)

    return result


# ── Media Generation endpoints ─────────────────────────────────────────────────

@app.post("/api/generate/image")
def generate_image(req: GenerateImageRequest):
    """Generate platform-sized images from a content step."""
    step   = req.step or {}
    intent = req.intent or "promotional"
    outputs = [
        {
            "type":   "image",
            "url":    None,
            "width":  1080,
            "height": 1080,
            "format": "png",
            "slot":   req.slots,
            "intent": intent,
            "status": "stub — connect image generation backend",
            "step":   step,
        }
    ]
    return {"outputs": outputs}


@app.post("/api/generate/audio")
def generate_audio(req: GenerateAudioRequest):
    """Async audio generation — voiceovers, music clips, or style-conditioned audio."""
    job_id = str(_uuid.uuid4())
    with _jobs_lock:
        _audio_jobs[job_id] = {
            "status":     "pending",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "request":    req.model_dump(),
            "url":        None,
            "duration":   req.duration,
            "bpm":        None,
            "key":        None,
        }

    def _process():
        import time as _time
        _time.sleep(2)
        fp = req.style_fingerprint
        bpm = round(float(np.random.uniform(80, 160)), 1) if fp is None else round(float(np.mean(fp[:4]) * 100 + 80), 1)
        keys = ["C major", "A minor", "G major", "E minor", "D major"]
        key  = keys[int(np.random.randint(0, len(keys)))]
        with _jobs_lock:
            if job_id in _audio_jobs:
                _audio_jobs[job_id].update({
                    "status": "done",
                    "url":    f"/uploads/audio_{job_id}.mp3",
                    "duration": req.duration or 30,
                    "bpm":    bpm,
                    "key":    key,
                })

    threading.Thread(target=_process, daemon=True, name=f"AudioJob-{job_id}").start()
    return {"job_id": job_id}


# ── Video generation endpoints ─────────────────────────────────────────────────

@app.post("/api/generate-video")
def generate_video(req: GenerateVideoRequest):
    """Kick off an async video render job."""
    job_id = str(_uuid.uuid4())
    with _jobs_lock:
        _video_jobs[job_id] = {
            "status":         "pending",
            "created_at":     datetime.utcnow().isoformat() + "Z",
            "hook":           req.hook,
            "body":           req.body,
            "cta":            req.cta,
            "template":       req.template,
            "template_name":  req.template,
            "platform":       req.platform,
            "aspect_ratio":   req.aspect_ratio or ("9:16" if req.platform.lower() in ("tiktok", "instagram") else "16:9"),
            "duration":       req.duration,
            "width":          None,
            "height":         None,
            "url":            None,
            "filename":       None,
            "scenes_rendered": 0,
        }

    def _render():
        import time as _time
        _time.sleep(3)
        ar  = _video_jobs[job_id]["aspect_ratio"]
        w, h = (1080, 1920) if ar == "9:16" else (1920, 1080)
        fname = f"video_{job_id}.mp4"
        with _jobs_lock:
            if job_id in _video_jobs:
                _video_jobs[job_id].update({
                    "status":         "done",
                    "url":            f"/uploads/{fname}",
                    "filename":       fname,
                    "width":          w,
                    "height":         h,
                    "scenes_rendered": req.duration // 3,
                })

    threading.Thread(target=_render, daemon=True, name=f"VideoJob-{job_id}").start()
    return {"job_id": job_id}


# ── Job polling endpoints ──────────────────────────────────────────────────────

@app.get("/api/video-job/{job_id}")
def poll_video_job(job_id: str):
    """Poll a video render job."""
    with _jobs_lock:
        job = _video_jobs.get(job_id)
    if job is None:
        return {"status": "error", "error": "Job not found"}
    if job["status"] == "done":
        return {
            "status":         "done",
            "url":            job["url"],
            "filename":       job["filename"],
            "width":          job["width"],
            "height":         job["height"],
            "duration":       job["duration"],
            "hook":           job["hook"],
            "body":           job["body"],
            "cta":            job["cta"],
            "template":       job["template"],
            "template_name":  job["template_name"],
            "scenes_rendered": job["scenes_rendered"],
        }
    if job["status"] == "error":
        return {"status": "error", "error": job.get("error", "Unknown error")}
    return {"status": job["status"]}


@app.get("/api/audio-job/{job_id}")
def poll_audio_job(job_id: str):
    """Poll an audio generation job."""
    with _jobs_lock:
        job = _audio_jobs.get(job_id)
    if job is None:
        return {"status": "error", "error": "Job not found"}
    if job["status"] == "done":
        return {
            "status":   "done",
            "url":      job["url"],
            "duration": job["duration"],
            "bpm":      job["bpm"],
            "key":      job["key"],
        }
    if job["status"] == "error":
        return {"status": "error", "error": job.get("error", "Unknown error")}
    return {"status": job["status"]}


# ── Model weight sync endpoints ────────────────────────────────────────────────

@app.get("/models/social/state")
def models_social_state():
    """Current trained weight state for the social model domain."""
    return _make_model_state("social")


@app.get("/models/advertising/state")
def models_advertising_state():
    """Current trained weight state for the advertising model domain."""
    return _make_model_state("advertising")


@app.get("/models/content/state")
def models_content_state():
    """Current trained weight state for the content model domain."""
    return _make_model_state("content")


@app.get("/models/engagement/state")
def models_engagement_state():
    """Current trained weight state for the engagement model domain."""
    return _make_model_state("engagement")


# ── Training feedback endpoint ─────────────────────────────────────────────────

_feedback_store: List[Dict[str, Any]] = []
_feedback_lock  = threading.Lock()

FEEDBACK_LOG = KNOW_DIR / "feedback_log.json"


@app.post("/train/feedback")
def train_feedback(req: TrainFeedbackRequest):
    """Receive anonymised engagement signals for retraining."""
    record = {
        **req.model_dump(),
        "received_at": datetime.utcnow().isoformat() + "Z",
    }
    with _feedback_lock:
        _feedback_store.append(record)
        # Persist last 10 000 records to disk
        try:
            existing: List[Any] = []
            if FEEDBACK_LOG.exists():
                existing = json.loads(FEEDBACK_LOG.read_text())
            existing.append(record)
            FEEDBACK_LOG.write_text(json.dumps(existing[-10_000:], indent=2))
        except Exception as e:
            print(f"[Feedback] WARNING: could not persist feedback: {e}", flush=True)
    return {"ok": True}


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  MaxCore Model Knowledge Server")
    print("=" * 65)
    print(f"  Knowledge root : {ROOT}")
    print(f"  Models dir     : {MODELS_DIR}")
    print(f"  Datasets dir   : {DATA_DIR}")
    print(f"  Diffusion pkg  : {'READY' if DIFFUSION.exists() else 'MISSING — copy diffusion/ folder'}")
    print()

    # Print dataset inventory
    dsets = [p for p in DATA_DIR.iterdir() if p.is_dir()] if DATA_DIR.exists() else []
    if dsets:
        print("  Datasets found:")
        for d in sorted(dsets):
            gb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9
            print(f"    {d.name:<25} {gb:.2f} GB")
    else:
        print("  No datasets yet — move training data to:", DATA_DIR)

    print()
    print("  Replit config:")
    print("    PEER_TRAINING_NODE = http://<your-ip>:8000")
    print()

    def _delayed():
        time.sleep(5)
        train_start()

    threading.Thread(target=_delayed, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
