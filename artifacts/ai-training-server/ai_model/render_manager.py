"""
RenderManager — real async job queue for thumbnail and video renders.

Manages a thread pool, tracks job status by UUID, and routes work to the
image engine (thumbnails) and video engine (cinematic renders).

90M-scale design:
- Submission coalescing: identical (sheet_id, type) submissions while a job is
  queued/running return the existing job_id — zero redundant GPU work.
- TTL eviction: completed/errored jobs are removed after _DONE_TTL_S seconds
  by a background GC thread, so _jobs never grows unboundedly.
- Queue depth cap: submissions beyond _MAX_QUEUE_DEPTH are rejected with a
  structured error so callers can back off cleanly.
"""
from __future__ import annotations
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

from .boostsheets.boostsheet import BoostSheet


# Jobs older than this (seconds) that have finished are evicted from memory.
_DONE_TTL_S:     int = 600   # 10 minutes
# Hard cap on in-flight (queued + running) jobs; beyond this, submissions
# get a structured queue_full response so clients can apply backpressure.
_MAX_QUEUE_DEPTH: int = 50_000
# How often the GC thread wakes up to sweep expired entries.
_GC_INTERVAL_S:  int = 60


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _now_ts() -> float:
    return time.monotonic()


class RenderManager:
    """
    Async render queue.  All heavy work runs on a background thread pool
    so that the FastAPI event loop is never blocked.

    Thread-safety: all mutations to _jobs and _active go through _lock.
    """

    _MAX_WORKERS = 2

    def __init__(self):
        self._jobs:   Dict[str, Dict[str, Any]] = {}
        self._active: Dict[Tuple[str, str], str] = {}   # (sheet_id, type) -> job_id
        self._lock    = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=self._MAX_WORKERS,
            thread_name_prefix="render-mgr",
        )
        # Background GC: evicts expired completed/errored jobs.
        self._gc_thread = threading.Thread(
            target=self._gc_loop, daemon=True, name="render-mgr-gc"
        )
        self._gc_thread.start()

    # ── Public API ──────────────────────────────────────────────────────────

    def render_thumbnail(
        self,
        sheet: BoostSheet,
        image_engine=None,
    ) -> Dict[str, Any]:
        key = (sheet.sheet_id, "thumbnail")
        with self._lock:
            # Coalesce: if an identical job is already in flight, return it.
            existing = self._active.get(key)
            if existing and self._jobs.get(existing, {}).get("status") in ("queued", "running"):
                return {
                    "status":   "coalesced",
                    "job_id":   existing,
                    "sheet_id": sheet.sheet_id,
                    "type":     "thumbnail",
                }
            # Backpressure: reject when queue is too deep.
            if self._queue_depth_locked() >= _MAX_QUEUE_DEPTH:
                return {
                    "status":    "queue_full",
                    "job_id":    None,
                    "sheet_id":  sheet.sheet_id,
                    "type":      "thumbnail",
                    "retry_after": 5,
                }
            job_id = str(uuid.uuid4())
            self._set_job_locked(job_id, "thumbnail", sheet.sheet_id, "queued")
            self._active[key] = job_id

        sheet.add_history(f"Thumbnail render queued — job {job_id}")
        self._executor.submit(self._run_thumbnail, job_id, key, sheet, image_engine)
        return {"status": "queued", "job_id": job_id, "sheet_id": sheet.sheet_id, "type": "thumbnail"}

    def render_video(
        self,
        sheet: BoostSheet,
        video_agent=None,
        video_agent_request=None,
    ) -> Dict[str, Any]:
        key = (sheet.sheet_id, "video")
        with self._lock:
            # Coalesce: identical in-flight job → return existing.
            existing = self._active.get(key)
            if existing and self._jobs.get(existing, {}).get("status") in ("queued", "running"):
                return {
                    "status":   "coalesced",
                    "job_id":   existing,
                    "sheet_id": sheet.sheet_id,
                    "type":     "video",
                }
            # Backpressure.
            if self._queue_depth_locked() >= _MAX_QUEUE_DEPTH:
                return {
                    "status":    "queue_full",
                    "job_id":    None,
                    "sheet_id":  sheet.sheet_id,
                    "type":      "video",
                    "retry_after": 5,
                }
            job_id = str(uuid.uuid4())
            self._set_job_locked(job_id, "video", sheet.sheet_id, "queued")
            self._active[key] = job_id

        sheet.add_history(f"Video render queued — job {job_id}")
        self._executor.submit(
            self._run_video, job_id, key, sheet, video_agent, video_agent_request
        )
        return {"status": "queued", "job_id": job_id, "sheet_id": sheet.sheet_id, "type": "video"}

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return {"status": "not_found", "job_id": job_id}
        return dict(job)

    def list_jobs(self, limit: int = 50) -> list:
        with self._lock:
            items = list(self._jobs.values())
        items.sort(key=lambda j: j.get("created_at", ""), reverse=True)
        return items[:limit]

    def stats(self) -> Dict[str, Any]:
        """Return live queue statistics for monitoring."""
        with self._lock:
            total   = len(self._jobs)
            queued  = sum(1 for j in self._jobs.values() if j["status"] == "queued")
            running = sum(1 for j in self._jobs.values() if j["status"] == "running")
            done    = sum(1 for j in self._jobs.values() if j["status"] == "done")
            error   = sum(1 for j in self._jobs.values() if j["status"] == "error")
        return {
            "total": total, "queued": queued, "running": running,
            "done": done, "error": error,
            "max_queue_depth": _MAX_QUEUE_DEPTH,
            "done_ttl_s": _DONE_TTL_S,
        }

    def shutdown(self, wait: bool = False):
        self._executor.shutdown(wait=wait)

    # ── Internal runners ────────────────────────────────────────────────────

    def _run_thumbnail(self, job_id: str, key: Tuple[str, str],
                       sheet: BoostSheet, image_engine):
        self._update_job(job_id, status="running", started_at=_now_iso())
        try:
            result_path = None
            if image_engine is not None:
                try:
                    from .agents.visual_spec_agent import VisualSpecRequest
                    spec_req = VisualSpecRequest(
                        idea=getattr(sheet, "title", "") or sheet.sheet_id,
                        platform=getattr(sheet, "platform", "instagram"),
                        tone=getattr(sheet, "tone", "professional"),
                    )
                    render_result = image_engine.render(spec_req)
                    result_path = getattr(render_result, "file_path", None)
                except Exception as e:
                    self._update_job(job_id, status="error", error=str(e),
                                     finished_at=_now_iso())
                    self._release_active(key, job_id)
                    return

            sheet.add_history(f"Thumbnail render complete — job {job_id}")
            self._update_job(
                job_id, status="done", finished_at=_now_iso(),
                result={"file_path": result_path, "sheet_id": sheet.sheet_id},
            )
        except Exception as exc:
            self._update_job(job_id, status="error", error=str(exc),
                             finished_at=_now_iso())
        finally:
            self._release_active(key, job_id)

    def _run_video(self, job_id: str, key: Tuple[str, str],
                   sheet: BoostSheet, video_agent, video_agent_request):
        self._update_job(job_id, status="running", started_at=_now_iso())
        try:
            if video_agent is not None and video_agent_request is not None:
                render_result = video_agent.render(video_agent_request)
                if render_result.success:
                    sheet.add_history(
                        f"Video render complete — job {job_id}: {render_result.filename}"
                    )
                    self._update_job(
                        job_id, status="done", finished_at=_now_iso(),
                        result={
                            "file_path":       render_result.file_path,
                            "filename":        render_result.filename,
                            "url":             f"/uploads/videos/{render_result.filename}",
                            "duration":        render_result.duration,
                            "width":           render_result.width,
                            "height":          render_result.height,
                            "template":        render_result.template_name,
                            "scenes_rendered": render_result.scenes_rendered,
                            "render_ms":       render_result.render_time_ms,
                        },
                    )
                else:
                    sheet.add_history(
                        f"Video render failed — job {job_id}: {render_result.error}"
                    )
                    self._update_job(job_id, status="error",
                                     error=render_result.error,
                                     finished_at=_now_iso())
            else:
                self._update_job(job_id, status="error",
                                 error="No video agent configured",
                                 finished_at=_now_iso())
        except Exception as exc:
            self._update_job(job_id, status="error", error=str(exc),
                             finished_at=_now_iso())
        finally:
            self._release_active(key, job_id)

    # ── GC: evict expired completed/errored jobs ────────────────────────────

    def _gc_loop(self) -> None:
        """Daemon thread: sweep expired jobs every _GC_INTERVAL_S seconds."""
        while True:
            time.sleep(_GC_INTERVAL_S)
            try:
                self._evict_expired()
            except Exception:
                pass  # never crash the GC thread

    def _evict_expired(self) -> int:
        """Remove done/error jobs whose finished_at is older than _DONE_TTL_S.
        Returns the count of evicted entries (useful for metrics/tests)."""
        threshold_iso = datetime.fromtimestamp(
            time.time() - _DONE_TTL_S, tz=timezone.utc
        ).isoformat()
        with self._lock:
            expired = [
                jid for jid, j in self._jobs.items()
                if j["status"] in ("done", "error")
                and (j.get("finished_at") or "") < threshold_iso
            ]
            for jid in expired:
                del self._jobs[jid]
        return len(expired)

    # ── Internal state helpers ───────────────────────────────────────────────

    def _queue_depth_locked(self) -> int:
        """Count queued+running jobs. Must be called with _lock held."""
        return sum(
            1 for j in self._jobs.values()
            if j["status"] in ("queued", "running")
        )

    def _release_active(self, key: Tuple[str, str], job_id: str) -> None:
        """Remove key from _active only if it still points to this job_id."""
        with self._lock:
            if self._active.get(key) == job_id:
                del self._active[key]

    def _set_job_locked(self, job_id: str, job_type: str,
                        sheet_id: str, status: str) -> None:
        """Create a new job record. Must be called with _lock held."""
        self._jobs[job_id] = {
            "job_id":      job_id,
            "type":        job_type,
            "sheet_id":    sheet_id,
            "status":      status,
            "created_at":  _now_iso(),
            "started_at":  None,
            "finished_at": None,
            "result":      None,
            "error":       None,
        }

    def _update_job(self, job_id: str, **kwargs) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(kwargs)
