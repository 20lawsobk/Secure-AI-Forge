"""
RenderManager — real async job queue for thumbnail and video renders.

Manages a thread pool, tracks job status by UUID, and routes work to the
image engine (thumbnails) and video engine (cinematic renders).
"""
from __future__ import annotations
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from .boostsheets.boostsheet import BoostSheet


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RenderManager:
    """
    Async render queue.  All heavy work runs on a background thread pool
    so that the FastAPI event loop is never blocked.
    """

    _MAX_WORKERS = 2

    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=self._MAX_WORKERS,
            thread_name_prefix="render-mgr",
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def render_thumbnail(
        self,
        sheet: BoostSheet,
        image_engine=None,
    ) -> Dict[str, Any]:
        job_id = str(uuid.uuid4())
        self._set_job(job_id, "thumbnail", sheet.sheet_id, "queued")
        sheet.add_history(f"Thumbnail render queued — job {job_id}")
        self._executor.submit(self._run_thumbnail, job_id, sheet, image_engine)
        return {"status": "queued", "job_id": job_id, "sheet_id": sheet.sheet_id, "type": "thumbnail"}

    def render_video(
        self,
        sheet: BoostSheet,
        video_agent=None,
        video_agent_request=None,
    ) -> Dict[str, Any]:
        job_id = str(uuid.uuid4())
        self._set_job(job_id, "video", sheet.sheet_id, "queued")
        sheet.add_history(f"Video render queued — job {job_id}")
        self._executor.submit(
            self._run_video, job_id, sheet, video_agent, video_agent_request
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

    def shutdown(self, wait: bool = False):
        self._executor.shutdown(wait=wait)

    # ── Internal runners ────────────────────────────────────────────────────

    def _run_thumbnail(self, job_id: str, sheet: BoostSheet, image_engine):
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
                    self._update_job(job_id, status="error", error=str(e), finished_at=_now_iso())
                    return

            sheet.add_history(f"Thumbnail render complete — job {job_id}")
            self._update_job(
                job_id,
                status="done",
                finished_at=_now_iso(),
                result={"file_path": result_path, "sheet_id": sheet.sheet_id},
            )
        except Exception as exc:
            self._update_job(job_id, status="error", error=str(exc), finished_at=_now_iso())

    def _run_video(self, job_id: str, sheet: BoostSheet, video_agent, video_agent_request):
        self._update_job(job_id, status="running", started_at=_now_iso())
        try:
            if video_agent is not None and video_agent_request is not None:
                render_result = video_agent.render(video_agent_request)
                if render_result.success:
                    sheet.add_history(f"Video render complete — job {job_id}: {render_result.filename}")
                    self._update_job(
                        job_id,
                        status="done",
                        finished_at=_now_iso(),
                        result={
                            "file_path":      render_result.file_path,
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
                    sheet.add_history(f"Video render failed — job {job_id}: {render_result.error}")
                    self._update_job(
                        job_id, status="error",
                        error=render_result.error,
                        finished_at=_now_iso(),
                    )
            else:
                self._update_job(
                    job_id, status="error",
                    error="No video agent configured",
                    finished_at=_now_iso(),
                )
        except Exception as exc:
            self._update_job(job_id, status="error", error=str(exc), finished_at=_now_iso())

    # ── Internal state helpers ───────────────────────────────────────────────

    def _set_job(self, job_id: str, job_type: str, sheet_id: str, status: str):
        with self._lock:
            self._jobs[job_id] = {
                "job_id":     job_id,
                "type":       job_type,
                "sheet_id":   sheet_id,
                "status":     status,
                "created_at": _now_iso(),
                "started_at": None,
                "finished_at": None,
                "result":     None,
                "error":      None,
            }

    def _update_job(self, job_id: str, **kwargs):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(kwargs)
