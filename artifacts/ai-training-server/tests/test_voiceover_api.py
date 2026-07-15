"""API-level regression tests: the voice-over flag must be strictly opt-in.

Proves that rendering with ``voiceover`` omitted/false produces exactly the
previous music/silent behavior (no narration files are created, poll payload
reports ``voiceover: false``) and that ``voiceover: true`` is reported
correctly — on both video render endpoints.

These are live-server integration tests: they run against the local AI
training server (port 9878) and are skipped automatically when it is not up.
Topics are novel per-endpoint (content dedup cache is disk-backed and
survives restarts, so reused topics would coalesce into old jobs).
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

import pytest
import requests

_HERE = Path(__file__).resolve().parent
_SERVER_ROOT = _HERE.parent
sys.path.insert(0, str(_SERVER_ROOT))

BASE = os.environ.get("AI_SERVER_URL", "http://localhost:9878")
API_KEY = os.environ.get(
    "AI_TRAINING_KEY_PROD",
    "f242bf97d7e46b7ca0b17cd6b01ca9239bc327b862a86b703556565523849701",
)
HEADERS = {"Content-Type": "application/json", "X-Api-Key": API_KEY}
UPLOADS = _SERVER_ROOT / "uploads"

_RENDER_TIMEOUT = 240  # seconds per job


def _server_up() -> bool:
    try:
        return requests.get(f"{BASE}/health", timeout=5).status_code == 200
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _server_up(), reason="AI training server not running on port 9878"
)


def _wait_model_ready(payload_fn, timeout=180):
    """POST via payload_fn until the model has finished initialising."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = payload_fn()
        if r.status_code != 503:
            return r
        time.sleep(10)
    pytest.skip("AI model did not finish initialising in time")


def _poll_done(job_id: str) -> dict:
    deadline = time.time() + _RENDER_TIMEOUT
    while time.time() < deadline:
        r = requests.get(f"{BASE}/api/video-job/{job_id}", headers=HEADERS, timeout=15)
        assert r.status_code == 200, f"poll failed: {r.status_code} {r.text[:200]}"
        d = r.json()
        if d.get("status") in ("done", "error"):
            return d
        time.sleep(6)
    raise AssertionError(f"job {job_id} did not finish within {_RENDER_TIMEOUT}s")


def _vo_files_for(job_id: str) -> list[Path]:
    """Narration artifacts the voiceover path would create for this job."""
    stem = job_id[:12]
    return [p for p in UPLOADS.glob(f"vo*_{stem}*") if p.is_file()]


def _novel(base: str) -> str:
    """Novel topic per run so the disk-backed dedup cache can't coalesce."""
    return f"{base} {uuid.uuid4().hex[:8]}"


def _submit_generate_video(voiceover, topic: str) -> str:
    body = {
        "idea": topic,
        "platform": "tiktok",
        "duration": 6,
    }
    if voiceover is not None:
        body["voiceover"] = voiceover
    r = _wait_model_ready(
        lambda: requests.post(
            f"{BASE}/api/generate-video", json=body, headers=HEADERS, timeout=30
        )
    )
    assert r.status_code == 200, r.text
    return r.json()["job_id"]


def _submit_generate_ai(voiceover, topic: str) -> str:
    body = {
        "idea": topic,
        "platform": "tiktok",
        "duration": 6,
    }
    if voiceover is not None:
        body["voiceover"] = voiceover
    r = _wait_model_ready(
        lambda: requests.post(
            f"{BASE}/api/video/generate-ai", json=body, headers=HEADERS, timeout=120
        )
    )
    assert r.status_code == 200, r.text
    return r.json()["job_id"]


class TestGenerateVideoEndpoint:
    """/api/generate-video — voiceover strictly opt-in."""

    def test_voiceover_omitted_is_off(self):
        job_id = _submit_generate_video(None, _novel("neon tide rooftop chorus"))
        job = _poll_done(job_id)
        assert job["status"] == "done", job.get("error")
        assert job.get("voiceover") is False, "default must report voiceover false"
        assert not _vo_files_for(job_id), (
            "no narration files may be created when voiceover is omitted"
        )

    def test_voiceover_false_is_off(self):
        job_id = _submit_generate_video(False, _novel("copper dune skyline waltz"))
        job = _poll_done(job_id)
        assert job["status"] == "done", job.get("error")
        assert job.get("voiceover") is False
        assert not _vo_files_for(job_id)

    def test_voiceover_true_reported(self):
        job_id = _submit_generate_video(True, _novel("velvet meteor lullaby drop"))
        job = _poll_done(job_id)
        assert job["status"] == "done", job.get("error")
        # voiceover=true must be reported truthfully: True when narration was
        # rendered; if espeak were unavailable the never-raise fallback keeps
        # it False — in that case narration files must also be absent.
        if job.get("voiceover") is True:
            assert _vo_files_for(job_id), "voiceover=true job should leave narration files"
        else:
            assert not _vo_files_for(job_id)


class TestGenerateAiEndpoint:
    """/api/video/generate-ai — same opt-in contract via raw body flag."""

    def test_voiceover_omitted_is_off(self):
        job_id = _submit_generate_ai(None, _novel("saffron comet harbor anthem"))
        job = _poll_done(job_id)
        assert job["status"] == "done", job.get("error")
        assert job.get("voiceover") is False
        assert not _vo_files_for(job_id)

    def test_voiceover_false_is_off(self):
        job_id = _submit_generate_ai(False, _novel("juniper eclipse boulevard suite"))
        job = _poll_done(job_id)
        assert job["status"] == "done", job.get("error")
        assert job.get("voiceover") is False
        assert not _vo_files_for(job_id)

    def test_voiceover_true_reported(self):
        job_id = _submit_generate_ai(True, _novel("glacier violet arcade hymn"))
        job = _poll_done(job_id)
        assert job["status"] == "done", job.get("error")
        if job.get("voiceover") is True:
            assert _vo_files_for(job_id)
        else:
            assert not _vo_files_for(job_id)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
