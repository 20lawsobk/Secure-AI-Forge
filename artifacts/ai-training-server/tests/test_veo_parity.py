"""Veo-parity feature guards: image conditioning, cinematography fusion,
multi-sample seeds, and the /api/video/extend endpoint contract.

Fast tests only — no rendering. Endpoint checks hit the live server on 9878.
"""
import base64
import io
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = "http://127.0.0.1:9878"
API_KEY = "f242bf97d7e46b7ca0b17cd6b01ca9239bc327b862a86b703556565523849701"
HEADERS = {"Content-Type": "application/json", "X-Api-Key": API_KEY}


def _post(path: str, payload: dict) -> tuple[int, dict]:
    rq = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(), headers=HEADERS, method="POST"
    )
    try:
        with urllib.request.urlopen(rq, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


# ── SDEdit init-frame conditioning ─────────────────────────────────────────

def test_sdedit_decodes_init_frame_and_never_raises():
    from PIL import Image
    from ai_model.video.diffusion.sdedit_prior import _init_frame_from_context

    img = Image.new("RGB", (64, 64), (200, 30, 30))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    frame = _init_frame_from_context(128, 96, {"init_frame_b64": b64})
    assert frame is not None and frame.shape == (96, 128, 3)
    assert frame[0, 0, 0] > 150  # red channel survived decode+resize

    # data-URL form accepted, garbage and absence return None (never raise)
    assert _init_frame_from_context(64, 64, {"init_frame_b64": "data:image/png;base64," + b64}) is not None
    assert _init_frame_from_context(64, 64, {"init_frame_b64": "!!notb64!!"}) is None
    assert _init_frame_from_context(64, 64, {}) is None


# ── Scene-builder conditioning plumbing ────────────────────────────────────

def test_build_scenes_routes_frames_and_cinematography():
    from ai_model.video import ai_scene_builder as asb

    scenes = [{"type": t, "text": f"scene {t}"} for t in ("hook", "verse", "chorus", "cta")]
    cfgs = asb.build_scenes(
        scenes, idea="night drive", genre="trap", tone="dark",
        platform="tiktok", artist_name="AX", total_duration=12.0,
        width=540, height=960,
        camera_motion="dolly_zoom", composition="close_up",
        reference_images=["REF1", "REF2"],
        first_frame_b64="FIRST", last_frame_b64="LAST",
    )
    metas = [c.diffusion_meta for c in cfgs]
    assert all(metas), "conditioning params must force diffusion_meta on"
    assert metas[0]["init_frame_b64"] == "FIRST"     # first scene anchors on first frame
    assert metas[-1]["init_frame_b64"] == "LAST"     # last scene anchors on last frame
    for m in metas[1:-1]:                            # middle scenes cycle references
        assert m["init_frame_b64"] in ("REF1", "REF2")
    # camera/composition fold into the awareness text the conditioner encodes
    assert "camera dolly zoom" in metas[0]["awareness"]
    assert "close up framing" in metas[0]["awareness"]
    assert metas[0]["composition"] == "close_up"


def test_build_scenes_backward_compatible_when_no_veo_params():
    from ai_model.video import ai_scene_builder as asb

    scenes = [{"type": "hook", "text": "x"}, {"type": "cta", "text": "y"}]
    cfgs = asb.build_scenes(
        scenes, idea="x", genre="pop", tone="chill", platform="tiktok",
        artist_name="", total_duration=6.0, width=540, height=960,
    )
    assert all(c.diffusion_meta is None for c in cfgs)


# ── Multi-sample generation ────────────────────────────────────────────────

def test_sample_count_returns_distinct_seeded_jobs():
    status, body = _post("/api/generate-video", {
        "idea": "test veo parity sample fanout",
        "platform": "tiktok", "duration": 6,
        "sample_count": 3, "seed": 42,
    })
    assert status == 200, body
    jobs = body.get("sample_jobs")
    assert jobs and len(jobs) == 3
    assert len({j["job_id"] for j in jobs}) == 3
    assert len({j["seed"] for j in jobs}) == 3      # derived seeds all differ
    assert jobs[0]["seed"] == 42                     # sample 0 honours explicit seed


# ── Video extension endpoint contract ──────────────────────────────────────

def test_extend_requires_idea_and_real_source():
    status, body = _post("/api/video/extend", {"source": "whatever.mp4"})
    assert status == 422 and "idea" in body["detail"]

    status, body = _post("/api/video/extend", {"source": "no_such_file.mp4", "idea": "go on"})
    assert status == 404

    # path traversal must not escape uploads/videos
    status, _ = _post("/api/video/extend", {"source": "../../etc/passwd", "idea": "x"})
    assert status == 404
