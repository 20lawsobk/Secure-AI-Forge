"""Regression guard: get_diffusion_frame must denoise at native resolution.

The diffusion latent grid is fixed (LATENT_H x LATENT_W), so decoding at the
caller's output resolution (e.g. 1920) only multiplies DDIM/VAE pixel work
(~10x slower measured) without adding detail — the caller-facing bilinear
resize produces the same image either way. These tests lock in the contract:
callers always get a frame at exactly their requested dimensions, and the
heavy generate() call always runs at the pipeline's native RESOLUTION.
"""

import time

import pytest

from ai_model.video.diffusion import maxcore_diffusion as md


def _pipeline_available() -> bool:
    try:
        return md._get_pipeline() is not None
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _pipeline_available(), reason="diffusion pipeline unavailable (torch)"
)


@pytest.mark.parametrize("width,height", [(1080, 1920), (1920, 1080), (256, 256)])
def test_output_matches_requested_dimensions(width, height):
    frame = md.get_diffusion_frame(
        idea="resolution contract test",
        platform="tiktok",
        tone="hype",
        awareness="",
        width=width,
        height=height,
        context={},
    )
    assert frame is not None
    assert frame.shape == (height, width, 3)
    assert frame.dtype.kind == "u"


def test_generate_called_at_native_resolution(monkeypatch):
    """The heavy generate() must receive resolution=RESOLUTION, never max(w,h)."""
    seen = {}
    pipeline = md._get_pipeline()
    real_generate = pipeline.generate

    def spy(*args, **kwargs):
        seen["resolution"] = kwargs.get("resolution")
        return real_generate(*args, **kwargs)

    monkeypatch.setattr(pipeline, "generate", spy)
    frame = md.get_diffusion_frame(
        idea="native res spy",
        platform="tiktok",
        tone="hype",
        awareness="",
        width=1080,
        height=1920,
        context={},
    )
    assert frame is not None
    assert seen["resolution"] == md.RESOLUTION


def test_hi_res_request_stays_fast():
    """A 1920-tall request must cost ~native-res time (was ~28s pre-fix)."""
    t0 = time.time()
    frame = md.get_diffusion_frame(
        idea="speed guard",
        platform="tiktok",
        tone="hype",
        awareness="",
        width=1080,
        height=1920,
        context={},
    )
    elapsed = time.time() - t0
    assert frame is not None
    # Native-res run measures ~3s; 15s leaves generous CI headroom while
    # still catching a regression to full-res denoising (~28s).
    assert elapsed < 15, f"diffusion frame took {elapsed:.1f}s — hi-res regression?"
