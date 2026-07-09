"""Regression tests for the RTA-1 fabric's strict correctness contracts.

These guard three properties that were bugs found and fixed during the build and
that are easy to silently re-break in future edits:

  1. Cross-process determinism — the same inputs must produce byte-identical
     images across fresh processes with *different* ``PYTHONHASHSEED`` values.
     (Seeding must never fall back to Python's per-process-salted ``hash()``.)

  2. Spectral tail reconstruction — the ARC overlap-add pass must reconstruct the
     tail of a non-hop-aligned clip as faithfully as the rest of the signal.

  3. Fabric self-test smoke — all three mediums (image / video / audio) render
     and the Digital-GPU GEMM counter actually increments.

Run standalone:
    cd artifacts/ai-training-server
    uv run python -u -m ai_model.rta.test_render_contracts
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from typing import List

import numpy as np

from ai_model.rta.api import render_image, self_test
from ai_model.rta.audio.spectral import SpectralConfig, SpectralEngine

_FAILS: List[str] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    status = "ok  " if cond else "FAIL"
    line = f"  [{status}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    if not cond:
        _FAILS.append(name)


# ── 1. Cross-process image determinism ──────────────────────────────────────

# A tiny render whose bytes we hash. Kept small so the two subprocesses are fast.
_RENDER_SNIPPET = (
    "import hashlib;"
    "from ai_model.rta.api import render_image;"
    "img=render_image('dark_neon','cinematic',width=48,height=48,"
    "samples=1,max_bounces=1,seed=7);"
    "print(hashlib.sha256(img.tobytes()).hexdigest())"
)


def _render_hash_in_subprocess(hash_seed: str) -> str:
    """Render in a fresh interpreter with an explicit PYTHONHASHSEED."""
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    # Run from the package root so ``ai_model`` imports resolve, regardless of
    # the caller's CWD.
    pkg_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    proc = subprocess.run(
        [sys.executable, "-c", _RENDER_SNIPPET],
        env=env,
        cwd=pkg_root,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"render subprocess (PYTHONHASHSEED={hash_seed}) failed:\n{proc.stderr}"
        )
    return proc.stdout.strip().splitlines()[-1]


def test_image_bytes_identical_across_processes() -> None:
    """Same inputs → byte-identical image across processes with different hash seeds."""
    h0 = _render_hash_in_subprocess("0")
    h1 = _render_hash_in_subprocess("12345")
    _check(
        "image bytes identical across fresh processes (differing PYTHONHASHSEED)",
        h0 == h1,
        f"{h0[:12]} vs {h1[:12]}",
    )


# ── 2. Spectral tail reconstruction ─────────────────────────────────────────

def test_spectral_tail_reconstruction_non_hop_aligned() -> None:
    """The reconstructed tail of a non-hop-aligned clip is as accurate as the whole.

    With ``reduction_db=0`` the gate is the identity, so ``denoise`` reduces to a
    pure STFT→iSTFT overlap-add round-trip. The tail must be fully reconstructed
    (the ``_frame`` padding bug left it under-covered), so the mean error over the
    last window should not exceed the whole-signal mean error.
    """
    cfg = SpectralConfig(reduction_db=0.0)
    win, hop = cfg.win, cfg.hop
    # Length deliberately NOT a multiple of the hop (the failure case).
    n = win * 7 + 137
    assert n % hop != 0, "test length must be non-hop-aligned"
    t = np.arange(n) / 8000.0
    x = (0.3 * np.sin(2 * np.pi * 330 * t)
         + 0.2 * np.sin(2 * np.pi * 550 * t)).astype(np.float32)

    y = SpectralEngine().denoise(x, 8000, cfg)
    _check("spectral output length preserved", len(y) == n, f"{len(y)} vs {n}")

    err = np.abs(x[:len(y)] - y[:len(x)])
    # Ignore the outermost few samples: the Hann window is ~0 there, so OLA
    # normalisation is ill-conditioned at both signal ends by design — this is a
    # tail-*coverage* test, not an edge-taper test.
    edge = hop
    whole_err = float(err[edge:-edge].mean())
    tail_err = float(err[-win:-edge].mean())
    # Tail must be reconstructed at least as well as the body (with slack).
    _check(
        "spectral tail error <= whole-signal error (non-hop-aligned length)",
        tail_err <= whole_err * 2.0 + 1e-6,
        f"tail={tail_err:.2e} whole={whole_err:.2e}",
    )
    # And the body reconstruction is genuinely accurate, not merely equal-and-bad.
    _check(
        "spectral round-trip is accurate",
        whole_err < 1e-3,
        f"whole={whole_err:.2e}",
    )


# ── 3. Fabric self-test smoke ───────────────────────────────────────────────

def test_fabric_self_test_all_mediums_and_gpu() -> None:
    """All three mediums render and the Digital-GPU GEMM counter increments."""
    res = self_test()
    _check("self_test image renders", len(res.get("image_shape", [])) == 3,
           str(res.get("image_shape")))
    _check("self_test video renders", len(res.get("video_shape", [])) == 3,
           str(res.get("video_shape")))
    _check("self_test audio renders", int(res.get("audio_len", 0)) > 0,
           str(res.get("audio_len")))
    _check("self_test Digital-GPU GEMM counter incremented",
           int(res.get("digital_gpu_gemms", 0)) > 0,
           str(res.get("digital_gpu_gemms")))


def main() -> int:
    print("RTA-1 render-contract tests")
    for fn in (
        test_image_bytes_identical_across_processes,
        test_spectral_tail_reconstruction_non_hop_aligned,
        test_fabric_self_test_all_mediums_and_gpu,
    ):
        print(f"\n{fn.__name__}:")
        fn()
    print()
    if _FAILS:
        print(f"FAILED ({len(_FAILS)}): {', '.join(_FAILS)}")
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
