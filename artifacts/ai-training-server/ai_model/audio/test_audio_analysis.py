"""Tests for the Conductor (audio_analysis). Run:

    uv run python -m ai_model.audio.test_audio_analysis

Verifies correctness on structured signals, determinism (same input -> same
output), and totality (never raises on any input).
"""
from __future__ import annotations

import sys

import numpy as np

from ai_model.audio.audio_analysis import MusicalTimeline, analyze_audio

SR = 22050


def _click_track(bpm: float, duration: float, sr: int = SR) -> np.ndarray:
    y: np.ndarray = np.zeros(int(duration * sr), dtype=np.float32)
    period = int(sr * 60.0 / bpm)
    click: np.ndarray = np.hanning(64).astype(np.float32)
    for start in range(0, len(y) - 64, period):
        y[start : start + 64] += click
    return y


def _bpm_matches(detected: float, target: float, tol: float = 0.08) -> bool:
    for mult in (1.0, 0.5, 2.0, 1.5, 1.0 / 1.5, 3.0, 1.0 / 3.0):
        if abs(detected - target * mult) <= tol * target * mult:
            return True
    return False


def _check(name: str, cond: bool) -> None:
    if not cond:
        raise AssertionError(f"FAILED: {name}")
    print(f"  ok: {name}")


def test_bpm_detection() -> None:
    for target in (90.0, 120.0, 140.0):
        tl = analyze_audio(_click_track(target, 7.0), SR)
        _check(
            f"bpm≈{target} (got {tl.bpm})",
            _bpm_matches(tl.bpm, target) and tl.analysis_ok,
        )
        _check(f"beats found for {target}bpm", len(tl.beats) > 5)


def test_determinism() -> None:
    sig = _click_track(128.0, 6.0)
    a = analyze_audio(sig, SR)
    b = analyze_audio(sig, SR)
    _check("bpm deterministic", a.bpm == b.bpm)
    _check("beats deterministic", a.beats == b.beats)
    _check("key deterministic", (a.key, a.mode) == (b.key, b.mode))
    _check("sections deterministic", a.to_dict()["sections"] == b.to_dict()["sections"])


def test_energy_and_sections() -> None:
    sr = SR
    quiet = (np.random.RandomState(0).randn(sr * 4) * 0.02).astype(np.float32)
    t = np.linspace(0, 4, sr * 4, endpoint=False)
    loud = (0.6 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sig = np.concatenate([quiet, loud])
    tl = analyze_audio(sig, sr)
    env = np.asarray(tl.energy_envelope)
    half = len(env) // 2
    _check("energy rises in 2nd half", float(env[half:].mean()) > float(env[:half].mean()))
    _check("multiple sections detected", len(tl.sections) >= 2)
    _check("sections cover timeline", abs(tl.sections[-1].end - tl.duration_sec) < 1.5)


def test_band_envelopes() -> None:
    t = np.linspace(0, 4, SR * 4, endpoint=False)
    sig = (0.5 * np.sin(2 * np.pi * 50 * t)).astype(np.float32)
    tl = analyze_audio(sig, SR)
    _check("band keys present", set(tl.band_envelopes) == {"sub", "low", "mid", "high"})
    _check("sub band has energy", len(tl.band_envelopes["sub"]) > 0)


def test_totality() -> None:
    cases = {
        "empty": np.array([], dtype=np.float32),
        "silence": np.zeros(SR * 3, dtype=np.float32),
        "nan": np.full(SR * 2, np.nan, dtype=np.float32),
        "inf": np.full(SR * 2, np.inf, dtype=np.float32),
        "tiny": np.array([0.1, -0.2, 0.05], dtype=np.float32),
        "noise": (np.random.RandomState(1).randn(SR * 4)).astype(np.float32),
        "stereo": (np.random.RandomState(2).randn(SR * 3, 2)).astype(np.float32),
        "huge_amp": (np.ones(SR * 2, dtype=np.float32) * 1e9),
    }
    for name, sig in cases.items():
        tl = analyze_audio(sig, SR)
        _check(f"totality[{name}] returns timeline", isinstance(tl, MusicalTimeline))
        _check(f"totality[{name}] finite bpm", np.isfinite(tl.bpm) and tl.bpm > 0)
        _check(f"totality[{name}] beat positions valid",
               all(0.0 <= p <= 1.0 for p in tl.beat_positions_normalized()))
    _check("bad samplerate handled", isinstance(analyze_audio(np.zeros(100), 0), MusicalTimeline))


def test_encoder_integration() -> None:
    from maxbooster_veo_music.model.audio_encoder import AudioEncoder

    class _FakeGPU:
        def gemm(self, a, b):  # type: ignore[no-untyped-def]
            return np.asarray(a) @ np.asarray(b)

    enc = AudioEncoder(_FakeGPU())  # type: ignore[arg-type]
    sig = _click_track(120.0, 4.0)
    out = enc.encode(sig, SR)
    for k in ("time_embeddings", "section_labels", "energy_curve",
              "beat_positions", "bpm", "key", "mode"):
        _check(f"encode returns '{k}'", k in out)
    _check("section_labels length == n_frames", out["section_labels"].shape[0] == enc.n_frames)
    _check("beat_positions in [0,1]", bool(np.all((out["beat_positions"] >= 0) & (out["beat_positions"] <= 1))))

    import ai_model.audio.audio_analysis as aa

    def _boom(*_a, **_k):
        raise RuntimeError("forced failure")

    original = aa.analyze_audio
    aa.analyze_audio = _boom  # type: ignore[assignment]
    try:
        out2 = enc.encode(sig, SR)
    finally:
        aa.analyze_audio = original
    _check("fallback returns full schema",
           all(k in out2 for k in ("section_labels", "beat_positions", "bpm", "key", "mode")))
    _check("fallback uses quarter sections", int(out2["section_labels"].max()) == 3)


def main() -> int:
    tests = [
        test_bpm_detection,
        test_determinism,
        test_energy_and_sections,
        test_band_envelopes,
        test_totality,
        test_encoder_integration,
    ]
    failed = 0
    for t in tests:
        print(f"\n[{t.__name__}]")
        try:
            t()
        except AssertionError as exc:
            print(f"  {exc}")
            failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}")
            failed += 1
    print("\n" + ("ALL PASSED" if failed == 0 else f"{failed} TEST GROUP(S) FAILED"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
