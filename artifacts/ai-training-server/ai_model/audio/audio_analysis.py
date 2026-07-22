"""Conductor: deterministic, total musical-timeline analysis.

Turns a raw audio waveform into a structured :class:`MusicalTimeline` (tempo,
beat grid, downbeats, onsets, energy envelope, frequency-band envelopes,
structural sections, and musical key) that drives audio-conducted generation.

Engineered as a *total* function: :func:`analyze_audio` never raises and always
returns a valid timeline for any input -- including empty, silence, NaN, noise,
mono, or stereo. Every feature stage degrades to a sane default instead of
failing, so callers never need a separate fallback path.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Digital GPU backend singleton ─────────────────────────────────────────────
_GPU_BACKEND = None
_GPU_BACKEND_LOCK = threading.Lock()

def _get_gpu():
    global _GPU_BACKEND
    if _GPU_BACKEND is None:
        with _GPU_BACKEND_LOCK:
            if _GPU_BACKEND is None:
                try:
                    from ai_model.gpu.torch_backend import DigitalGPUBackend
                    _GPU_BACKEND = DigitalGPUBackend()
                except Exception:
                    pass
    return _GPU_BACKEND

_ANALYSIS_SR = 22050
_MAX_ANALYSIS_SEC = 300.0
_DEFAULT_BPM = 120.0
_MIN_BPM = 40.0
_MAX_BPM = 220.0

_KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_BANDS: dict[str, tuple[float, float]] = {
    "sub": (0.0, 60.0),
    "low": (60.0, 250.0),
    "mid": (250.0, 2000.0),
    "high": (2000.0, float(_ANALYSIS_SR)),
}

_CACHE_MAX = 64
_cache: "OrderedDict[str, MusicalTimeline]" = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(waveform: Any, sample_rate: Any, max_seconds: float) -> str | None:
    try:
        arr = np.ascontiguousarray(_to_mono(waveform), dtype=np.float32)
        digest = hashlib.blake2b(arr.tobytes(), digest_size=16)
        meta = np.asarray(
            [float(sample_rate or 0.0), float(max_seconds or 0.0)], dtype=np.float64
        )
        digest.update(meta.tobytes())
        return digest.hexdigest()
    except Exception:
        return None


@dataclass
class Section:
    start: float
    end: float
    label: str


@dataclass
class MusicalTimeline:
    """Structured musical description of a track. Always valid."""

    duration_sec: float
    sample_rate: int
    bpm: float
    beats: list[float]
    downbeats: list[float]
    onsets: list[float]
    energy_times: list[float]
    energy_envelope: list[float]
    band_envelopes: dict[str, list[float]]
    sections: list[Section]
    key: str
    mode: str
    key_confidence: float
    analysis_ok: bool
    notes: str = ""

    def beat_positions_normalized(self) -> list[float]:
        """Beat times as fractions of total duration in ``[0, 1]``."""
        if self.duration_sec <= 0 or not self.beats:
            return []
        return [min(1.0, max(0.0, b / self.duration_sec)) for b in self.beats]

    def section_label_at(self, t: float) -> str:
        for s in self.sections:
            if s.start <= t < s.end:
                return s.label
        return self.sections[-1].label if self.sections else "full"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _section_labels(n: int) -> list[str]:
    if n <= 0:
        return []
    if n == 1:
        return ["full"]
    body = ["verse", "chorus", "verse", "chorus", "bridge", "chorus"]
    labels: list[str] = []
    for i in range(n):
        if i == 0:
            labels.append("intro")
        elif i == n - 1:
            labels.append("outro")
        else:
            labels.append(body[(i - 1) % len(body)])
    return labels


def _even_sections(duration: float, k: int) -> list[Section]:
    k = max(1, int(k))
    if duration <= 0:
        return []
    step = duration / k
    labels = _section_labels(k)
    return [
        Section(round(i * step, 3), round(min(duration, (i + 1) * step), 3), labels[i])
        for i in range(k)
    ]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - np.mean(a)
    b = b - np.mean(b)
    gpu = _get_gpu()
    if gpu is not None:
        # Use the underlying DigitalGPU numpy kernel directly (not the torch wrapper)
        engine = gpu.gpu
        a32 = np.ascontiguousarray(a, dtype=np.float32)
        b32 = np.ascontiguousarray(b, dtype=np.float32)
        da = float(np.sqrt(abs(engine.gemm(a32.reshape(1, -1), a32.reshape(-1, 1)).ravel()[0])))
        db = float(np.sqrt(abs(engine.gemm(b32.reshape(1, -1), b32.reshape(-1, 1)).ravel()[0])))
        if da == 0.0 or db == 0.0:
            return 0.0
        return float(engine.gemm(a32.reshape(1, -1), b32.reshape(-1, 1)).ravel()[0]) / (da * db)
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return float(np.dot(a, b) / (da * db))


def _to_mono(waveform: Any) -> np.ndarray:
    y = np.asarray(waveform, dtype=np.float32)
    if y.ndim > 1:
        axis = 0 if y.shape[0] < y.shape[1] else 1
        y = y.mean(axis=axis)
    return np.ascontiguousarray(y, dtype=np.float32)


def _sanitize(y: np.ndarray) -> np.ndarray:
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)


def _safe_duration(waveform: Any, sample_rate: Any) -> float:
    try:
        n = int(np.asarray(waveform).shape[0])
        sr = int(sample_rate) if sample_rate and int(sample_rate) > 0 else _ANALYSIS_SR
        return float(n) / float(sr)
    except Exception:
        return 0.0


def _default_timeline(duration: float, sr: int, notes: str) -> MusicalTimeline:
    duration = max(0.0, float(duration))
    bpm = _DEFAULT_BPM
    beat_period = 60.0 / bpm
    n_beats = int(duration // beat_period) if duration > 0 else 0
    beats = [round(i * beat_period, 4) for i in range(n_beats)]
    sections = (
        _even_sections(duration, max(1, min(6, int(duration // 20) + 1)))
        if duration > 0
        else []
    )
    return MusicalTimeline(
        duration_sec=round(duration, 3),
        sample_rate=int(sr),
        bpm=bpm,
        beats=beats,
        downbeats=beats[::4],
        onsets=[],
        energy_times=[],
        energy_envelope=[],
        band_envelopes={b: [] for b in _BANDS},
        sections=sections,
        key="C",
        mode="major",
        key_confidence=0.0,
        analysis_ok=False,
        notes=notes,
    )


def _tempo_beats(y: np.ndarray, sr: int, librosa: Any) -> tuple[float, list[float]]:
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])
        if not np.isfinite(bpm) or bpm <= 0:
            bpm = _DEFAULT_BPM
        while bpm < _MIN_BPM:
            bpm *= 2
        while bpm > _MAX_BPM:
            bpm /= 2
        times = librosa.frames_to_time(beat_frames, sr=sr)
        beats = [round(float(t), 4) for t in times if np.isfinite(t)]
        return bpm, beats
    except Exception:
        return _DEFAULT_BPM, []


def _onsets(y: np.ndarray, sr: int, librosa: Any) -> list[float]:
    try:
        ot = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
        return [round(float(t), 4) for t in ot if np.isfinite(t)]
    except Exception:
        return []


def _energy(y: np.ndarray, sr: int, librosa: Any) -> tuple[list[float], list[float]]:
    try:
        rms = np.nan_to_num(librosa.feature.rms(y=y)[0])
        times = librosa.times_like(rms, sr=sr)
        mx = float(np.max(rms)) if rms.size else 0.0
        norm = (rms / mx) if mx > 0 else rms
        return (
            [round(float(t), 3) for t in times],
            [round(float(v), 4) for v in norm],
        )
    except Exception:
        return [], []


def _band_envelopes(y: np.ndarray, sr: int, librosa: Any) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {b: [] for b in _BANDS}
    try:
        spec = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        for name, (lo, hi) in _BANDS.items():
            mask = (freqs >= lo) & (freqs < hi)
            if not mask.any():
                continue
            band = spec[mask, :].mean(axis=0)
            mx = float(np.max(band)) if band.size else 0.0
            norm = (band / mx) if mx > 0 else band
            out[name] = [round(float(v), 4) for v in norm]
        return out
    except Exception:
        return {b: [] for b in _BANDS}


def _sections(y: np.ndarray, sr: int, duration: float, librosa: Any) -> list[Section]:
    try:
        if duration < 4.0:
            return _even_sections(duration, 1)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        feat = np.vstack(
            [librosa.util.normalize(chroma, axis=0), librosa.util.normalize(mfcc, axis=0)]
        )
        k = int(np.clip(round(duration / 18.0), 2, 8))
        bounds = librosa.segment.agglomerative(feat, k)
        bound_times = librosa.frames_to_time(bounds, sr=sr)
        marks = sorted(
            {0.0, duration, *[float(t) for t in bound_times if 0.0 < t < duration]}
        )
        labels = _section_labels(len(marks) - 1)
        secs = [
            Section(round(marks[i], 3), round(marks[i + 1], 3), labels[i])
            for i in range(len(marks) - 1)
            if marks[i + 1] - marks[i] > 0.5
        ]
        return secs if secs else _even_sections(duration, k)
    except Exception:
        return _even_sections(duration, max(1, int(duration // 20) + 1))


def _key(y: np.ndarray, sr: int, librosa: Any) -> tuple[str, str, float]:
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        prof = np.mean(chroma, axis=1)
        s = float(np.sum(prof))
        if s <= 0:
            return "C", "major", 0.0
        prof = prof / s
        best_score = -2.0
        best_key = "C"
        best_mode = "major"
        for i in range(12):
            maj = _corr(prof, np.roll(_KS_MAJOR, i))
            mnr = _corr(prof, np.roll(_KS_MINOR, i))
            if maj > best_score:
                best_score, best_key, best_mode = maj, _PITCH_CLASSES[i], "major"
            if mnr > best_score:
                best_score, best_key, best_mode = mnr, _PITCH_CLASSES[i], "minor"
        conf = max(0.0, min(1.0, (best_score + 1.0) / 2.0))
        return best_key, best_mode, conf
    except Exception:
        return "C", "major", 0.0


def _analyze(
    waveform: Any, sample_rate: Any, max_seconds: float, librosa: Any
) -> MusicalTimeline:
    sr_in = int(sample_rate) if sample_rate and int(sample_rate) > 0 else _ANALYSIS_SR
    y = _sanitize(_to_mono(waveform))
    if y.size == 0:
        return _default_timeline(0.0, _ANALYSIS_SR, "empty_input")

    if sr_in != _ANALYSIS_SR:
        try:
            y = librosa.resample(y, orig_sr=sr_in, target_sr=_ANALYSIS_SR)
        except Exception:
            pass
    sr = _ANALYSIS_SR

    full_duration = len(y) / sr
    if max_seconds and full_duration > max_seconds:
        y = y[: int(max_seconds * sr)]
    duration = len(y) / sr

    if float(np.max(np.abs(y))) < 1e-5:
        return _default_timeline(full_duration, sr, "silence")

    bpm, beats = _tempo_beats(y, sr, librosa)
    onsets = _onsets(y, sr, librosa)
    energy_times, energy_env = _energy(y, sr, librosa)
    bands = _band_envelopes(y, sr, librosa)
    sections = _sections(y, sr, duration, librosa)
    key, mode, kconf = _key(y, sr, librosa)

    return MusicalTimeline(
        duration_sec=round(full_duration, 3),
        sample_rate=sr,
        bpm=round(bpm, 2),
        beats=beats,
        downbeats=beats[::4] if beats else [],
        onsets=onsets,
        energy_times=energy_times,
        energy_envelope=energy_env,
        band_envelopes=bands,
        sections=sections,
        key=key,
        mode=mode,
        key_confidence=round(kconf, 3),
        analysis_ok=True,
    )


def _compute_timeline(
    waveform: Any, sample_rate: int, max_seconds: float
) -> MusicalTimeline:
    try:
        import librosa
    except Exception as exc:  # pragma: no cover - dependency guard
        dur = _safe_duration(waveform, sample_rate)
        sr = int(sample_rate) if sample_rate and int(sample_rate) > 0 else _ANALYSIS_SR
        return _default_timeline(dur, sr, f"librosa_unavailable:{exc}")

    try:
        return _analyze(waveform, sample_rate, max_seconds, librosa)
    except Exception as exc:
        logger.warning("[Conductor] analysis failed; returning default timeline: %s", exc)
        dur = _safe_duration(waveform, sample_rate)
        return _default_timeline(dur, _ANALYSIS_SR, f"analysis_error:{exc}")


def analyze_audio(
    waveform: Any,
    sample_rate: int,
    *,
    max_seconds: float = _MAX_ANALYSIS_SEC,
    use_cache: bool = True,
) -> MusicalTimeline:
    """Analyze a waveform into a :class:`MusicalTimeline`. Never raises.

    Results are memoized in a bounded in-process LRU keyed by a content hash of
    the waveform, so re-analyzing the same track (the deterministic common case)
    is effectively free and avoids repeated heavy beat tracking.

    Args:
        waveform: 1-D mono or 2-D multi-channel float array of audio samples.
        sample_rate: Sample rate of ``waveform`` in Hz.
        max_seconds: Cap the span analyzed (the reported duration is still full).
        use_cache: Set ``False`` to bypass the content-hash cache.

    Returns:
        A valid :class:`MusicalTimeline` for any input. On any internal failure
        it returns a deterministic default timeline with ``analysis_ok=False``.
    """
    key = _cache_key(waveform, sample_rate, max_seconds) if use_cache else None
    if key is not None:
        try:
            with _cache_lock:
                cached = _cache.get(key)
                if cached is not None:
                    _cache.move_to_end(key)
                    return cached
        except Exception:
            key = None

    timeline = _compute_timeline(waveform, sample_rate, max_seconds)

    if key is not None:
        try:
            with _cache_lock:
                _cache[key] = timeline
                _cache.move_to_end(key)
                while len(_cache) > _CACHE_MAX:
                    _cache.popitem(last=False)
        except Exception:
            pass

    return timeline
