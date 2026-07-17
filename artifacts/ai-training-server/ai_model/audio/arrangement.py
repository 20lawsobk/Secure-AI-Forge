"""
Awareness-driven song arrangement for the audio render pipeline.

Turns a single retuned dataset loop into a STRUCTURED track (intro → verse →
hook → bridge → outro …) instead of one loop sustained for the whole duration.

Knowledge sourcing follows the project-wide borrowed-knowledge contract:

* The section grammars below are world-studied genre conventions (the same
  class of borrowed knowledge as ``content_playbook``).  Every injection
  point gates on ``quality_awareness.self_sufficiency()["retired"]`` — when
  the own corpus becomes self-sufficient the genre grammars retire and only
  the minimal generic arc (the ungated safety net) remains until
  dataset-derived structure learning replaces it.
* Live awareness leads: the caller passes the trending genres/mood already
  extracted from the ContentAwarenessService context (the same live
  music-industry feed that conditions track selection), and those refine
  which grammar is chosen for the request.

Realization is a single never-raise ffmpeg pass: HPSS stems (drums / bass /
melody) are looped for the full duration and mixed with per-section volume
automation (``volume=<piecewise between(t,..)>:eval=frame``), so section
changes are sample-accurate with no concat seams.  Any failure returns
``False`` and the caller keeps the plain-loop path — an arrangement problem
must never fail a render.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["Section", "build_plan", "realize", "plan_summary"]


# ─── Section model ────────────────────────────────────────────────────────────

@dataclass
class Section:
    kind: str          # intro / verse / build / hook / bridge / outro …
    start: float       # seconds
    length: float      # seconds
    energy: float      # 0–1, drives the stem mix
    # Per-stem gains (drums, bass, melody) 0–1
    gains: Tuple[float, float, float] = (1.0, 1.0, 1.0)


# Stem-mix per section kind: (drums, bass, melody).  These express the
# universal arrangement device "mute/unmute layers per section" — evaluation
# mechanics, not borrowed content (they do not retire; the GRAMMARS do).
_SECTION_MIX: Dict[str, Tuple[float, float, float]] = {
    "intro":  (0.00, 0.45, 0.90),
    "verse":  (0.85, 0.90, 0.75),
    "build":  (0.55, 0.75, 1.00),
    "hook":   (1.00, 1.00, 1.00),
    "drop":   (1.00, 1.00, 1.00),
    "chorus": (1.00, 1.00, 1.00),
    "bridge": (0.00, 0.80, 1.00),
    "break":  (0.00, 0.70, 1.00),
    "outro":  (0.50, 0.60, 0.80),
}

_SECTION_ENERGY: Dict[str, float] = {
    "intro": 0.3, "verse": 0.6, "build": 0.7, "hook": 1.0, "drop": 1.0,
    "chorus": 1.0, "bridge": 0.4, "break": 0.4, "outro": 0.35,
}


# ─── Borrowed genre grammars (world-studied conventions — RETIRE) ────────────
# Each grammar is a list of (section_kind, bars) drawn from studied chart
# arrangements per genre family. Chosen grammar is scaled to the requested
# duration on bar boundaries.

_GENRE_GRAMMARS: Dict[str, List[Tuple[str, int]]] = {
    # trap / hip-hop / drill: sparse intro, verse/hook alternation
    "hiphop": [("intro", 2), ("verse", 8), ("hook", 8), ("verse", 8),
               ("hook", 8), ("outro", 2)],
    # EDM / electronic: build → drop cycles with a break
    "electronic": [("intro", 4), ("build", 8), ("drop", 8), ("break", 4),
                   ("build", 8), ("drop", 8), ("outro", 4)],
    # pop / afrobeats / dancehall: verse-chorus with a bridge
    "pop": [("intro", 2), ("verse", 8), ("chorus", 8), ("verse", 8),
            ("chorus", 8), ("bridge", 4), ("chorus", 8), ("outro", 2)],
    # ambient / lofi / chill: long evolving arcs
    "ambient": [("intro", 4), ("verse", 12), ("bridge", 6), ("verse", 12),
                ("outro", 6)],
}

_GENRE_FAMILY: Dict[str, str] = {
    "trap": "hiphop", "drill": "hiphop", "hiphop": "hiphop",
    "hip-hop": "hiphop", "rap": "hiphop", "grime": "hiphop",
    "boom bap": "hiphop",
    "edm": "electronic", "house": "electronic", "techno": "electronic",
    "dubstep": "electronic", "electronic": "electronic",
    "drum and bass": "electronic", "dnb": "electronic",
    "synthwave": "electronic", "amapiano": "electronic",
    "pop": "pop", "afrobeats": "pop", "afrobeat": "pop", "dancehall": "pop",
    "r&b": "pop", "rnb": "pop", "reggaeton": "pop", "gospel": "pop",
    "country": "pop", "rock": "pop", "indie": "pop",
    "lofi": "ambient", "lo-fi": "ambient", "ambient": "ambient",
    "chill": "ambient", "chillout": "ambient", "classical": "ambient",
    "jazz": "ambient", "soul": "ambient",
}

# Ungated safety net — the minimal generic arc that never retires (matches
# the buffer contract: keep a minimal generic fallback so output is never
# structureless even after retirement, until dataset-derived structure
# learning exists).
_GENERIC_ARC: List[Tuple[str, int]] = [
    ("intro", 2), ("verse", 8), ("hook", 8), ("verse", 8), ("outro", 2),
]


def _grammar_for(genre: Optional[str], mood: Optional[str],
                 trending_genres: Optional[Sequence[str]]) -> List[Tuple[str, int]]:
    """Pick a grammar.  Borrowed genre grammars gate on the awareness
    retirement contract; live awareness genres refine the choice when the
    request itself has no genre (caller awareness leads)."""
    retired = True
    try:
        from ai_model.quality_awareness import self_sufficiency
        retired = bool(self_sufficiency().get("retired"))
    except Exception:
        retired = True  # awareness unavailable → only the safety net
    if retired:
        return list(_GENERIC_ARC)

    candidates: List[str] = []
    if genre:
        candidates.append(str(genre).lower().strip())
    for g in (trending_genres or []):
        candidates.append(str(g).lower().strip())
    if mood and str(mood).lower() in ("chill", "calm", "ambient", "relaxed"):
        candidates.append("ambient")
    for c in candidates:
        fam = _GENRE_FAMILY.get(c)
        if fam:
            return list(_GENRE_GRAMMARS[fam])
    return list(_GENERIC_ARC)


# ─── Plan building ────────────────────────────────────────────────────────────

def build_plan(duration_sec: float, bpm: float, *,
               genre: Optional[str] = None, mood: Optional[str] = None,
               trending_genres: Optional[Sequence[str]] = None,
               seed: Optional[int] = None) -> List[Section]:
    """Build a bar-aligned section plan filling ``duration_sec``.

    Deterministic for a given (seed, request): the seed only shuffles which
    mid sections absorb rounding slack.  Never raises — any internal problem
    degrades to a single full-mix section (== the old plain-loop behaviour).
    """
    try:
        duration_sec = float(duration_sec)
        bpm = float(bpm) if bpm and bpm > 0 else 120.0
        bar = 4.0 * 60.0 / max(40.0, min(300.0, bpm))  # 4/4 assumption
        grammar = _grammar_for(genre, mood, trending_genres)

        total_bars = max(2, int(duration_sec // bar))
        gram_bars = sum(b for _, b in grammar)
        # Scale grammar to available bars (floor 1 bar per section), then
        # give leftover bars to the highest-energy sections.
        scale = total_bars / float(gram_bars)
        alloc = [max(1, int(b * scale)) for _, b in grammar]
        rng = random.Random(seed if seed is not None else 0)
        while sum(alloc) > total_bars and max(alloc) > 1:
            i = max(range(len(alloc)), key=lambda j: alloc[j])
            alloc[i] -= 1
        hot = [i for i, (k, _) in enumerate(grammar)
               if _SECTION_ENERGY.get(k, 0.5) >= 0.9] or list(range(len(alloc)))
        while sum(alloc) < total_bars:
            alloc[rng.choice(hot)] += 1

        plan: List[Section] = []
        t = 0.0
        for (kind, _), bars in zip(grammar, alloc):
            length = bars * bar
            plan.append(Section(
                kind=kind, start=round(t, 3), length=round(length, 3),
                energy=_SECTION_ENERGY.get(kind, 0.6),
                gains=_SECTION_MIX.get(kind, (1.0, 1.0, 1.0)),
            ))
            t += length
        # Stretch the final section to cover any sub-bar remainder so the
        # arrangement always fills the full requested duration.
        if plan and t < duration_sec:
            plan[-1].length = round(plan[-1].length + (duration_sec - t), 3)
        return plan
    except Exception:
        return [Section(kind="full", start=0.0, length=float(duration_sec),
                        energy=1.0, gains=(1.0, 1.0, 1.0))]


def plan_summary(plan: List[Section]) -> List[Dict]:
    return [{"section": s.kind, "start": s.start, "length": s.length,
             "energy": s.energy} for s in plan]


# ─── Realization (never-raise) ───────────────────────────────────────────────

_RAMP = 0.35  # seconds of linear gain ramp at each section boundary (declick)
_DOMINANCE_THRESHOLD = 0.55  # energy share above which a stem never mutes


def _stem_energy_shares(stem_paths: List[Path]) -> List[float]:
    """RMS-energy share of each stem (sums to 1). Never raises — equal
    shares on any failure (guard becomes a no-op tilt toward safety)."""
    import wave as _wave
    import numpy as _np
    rms = []
    try:
        for p in stem_paths:
            with _wave.open(str(p), "rb") as wf:
                n = wf.getnframes()
                # Sample up to ~30 s from the middle — enough for a share
                # estimate without decoding a full 3-minute stem.
                take = min(n, wf.getframerate() * wf.getnchannels() * 30)
                wf.setpos(max(0, (n - take) // 2))
                raw = wf.readframes(take)
            x = _np.frombuffer(raw, dtype="<i2").astype(_np.float32)
            rms.append(float(_np.sqrt(_np.mean(x * x))) if x.size else 0.0)
        total = sum(rms)
        if total <= 0:
            return [1.0 / len(stem_paths)] * len(stem_paths)
        return [v / total for v in rms]
    except Exception:
        return [1.0 / len(stem_paths)] * len(stem_paths)


def _apply_dominance_floor(plan: List[Section],
                           shares: List[float]) -> List[Section]:
    """Return a plan copy where any stem holding > _DOMINANCE_THRESHOLD of
    the total energy keeps a proportional minimum gain in every section, so
    no section collapses to near-silence when HPSS separation is lopsided."""
    floors = [min(0.7, max(0.0, (s - _DOMINANCE_THRESHOLD) * 2.0))
              for s in shares]
    if all(f <= 0.0 for f in floors):
        return plan
    out = []
    for s in plan:
        g = tuple(max(s.gains[i], floors[i]) for i in range(3))
        out.append(Section(kind=s.kind, start=s.start, length=s.length,
                           energy=s.energy, gains=g))
    return out


def _vol_expr(plan: List[Section], stem_idx: int) -> str:
    """Piecewise volume expression over t for one stem, with short linear
    ramps between sections so gain changes never click."""
    terms = []
    for i, s in enumerate(plan):
        g = s.gains[stem_idx]
        g_prev = plan[i - 1].gains[stem_idx] if i > 0 else 0.0
        st, en = s.start, s.start + s.length
        ramp = min(_RAMP, s.length / 2.0)
        if abs(g - g_prev) > 1e-3 and ramp > 0.01:
            # ramp from previous gain to this gain over [st, st+ramp]
            terms.append(
                f"between(t,{st:.3f},{st + ramp:.3f})*"
                f"({g_prev:.3f}+({g - g_prev:.3f})*(t-{st:.3f})/{ramp:.3f})"
            )
            terms.append(f"between(t,{st + ramp:.3f},{en:.3f})*{g:.3f}")
        else:
            terms.append(f"between(t,{st:.3f},{en:.3f})*{g:.3f}")
    return "+".join(terms) if terms else "1.0"


def realize(stage_in: Path, out_wav: Path, plan: List[Section], *,
            sample_rate: int = 44100, run_ffmpeg=None,
            uploads_dir: Optional[Path] = None,
            job_tag: str = "arr") -> bool:
    """Render the arrangement: stems → looped → per-section volume automation
    → mix.  Returns True when ``out_wav`` was produced; False on ANY failure
    (caller falls back to the plain loop).  Never raises."""
    tmp_stems: Dict[str, Path] = {}
    try:
        if not plan or len(plan) < 2:
            return False  # nothing structural to do — plain loop is identical
        if run_ffmpeg is None:
            from ai_model.video.ffmpeg_util import run_ffmpeg as _rf
            run_ffmpeg = _rf
        from ai_model.audio import producer_tools as _pt

        duration = plan[-1].start + plan[-1].length
        out_dir = uploads_dir or Path(stage_in).parent
        tmp_stems = _pt.separate_stems(Path(stage_in), out_dir,
                                       f"arrstem_{job_tag}", fmt="wav",
                                       bit_depth=16)
        drums = tmp_stems.get("drums")
        bass = tmp_stems.get("bass")
        melody = tmp_stems.get("melody")
        if not (drums and bass and melody):
            return False

        # Dominance guard: HPSS is imperfect — on some material one stem
        # carries nearly ALL the energy (e.g. everything lands in "drums").
        # Fully muting that stem would leave whole sections near-silent, so
        # floor the dominant stem's gain in proportion to its energy share.
        shares = _stem_energy_shares([drums, bass, melody])
        plan = _apply_dominance_floor(plan, shares)

        # One pass: loop each stem for the full duration, apply its section
        # volume automation, mix, and hard-trim.
        fc = (
            f"[0:a]volume='{_vol_expr(plan, 0)}':eval=frame[d];"
            f"[1:a]volume='{_vol_expr(plan, 1)}':eval=frame[b];"
            f"[2:a]volume='{_vol_expr(plan, 2)}':eval=frame[m];"
            f"[d][b][m]amix=inputs=3:normalize=0,"
            f"atrim=0:{duration:.3f}[out]"
        )
        # Timeout scales with duration (long-form contract).
        to = max(90, int(duration * 1.5) + 30)
        r = run_ffmpeg(
            ["ffmpeg", "-y",
             "-stream_loop", "-1", "-i", str(drums),
             "-stream_loop", "-1", "-i", str(bass),
             "-stream_loop", "-1", "-i", str(melody),
             "-filter_complex", fc, "-map", "[out]",
             "-t", f"{duration:.3f}", "-ac", "2", "-ar", str(int(sample_rate)),
             str(out_wav)],
            timeout=to,
        )
        return r.returncode == 0 and Path(out_wav).exists()
    except Exception as err:  # never-raise contract
        print(f"[Arrange] realization failed — plain loop fallback: {err}",
              flush=True)
        return False
    finally:
        for p in tmp_stems.values():
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass
