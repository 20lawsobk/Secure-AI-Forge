"""Media-technique ("Visual/Sonic DNA") extraction from real reference assets.

The awareness/quality buffer tells the strategy layer *what* peak content looks
like in words. This module closes the other half of the loop: it reads the
**actual reference media we hold** (real frames from the RCGS asset index, and —
optionally — real audio samples) and distils their *technique* into a normalized
descriptor set (energy / darkness / warmth / saturation / contrast, plus
tempo / key / spectral tilt). Those descriptors then condition every renderer.

Design invariants (mirroring the rest of MaxCore):

  * **Total / never-raise.** Any failure — no index, no readable asset, PIL
    missing — degrades cleanly to the genre/tone *prior*; extraction can only
    add grounding, never break a request.
  * **Blend, don't replace.** The real-asset evidence is mixed with the
    genre/tone prior weighted by the quality buffer's ``buffer_weight``: while
    the own corpus is small the live real-asset signal dominates (the buffer's
    reason to exist); as the corpus graduates, the prior/own content takes over
    and the weight decays to 0 — the same retirement contract the buffer uses.
  * **Cached.** Extraction does asset I/O, so results are memoised with a short
    TTL keyed by the request's stylistic fingerprint.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── tuning ──────────────────────────────────────────────────────────────────
_CACHE_TTL = 300.0          # seconds a technique profile stays fresh
_CACHE_CAP = 256
_SWATCH = 96                # working resolution for the query swatch
# Real evidence can dominate, but never so completely that the genre/tone prior
# stops shaping output — otherwise a non-diverse index (only bootstrap seeds)
# would flatten every genre onto the same look. The prior always keeps ≥ 1-cap.
_REAL_WEIGHT_CAP = 0.6

_cache: "OrderedDict[str, Tuple[float, 'TechniqueProfile']]" = OrderedDict()
_cache_lock = threading.Lock()


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return float(max(lo, min(hi, x)))
    except (TypeError, ValueError):
        return lo


@dataclass
class TechniqueProfile:
    """Normalized, renderer-agnostic conditioning descriptors.

    Every visual field is in ``[0, 1]``. ``source`` records provenance so the
    response metadata can show whether real reference media grounded the
    conditioning (``blended_real``) or only the genre/tone prior did (``prior``).
    """

    # ── Visual DNA ──────────────────────────────────────────────────────────
    energy: float = 0.6
    darkness: float = 0.55
    warmth: float = 0.5
    saturation: float = 0.65
    contrast: float = 0.5
    grain: float = 0.1
    # ── Sonic DNA (optional) ────────────────────────────────────────────────
    tempo: Optional[float] = None       # BPM
    key: Optional[str] = None
    brightness_tilt: float = 0.5        # spectral centroid proxy, 0 dark .. 1 bright
    # ── Provenance ──────────────────────────────────────────────────────────
    source: str = "prior"               # "prior" | "blended_real"
    asset_ref: Optional[str] = None
    real_weight: float = 0.0
    notes: List[str] = field(default_factory=list)

    # -- renderer mappings ----------------------------------------------------
    def dna_dict(self) -> Dict[str, float]:
        """The 4-float Visual DNA the diffusion pipeline conditions on."""
        return {
            "energy": round(self.energy, 4),
            "darkness": round(self.darkness, 4),
            "warmth": round(self.warmth, 4),
            "saturation": round(self.saturation, 4),
        }

    def color_scheme(self) -> str:
        """Map descriptors onto one of the ImageEngine COLOR_SCHEMES keys."""
        if self.saturation < 0.22:
            return "monochrome"
        if self.contrast > 0.7 and self.saturation > 0.5:
            return "high_contrast"
        if self.warmth > 0.62 and self.saturation > 0.55:
            return "bold_red_gold" if self.energy > 0.6 else "warm_earth"
        if self.warmth < 0.42:
            return "corporate_blue"
        if self.darkness > 0.6 and self.saturation > 0.5:
            return "dark_neon"
        if self.darkness < 0.4 and self.saturation > 0.5:
            return "vibrant_pastel"
        return "dark_neon"

    def mood(self) -> str:
        """A short RTA mood token derived from energy/darkness."""
        if self.energy > 0.72:
            return "vibrant"
        if self.darkness > 0.7:
            return "moody"
        if self.energy < 0.32:
            return "calm"
        return "cinematic"

    def color_grade(self) -> str:
        """VRC / ffmpeg colour-grade tag consistent with the descriptors."""
        if self.grain > 0.4:
            return "vintage"
        if self.darkness > 0.72 and self.saturation > 0.6:
            return "neon"
        if self.warmth > 0.68:
            return "warm"
        if self.warmth < 0.4:
            return "cool"
        return "cinematic"

    def audio_conditioning(self) -> Dict[str, Any]:
        """Tempo / key / spectral tilt for the audio producer."""
        return {
            "tempo": self.tempo,
            "key": self.key,
            "brightness_tilt": round(self.brightness_tilt, 4),
            "energy": round(self.energy, 4),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "energy": round(self.energy, 4),
            "darkness": round(self.darkness, 4),
            "warmth": round(self.warmth, 4),
            "saturation": round(self.saturation, 4),
            "contrast": round(self.contrast, 4),
            "grain": round(self.grain, 4),
            "tempo": self.tempo,
            "key": self.key,
            "brightness_tilt": round(self.brightness_tilt, 4),
            "color_scheme": self.color_scheme(),
            "mood": self.mood(),
            "color_grade": self.color_grade(),
            "source": self.source,
            "asset_ref": self.asset_ref,
            "real_weight": round(self.real_weight, 4),
            "notes": list(self.notes),
        }


# ── prior (genre/tone) ───────────────────────────────────────────────────────

def _prior_dna(idea: str, genre: str, tone: str, energy_hint: Optional[float]) -> Dict[str, float]:
    """Genre/tone Visual DNA from the existing scene-builder tables."""
    base = {"energy": 0.62, "darkness": 0.55, "warmth": 0.5, "saturation": 0.65, "grain": 0.1}
    try:
        from ai_model.video.ai_scene_builder import build_dna
        dna = build_dna(idea or "music content", genre or "", tone or "")
        base = {
            "energy": float(dna.energy),
            "darkness": float(dna.darkness),
            "warmth": float(dna.warmth),
            "saturation": float(dna.saturation),
            "grain": float(dna.grain),
        }
    except Exception:
        pass
    # Producer-derived energy (mood/bpm) nudges the genre energy.
    if energy_hint is not None:
        try:
            base["energy"] = _clamp(0.6 * base["energy"] + 0.4 * float(energy_hint))
        except (TypeError, ValueError):
            pass
    return base


def _query_swatch(prior: Dict[str, float], seed: int) -> Optional[np.ndarray]:
    """Render a tiny procedural swatch from the prior palette to embed & query.

    The swatch encodes the prior's palette so the nearest real asset retrieved
    is stylistically close — grounding technique in a *relevant* reference,
    not a random one.
    """
    try:
        from ai_model.video.ai_scene_builder import derive_palette
        from ai_model.retrieval.image_features import _to_rgb  # noqa: F401 (availability)

        class _D:  # minimal VisualDNA-shaped object for derive_palette
            pass

        # Rebuild a VisualDNA via the real class so derive_palette stays honest.
        from ai_model.video.ai_scene_builder import VisualDNA
        vdna = VisualDNA(
            energy=prior["energy"], darkness=prior["darkness"], warmth=prior["warmth"],
            saturation=prior["saturation"], grain=prior["grain"],
            complexity=_clamp(prior["energy"] * 0.7 + prior["saturation"] * 0.3),
            seed=int(seed),
        )
        pal = derive_palette(vdna)

        def _hex(h: str) -> Tuple[int, int, int]:
            c = h.lstrip("0x").lstrip("#").zfill(6)
            return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)

        c1 = np.array(_hex(pal.bg1), dtype=np.float32)
        c2 = np.array(_hex(pal.bg2), dtype=np.float32)
        t = np.linspace(0.0, 1.0, _SWATCH, dtype=np.float32).reshape(-1, 1, 1)
        grad = (c1.reshape(1, 1, 3) * (1.0 - t) + c2.reshape(1, 1, 3) * t)
        arr = np.broadcast_to(grad, (_SWATCH, _SWATCH, 3)).astype(np.uint8).copy()
        # A small accent block so saturation/contrast register in the embedding.
        ar, ag, ab = _hex(pal.accent)
        arr[_SWATCH // 3: 2 * _SWATCH // 3, _SWATCH // 3: 2 * _SWATCH // 3] = (ar, ag, ab)
        return arr
    except Exception:
        return None


# How far to trust a retrieved asset by which cascade rung answered. An
# "anchor" is the always-loaded *generic* domain fallback — not genre-matched
# peak content — so it must never override the genre/tone prior; a real
# "nearest"/"exact" match earns most of the buffer weight.
_RUNG_RELEVANCE = {"exact": 1.0, "nearest": 0.85, "brand_prior": 0.7, "anchor": 0.3}


def _relevance(rung: str) -> float:
    return _RUNG_RELEVANCE.get(rung, 0.5)


def _real_descriptors(prior: Dict[str, float], brand: Optional[str], seed: int
                      ) -> Optional[Tuple[Dict[str, float], str, str]]:
    """Retrieve the closest real asset and describe its technique. Never raises.

    Returns ``(descriptors, asset_ref, rung)`` or ``None`` when no real asset is
    usable (empty index, unreadable file, PIL missing …). ``rung`` records which
    cascade level answered so the caller can scale how much to trust it.
    """
    try:
        swatch = _query_swatch(prior, seed)
        if swatch is None:
            return None
        from ai_model.retrieval.image_features import image_to_vector, describe_image
        from ai_model.retrieval.asset_pipeline import get_asset_index
        from ai_model.retrieval.rcgs import _resolve_local_path

        vec = image_to_vector(swatch)
        if vec is None:
            return None
        index = get_asset_index()
        asset = index.query(vec, brand=(brand or None))
        if asset is None:
            return None
        meta = asset.metadata or {}
        path = _resolve_local_path(meta)
        if path is None:
            return None
        desc = describe_image(path)
        if desc is None:
            return None
        ref = str(meta.get("filename") or meta.get("path") or path.name)
        return desc, ref, getattr(asset, "rung", "nearest")
    except Exception:
        return None


def _buffer_weight() -> float:
    """How strongly to trust live real-asset evidence over the prior.

    Reuses the quality buffer's self-sufficiency: high while the own corpus is
    small, decaying to 0 at retirement. On any failure, defaults to a moderate
    0.5 so real grounding still contributes.
    """
    try:
        from ai_model.quality_awareness import self_sufficiency
        w = float(self_sufficiency().get("buffer_weight", 0.5))
        return _clamp(w)
    except Exception:
        return 0.5


def _real_sonic(genre: str, tempo: Optional[float], key: Optional[str]) -> Dict[str, Any]:
    """Tempo/key grounded in the *real* seeded audio dataset we hold.

    Off by default (the dataset must be seeded into storage); enabled with
    ``TECHNIQUE_REAL_AUDIO=1``. Reads the per-sample ``bpm``/``key`` recorded in
    the audio dataset index (``mb:dataset:audio:meta``) and returns their
    central tendency — the tempo/key our real reference corpus actually sits at.
    Never raises — returns ``{}`` when unavailable, leaving the caller's
    genre/brief-derived sonic values in place.
    """
    import os
    if os.environ.get("TECHNIQUE_REAL_AUDIO", "0") != "1":
        return {}
    try:
        from storage_client import get_storage
        meta = get_storage().get("mb:dataset:audio:meta")
        if not meta or int(meta.get("num_chunks", 0)) <= 0:
            return {}
        index = meta.get("index") or []
        bpms = [float(e["bpm"]) for e in index if isinstance(e, dict) and e.get("bpm")]
        keys = [str(e["key"]) for e in index if isinstance(e, dict) and e.get("key")]
        out: Dict[str, Any] = {}
        if bpms:
            import statistics
            out["tempo"] = round(float(statistics.median(bpms)), 1)
        if keys:
            from collections import Counter
            out["key"] = Counter(keys).most_common(1)[0][0]
        return out
    except Exception:
        return {}


def extract_technique(
    *,
    idea: str = "",
    genre: Optional[str] = None,
    tone: Optional[str] = None,
    energy: Optional[float] = None,
    mood: Optional[str] = None,
    bpm: Optional[float] = None,
    key: Optional[str] = None,
    brand: Optional[str] = None,
    seed: int = 0,
    with_audio: bool = False,
    use_cache: bool = True,
) -> TechniqueProfile:
    """Produce a :class:`TechniqueProfile` for a request. Never raises.

    The profile blends a genre/tone prior with descriptors extracted from the
    closest real reference asset, weighted by the quality buffer. When no real
    asset is available the profile is the pure prior (``source="prior"``).
    """
    genre = (genre or "").strip()
    tone = (tone or "").strip()

    def _fnum(v: Any, fmt: str) -> str:
        try:
            return format(float(v), fmt)
        except (TypeError, ValueError):
            return "-"

    key_id = "|".join([
        genre.lower(), tone.lower(), (mood or "").lower(),
        _fnum(energy, ".2f") if energy is not None else "-",
        _fnum(bpm, ".0f") if bpm else "-", (key or ""), (brand or ""),
        str(seed % 997), "a" if with_audio else "",
    ])
    now = time.time()
    if use_cache:
        with _cache_lock:
            hit = _cache.get(key_id)
            if hit is not None and now - hit[0] < _CACHE_TTL:
                _cache.move_to_end(key_id)
                return hit[1]

    # Whole-body guard: this function is contractually never-raise. Any failure
    # (retrieval, buffer weight, malformed hints) degrades to the pure prior
    # profile rather than propagating into the request.
    profile = TechniqueProfile()
    try:
        prior = _prior_dna(idea, genre, tone, energy)
        contrast_prior = _clamp(0.35 + prior["energy"] * 0.4)

        profile = TechniqueProfile(
            energy=prior["energy"], darkness=prior["darkness"], warmth=prior["warmth"],
            saturation=prior["saturation"], contrast=contrast_prior, grain=prior["grain"],
        )

        real = _real_descriptors(prior, brand, seed)
        if real is not None:
            desc, ref, rung = real
            # Effective weight = how empty our own corpus is (buffer_weight) × how
            # relevant this retrieval actually is (rung). A generic anchor can only
            # nudge; a real genre-matched asset can dominate. This keeps EDM ≠
            # acoustic while the index is sparse (both would otherwise collapse
            # onto the single always-loaded anchor).
            w = _clamp(min(_REAL_WEIGHT_CAP, _buffer_weight() * _relevance(rung)))
            real_energy = _clamp(0.4 * desc["colorfulness"] + 0.3 * desc["contrast"]
                                 + 0.3 * desc["saturation"])
            profile.energy = _clamp((1 - w) * prior["energy"] + w * real_energy)
            profile.darkness = _clamp((1 - w) * prior["darkness"] + w * desc["darkness"])
            profile.warmth = _clamp((1 - w) * prior["warmth"] + w * desc["warmth"])
            profile.saturation = _clamp((1 - w) * prior["saturation"] + w * desc["saturation"])
            profile.contrast = _clamp((1 - w) * contrast_prior + w * desc["contrast"])
            profile.source = "blended_real" if w >= 0.05 else "prior"
            profile.asset_ref = ref
            profile.real_weight = round(w, 4)
            profile.notes.append(
                f"grounded on real asset via '{rung}' rung (weight={w:.2f})" if w >= 0.05
                else f"real asset matched ('{rung}') but weight retired — prior dominates"
            )
        else:
            profile.notes.append("no real reference asset available — genre/tone prior")

        # ── Sonic ──────────────────────────────────────────────────────────────
        try:
            profile.tempo = float(bpm) if bpm else None
        except (TypeError, ValueError):
            profile.tempo = None
        profile.key = key or None
        profile.brightness_tilt = _clamp(0.35 + profile.energy * 0.5)
        if with_audio:
            sonic = _real_sonic(genre, profile.tempo, profile.key)
            if sonic.get("tempo"):
                profile.tempo = sonic["tempo"]
                profile.notes.append("tempo from real audio sample")
            if sonic.get("key"):
                profile.key = sonic["key"]
    except Exception:
        pass

    if use_cache:
        try:
            with _cache_lock:
                _cache[key_id] = (now, profile)
                _cache.move_to_end(key_id)
                while len(_cache) > _CACHE_CAP:
                    _cache.popitem(last=False)
        except Exception:
            pass

    return profile
