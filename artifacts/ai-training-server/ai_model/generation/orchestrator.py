"""Unified generation orchestrator — one brief, one awareness merge, one bus.

Every modality (content, text, image, audio, video) used to hand-assemble the
same three things independently: a :class:`GenerationBrief` via
``request_intelligence.build_brief``, a merged awareness string, and — for media
— ad-hoc conditioning. :func:`build_context` centralises all of it and returns a
:class:`GenerationContext` carrying the shared *conditioning bus*
(:class:`~ai_model.generation.technique.TechniqueProfile`), plus helpers that map
that bus onto each concrete renderer (diffusion, RTA, the PIL image engine, the
audio producer).

Nothing here changes an endpoint's request/response contract — it only removes
the duplication and gives every modality the *same* real-asset conditioning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .technique import TechniqueProfile, extract_technique


def _as_text(v: Any) -> str:
    return v.strip() if isinstance(v, str) else ""


def merge_awareness(req: Any) -> str:
    """Canonical awareness-bridge input for a request.

    The caller's creative direction (``instruction`` / ``extra_context`` /
    ``content_themes``) is serialised FIRST via
    ``request_intelligence.awareness_from_direction`` so it outranks generic
    trend context, then genuine external live-signal awareness is appended.
    ``brief.directives`` are deliberately excluded (they are internal
    prompt-engineering instructions the awareness parser would quote verbatim).
    Returns ``""`` when there is no real context. Never raises.
    """
    try:
        from ai_model import request_intelligence as ri
        direction = " ".join(s for s in (
            _as_text(getattr(req, "instruction", None)),
            _as_text(getattr(req, "extra_context", None)),
        ) if s)
        return "\n".join(p for p in (
            ri.awareness_from_direction(direction, getattr(req, "content_themes", None)),
            (getattr(req, "awareness", "") or "").strip(),
        ) if p)
    except Exception:
        return (getattr(req, "awareness", "") or "").strip()


@dataclass
class GenerationContext:
    """The unified per-request conditioning bus.

    Carries the brief, the merged awareness string, and the extracted technique
    profile. Helper methods translate the shared bus into renderer-specific
    parameters so each endpoint dispatches through one consistent surface.
    """

    modality: str
    platform: str
    brief: Any                       # request_intelligence.GenerationBrief
    awareness: str
    technique: Optional[TechniqueProfile] = None
    seed: int = 0

    # -- renderer mappings ----------------------------------------------------
    def diffusion_meta(self, idea: str, tone: str, brand: str = "",
                       extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build the ``diffusion_meta`` dict scenes.py feeds to the diffusion
        pipeline, with real-asset Visual DNA instead of a genre-only lookup."""
        meta: Dict[str, Any] = {
            "idea": idea,
            "platform": self.platform,
            "tone": tone,
            "awareness": self.awareness,
            "brand": brand or "",
            "dna": (self.technique.dna_dict() if self.technique
                    else {"energy": 0.5, "darkness": 0.5, "warmth": 0.5, "saturation": 0.5}),
            "technique_source": (self.technique.source if self.technique else "none"),
        }
        if extra:
            meta.update(extra)
        return meta

    def rta_image_params(self, width: int, height: int, seed: int) -> Dict[str, Any]:
        """Parameters for the RTA IRC path tracer, conditioned by technique."""
        t = self.technique
        return {
            "color_scheme": t.color_scheme() if t else "dark_neon",
            "mood": t.mood() if t else "cinematic",
            "width": width,
            "height": height,
            "seed": seed,
        }

    def image_color_scheme(self, default: str = "dark_neon") -> str:
        return self.technique.color_scheme() if self.technique else default

    def color_grade(self) -> str:
        return self.technique.color_grade() if self.technique else "cinematic"

    def audio_conditioning(self) -> Dict[str, Any]:
        return self.technique.audio_conditioning() if self.technique else {}

    def to_meta(self) -> Dict[str, Any]:
        """Compact, transparent view for API response metadata."""
        out: Dict[str, Any] = {}
        try:
            out["brief"] = self.brief.to_dict()
        except Exception:
            pass
        if self.technique is not None:
            out["technique"] = self.technique.to_dict()
        return out


def _stub_brief(modality: str, platform: str) -> Any:
    """Absolute last-resort GenerationBrief with every required field defaulted.

    Reached only if BOTH the full and the minimal ``build_brief`` calls raise —
    effectively impossible, but it keeps ``build_context`` never-raise so a total
    intelligence-layer failure degrades to safe procedural copy rather than a 5xx.
    """
    from ai_model.request_intelligence import GenerationBrief
    return GenerationBrief(
        modality=modality, platform=platform, intent="promote",
        intent_label="promotion", intent_confidence=0.0, audience="music fans",
        tone="", keywords=[], aspect_ratio="1:1", layout="standard",
        hook_style="direct", cta_style="soft", suggested_cta="Follow for more.",
        word_count_target=(20, 60), hashtags_target=5, tempo="medium",
        candidate_count=1, temperature=0.7, directives=[], augmented_idea="",
    )


def build_context(
    modality: str,
    req: Any,
    *,
    genre: Optional[str] = None,
    brand: Optional[str] = None,
    seed: int = 0,
    with_technique: bool = True,
    with_audio: bool = False,
    **brief_kwargs: Any,
) -> GenerationContext:
    """Build the unified generation context for one request. Never raises.

    ``brief_kwargs`` are forwarded verbatim to ``request_intelligence.build_brief``
    (each modality supplies its own inherent inputs — topic, goal, tone, themes,
    narrative, …). Awareness is merged from ``req`` and — unless
    ``with_technique=False`` (the cheap text/content path) — the technique
    profile is extracted from real reference assets.
    """
    from ai_model import request_intelligence as ri

    # Brief construction must never take an endpoint down — content/text now
    # depend on this path with no local fallback of their own. Degrade to a
    # minimal brief if the intelligence layer errors. `platform` and `topic` are
    # REQUIRED by build_brief, so the fallback must supply them (a bare
    # `build_brief(modality=...)` would itself raise TypeError). As an absolute
    # last resort, hand back a stub brief with every required field defaulted so
    # this function keeps its never-raise contract no matter what.
    _plat = brief_kwargs.get("platform") or "general"
    try:
        brief = ri.build_brief(modality=modality, **brief_kwargs)
    except Exception:
        try:
            brief = ri.build_brief(
                modality=modality, platform=_plat,
                topic=brief_kwargs.get("topic") or "",
            )
        except Exception:
            brief = _stub_brief(modality, _plat)

    try:
        awareness = merge_awareness(req)
    except Exception:
        awareness = ""

    technique: Optional[TechniqueProfile] = None
    if with_technique:
        # extract_technique is itself never-raise, but guard the attribute reads
        # around it too so a malformed brief can't escape here.
        try:
            technique = extract_technique(
                idea=getattr(brief, "track", None) or brief_kwargs.get("topic", "") or "",
                genre=genre,
                tone=getattr(brief, "tone", None),
                energy=getattr(brief, "energy", None),
                mood=getattr(brief, "mood", None),
                bpm=getattr(brief, "bpm", None),
                key=getattr(brief, "key", None),
                brand=brand,
                seed=seed,
                with_audio=with_audio,
            )
        except Exception:
            technique = None

    return GenerationContext(
        modality=modality,
        platform=brief_kwargs.get("platform", getattr(brief, "platform", "general")),
        brief=brief,
        awareness=awareness,
        technique=technique,
        seed=seed,
    )
