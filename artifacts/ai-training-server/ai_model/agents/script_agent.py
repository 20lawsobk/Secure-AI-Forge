from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List
from ..model.creative_model import CreativeModel

# ── Static dicts — dead code in normal operation (awareness is always active) ──
# Kept as a true last resort for the case where awareness is completely absent.

PLATFORM_HOOKS = {
    "tiktok": "Stop scrolling — you need to hear this",
    "instagram": "This is what you've been waiting for",
    "youtube": "In this video, I'm going to show you something incredible",
    "facebook": "I've got something special to share today",
    "twitter": "Thread time",
    "linkedin": "Here's a lesson from my music career",
    "google_business": "Exciting update from the studio",
    "threads": "Let me tell you about this",
}

PLATFORM_CTAS = {
    "tiktok": "Follow for more content like this! Link in bio",
    "instagram": "Double tap if you feel this! Save for later",
    "youtube": "Like and subscribe for more! Hit the bell",
    "facebook": "Share this with someone who needs to hear it",
    "twitter": "RT if you agree. Drop a reply with your take",
    "linkedin": "What are your thoughts? Comment below",
    "google_business": "Visit us today and experience it yourself",
    "threads": "Repost if this hits different",
}


@dataclass
class ScriptRequest:
    idea: str
    platform: str
    goal: str
    tone: str
    awareness: str = ""
    variant_idx: int = 0  # offset into signal list; breaks determinism for multi-variant calls


@dataclass
class ScriptResponse:
    hook: str
    body: str
    cta: str
    source: str = "template"


def _clean_text(text: str) -> str:
    # Remove any XML-style control tokens — upper-case, lower-case, or mixed
    # e.g. <UNK>, <pad>, <|endoftext|>, <extra_id_0>
    text = re.sub(r"<[A-Za-z_|][A-Za-z0-9_|/]*>", "", text)
    # Remove numeric special tokens like [UNK], [PAD], [CLS], [SEP], [MASK]
    text = re.sub(r"\[(?:UNK|PAD|CLS|SEP|MASK|BOS|EOS)\]", "", text, flags=re.IGNORECASE)
    # Collapse whitespace left by removed tokens
    text = re.sub(r"\s+", " ", text).strip()
    # Stage 8 constraint enforcement (post-generation): redact/refuse unsafe
    # content on every agent output path, independent of downstream ranking.
    try:
        from ai_model.safety import enforce as _safety_enforce
        text = _safety_enforce(text)
    except Exception:
        pass
    return text


# ── Awareness parsing helpers ──────────────────────────────────────────────────
# All functions guarantee a non-empty result when `awareness` is non-empty.

def _any_lines(awareness: str, min_len: int = 15) -> List[str]:
    """Broadest possible extraction — any non-trivial, non-header line."""
    return [
        line.strip() for line in awareness.splitlines()
        if len(line.strip()) >= min_len and not line.strip().startswith("===")
    ]


def _parse_signals_for_platform(awareness: str, platform: str) -> List[str]:
    """
    Extract action phrases and signal titles from the awareness context.
    When nothing platform-specific is found, widens to all signals so the
    result is always non-empty when awareness is non-empty.
    """
    if not awareness:
        return []

    signals: List[str] = []
    plat_lower = platform.lower()
    other_platforms = [
        p for p in ["tiktok", "instagram", "youtube", "facebook", "twitter", "linkedin", "threads"]
        if p != plat_lower
    ]

    for line in awareness.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(r"\[(HIGH|MEDIUM|LOW)\]\s+(.+)", stripped)
        if m:
            headline = m.group(2).strip()
            if plat_lower in headline.lower() or not any(p in headline.lower() for p in other_platforms):
                signals.append(headline)
        if stripped.startswith("Action:") or "↳ Action:" in stripped:
            action = re.sub(r"^(Action:|↳ Action:)\s*", "", stripped).strip()
            if action and len(action) > 15:
                signals.append(action)
        if stripped.startswith("•") and len(stripped) > 20:
            rec = stripped.lstrip("•").strip()
            if rec:
                signals.append(rec)

    if not signals:
        # Widen: include any signal regardless of platform specificity
        for line in awareness.splitlines():
            stripped = line.strip()
            m = re.match(r"\[(HIGH|MEDIUM|LOW)\]\s+(.+)", stripped)
            if m:
                signals.append(m.group(2).strip())
        # Final widening: any non-header line
        if not signals:
            signals = _any_lines(awareness)

    return signals[:5]


def _parse_hook_from_awareness(
    awareness: str, platform: str, idea: str, signal_offset: int = 0
) -> str:
    """
    Build a data-informed hook sentence from live industry signals.
    ``signal_offset`` rotates which signal is used as the primary hook source;
    pass ``variant_idx`` here to get distinct hooks across multi-variant calls.
    Guaranteed to return a non-empty string when ``awareness`` is non-empty.
    """
    if not awareness:
        return ""

    signals = _parse_signals_for_platform(awareness, platform)

    if signals:
        # Rotate through available signals so consecutive variant calls with the
        # same awareness string pick a different starting signal each time.
        best = signals[signal_offset % len(signals)]
        if re.search(r"algorithm|watch time|engagement|trending|viral", best, re.IGNORECASE):
            _hooks = [
                f"The algorithm is favouring this right now — {idea}",
                f"This is what the algorithm wants right now: {idea}",
                f"The algorithm keeps pushing {idea} — and here's why",
            ]
            return _hooks[signal_offset % len(_hooks)]
        if re.search(r"playlist|editorial|spotify|apple music", best, re.IGNORECASE):
            _hooks = [
                f"Playlist editors are watching — here's {idea}",
                f"Editorial playlists are picking this up — {idea}",
                f"The playlist curators are paying attention to {idea}",
            ]
            return _hooks[signal_offset % len(_hooks)]
        if re.search(r"short.form|reels|shorts|vertical", best, re.IGNORECASE):
            _hooks = [
                f"Short-form is dominating — and {idea} is why",
                f"Vertical content is winning right now. {idea} is ready.",
                f"Reels are everywhere — {idea} was made for this moment",
            ]
            return _hooks[signal_offset % len(_hooks)]
        if re.search(r"collab|feature|duet|trend", best, re.IGNORECASE):
            _hooks = [
                f"The biggest trend on {platform} right now: {idea}",
                f"Collabs are dominating {platform} — and {idea} delivers",
                f"Everyone on {platform} is watching this space — {idea}",
            ]
            return _hooks[signal_offset % len(_hooks)]
        # Use the signal headline directly as a hook
        hook = best.split(":")[0].strip() if ":" in best else best
        hook = hook[:80].rstrip(",.")
        if len(hook) > 10:
            return hook
        _generic = [
            f"The industry is moving — {idea}",
            f"The moment is right for {idea}",
            f"Everything is pointing to {idea}",
        ]
        return _generic[signal_offset % len(_generic)] if idea else hook or "The moment is now"

    # awareness is non-empty but yielded no parsed signals — synthesize
    trending_tags = re.findall(r"#(\w+)", awareness)
    if trending_tags:
        tag = trending_tags[signal_offset % len(trending_tags)]
        _synth = [
            f"#{tag} is everywhere right now — and {idea} is ready",
            f"#{tag} is the story this week — {idea} is the soundtrack",
            f"Every feed is showing #{tag} — here comes {idea}",
        ]
        return _synth[signal_offset % len(_synth)]
    _fallback_hooks = [
        f"The industry is moving — {idea}",
        f"The moment is right for {idea}",
        f"Everything is pointing to {idea}",
    ]
    return _fallback_hooks[signal_offset % len(_fallback_hooks)] if idea else "The moment is now"


def _parse_cta_from_awareness(awareness: str, platform: str) -> str:
    """
    Extract a CTA from awareness recommendations.
    Guaranteed to return a non-empty string when awareness is non-empty.
    """
    if not awareness:
        return ""

    plat_lower = platform.lower()

    # 1. Platform-specific CTA line
    for line in awareness.splitlines():
        stripped = line.strip()
        if plat_lower in stripped.lower() and any(
            kw in stripped.lower()
            for kw in ["follow", "subscribe", "like", "share", "save", "comment", "link", "stream"]
        ):
            cta = re.sub(r"^(•|↳|Action:)\s*", "", stripped).strip()
            if cta and len(cta) > 10:
                return cta[:120]

    # 2. Any action recommendation
    for line in awareness.splitlines():
        stripped = line.strip()
        if stripped.startswith("Action:") or "↳ Action:" in stripped:
            cta = re.sub(r"^(Action:|↳ Action:)\s*", "", stripped).strip()
            if cta and len(cta) > 10:
                return cta[:120]

    # 3. Synthesize from platform context — always produces something
    _cta_verbs = {
        "tiktok": "Follow for more — link in bio",
        "instagram": "Save this and follow for more drops",
        "youtube": "Subscribe and hit the bell",
        "facebook": "Like the page for more releases",
        "twitter": "RT if this goes hard",
        "linkedin": "Follow for music industry insights",
        "threads": "Repost if this hits different",
    }
    return _cta_verbs.get(plat_lower, "Follow for more content")


def _build_awareness_body(
    awareness: str, platform: str, idea: str, tone: str, signal_offset: int = 0
) -> str:
    """
    Build a body paragraph from live industry context.
    ``signal_offset`` rotates which pair of signals anchors the body so that
    multi-variant calls with the same awareness produce distinct copy.
    Guaranteed to return a non-empty string when ``awareness`` is non-empty.
    """
    if not awareness:
        return ""

    signals = _parse_signals_for_platform(awareness, platform)
    trending_tags = re.findall(r"#(\w+)", awareness)
    topic_words = [t for t in trending_tags if len(t) > 4][:3]
    context_phrase = f" ({', '.join('#' + t for t in topic_words)} trending)" if topic_words else ""

    if len(signals) >= 2:
        # Rotate the signal pair by offset so variant 0 uses (0,1), variant 1 uses (1,2), etc.
        i0 = signal_offset % len(signals)
        i1 = (signal_offset + 1) % len(signals)
        s1 = signals[i0].split(":")[0].rstrip(",.").strip()[:100]
        s2 = signals[i1].split(":")[0].rstrip(",.").strip()[:100]
        if i0 == i1:  # only one signal available
            return f"{idea}{context_phrase}. {s1}."
        return f"{idea}{context_phrase}. {s1}. {s2}."
    elif signals:
        s1 = signals[0].split(":")[0].rstrip(",.").strip()[:120]
        return f"{idea}{context_phrase}. {s1}."

    # No signals parsed — tone-aware synthesis from hashtag/keyword context.
    # Every pool has ≥3 entries so offsets 0, 1, 2 are always distinct.
    if topic_words:
        tag_str = ", ".join("#" + t for t in topic_words)
        fallbacks = [
            f"{idea} — riding the wave of {tag_str} that's dominating right now.",
            f"{idea} connects with the {tag_str} conversation happening across every feed.",
            f"{idea} is exactly where {tag_str} is pointing — and it's ready.",
        ]
        return fallbacks[signal_offset % len(fallbacks)]
    if tone == "energetic":
        pool = [
            f"{idea} is exactly what the moment needs{context_phrase}. The energy is undeniable.",
            f"{idea} is arriving at exactly the right time{context_phrase}. Don't sleep on this.",
            f"Everything is pointing toward {idea}{context_phrase}. The timing is perfect.",
        ]
    elif tone == "professional":
        pool = [
            f"Presenting {idea}{context_phrase} — crafted for today's landscape.",
            f"{idea}{context_phrase} — built around what the industry is responding to right now.",
            f"{idea}{context_phrase} — positioned for where the market is moving.",
        ]
    elif tone == "casual":
        pool = [
            f"So {idea} just dropped{context_phrase} — and yeah, it's that good.",
            f"Not gonna lie — {idea}{context_phrase} is hitting different right now.",
            f"Real talk: {idea}{context_phrase} showed up and delivered.",
        ]
    elif tone == "promotional":
        pool = [
            f"Introducing {idea}{context_phrase} — available now on all platforms.",
            f"{idea} is out now{context_phrase} — stream it everywhere.",
            f"{idea} just landed{context_phrase}. Go listen now.",
        ]
    else:
        pool = [
            f"{idea}{context_phrase} — shaped by what's working right now.",
            f"{idea}{context_phrase} — made for this moment.",
            f"{idea}{context_phrase} — the timing couldn't be better.",
        ]
    return pool[signal_offset % len(pool)]


# ── Agent ──────────────────────────────────────────────────────────────────────

class ScriptAgent:
    def __init__(self, model: CreativeModel):
        self.model = model

    def run(self, req: ScriptRequest) -> ScriptResponse:
        platform_token = f"<PLATFORM_{req.platform.upper()}>"
        goal_token = f"<GOAL_{req.goal.upper()}>"
        tone_token = f"<TONE_{req.tone.upper()}>"

        # Always inject live awareness signals into the primary LLM prompt so the
        # model conditions on them even when it generates successfully.
        # variant_idx rotates which signals lead the prompt so multi-variant
        # calls with identical inputs sample different generation contexts.
        awareness_prefix = ""
        if req.awareness:
            signals = _parse_signals_for_platform(req.awareness, req.platform)
            if signals:
                i0 = req.variant_idx % len(signals)
                i1 = (req.variant_idx + 1) % len(signals)
                awareness_prefix = f"Industry context: {signals[i0][:120]}\n"
                if len(signals) > 1 and i1 != i0:
                    awareness_prefix += f"Trend: {signals[i1][:80]}\n"
            trending_tags = re.findall(r"#(\w+)", req.awareness)
            if trending_tags:
                awareness_prefix += f"Trending: {', '.join(trending_tags[:4])}\n"

        prompt = f"{awareness_prefix}{platform_token} {goal_token} {tone_token} <STAGE_HOOK>"

        try:
            output = self.model.generate(prompt, max_new_tokens=80, temperature=0.8, top_p=0.92)

            hook = ""
            body = ""
            cta = ""

            if "<STAGE_BODY>" in output:
                hook_section = output.split("<STAGE_BODY>")[0]
                hook_section = hook_section.split("<STAGE_HOOK>")[-1] if "<STAGE_HOOK>" in hook_section else hook_section
                hook = _clean_text(hook_section)

                remainder = output.split("<STAGE_BODY>", 1)[1]
                if "<STAGE_CTA>" in remainder:
                    body = _clean_text(remainder.split("<STAGE_CTA>")[0])
                    cta = _clean_text(remainder.split("<STAGE_CTA>", 1)[1])
                else:
                    body = _clean_text(remainder)

            if self._is_meaningful(hook) and self._is_meaningful(body):
                # Undertrained-model garble guard: glued tokens ("beingpre-save",
                # "beingstarting") or letter-digit fusions ("Frequency82") pass
                # the length/repetition checks above but read as gibberish in
                # user-facing overlays. Words from the request itself (idea,
                # awareness) are whitelisted, so real names never trip it.
                # On garble, fall through to the awareness composition below.
                from ai_model.request_intelligence import looks_garbled
                _wl = f"{req.idea} {req.awareness or ''}"
                if not looks_garbled(f"{hook}\n{body}", whitelist=_wl):
                    if not cta or not self._is_meaningful(cta) or looks_garbled(cta, whitelist=_wl):
                        cta = _parse_cta_from_awareness(req.awareness, req.platform)
                        if not cta:
                            cta = PLATFORM_CTAS.get(req.platform.lower(), "Let me know what you think!")
                    return ScriptResponse(hook=hook, body=body, cta=cta, source="ai_model")
        except Exception:
            pass

        return self._fallback(req)

    def _is_meaningful(self, text: str) -> bool:
        if not text or len(text) < 10:
            return False
        words = text.split()
        if len(words) < 3:
            return False
        control_count = sum(1 for w in words if w.startswith("<") and w.endswith(">"))
        if control_count / len(words) > 0.3:
            return False
        unique_words = set(w.lower() for w in words)
        if len(unique_words) < len(words) * 0.3:
            return False
        return True

    def _fallback(self, req: ScriptRequest) -> ScriptResponse:
        """
        Awareness is always active — resolve hook/body/cta entirely from live
        signals. Static dicts are only reached when awareness is absent, which
        should not happen in normal operation.
        ``req.variant_idx`` rotates signal selection so multi-variant calls
        with identical inputs produce distinct, non-duplicate copy.
        """
        platform = req.platform.lower().replace(" ", "_")
        awareness = req.awareness or ""

        if awareness:
            hook = _parse_hook_from_awareness(awareness, platform, req.idea,
                                              signal_offset=req.variant_idx)
            body = _build_awareness_body(awareness, platform, req.idea, req.tone,
                                         signal_offset=req.variant_idx)
            cta = _parse_cta_from_awareness(awareness, platform)
            return ScriptResponse(hook=hook, body=body, cta=cta, source="awareness")

        # ── True last resort: awareness absent (should not happen in production) ──
        hook = PLATFORM_HOOKS.get(platform, "Check this out")
        if req.tone == "energetic":
            body = f"{req.idea} — and it's going to blow your mind!"
        elif req.tone == "professional":
            body = f"I'm excited to present: {req.idea}"
        elif req.tone == "casual":
            body = f"So about {req.idea}... yeah, it's that good"
        elif req.tone == "promotional":
            body = f"Introducing: {req.idea} — available now!"
        else:
            body = req.idea
        cta = PLATFORM_CTAS.get(platform, "Let me know what you think!")
        return ScriptResponse(hook=hook, body=body, cta=cta, source="template")
