from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional
from ..model.creative_model import CreativeModel

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


@dataclass
class ScriptResponse:
    hook: str
    body: str
    cta: str
    source: str = "template"


def _clean_text(text: str) -> str:
    text = re.sub(r"<[A-Z_]+>", "", text)
    text = re.sub(r"<UNK>", "", text)
    text = re.sub(r"<PAD>", "", text)
    text = re.sub(r"<BOS>", "", text)
    text = re.sub(r"<EOS>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Awareness parsing helpers ─────────────────────────────────────────────────

def _parse_signals_for_platform(awareness: str, platform: str) -> List[str]:
    """
    Extract action phrases and signal titles from the awareness context
    that are relevant to this platform. Returns a list of usable sentences.
    """
    if not awareness:
        return []

    signals: List[str] = []
    plat_lower = platform.lower()

    for line in awareness.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Signal headlines (e.g. "[HIGH] TikTok Algorithm Update: ...")
        m = re.match(r"\[(HIGH|MEDIUM|LOW)\]\s+(.+)", stripped)
        if m:
            headline = m.group(2).strip()
            # Include if it mentions the platform or is broadly applicable
            if plat_lower in headline.lower() or not any(
                p in headline.lower() for p in
                ["tiktok", "instagram", "youtube", "facebook", "twitter", "linkedin", "threads"]
                if p != plat_lower
            ):
                signals.append(headline)
        # Action recommendations
        if stripped.startswith("Action:") or "↳ Action:" in stripped:
            action = re.sub(r"^(Action:|↳ Action:)\s*", "", stripped).strip()
            if action and len(action) > 15:
                signals.append(action)
        # Content recommendations
        if stripped.startswith("•") and len(stripped) > 20:
            rec = stripped.lstrip("•").strip()
            if rec:
                signals.append(rec)

    return signals[:5]


def _parse_hook_from_awareness(awareness: str, platform: str, idea: str) -> Optional[str]:
    """Build a data-informed hook sentence from live industry signals."""
    signals = _parse_signals_for_platform(awareness, platform)
    if not signals:
        return None

    # Pick the most relevant signal as a hook basis
    best = signals[0]

    # If it's an algorithm or timing signal, convert it into attention language
    if re.search(r"algorithm|watch time|engagement|trending|viral", best, re.IGNORECASE):
        return f"The algorithm is favouring this right now — {idea}"
    if re.search(r"playlist|editorial|spotify|apple music", best, re.IGNORECASE):
        return f"Playlist editors are watching — here's {idea}"
    if re.search(r"short.form|reels|shorts|vertical", best, re.IGNORECASE):
        return f"Short-form is dominating — and {idea} is why"
    if re.search(r"collab|feature|duet|trend", best, re.IGNORECASE):
        return f"The biggest trend in {platform} right now: {idea}"

    # Fall back to turning the signal into a hook sentence
    hook = best.split(":")[0].strip() if ":" in best else best
    hook = hook[:80].rstrip(",.")
    if hook and len(hook) > 15:
        return hook

    return None


def _parse_cta_from_awareness(awareness: str, platform: str) -> Optional[str]:
    """Extract a platform-relevant CTA from awareness recommendations."""
    if not awareness:
        return None

    plat_lower = platform.lower()
    for line in awareness.splitlines():
        stripped = line.strip()
        if plat_lower in stripped.lower() and any(
            kw in stripped.lower() for kw in
            ["follow", "subscribe", "like", "share", "save", "comment", "link", "stream"]
        ):
            cta = re.sub(r"^(•|↳|Action:)\s*", "", stripped).strip()
            if cta and len(cta) > 10:
                return cta[:120]

    return None


def _build_awareness_body(awareness: str, platform: str, idea: str, tone: str) -> Optional[str]:
    """
    Build a body paragraph that weaves the artist's idea with live
    industry context (trending topics, platform timing, recommendations).
    """
    signals = _parse_signals_for_platform(awareness, platform)
    if not signals:
        return None

    # Extract trending topic keywords from the awareness text
    trending_tags = re.findall(r"#(\w+)", awareness)
    topic_words = [t for t in trending_tags if len(t) > 4][:3]

    context_phrase = ""
    if topic_words:
        context_phrase = f" ({', '.join('#' + t for t in topic_words)} is trending)"

    if tone == "energetic":
        return f"{idea} is exactly what the moment needs{context_phrase}. The energy is undeniable."
    elif tone == "professional":
        body_signal = signals[0].split(":")[0] if signals else ""
        return f"{idea} arrives at the right time{context_phrase}. {body_signal}." if body_signal else f"Presenting {idea}{context_phrase} — crafted for today's landscape."
    elif tone == "casual":
        return f"So {idea} just dropped{context_phrase} — and yeah, it's that good."
    elif tone == "promotional":
        return f"Introducing {idea}{context_phrase} — available now on all platforms."
    else:
        return f"{idea}{context_phrase} — shaped by what's working right now."


# ── Agent ─────────────────────────────────────────────────────────────────────

class ScriptAgent:
    def __init__(self, model: CreativeModel):
        self.model = model

    def run(self, req: ScriptRequest) -> ScriptResponse:
        platform_token = f"<PLATFORM_{req.platform.upper()}>"
        goal_token = f"<GOAL_{req.goal.upper()}>"
        tone_token = f"<TONE_{req.tone.upper()}>"

        prompt = f"{platform_token} {goal_token} {tone_token} <STAGE_HOOK>"

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
                if not cta or not self._is_meaningful(cta):
                    # Prefer awareness-derived CTA over static
                    cta = (
                        _parse_cta_from_awareness(req.awareness, req.platform)
                        or PLATFORM_CTAS.get(req.platform.lower(), "Let me know what you think!")
                    )
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
        Awareness-first fallback: uses live industry signals from the awareness
        context to construct hook/body/cta. Only falls back to static strings
        when no awareness data is available.
        """
        platform = req.platform.lower().replace(" ", "_")
        awareness = req.awareness or ""

        # 1. Try to build hook from awareness signals
        hook = _parse_hook_from_awareness(awareness, platform, req.idea)
        if not hook:
            hook = PLATFORM_HOOKS.get(platform, "Check this out")

        # 2. Try to build body from awareness + idea
        body = _build_awareness_body(awareness, platform, req.idea, req.tone)
        if not body:
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

        # 3. Try to build CTA from awareness
        cta = (
            _parse_cta_from_awareness(awareness, platform)
            or PLATFORM_CTAS.get(platform, "Let me know what you think!")
        )

        source = "awareness" if awareness else "template"
        return ScriptResponse(hook=hook, body=body, cta=cta, source=source)
