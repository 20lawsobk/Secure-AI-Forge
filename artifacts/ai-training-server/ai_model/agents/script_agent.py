from __future__ import annotations
import re
from dataclasses import dataclass
from ..model.creative_model import CreativeModel

PLATFORM_HOOKS = {
    "tiktok": "Stop scrolling - you need to hear this",
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
        platform = req.platform.lower().replace(" ", "_")
        hook = PLATFORM_HOOKS.get(platform, "Check this out")
        cta = PLATFORM_CTAS.get(platform, "Let me know what you think!")

        body = req.idea
        if req.tone == "energetic":
            body = f"{req.idea} - and it's going to blow your mind!"
        elif req.tone == "professional":
            body = f"I'm excited to present: {req.idea}"
        elif req.tone == "casual":
            body = f"So about {req.idea}... yeah, it's that good"
        elif req.tone == "promotional":
            body = f"Introducing: {req.idea} - available now!"

        return ScriptResponse(hook=hook, body=body, cta=cta, source="template")
