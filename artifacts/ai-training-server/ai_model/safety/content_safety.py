"""Content-safety / policy enforcement (pipeline Stage 8).

Deterministic, dependency-free policy engine. Two enforcement surfaces:

  • During generation — ``ContentSafety.bad_token_ids(tokenizer)`` yields the
    token ids the decoder should mask to ``-inf`` so the worst tokens can never
    be emitted (wired into ``CreativeModel._sample_next``).
  • After generation — ``screen`` / ``enforce`` inspect finished text and either
    pass it, redact matched spans, or replace the whole output with a safe
    refusal (wired into candidate ranking and agent post-processing).

Design goals
------------
* No silent fakes: a violation is reported honestly (category + severity +
  matched span). Nothing is quietly dropped without being counted.
* Low false-positives on a *creative* generator: SEVERE rules require
  instructional / exploitative intent (e.g. "how to build a bomb"), not the
  mere mention of a word that can appear in lyrics or marketing copy.
* Configurable via env: ``MB_SAFETY_ENABLED`` (default on) and
  ``MB_SAFETY_MODE`` ∈ {enforce, monitor, off}. In ``monitor`` mode violations
  are detected and counted but text is passed through unchanged.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, FrozenSet, List, Optional, Tuple


class Severity(IntEnum):
    """Ordered so ``max()`` picks the worst match."""
    NONE = 0
    REDACT = 1   # moderate: redact the offending span, keep the rest
    BLOCK = 2    # severe: refuse — replace the whole output


# ── Policy definitions ───────────────────────────────────────────────────────
# Each rule: (category, severity, compiled pattern). Patterns are intentionally
# narrow. SEVERE rules pair an intent verb with a harmful object so ordinary
# creative mentions don't trip them.

_INTENT = r"(?:how\s+(?:to|do\s+i)|instructions?\s+for|step[-\s]?by[-\s]?step|guide\s+to|recipe\s+for|teach\s+me\s+to|ways?\s+to)"

_RAW_RULES: List[Tuple[str, Severity, str]] = [
    # ── SEVERE / BLOCK ──────────────────────────────────────────────────────
    # Weapons, explosives, chemical/bio agents — instructional only.
    ("weapons_explosives", Severity.BLOCK,
     rf"\b{_INTENT}\b[^.\n]{{0,60}}\b(bomb|explosive|pipe\s?bomb|ied|detonator|"
     r"nerve\s?(?:agent|gas)|sarin|ricin|anthrax|chemical\s+weapon|"
     r"biological\s+weapon|silencer|untraceable\s+(?:gun|firearm)|ghost\s+gun)\b"),
    # Illicit drug synthesis — instructional only.
    ("illicit_drug_synthesis", Severity.BLOCK,
     rf"\b{_INTENT}\b[^.\n]{{0,60}}\b(?:synthesi[sz]e|manufacture|make|cook|produce)?\s*"
     r"(meth(?:amphetamine)?|fentanyl|heroin|cocaine|mdma|lsd)\b"),
    # Self-harm / suicide methods — instructional only.
    ("self_harm_instructions", Severity.BLOCK,
     rf"\b{_INTENT}\b[^.\n]{{0,40}}\b(kill\s+myself|commit\s+suicide|"
     r"hang\s+myself|overdose\s+on|end\s+my\s+life|painless(?:ly)?\s+(?:die|death))\b"),
    # Child sexual content — any co-occurrence of a minor term with sexual term.
    ("csae", Severity.BLOCK,
     r"\b(child|children|minor|underage|preteen|pre-teen|toddler|infant|"
     r"(?:\b(?:8|9|10|11|12|13|14|15|16)[-\s]?year[-\s]?old))\b[^.\n]{0,40}"
     r"\b(sex|sexual|nude|naked|porn|explicit|molest|fondl)\w*"),
    ("csae_rev", Severity.BLOCK,
     r"\b(sexual|nude|naked|porn|explicit|molest|fondl)\w*[^.\n]{0,40}"
     r"\b(child|children|minor|underage|preteen|pre-teen|toddler|infant)\b"),
    # Credible incitement / threat of mass violence — instructional only.
    ("violence_incitement", Severity.BLOCK,
     rf"\b{_INTENT}\b[^.\n]{{0,40}}\b(?:carry\s+out\s+a?\s*)?"
     r"(mass\s+shooting|school\s+shooting|terror\s+attack|kill\s+as\s+many)\b"),

    # ── MODERATE / REDACT ───────────────────────────────────────────────────
    # Hate slurs / dehumanising harassment. Compact, well-known set; a hate-
    # speech filter genuinely requires the terms to function. Word-boundaried.
    ("hate_slur", Severity.REDACT,
     r"\b(n[i1]gg(?:er|a)s?|f[a4]gg?(?:ot|ots)?|k[i1]kes?|sp[i1]cs?|"
     r"ch[i1]nks?|wetbacks?|tr[a4]nn(?:y|ies)|retards?)\b"),
]

_RULES: List[Tuple[str, Severity, "re.Pattern[str]"]] = [
    (cat, sev, re.compile(pat, re.IGNORECASE)) for (cat, sev, pat) in _RAW_RULES
]

# Terms hard-blocked at the token level during decoding (subset of the slur
# list that we never want the model to emit at all).
_HARD_BLOCK_TERMS: Tuple[str, ...] = (
    "nigger", "nigga", "faggot", "kike", "spic", "chink", "wetback",
    "tranny", "retard",
)

_REFUSAL = (
    "This request can't be fulfilled because it would violate the content "
    "safety policy."
)
_REDACTION = "\u2588\u2588\u2588\u2588"  # ████


@dataclass
class SafetyResult:
    """Outcome of screening a piece of text."""
    allowed: bool
    severity: Severity
    categories: List[str] = field(default_factory=list)
    matches: List[str] = field(default_factory=list)
    text: str = ""            # the enforced (possibly redacted/refused) text
    original: str = ""

    @property
    def flagged(self) -> bool:
        return self.severity > Severity.NONE

    def to_dict(self) -> Dict[str, object]:
        return {
            "allowed": self.allowed,
            "severity": self.severity.name.lower(),
            "categories": sorted(set(self.categories)),
            "flagged": self.flagged,
        }


class ContentSafety:
    """Deterministic policy engine. Thread-safe; cheap to construct."""

    def __init__(self) -> None:
        self._counts: Dict[str, int] = {}
        self._lock = threading.Lock()
        # tokenizer id -> frozenset of hard-blocked token ids
        self._bad_id_cache: Dict[int, FrozenSet[int]] = {}

    # ── config ──────────────────────────────────────────────────────────────
    @staticmethod
    def _mode() -> str:
        if os.environ.get("MB_SAFETY_ENABLED", "1") == "0":
            return "off"
        return os.environ.get("MB_SAFETY_MODE", "enforce").lower()

    @property
    def enabled(self) -> bool:
        return self._mode() != "off"

    # ── screening ───────────────────────────────────────────────────────────
    def screen(self, text: str) -> SafetyResult:
        """Detect violations without modifying text."""
        if not text or self._mode() == "off":
            return SafetyResult(allowed=True, severity=Severity.NONE, text=text or "", original=text or "")

        worst = Severity.NONE
        cats: List[str] = []
        matches: List[str] = []
        for cat, sev, pat in _RULES:
            for m in pat.finditer(text):
                worst = max(worst, sev)
                cats.append(cat)
                matches.append(m.group(0))

        if cats:
            with self._lock:
                for c in cats:
                    self._counts[c] = self._counts.get(c, 0) + 1

        allowed = worst < Severity.BLOCK
        return SafetyResult(
            allowed=allowed, severity=worst, categories=cats,
            matches=matches, text=text, original=text,
        )

    def enforce(self, text: str) -> SafetyResult:
        """Screen and apply the policy: pass, redact, or refuse.

        In ``monitor`` mode the text is returned unchanged but the violation is
        still detected and counted.
        """
        res = self.screen(text)
        mode = self._mode()
        if mode != "enforce" or res.severity == Severity.NONE:
            return res

        if res.severity >= Severity.BLOCK:
            res.text = _REFUSAL
            res.allowed = False
            return res

        # REDACT: replace each moderate match span in place.
        redacted = text
        for cat, sev, pat in _RULES:
            if sev == Severity.REDACT:
                redacted = pat.sub(_REDACTION, redacted)
        res.text = redacted
        res.allowed = True
        return res

    def penalty_of(self, res: SafetyResult) -> float:
        """Ranking penalty from an *already screened* result (no re-screen, so
        counters are not inflated). Zero unless in ``enforce`` mode."""
        if self._mode() != "enforce":
            return 0.0
        if res.severity >= Severity.BLOCK:
            return -1000.0
        if res.severity == Severity.REDACT:
            return -50.0 * len(res.matches)
        return 0.0

    def penalty(self, text: str) -> float:
        """Ranking penalty: 0 for clean text, large-negative for violations so a
        safe candidate always outranks an unsafe one. Prefer ``penalty_of`` when
        you already hold a screened result, to avoid double-counting."""
        return self.penalty_of(self.screen(text))

    # ── token-level masking (during generation) ─────────────────────────────
    def bad_token_ids(self, tokenizer) -> FrozenSet[int]:
        """Token ids to mask to -inf during decoding. Best-effort: only ids that
        decode to a hard-blocked term as a *whole* token are masked (so we never
        clobber a common subword). Cached per tokenizer instance."""
        # Only mask during generation in full enforce mode; monitor/off must not
        # change what the decoder can emit (detect/count only).
        if self._mode() != "enforce":
            return frozenset()
        key = id(tokenizer)
        cached = self._bad_id_cache.get(key)
        if cached is not None:
            return cached

        ids: set[int] = set()
        for term in _HARD_BLOCK_TERMS:
            for variant in (term, " " + term):
                try:
                    enc = tokenizer.encode(variant)
                    tok_ids = getattr(enc, "ids", enc)
                    if isinstance(tok_ids, list) and len(tok_ids) == 1:
                        ids.add(int(tok_ids[0]))
                except Exception:
                    continue
        frozen = frozenset(ids)
        with self._lock:
            self._bad_id_cache[key] = frozen
        return frozen

    def stats(self) -> Dict[str, object]:
        with self._lock:
            counts = dict(self._counts)
        return {"mode": self._mode(), "violation_counts": counts}


# ── module-level singleton + convenience wrappers ────────────────────────────
_SINGLETON: Optional[ContentSafety] = None
_SINGLETON_LOCK = threading.Lock()


def get_safety() -> ContentSafety:
    global _SINGLETON
    if _SINGLETON is None:
        with _SINGLETON_LOCK:
            if _SINGLETON is None:
                _SINGLETON = ContentSafety()
    return _SINGLETON


def screen(text: str) -> SafetyResult:
    return get_safety().screen(text)


def enforce(text: str) -> str:
    """Return policy-safe text (redacted or refused as needed)."""
    return get_safety().enforce(text).text


def safety_penalty(text: str) -> float:
    return get_safety().penalty(text)
