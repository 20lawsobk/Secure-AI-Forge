"""Stage 8 — Constraint Enforcement.

Content-safety / policy layer applied *during* generation (logit masking of
hard-blocked tokens) and *after* generation (output screening + redaction /
refusal). See ``content_safety.py`` for the policy and ``docs/PIPELINE.md`` for
where this sits in the 10-stage inference pipeline.
"""

from .content_safety import (
    ContentSafety,
    SafetyResult,
    Severity,
    get_safety,
    screen,
    enforce,
    safety_penalty,
)

__all__ = [
    "ContentSafety",
    "SafetyResult",
    "Severity",
    "get_safety",
    "screen",
    "enforce",
    "safety_penalty",
]
