---
name: Intent ↔ awareness synchronization
description: Tiers must blend, not override; caller awareness leads the platform buffer; [INTENT] machine lines must never become caption text
---

## Rules

1. **Tiers synchronize, never override.** When both `[INTENT]` signals and chart/trend (`[HIGH]`/`TRENDS:`) signals exist in a brief, they are reconciled into fused directives: agreement → reinforce ("commit fully"), divergence → intent is the foundation, trend is flavor ("anchor on X, fold in Y textures"), numeric energy → blended 65/35 intent-weighted. Single-tier input keeps the plain single-tier directive.
   **Why:** user explicitly rejected priority-override semantics ("they should synchronize with each other"); flat override discards live-market signal.
   **How to apply:** reconciliation lives in `build_brief`'s awareness parser via a never-raise sync helper that runs the intent detector over the trend text itself. Extend the helper, don't add new override branches.

2. **Caller awareness leads the platform quality buffer.** The always-on platform buffer supplies `[HIGH]` signals, which made the script agent's plain-line fallback unreachable — caller free-text awareness silently became a no-op (bare vs aware output byte-identical). Fix pattern: caller awareness goes FIRST in the merged string, and the signal parser collects substantial plain (unprefixed) lines alongside prefixed ones.
   **Why:** any "always-present" enrichment tier will drown optional request-specific input if the parser stops at the first tier that matches.

3. **`[INTENT]` key=value lines are machine-readable only.** They must be excluded from any parser that promotes awareness lines into user-facing hook/body/cta text (exclude lines starting with `[`), or `genre=trap energy=0.85` leaks into captions.

Regression guard: `test_plain_awareness_used_and_intent_lines_never_leak` in the awareness quality suite.
