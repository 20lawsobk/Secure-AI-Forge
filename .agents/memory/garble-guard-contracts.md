---
name: Garble-guard & seeding-guard test contracts
description: tests/test_url_label_garble_guard.py and test_audio_seeding_guard.py assert source-level contracts; how they're satisfied without hot-path latency
---

Some tests assert contracts by reading source (`server.py` must contain `seeding_in_progress` + `_is_seeding_render`) or by capturing WARN logs on loggers `script_agent` / `distribution_agent` containing `[garble-guard] ... reason=label_echo`.

**Why:** these contracts were written ahead of code and once drifted out — 4 tests failed with no functional bug visible. The observable trail (never suppress model output silently) is the point.

**How to apply:**
- ScriptAgent/DistributionAgent run a one-shot model-health probe (once per agent instance; agents are process singletons) that samples a small model draft, runs `garble_reason()` from request_intelligence, and WARNs with the reason. Awareness copy stays authoritative; probe is never-raise. Don't make the probe per-request — it adds model latency to hot paths.
- `_render_audio_from_dataset` waits (bounded ~12s, NOT 45 — video auto-soundtrack path has no early-exit guard and jobs must finish <30s) for the first seeded chunk only while `is_seeding()`.
- The audio early-exit 202 payload must keep `"error": "seeding_in_progress"`.
