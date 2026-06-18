---
name: Deterministic/seeded output reuse must validate + write atomically
description: The correctness rule for any cache that reuses a previously-rendered file by deterministic name (image engine seeded renders, and similar)
---

# Reuse a deterministic output only after validating it, and always write it atomically
When a renderer derives a deterministic filename from its inputs and skips re-rendering if that file already exists, two hazards break the "no broken fallback" rule:

1. **Trusting `exists()` alone serves junk forever.** A prior crash, a killed process, or a concurrent half-write can leave a zero-byte/truncated/corrupt file. `exists()` is true, so the renderer reuses it on every future call and downstream (embedding, anchor load, display) fails repeatedly instead of self-correcting. Reuse only after a lightweight *validity* check (non-empty + decodes, e.g. `Image.open(...).verify()` for PNG). If invalid, re-render — that restores pre-cache behavior.

2. **Direct writes to the final path expose partials.** Render to a temp sibling and `os.replace()` into place (atomic rename on POSIX/same-fs). A concurrent reader or a crash then never observes a partial file; on failure, unlink the temp and re-raise (matches prior render-can-raise behavior, caught by the endpoint).

# Locks that guard a deterministic filename must be process-global, not per-instance
Per-key locks dedupe concurrent identical renders, but if they live on the renderer *instance*, two instances in the same process (or two requests holding different instances) race on the same file. Keep the lock map at **module level** so every instance in the process serializes on the same key. Cross-process races are still covered by the atomic `os.replace` (readers always see a complete file; worst case two processes render identical bytes and one harmlessly replaces the other).

**Why:** caught in code review — the original seeded-dedupe trusted `exists()` and wrote directly to the final path with instance-local locks. Both violate constraint #4 ("build so well it never breaks; degrade to prior behavior, never serve broken output").
**How to apply:** any new deterministic-name output cache (images, video frames, audio stems) follows the same validate-before-reuse + temp-then-rename + module-level-lock recipe.
