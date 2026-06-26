---
name: mypy object-from-dict pattern
description: How to clear mypy "object"-operand errors in the ai-training-server without changing runtime behavior
---

# Clearing mypy "object" operand errors

The Python AI engine (`artifacts/ai-training-server/`) is fully mypy-clean and must
stay that way. Most non-trivial mypy errors there are `Unsupported operand types ... ("object")`
and stem from heterogeneous dicts whose values mypy widens to `object`.

**Rule — pick the fix by dict kind:**
- **Mutable state maps** that are mutated/read all over (e.g. `self.state` in
  `workers/continuous_trainer.py`, `_training_state` in `server.py`, a per-loop
  `variant` dict): annotate the dict itself as `dict[str, Any]` at its definition.
  One annotation clears every downstream access.
- **Read-only constant config dicts** with genuinely mixed value types
  (`TEMPLATE_STYLES`, `PLATFORM_DEFAULTS`, `GOAL_SPECS`): do NOT annotate the
  constant. Use `typing.cast(int|float|str, d["key"])` at the read site. `cast`
  is an identity at runtime, so behavior is unchanged.

**Why:** annotating a constant's values forces a single value type and breaks the
mixed literal; casting at use is local and behavior-preserving. Avoid `int(x)`/`float(x)`
to "fix" an `object` — mypy rejects `int(object)` (no matching overload).

**How to apply:** mypy check command lives in the `python-typecheck` workflow:
`uv tool run mypy artifacts/ai-training-server/ --ignore-missing-imports --no-strict-optional --follow-imports=skip`.
For `Need type annotation` on numpy assignments, annotate `: np.ndarray`.

**Bonus — two real bugs this surfaced (call-arg errors are often real):**
mypy `call-arg` errors here were genuine runtime `TypeError`s masked by broad
`except`/unused code paths. `TrainConfig` takes one nested cfg dict
(`{"model":..., "train":...}`), not kwargs, and has no `grad_clip` field
(`max_grad_norm` is hardcoded 1.0). Treat `call-arg` mypy errors as suspected real
bugs, verify the real signature, fix the call site.
