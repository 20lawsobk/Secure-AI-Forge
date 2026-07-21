---
name: Guaranteed-completion (no-timeout) policy
description: How request/job timeout aborts were removed so every job completes on first try; which timeouts must stay
---

Policy: no caller-visible timeout may ever fail a request or job. One submission = one guaranteed completion, regardless of size/complexity/quantity.

**Rules:**
- Python middleware is observability-only: logs SLOW/VERY SLOW at latency-class marks, then awaits to completion. Never returns 504.
- Adaptive-concurrency gates wait indefinitely (`gate.slot(timeout=None)`) — bursts queue and drain, never GateBusy/503.
- Readiness parks (`_wait_for_model_ready`, `_wait_for_workers`) poll forever with periodic slow-logs — never 503.
- Audio jobs have NO render deadline timer. Render thread is the sole terminal-state writer and retries with backoff until success; a 30s heartbeat updates `status: rendering` + `elapsed_seconds` so pollers see life on long renders.
- Node proxy `waitForRecovery` holds requests indefinitely during Python restarts (300ms poll, periodic log) — never 503 on a timer. undici headersTimeout/bodyTimeout stay 0; server.timeout=0.
- Agent calls (ScriptAgent etc.) are plain awaits — no `asyncio.wait_for` wrappers.

**What must NOT be removed** (they are the sensors that make indefinite waiting safe):
- ffmpeg/subprocess timeouts inside never-raise renderers — they detect frozen subprocesses and feed internal retry, converting hangs into re-attempts.
- Watchdog probe timeouts (`as_completed(timeout=...)`) — hang detection powering self-healing.

**Why:** deadline timers were marking healthy long renders as "error" and gates were 503ing bursts; user directive is 100% first-try completion.

**How to apply:** any new endpoint/job must follow the same split — no caller-facing deadline, internal hang-sensors + retry loops only. Retry loops need: terminal-state handoff (stop+JOIN heartbeat before the final write so "done" can't be overwritten), and fatal-fault escalation (same error repeated → loud watchdog escalation, capped backoff) so deterministic faults surface instead of spinning silently.
