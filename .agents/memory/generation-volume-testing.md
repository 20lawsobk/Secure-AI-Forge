---
name: Generation volume testing
description: How to reliably retest the slow in-house generation endpoints at LOW/MED/HIGH without losing results to reaped background processes
---

# Retesting the generation endpoints (multimodal/audio/video/social/ads)

The in-house transformer is undertrained and slow: multimodal/audio/video calls do
synchronous per-slot model inference, so a full LOW/MED/HIGH matrix takes many minutes.

**Rule:** Do NOT drive long test matrices from a single long-lived detached harness
(`setsid nohup python3 big_suite.py &`). Those background processes get reaped mid-run
(observed dying consistently at ~85–120s of lifetime, regardless of what step they were
on — it always *looked* like "the audio step hangs" because audio happened to be running
when the reaper fired). The server stayed healthy the whole time; only the client died.

Instead, use one of these:
- **Short synchronous bash batches**: run a small Python script with `timeout 90 python3 x.py`
  directly in the bash call (NOT detached). Children of an active bash call are not reaped.
  Group fast endpoints (social/ads/video-plan, the 13 single-shot regression endpoints,
  audio 3 tiers — audio renders in ~5s each) so each batch finishes well under ~90s.
- **Submit-then-poll for slow async jobs** (video renders, and audio if needed): POST the
  job (returns `job_id` instantly), save the ids to a file, then poll
  `GET /api/video-job/{id}` / `GET /api/audio-job/{id}` across *separate* short bash calls.
  Jobs run server-side independent of the client, so nothing is lost if a poll call ends.

**Why:** Two consecutive sessions burned significant time chasing a phantom "audio/MED hang"
that was really (a) a SIGKILLed first harness leaving the server in a degraded idle state
(fixed only by `restart_workflow "Start application"`), and (b) the reaper killing
long-lived detached clients. Neither was a code bug.

**How to apply:** When asked to "retest all generation endpoints at volume," structure the
run as several short sync batches + submit/poll for video. Verify served media by GET-ing
the `/uploads/...` URL and asserting HTTP 200 + non-zero bytes (sizes should scale with
duration/count). If something genuinely hangs, suspect a degraded server (check
`GET /api/health` uptime + a fresh `restart_workflow`) before suspecting the code.

## Honest pass criteria — two independent verdicts, never one

A load test must split **crash/backpressure** from **correctness**, and a phase passes
only if BOTH pass. Do NOT use `status < 500` as a correctness gate — it launders a 503/504
or a 422 (bad request body) into a green result.
- **crash/backpressure verdict:** no `-1` (connection death) and no real 5xx. A `503` from
  the AdaptiveGate IS backpressure working → allowed, not a failure.
- **correctness verdict:** the endpoint actually produced real output. Assert modality
  invariants, not a self-reported `source`: text → non-empty caption/hook/body; image →
  real `image_url`/`path`; audio/video → a `job_id` whose job polls through to COMPLETED
  with a real `audio_url` / video `filename`/`scenes_rendered`. For concurrency bursts,
  require `accepted + 503 == N` (every request got a definitive answer) and `accepted >= 1`.
- Observable fallbacks are fine and expected: completed audio jobs report `source=template`,
  video `source=datasets`. Labeled ≠ silent → satisfies the honesty contract.
- The harness lives at `ai_model/maxcore/tests/endpoint_load_test.py` (phases text|image|
  audio|video|all); its in-process sibling is `load_test.py`.

## Proxy (:8080) has a hard 45s synchronous ceiling — engine (:9878) does not

`api-server/src/routes/model-proxy.ts` wraps each proxied call in a 45s `AbortController`;
on abort it returns **504** ("Upstream timeout") and records a circuit-breaker failure.
**Why:** synchronous generation (`/api/generate/content|image`) under a saturated box can
exceed 45s on cold compute and get shed as 504 — this is a full-stack SLA limit, not an
engine bug; direct-to-:9878 the same call succeeds. **How to apply:** (1) job-based audio/
video are immune (submit returns a `job_id` instantly, render happens server-side); (2) the
text dedup cache makes the warm path instant through the proxy — to prove the proxy works,
pre-warm the dedup cache directly on :9878 then drive identical requests through :8080 (all
return ~ms cache hits); (3) `curl 000` in a burst is the CLIENT `-m` timeout firing before
the proxy's 45s, not a server crash — never read `000` as a 5xx.

## uploads/ is runtime media — gitignored

`artifacts/ai-training-server/uploads/` holds AI-generated media regenerated on demand; it
is gitignored. Load tests dump dozens of files there — they will NOT pollute the commit, but
delete them for disk hygiene after a run.
