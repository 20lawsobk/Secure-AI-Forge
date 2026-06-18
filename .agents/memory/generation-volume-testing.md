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
