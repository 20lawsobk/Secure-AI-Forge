---
name: ffmpeg spawn under memory pressure
description: Why video ffmpeg subprocess must use posix_spawn, not fork, in the deployed AI server
---

# ffmpeg subprocess must avoid fork() in the deployed AI server

In production (Nix GCE container, single Python uvicorn worker holding the ~1.7GB
transformer model in memory), every video scene failed with
`OSError: [Errno 5] Input/output error: 'ffmpeg'` in <1s, while `/api/generate/content`
requests flooded and timed out at 45s. Dev never reproduced it because dev isn't under
the same memory/cgroup pressure.

**Rule:** Any subprocess that runs ffmpeg (or any external binary) from the AI server
must be spawned via `posix_spawn`, never `fork()+exec()`. Use the shared helper
`ai_model/video/ffmpeg_util.py::run_ffmpeg()`.

**Why:** `subprocess.run(cmd, capture_output=True)` with a bare `"ffmpeg"` command and
default `close_fds=True` makes CPython use `fork()`. Forking a process that holds a large
in-memory model forces the kernel to account for a full copy of the address space; under
the container's memory cgroup limit this fails (EIO/ENOMEM). `posix_spawn` uses
vfork/clone semantics that share the parent address space, so it never duplicates the
model's memory.

**How to apply:** CPython only takes the `posix_spawn` path when ALL hold: absolute
executable path (`os.path.dirname(exe)` truthy — resolve via `shutil.which`), `close_fds=False`,
no `pass_fds`, `cwd=None`, and NO PIPE redirections (use `DEVNULL` for stdout + a temp
file for stderr instead of `capture_output=True`). Verify with a monkeypatched
`os.posix_spawn` counter. The helper also retries only on transient errnos
(EIO/ENOMEM/EAGAIN) and fails fast on permanent ones (ENOENT/EACCES).

## posix_spawn alone was NOT enough — the deeper cause is duplicate model loads
If ffmpeg EIO recurs in prod even though `run_ffmpeg` already uses posix_spawn (verified:
posix_spawn=1, fork=0 with the helper's exact args), the real cause is **two copies of the
~1.7GB model loaded at once**, which alone tips the cgroup so even a vfork'd execve of the
ffmpeg binary fails with EIO. How it happens: the API server can run >1 cluster *primary*
in prod (deploy logs showed `[Cluster] Primary` on BOTH 8080 and 3000), each independently
runs `ensurePythonServer()`. uvicorn starts the model load before port 9878 is reliably bound,
so two near-simultaneous spawns both pass the "is port open?" check and both load the model.
**Fix in place:** a single-instance guard in `server.py`'s `__main__` (before `uvicorn.run`)
takes an exclusive `fcntl.flock` on `/tmp/maxcore_model_<port>.lock`; a duplicate waits for the
owner to bind the port then `os._exit(0)`, so exactly one model ever loads. Must stay in
`__main__` (NOT module top-level) or uvicorn's re-import of `server:app` would lock out the winner.
**Remaining follow-up (not done):** also stop the 2nd API primary from managing Python at the
Node layer — the flock is containment, not a control-plane fix.

## ffmpeg must be a DECLARED deployment dependency (the real prod EIO cause)
Prod video failed every scene with `OSError: [Errno 5] Input/output error: '/nix/store/...ffmpeg-full.../bin/ffmpeg'` while content/image worked — even though `run_ffmpeg` already uses posix_spawn (verified posix_spawn=1/fork=0) and local renders pass end-to-end. Root cause: **`replit.nix` declared only `pkgs.dejavu_fonts`, NOT ffmpeg.** Dev works because the dev shell provides ffmpeg, so `shutil.which("ffmpeg")` resolves the absolute /nix/store path at import; but the GCE deployment closure didn't include it, so the path exists as a dirent yet its content isn't materialized → execve faults with **EIO (not ENOENT)**.
- **Tell-tale:** EIO with the FULL absolute /nix/store binary path = dirent-present-but-unmaterialized = a binary the code execs that isn't a declared deploy dep. (ENOENT would mean the path itself is gone.) This is independent of memory pressure — don't chase the cgroup/duplicate-model theory first when replit.nix is missing the dep.
- **Fix:** `installSystemDependencies(["ffmpeg-full"])` (adds `pkgs.ffmpeg-full` to replit.nix — direct edits to replit.nix are blocked, must go through package management). Use `ffmpeg-full` to match the dev codec set the renderer relies on.
- **How to apply:** any external binary the server execs in prod (ffmpeg, etc.) MUST be a declared dep in replit.nix, not just whatever the dev shell happens to provide. Takes effect only on the NEXT publish — an already-published deployment stays broken until re-deployed.

## Cold-start ffmpeg timeout (transient, NOT the EIO bug)
The audio encode (`_render_audio_clip` → `run_ffmpeg` libmp3lame) normally finishes in <1s
and a full clip job in ~5s on a warm server. But an audio request that lands in the first
~15–20s after `restart_workflow` (model just loaded, `uptime_seconds` tiny) can hit the 45s
ffmpeg timeout because model warmup saturates the CPU and starves the ffmpeg subprocess.
This surfaces as job `status=error` with "...timed out after 45s" — handled correctly (no
false "done", WAV cleaned up via `try/finally`). It is transient: the same durations pass in
~5s once the server is warm. Don't mistake it for the posix_spawn/EIO issue above and don't
"fix" it by gutting the timeout — just let cold requests fail explicitly and retry warm.

## CPU-contention starvation of short encodes (July 2026)
Short audio ffmpeg encodes (<1s warm) hit their timeout in prod when a concurrent video render saturated CPU (a 4s video encode stretched to 38s). Fix: `run_ffmpeg(niceness=10)` prefixes the ABSOLUTE `nice` binary (never preexec_fn — that forces fork() and reintroduces EIO); video scene/fallback/composite encodes run at nice+10 (timeouts 90/90/180s), scene render threads self-deprioritize via os.setpriority on native tid (never-raise), audio encode timeout 45→120s as headroom. Verified: audio render done in 7s while video render in flight. Prefix must be applied AFTER FFMPEG_BIN substitution so cmd[0] stays absolute for posix_spawn.
