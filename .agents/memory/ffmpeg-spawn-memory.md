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
