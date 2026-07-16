from __future__ import annotations
import os
import sys
import time
import errno
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List

FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"
NICE_BIN = shutil.which("nice")  # absolute path keeps posix_spawn eligibility

if not os.path.isabs(FFMPEG_BIN):
    print(
        f"[VideoRender][WARN] ffmpeg resolved to a non-absolute path ({FFMPEG_BIN!r}); "
        "posix_spawn eligibility (and fork-avoidance under memory pressure) is not guaranteed.",
        file=sys.stderr,
    )

# Transient spawn failures worth retrying; permanent errors (ENOENT, EACCES, ...) fail fast.
_RETRYABLE_ERRNOS = {errno.EIO, errno.ENOMEM, errno.EAGAIN}


@dataclass
class FfmpegResult:
    returncode: int
    stderr: str


def run_ffmpeg(
    cmd: List[str],
    timeout: float = 60.0,
    retries: int = 2,
    niceness: int = 0,
) -> FfmpegResult:
    """Run an ffmpeg command resiliently under heavy memory pressure.

    The default ``subprocess.run(cmd, capture_output=True)`` with a bare
    ``"ffmpeg"`` command and ``close_fds=True`` forces CPython to use
    ``fork()`` + ``exec()``. Forking a process that holds the large in-memory
    transformer model can fail with ``OSError`` (EIO / ENOMEM) inside a
    memory-constrained container because the kernel must account for a full
    copy of the parent address space.

    This helper avoids ``fork()`` by satisfying CPython's ``posix_spawn()``
    eligibility conditions:
      * absolute executable path (``os.path.dirname(executable)`` is truthy)
      * ``close_fds=False``
      * no PIPE redirections (stdout to DEVNULL, stderr to a temp file)
    ``posix_spawn()`` uses ``vfork``/``clone`` semantics that share the parent
    address space, so it does not duplicate the model's memory and is immune to
    the overcommit failure. Transient spawn failures are retried with backoff.
    """
    if cmd and cmd[0] == "ffmpeg":
        cmd = [FFMPEG_BIN] + list(cmd[1:])

    # niceness>0 deprioritizes long-running encodes (video scenes/composites)
    # so short interactive encodes (audio clips, voiceover) are never CPU-starved
    # into their timeout when a video render is in flight. Implemented by
    # prefixing the absolute `nice` binary — NOT preexec_fn, which would force
    # CPython back onto fork() and reintroduce the EIO-under-memory-pressure bug.
    if niceness > 0 and NICE_BIN and cmd and os.path.isabs(cmd[0]):
        cmd = [NICE_BIN, "-n", str(niceness)] + list(cmd)

    last_exc: BaseException | None = None
    for attempt in range(retries + 1):
        err_path = None
        # CPU starvation on the 2-CPU prod VM (model inference + concurrent
        # generations) can make even a trivial decode blow its budget.  Each
        # retry doubles the timeout so a starved-but-healthy ffmpeg gets room
        # to finish instead of failing the whole render.
        attempt_timeout = timeout * (2 ** attempt)
        try:
            fd, err_path = tempfile.mkstemp(suffix=".ffmpeg.err")
            with os.fdopen(fd, "w+") as err_file:
                proc = subprocess.run(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=err_file,
                    close_fds=False,
                    timeout=attempt_timeout,
                )
                err_file.seek(0)
                stderr_text = err_file.read()
            return FfmpegResult(returncode=proc.returncode, stderr=stderr_text)
        except subprocess.TimeoutExpired as exc:
            last_exc = exc
            print(
                f"[VideoRender][WARN] ffmpeg timed out after {attempt_timeout}s "
                f"(attempt {attempt + 1}/{retries + 1}) — likely CPU starvation; "
                f"retrying with doubled budget",
                file=sys.stderr,
            )
            time.sleep(0.5 * (attempt + 1))
        except OSError as exc:
            last_exc = exc
            if exc.errno not in _RETRYABLE_ERRNOS:
                # Permanent error (ENOENT, EACCES, ...) — no point retrying.
                raise
            print(
                f"[VideoRender][WARN] ffmpeg spawn OSError (attempt {attempt + 1}/{retries + 1}): {exc}",
                file=sys.stderr,
            )
            time.sleep(0.5 * (attempt + 1))
        finally:
            if err_path and os.path.exists(err_path):
                try:
                    os.remove(err_path)
                except OSError:
                    pass

    raise last_exc if last_exc else RuntimeError("ffmpeg failed to spawn")
