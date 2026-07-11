"""Native SIMD compiler — the Path-A substrate.

Compiles C source to a shared library that gcc auto-vectorizes to the host's
widest SIMD (AVX-512 here), caches the ``.so`` keyed by source+flags+compiler
version, and loads it via ctypes. Everything is never-raise: if no compiler is
present or a build fails, ``compile()`` returns ``None`` and the caller falls
back to numpy. This is a *real* compiled CPU kernel path — an honest speedup over
interpreting ops as numpy call chains — but it is still CPU, not GPU hardware.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from typing import List, Optional, Sequence


class NativeCompiler:
    def __init__(self, extra_flags: Optional[Sequence[str]] = None,
                 use_openmp: bool = False, fast_math: bool = True,
                 cache_dir: Optional[str] = None):
        self.cc = shutil.which("gcc") or shutil.which("cc")
        self.available = self.cc is not None
        self.use_openmp = use_openmp
        self.fast_math = fast_math
        self.cc_version = self._probe_version()
        self.last_error: Optional[str] = None

        # NOTE: -march=native is silently stripped on NixOS (NIX_ENFORCE_NO_NATIVE),
        # which quietly caps kernels at baseline SSE. We instead detect the CPU's
        # ISA from /proc/cpuinfo and pass explicit -mavx512*/-mavx2 flags, which
        # are honored. This is what actually unlocks AVX-512 width.
        self.isa_flags = self._detect_isa_flags()
        self.base_flags: List[str] = ["-O3", "-funroll-loops", "-shared", "-fPIC",
                                      *self.isa_flags]
        if fast_math:
            self.base_flags.append("-ffast-math")
        if extra_flags:
            self.base_flags.extend(extra_flags)

        self.cache_dir = cache_dir or os.path.join(
            tempfile.gettempdir(), "digital_gpu_native")
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except OSError as e:
            # cache dir unusable -> disable native path; callers fall back to numpy
            self.available = False
            self.last_error = f"cannot create cache dir {self.cache_dir!r}: {e}"

    @staticmethod
    def _detect_isa_flags() -> List[str]:
        """Return explicit SIMD ISA flags for the host CPU (widest available)."""
        try:
            with open("/proc/cpuinfo") as f:
                info = f.read()
        except OSError:
            return []
        cpu_flags: set = set()
        for line in info.splitlines():
            if line.startswith(("flags", "Features")):
                cpu_flags = set(line.split(":", 1)[1].split())
                break
        if "avx512f" in cpu_flags:
            out = ["-mavx512f"]
            out += [f"-m{x}" for x in ("avx512bw", "avx512vl", "avx512dq")
                    if x in cpu_flags]
            if "fma" in cpu_flags:
                out.append("-mfma")
            return out
        if "avx2" in cpu_flags:
            return ["-mavx2"] + (["-mfma"] if "fma" in cpu_flags else [])
        if "avx" in cpu_flags:
            return ["-mavx"]
        return []

    def _probe_version(self) -> str:
        if not self.available:
            return ""
        try:
            r = subprocess.run([self.cc, "-dumpversion"],
                               capture_output=True, text=True, timeout=15)
            return r.stdout.strip()
        except Exception:
            return ""

    def _flags(self) -> List[str]:
        flags = list(self.base_flags)
        if self.use_openmp:
            flags.append("-fopenmp")
        return flags

    def compile(self, source: str,
                link: Sequence[str] = ("-lm",)) -> Optional[ctypes.CDLL]:
        """Compile ``source`` to a cached shared lib and return the loaded CDLL,
        or ``None`` on any failure (with the reason in ``last_error``)."""
        if not self.available:
            self.last_error = self.last_error or "no C compiler (gcc/cc) on PATH"
            return None

        # Truly never-raise: every filesystem/compile/load step is inside the
        # try, so any failure (unwritable cache dir, compile error, bad load)
        # returns None with a reason and lets the caller fall back to numpy.
        try:
            flags = self._flags()
            sig = "|".join([source, self.cc_version, *flags, *link])
            key = hashlib.sha1(sig.encode()).hexdigest()[:16]
            so_path = os.path.join(self.cache_dir, f"lib_{key}.so")

            if not os.path.exists(so_path):
                # Invocation-unique temp names (via mkstemp) so concurrent
                # compiles of the same key in one process can't collide.
                fd_c, c_path = tempfile.mkstemp(
                    suffix=".c", prefix=f"src_{key}_", dir=self.cache_dir)
                fd_so, tmp_so = tempfile.mkstemp(
                    suffix=".so", prefix=f"lib_{key}_", dir=self.cache_dir)
                os.close(fd_so)
                try:
                    with os.fdopen(fd_c, "w") as f:
                        f.write(source)
                    cmd = [self.cc, *flags, "-o", tmp_so, c_path, *link]
                    subprocess.run(cmd, check=True, capture_output=True,
                                   text=True, timeout=120)
                    os.replace(tmp_so, so_path)          # atomic publish
                finally:
                    for p in (c_path, tmp_so):           # tmp_so gone after replace
                        try:
                            if os.path.exists(p):
                                os.remove(p)
                        except OSError:
                            pass

            return ctypes.CDLL(so_path)
        except Exception as e:                           # never propagate
            self.last_error = getattr(e, "stderr", None) or str(e)
            return None
