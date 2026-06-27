"""Tests for the runtime thread/hardware configuration.

Runnable two ways:
  * pytest:  uv run pytest ai_model/maxcore/tests/test_hardware.py
  * direct:  uv run python ai_model/maxcore/tests/test_hardware.py

Validates the pure thread-planning logic (single vs multi-worker, reserve, never
below 1) and that configuration respects pre-existing operator overrides.
"""
from __future__ import annotations

import os
import sys

_SERVER_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

from ai_model.maxcore.hardware import (  # noqa: E402
    _BLAS_ENV_VARS,
    configure_blas_threads,
    cpu_count,
    plan_blas_threads,
)


def test_single_worker_uses_all_cores():
    assert plan_blas_threads(16, 1) == 16
    assert plan_blas_threads(2, 1) == 2


def test_multi_worker_avoids_oversubscription():
    assert plan_blas_threads(16, 4) == 4
    assert plan_blas_threads(16, 5) == 3   # floor division
    assert plan_blas_threads(2, 4) == 1    # never below 1


def test_reserve_keeps_cores_free():
    assert plan_blas_threads(16, 1, reserve=2) == 14
    assert plan_blas_threads(2, 1, reserve=8) == 1


def test_invalid_args_raise():
    for bad in (lambda: plan_blas_threads(0, 1), lambda: plan_blas_threads(4, 0)):
        try:
            bad()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_configure_respects_existing_then_overrides():
    saved = {v: os.environ.get(v) for v in _BLAS_ENV_VARS}
    try:
        for v in _BLAS_ENV_VARS:
            os.environ.pop(v, None)
        os.environ["OMP_NUM_THREADS"] = "1"  # operator pin
        eff = configure_blas_threads(num_workers=1)  # no override
        assert eff["OMP_NUM_THREADS"] == "1"  # pin respected
        assert eff["OPENBLAS_NUM_THREADS"] == str(cpu_count())  # others set
        eff2 = configure_blas_threads(num_workers=1, override=True)
        assert all(val == str(cpu_count()) for val in eff2.values())
    finally:
        for v, val in saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
