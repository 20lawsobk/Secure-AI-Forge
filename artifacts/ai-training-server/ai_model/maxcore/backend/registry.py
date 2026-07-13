"""Backend registry — register and resolve named backends.

The Digital GPU backend is the primary backend on this host; GPU/cluster/ASIC
are registered as plug-points so callers can discover them and get a clear,
contract-bearing error if selected before they exist.
"""
from __future__ import annotations

from typing import Callable

from .base import Backend
from .cpu_backend import DigitalGPUBackend
from .device_backend import GPUBackend
from .future_backends import ASICBackend, ClusterBackend

_FACTORIES: dict[str, Callable[..., Backend]] = {}
_INSTANCES: dict[tuple, Backend] = {}


def register(name: str, factory: Callable[..., Backend]) -> None:
    _FACTORIES[name] = factory


def get_backend(name: str = "digital_gpu", **kwargs) -> Backend:
    key = (name, tuple(sorted(kwargs.items())))
    inst = _INSTANCES.get(key)
    if inst is None:
        if name not in _FACTORIES:
            raise ValueError(f"unknown backend '{name}'. registered: {available()}")
        inst = _FACTORIES[name](**kwargs)
        _INSTANCES[key] = inst
    return inst


def available() -> list[str]:
    return sorted(_FACTORIES.keys())


def select_backend(
    prefer: tuple[str, ...] = ("gpu", "digital_gpu"),
    **kwargs,
) -> Backend:
    """Return the highest-preference backend that can *actually* run on this host.

    This is the honest auto-selector the runtime uses to pick hardware without
    the caller having to know what is present:

      * On a CUDA host it returns the real ``GPUBackend`` (torch on the device).
      * On this host ``gpu`` reports ``is_available() == False``, so it falls
        through to the ``DigitalGPUBackend`` — the in-house engine.

    It walks ``prefer`` in order and returns the first backend whose
    ``is_available()`` is True. It NEVER returns a backend that cannot run and
    never pretends one tier is another — the returned object's ``name``/``info()``
    tell the truth about what you actually got.

    ``kwargs`` are forwarded to the preferred backends (e.g. ``device="cpu"`` to
    validate the GPU code path via torch-CPU). The final Digital GPU fallback is
    always constructed with no kwargs so a GPU-only kwarg can't break it.

    Honesty about failures: only *expected* "this backend doesn't apply here"
    conditions are skipped — an unknown backend name (``ValueError``) or a
    preference-specific kwarg the constructor rejects (``TypeError``). A backend
    reports hardware absence by returning ``is_available() == False`` (it does
    not raise). If an availability probe or constructor raises something
    *unexpected*, that is a real fault — it is allowed to propagate so an
    operational bug can never masquerade as a silent downgrade.
    """
    tried: list[str] = []
    for name in prefer:
        tried.append(name)
        try:
            backend = get_backend(name, **kwargs)
        except (ValueError, TypeError):
            continue
        if backend.is_available():
            return backend
    # Guaranteed honest fallback: the Digital GPU backend always runs.
    dgpu = get_backend("digital_gpu")
    if dgpu.is_available():
        return dgpu
    raise RuntimeError(
        f"select_backend: none of the preferred backends {tried} are runnable "
        f"on this host, and the Digital GPU fallback is unavailable. runtime "
        f"availability: {available_runtime()}"
    )


def available_runtime() -> dict[str, bool]:
    """Map of backend name -> whether it can actually run on this host."""
    out: dict[str, bool] = {}
    for name, factory in _FACTORIES.items():
        try:
            out[name] = bool(factory().is_available())
        except Exception:
            out[name] = False
    return out


register("digital_gpu", lambda **kw: DigitalGPUBackend(**kw))
register("cpu",         lambda **kw: DigitalGPUBackend(**kw))   # alias — same engine
register("gpu",         lambda **kw: GPUBackend(**kw))
register("cluster",     lambda **kw: ClusterBackend())
register("asic",        lambda **kw: ASICBackend())
