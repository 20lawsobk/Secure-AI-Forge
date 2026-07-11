"""Backend registry — register and resolve named backends.

The CPU backend is the only one runnable on this host; GPU/cluster/ASIC are
registered as plug-points so callers can discover them and get a clear,
contract-bearing error if selected before they exist.
"""
from __future__ import annotations

from typing import Callable

from .base import Backend
from .cpu_backend import CPUBackend
from .device_backend import GPUBackend
from .future_backends import ASICBackend, ClusterBackend

_FACTORIES: dict[str, Callable[..., Backend]] = {}
_INSTANCES: dict[tuple, Backend] = {}


def register(name: str, factory: Callable[..., Backend]) -> None:
    _FACTORIES[name] = factory


def get_backend(name: str = "cpu", **kwargs) -> Backend:
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


def select_backend(prefer: tuple[str, ...] = ("gpu", "cpu"), **kwargs) -> Backend:
    """Return the highest-preference backend that can *actually* run on this host.

    This is the honest auto-selector the runtime uses to pick hardware without
    the caller having to know what is present:

      * On a CUDA host it returns the real ``GPUBackend`` (torch on the device).
      * On a CPU-only host (like this one) ``gpu`` reports ``is_available()`` ==
        False, so it falls through to the numpy ``CPUBackend``.

    It walks ``prefer`` in order and returns the first backend whose
    ``is_available()`` is True. It NEVER returns a backend that cannot run and
    never pretends CPU is a GPU — the returned object's ``name``/``info()`` tell
    the truth about what you actually got.

    ``kwargs`` are forwarded to the preferred backends (e.g. ``device="cpu"`` to
    validate the GPU code path via torch-CPU). The final CPU fallback is always
    constructed with no kwargs so a GPU-only kwarg can't break the safety net.

    Honesty about failures: only *expected* "this backend doesn't apply here"
    conditions are skipped — an unknown backend name (``ValueError``) or a
    preference-specific kwarg the constructor rejects (``TypeError``, e.g.
    ``device="cuda"`` passed to the CPU backend). A backend reports hardware
    absence by returning ``is_available() == False`` (it does not raise). If an
    availability probe or constructor raises something *unexpected*, that is a
    real fault — it is allowed to propagate rather than being silently downgraded
    to CPU, so an operational bug can never masquerade as "no GPU here."
    """
    tried: list[str] = []
    for name in prefer:
        tried.append(name)
        try:
            backend = get_backend(name, **kwargs)
        except (ValueError, TypeError):
            # Expected + benign: unknown backend name, or a preference-specific
            # kwarg this constructor doesn't accept. Not applicable -> next.
            continue
        # NOTE: is_available() is deliberately NOT wrapped. A truthful backend
        # returns False when its hardware is absent; if it *raises*, that's a
        # genuine fault we must surface, not mask as CPU fallback.
        if backend.is_available():
            return backend
    # Guaranteed honest fallback: the numpy CPU backend always runs. Build it
    # with no kwargs so a preference-specific kwarg (e.g. device="cuda") that
    # was skipped above can't also break the safety net.
    cpu = get_backend("cpu")
    if cpu.is_available():
        return cpu
    raise RuntimeError(
        f"select_backend: none of the preferred backends {tried} are runnable "
        f"on this host, and the CPU fallback is unavailable. runtime "
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


register("cpu", lambda **kw: CPUBackend(**kw))
register("gpu", lambda **kw: GPUBackend(**kw))
register("cluster", lambda **kw: ClusterBackend())
register("asic", lambda **kw: ASICBackend())
