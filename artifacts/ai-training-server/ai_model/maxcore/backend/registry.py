"""Backend registry — register and resolve named backends.

The CPU backend is the only one runnable on this host; GPU/cluster/ASIC are
registered as plug-points so callers can discover them and get a clear,
contract-bearing error if selected before they exist.
"""
from __future__ import annotations

from typing import Callable

from .base import Backend
from .cpu_backend import CPUBackend
from .future_backends import ASICBackend, ClusterBackend, GPUBackend

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
register("gpu", lambda **kw: GPUBackend())
register("cluster", lambda **kw: ClusterBackend())
register("asic", lambda **kw: ASICBackend())
