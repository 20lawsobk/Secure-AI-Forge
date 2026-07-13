from .base import Backend
from .cpu_backend import DigitalGPUBackend, CPUBackend   # CPUBackend = alias
from .device_backend import GPUBackend
from .future_backends import ASICBackend, ClusterBackend
from .registry import (
    available,
    available_runtime,
    get_backend,
    register,
    select_backend,
)

__all__ = [
    "Backend",
    "DigitalGPUBackend",
    "CPUBackend",          # backwards-compatible alias
    "GPUBackend",
    "ClusterBackend",
    "ASICBackend",
    "get_backend",
    "select_backend",
    "register",
    "available",
    "available_runtime",
]
