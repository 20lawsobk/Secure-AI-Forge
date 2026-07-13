"""Backend abstraction — the contract every DigitalGPU backend must satisfy.

A backend implements the primitive kernels (gemm, attention, conv2d, mlp,
reduce, ...). The runtime dispatches IR ops to these methods, so all backends
are interchangeable behind the public API. The Digital GPU backend is the real
one on this host; GPU/cluster/ASIC backends are declared but raise until implemented.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Backend(ABC):
    name: str = "base"

    @abstractmethod
    def create_tensor(self, data: Any, dtype: str = "float32"): ...

    @abstractmethod
    def gemm(self, a, b, bias=None, activation=None): ...

    @abstractmethod
    def add(self, a, b): ...

    @abstractmethod
    def relu(self, x): ...

    @abstractmethod
    def softmax(self, x, axis: int = -1): ...

    @abstractmethod
    def attention(self, q, k, v, mask=None, causal: bool = False): ...

    @abstractmethod
    def conv2d(self, x, w, bias=None, stride: int = 1, padding: int = 0): ...

    @abstractmethod
    def mlp(self, x, w1, b1, w2, b2, activation: str = "relu"): ...

    @abstractmethod
    def reduce(self, x, op: str, axis, keepdims: bool = False): ...

    def is_available(self) -> bool:
        """Whether this backend can actually run on the current host."""
        return True

    def supports(self, op: str) -> bool:
        return True

    def info(self) -> dict:
        return {"name": self.name, "available": self.is_available()}
