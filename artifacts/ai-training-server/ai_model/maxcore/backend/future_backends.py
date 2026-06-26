"""Future backend plug-points — declared, contract-documented, NOT faked.

These backends are part of the architecture but are not implemented on this
CPU-only host. Per the "no placeholders / honest engineering" rule, they do not
silently pretend to work: ``is_available()`` returns ``False`` and every kernel
raises ``NotImplementedError`` carrying the contract a real implementation must
satisfy. They exist so the registry, runtime, and API are already
backend-agnostic the day a real device backend is added.
"""
from __future__ import annotations

from .base import Backend


class _FutureBackend(Backend):
    name = "future"
    contract = "Not implemented on this host."

    def is_available(self) -> bool:
        return False

    def _nyi(self, op: str):
        raise NotImplementedError(
            f"{self.name} backend: '{op}' is not implemented. {self.contract}"
        )

    def create_tensor(self, *a, **k):
        self._nyi("create_tensor")

    def gemm(self, *a, **k):
        self._nyi("gemm")

    def add(self, *a, **k):
        self._nyi("add")

    def relu(self, *a, **k):
        self._nyi("relu")

    def softmax(self, *a, **k):
        self._nyi("softmax")

    def attention(self, *a, **k):
        self._nyi("attention")

    def conv2d(self, *a, **k):
        self._nyi("conv2d")

    def mlp(self, *a, **k):
        self._nyi("mlp")

    def reduce(self, *a, **k):
        self._nyi("reduce")


class GPUBackend(_FutureBackend):
    name = "gpu"
    contract = (
        "Implement kernels via Triton / cuBLAS / cuDNN with CUDA graphs and "
        "persistent kernels. Must match CPUBackend numerics within fp tolerance "
        "and honor the runtime's deterministic mode."
    )


class ClusterBackend(_FutureBackend):
    name = "cluster"
    contract = (
        "Partition the MaxCoreGraph across nodes over an RPC layer, shard the "
        "KV-cache, and reduce/all-reduce at BARRIER ops. Each shard runs a "
        "device backend locally."
    )


class ASICBackend(_FutureBackend):
    name = "asic"
    contract = (
        "Lower MaxCore IR to ASIC microcode driving DMA engines, on-chip "
        "KV-cache, and specialized tensor units."
    )
