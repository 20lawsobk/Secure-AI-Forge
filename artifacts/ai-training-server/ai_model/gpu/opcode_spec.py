"""Opcode contract — the declarative spec every digital-GPU kernel is checked against.

An ``OpcodeSpec`` is *documentation the runtime can enforce*: shapes, dtypes,
numeric profile, determinism and — crucially — whether the op runs on real
hardware. On this host nothing does, so every spec here sets
``is_hardware_execution=False``.

Honesty rules baked into the contract
--------------------------------------
* ``numeric_profile`` describes the *numerics being modelled* (e.g. ``fp8_mixed``),
  not the substrate. A profile of ``fp8_mixed`` means "fp8 rounding emulated on
  CPU" (see :mod:`ai_model.gpu.precision`), NOT fp8 tensor-core execution.
* ``target_arch`` (e.g. ``"sm_102"``) records the architecture whose behaviour an
  op *models*. It is a label of intent, never a claim that this box is that GPU.
* ``is_hardware_execution`` is the single source of truth for "does this actually
  run on the named silicon". It is ``False`` for everything here. The graph
  scheduler refuses to run a spec that claims ``True`` unless a real device
  backend is present, so a mislabelled op fails loudly instead of masquerading.

``flop_formula``
----------------
An optional callable ``(ins: dict[str, np.ndarray]) -> float`` that derives the
analytic FLOP count from live input arrays. The scheduler calls it (via
``_derived_flops``) to produce telemetry; a missing formula silently records 0
rather than crashing (never-raise). Derived from each op's roofline arithmetic
intensity: elementwise ops at ~1–4 FLOPs/byte, GEMM at 2MNK, norms at 5×N.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np

# Numeric profile names. "*_mixed" == that format's rounding emulated on CPU
# (compute grid modelled), NOT hardware execution in that format.
NumericProfile = str  # one of the keys below; kept a str for forward-compat

VALID_PROFILES = ("fp32_strict", "fp16_mixed", "bf16_mixed", "fp8_mixed")


@dataclass(frozen=True)
class OpcodeSpec:
    name: str
    version: int
    input_shapes: Dict[str, Tuple]
    output_shapes: Dict[str, Tuple]
    dtypes: Dict[str, str]
    numeric_profile: NumericProfile
    deterministic: bool
    doc: str
    # Truth about the substrate. False everywhere on this CPU-only host.
    is_hardware_execution: bool = False
    # Architecture whose numerics/behaviour this op *models* (label of intent).
    target_arch: Optional[str] = None
    inputs: Tuple[str, ...] = field(default_factory=tuple)
    # Analytic FLOP formula derived from input shapes. Callable receives the
    # live ``ins`` dict (arg_name → ndarray) and returns a float FLOP count.
    # None = unknown; scheduler records flops=0 and flops_unknown=True.
    flop_formula: Optional[Callable] = None

    @property
    def key(self) -> str:
        return f"{self.name}:v{self.version}"

    def describe(self) -> str:
        substrate = (
            f"executes on {self.target_arch}" if self.is_hardware_execution
            else f"CPU numerics model"
            + (f" of {self.target_arch}" if self.target_arch else "")
        )
        return f"{self.key} [{self.numeric_profile}, {substrate}]"


def _spec(**kw) -> OpcodeSpec:
    s = OpcodeSpec(**kw)
    if s.numeric_profile not in VALID_PROFILES:
        raise ValueError(
            f"{s.key}: unknown numeric_profile {s.numeric_profile!r}; "
            f"valid: {VALID_PROFILES}"
        )
    return s


OPCODES: Dict[str, OpcodeSpec] = {}


def register(spec: OpcodeSpec) -> None:
    OPCODES[spec.key] = spec


def get_spec(opcode: str) -> OpcodeSpec:
    """Resolve an opcode key. Accepts ``"gemm"`` (latest version) or ``"gemm:v1"``.

    Raises :class:`~ai_model.gpu.digital_gpu.InvalidOpcodeError` if unknown, so the
    scheduler surfaces a bad opcode as a typed, catchable error.
    """
    from ai_model.gpu.digital_gpu import InvalidOpcodeError

    if ":" in opcode:
        spec = OPCODES.get(opcode)
        if spec is None:
            raise InvalidOpcodeError(
                f"unknown opcode {opcode!r}. registered: {sorted(OPCODES)}")
        return spec
    # bare name -> highest registered version
    candidates = [s for s in OPCODES.values() if s.name == opcode]
    if not candidates:
        raise InvalidOpcodeError(
            f"unknown opcode {opcode!r}. registered: {sorted(OPCODES)}")
    return max(candidates, key=lambda s: s.version)


# ─────────────────────────────────────────────────────────────────────────────
# Base opcode set (DigitalGPU / SIMDCore level)
# ─────────────────────────────────────────────────────────────────────────────

register(_spec(
    name="gemm", version=1,
    inputs=("A", "B"),
    input_shapes={"A": ("M", "K"), "B": ("K", "N")},
    output_shapes={"C": ("M", "N")},
    dtypes={"A": "fp32", "B": "fp32", "C": "fp32"},
    numeric_profile="fp32_strict", deterministic=True,
    doc="Tiled matrix multiply C = A @ B (full fp32).",
    flop_formula=lambda ins: 2.0 * ins["A"].shape[0] * ins["A"].shape[1] * ins["B"].shape[1],
))

register(_spec(
    name="add", version=1,
    inputs=("A", "B"),
    input_shapes={"A": ("*",), "B": ("*",)},
    output_shapes={"C": ("*",)},
    dtypes={"A": "fp32", "B": "fp32", "C": "fp32"},
    numeric_profile="fp32_strict", deterministic=True,
    doc="Elementwise C = A + B.",
    flop_formula=lambda ins: float(ins["A"].size),
))

register(_spec(
    name="softmax", version=1,
    inputs=("X",),
    input_shapes={"X": ("*",)},
    output_shapes={"Y": ("*",)},
    dtypes={"X": "fp32", "Y": "fp32"},
    numeric_profile="fp32_strict", deterministic=True,
    doc="Numerically-stable softmax over the last axis.",
    # Roofline: exp + normalise ≈ 5 FLOPs/element; memory-bound at ~4 FLOPs/byte.
    flop_formula=lambda ins: 5.0 * float(ins["X"].size),
))

register(_spec(
    name="attention", version=1,
    inputs=("Q", "K", "V"),
    input_shapes={"Q": ("B", "T", "D"), "K": ("B", "T", "D"), "V": ("B", "T", "D")},
    output_shapes={"O": ("B", "T", "D")},
    dtypes={"Q": "fp32", "K": "fp32", "V": "fp32", "O": "fp32"},
    numeric_profile="fp32_strict", deterministic=True,
    doc="Scaled-dot-product attention (full fp32).",
    flop_formula=lambda ins: (
        4.0 * float(np.prod(ins["Q"].shape[:-2])) * ins["Q"].shape[-2] ** 2 * ins["Q"].shape[-1]
    ),
))

register(_spec(
    name="conv2d", version=1,
    inputs=("X", "W"),
    input_shapes={"X": ("N", "C", "H", "W"), "W": ("F", "C", "KH", "KW")},
    output_shapes={"Y": ("N", "F", "OH", "OW")},
    dtypes={"X": "fp32", "W": "fp32", "Y": "fp32"},
    numeric_profile="fp32_strict", deterministic=True,
    doc="2D convolution via im2col + gemm (reference).",
    flop_formula=lambda ins: (
        2.0 * ins["X"].shape[0] * ins["W"].shape[0]
        * ins["W"].shape[1] * ins["W"].shape[2] * ins["W"].shape[3]
        * ins["X"].shape[2] * ins["X"].shape[3]
    ),
))

register(_spec(
    name="flash_attention_fp8_sm102", version=1,
    inputs=("Q", "K", "V"),
    input_shapes={"Q": ("B", "H", "T", "D"), "K": ("B", "H", "T", "D"),
                  "V": ("B", "H", "T", "D")},
    output_shapes={"O": ("B", "H", "T", "D")},
    dtypes={"Q": "fp8", "K": "fp8", "V": "fp8", "O": "fp16"},
    numeric_profile="fp8_mixed", deterministic=True,
    is_hardware_execution=False,          # <- the truth: modelled on CPU
    target_arch="sm_102",                 # <- the intent it models
    doc=("FlashAttention with fp8 inputs / fp16 accumulate / fp32 softmax, "
         "MODELLED on CPU (see precision.flash_attention_fp8_model). This "
         "reproduces sm_102 fp8 *numerics* for error study; it does not execute "
         "on sm_102 hardware and is not faster than the fp32 path here."),
    flop_formula=lambda ins: (
        4.0 * float(np.prod(ins["Q"].shape[:-2])) * ins["Q"].shape[-2] ** 2 * ins["Q"].shape[-1]
    ),
))

# ─────────────────────────────────────────────────────────────────────────────
# HyperGPU / HyperSIMDCore op set
# FLOP formulas derived from roofline arithmetic intensity analysis:
#   Elementwise (gelu, silu, relu): memory-bound, ~1–8 FLOPs/byte
#   Norms (layer_norm, batch_norm): memory-bound, ~5 FLOPs/element
#   Conv3d, gemm_batched, grouped_gemm: 2*M*N*K per group
#   fused_attention_norm: flash-attn + layer-norm
# ─────────────────────────────────────────────────────────────────────────────

register(_spec(
    name="gelu", version=1,
    inputs=("X",),
    input_shapes={"X": ("*",)},
    output_shapes={"Y": ("*",)},
    dtypes={"X": "fp32", "Y": "fp32"},
    numeric_profile="fp32_strict", deterministic=True,
    doc="GeLU activation (tanh approximation). Memory-bound: ~8 FLOPs/element.",
    # tanh approx: cube + 3 muls + 2 adds + tanh + scale = ~8 FLOPs/element
    flop_formula=lambda ins: 8.0 * float(ins["X"].size),
))

register(_spec(
    name="silu", version=1,
    inputs=("X",),
    input_shapes={"X": ("*",)},
    output_shapes={"Y": ("*",)},
    dtypes={"X": "fp32", "Y": "fp32"},
    numeric_profile="fp32_strict", deterministic=True,
    doc="SiLU activation x*sigmoid(x). Memory-bound: ~4 FLOPs/element.",
    # neg + exp + add + div + mul = 4 FLOPs/element
    flop_formula=lambda ins: 4.0 * float(ins["X"].size),
))

register(_spec(
    name="relu", version=1,
    inputs=("X",),
    input_shapes={"X": ("*",)},
    output_shapes={"Y": ("*",)},
    dtypes={"X": "fp32", "Y": "fp32"},
    numeric_profile="fp32_strict", deterministic=True,
    doc="ReLU activation max(0, X). Memory-bound: 1 FLOPs/element.",
    flop_formula=lambda ins: 1.0 * float(ins["X"].size),
))

register(_spec(
    name="layer_norm", version=1,
    inputs=("X", "gamma", "beta"),
    input_shapes={"X": ("*",), "gamma": ("D",), "beta": ("D",)},
    output_shapes={"Y": ("*",)},
    dtypes={"X": "fp32", "gamma": "fp32", "beta": "fp32", "Y": "fp32"},
    numeric_profile="fp32_strict", deterministic=True,
    doc="Layer normalisation. Memory-bound: ~5 FLOPs/element (mean+var+norm+scale+shift).",
    flop_formula=lambda ins: 5.0 * float(ins["X"].size),
))

register(_spec(
    name="batch_norm", version=1,
    inputs=("X", "gamma", "beta"),
    input_shapes={"X": ("N", "C", "*"), "gamma": ("C",), "beta": ("C",)},
    output_shapes={"Y": ("N", "C", "*")},
    dtypes={"X": "fp32", "gamma": "fp32", "beta": "fp32", "Y": "fp32"},
    numeric_profile="fp32_strict", deterministic=True,
    doc=("Batch normalisation (training mode in the scheduler; running stats "
         "are not tracked in the DAG — use the HyperGPU API directly for eval "
         "mode with running statistics)."),
    flop_formula=lambda ins: 5.0 * float(ins["X"].size),
))

register(_spec(
    name="conv3d", version=1,
    inputs=("X", "W"),
    input_shapes={"X": ("N", "C", "D", "H", "W"), "W": ("F", "C", "KD", "KH", "KW")},
    output_shapes={"Y": ("N", "F", "OD", "OH", "OW")},
    dtypes={"X": "fp32", "W": "fp32", "Y": "fp32"},
    numeric_profile="fp32_strict", deterministic=True,
    doc="3D convolution via im2col + gemm. FLOPs = 2*N*F*C*kD*kH*kW*OD*OH*OW.",
    flop_formula=lambda ins: (
        2.0 * ins["X"].shape[0] * ins["W"].shape[0]
        * ins["W"].shape[1] * ins["W"].shape[2] * ins["W"].shape[3] * ins["W"].shape[4]
        * max(ins["X"].shape[2] - ins["W"].shape[2] + 1, 1)
        * max(ins["X"].shape[3] - ins["W"].shape[3] + 1, 1)
        * max(ins["X"].shape[4] - ins["W"].shape[4] + 1, 1)
    ),
))

register(_spec(
    name="gemm_batched", version=1,
    inputs=("A", "B"),
    input_shapes={"A": ("G", "M", "K"), "B": ("G", "K", "N")},
    output_shapes={"C": ("G", "M", "N")},
    dtypes={"A": "fp32", "B": "fp32", "C": "fp32"},
    numeric_profile="fp32_strict", deterministic=True,
    doc="Batched GEMM over leading batch dims. FLOPs = 2*G*M*N*K.",
    flop_formula=lambda ins: (
        2.0 * float(np.prod(ins["A"].shape[:-2]))
        * ins["A"].shape[-2] * ins["A"].shape[-1] * ins["B"].shape[-1]
    ),
))

register(_spec(
    name="grouped_gemm", version=1,
    inputs=("A", "B"),
    input_shapes={"A": ("G", "M", "K"), "B": ("G", "K", "N")},
    output_shapes={"C": ("G", "M", "N")},
    dtypes={"A": "fp32", "B": "fp32", "C": "fp32"},
    numeric_profile="fp32_strict", deterministic=True,
    doc=("Grouped GEMM: G independent M×K @ K×N products on uniform-shape groups. "
         "Pass as stacked 3D tensors [G, M, K] and [G, K, N]. For variable-shape "
         "groups use the HyperGPU.grouped_gemm() API directly."),
    flop_formula=lambda ins: (
        2.0 * ins["A"].shape[0] * ins["A"].shape[1] * ins["A"].shape[2] * ins["B"].shape[2]
    ),
))

register(_spec(
    name="mixed_precision_gemm", version=1,
    inputs=("A", "B"),
    input_shapes={"A": ("M", "K"), "B": ("K", "N")},
    output_shapes={"C": ("M", "N")},
    dtypes={"A": "fp32", "B": "fp32", "C": "fp32"},
    numeric_profile="fp16_mixed", deterministic=True,
    doc=("GEMM with fp16 operand rounding + fp32 accumulate (numerics model). "
         "On this host dispatches to FP32 SGEMM by default; set "
         "MAXCORE_EMULATE_FP16=1 for bit-exact fp16 rounding for error study."),
    flop_formula=lambda ins: 2.0 * ins["A"].shape[0] * ins["A"].shape[1] * ins["B"].shape[1],
))

register(_spec(
    name="fused_attention_norm", version=1,
    inputs=("Q", "K", "V", "gamma", "beta"),
    input_shapes={
        "Q": ("B", "T", "D"), "K": ("B", "T", "D"), "V": ("B", "T", "D"),
        "gamma": ("D",), "beta": ("D",),
    },
    output_shapes={"O": ("B", "T", "D")},
    dtypes={"Q": "fp32", "K": "fp32", "V": "fp32",
            "gamma": "fp32", "beta": "fp32", "O": "fp32"},
    numeric_profile="fp32_strict", deterministic=True,
    doc="Flash-attention followed by layer-norm in a single fused op.",
    flop_formula=lambda ins: (
        # attention: 4*B*T²*D
        4.0 * float(np.prod(ins["Q"].shape[:-2])) * ins["Q"].shape[-2] ** 2 * ins["Q"].shape[-1]
        # layer_norm: 5*B*T*D
        + 5.0 * float(ins["Q"].size)
    ),
))
