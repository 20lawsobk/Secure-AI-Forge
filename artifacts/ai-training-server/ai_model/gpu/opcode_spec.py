"""Opcode contract — the declarative spec every digital-GPU kernel is checked against.

An ``OpcodeSpec`` is *documentation the runtime can enforce*: shapes, dtypes,
numeric profile, determinism and — crucially — whether the op runs on real
hardware. On this host nothing does, so every spec here sets
``is_hardware_execution=False``.

Honesty rules baked into the contract
-------------------------------------
* ``numeric_profile`` describes the *numerics being modelled* (e.g. ``fp8_mixed``),
  not the substrate. A profile of ``fp8_mixed`` means "fp8 rounding emulated on
  CPU" (see :mod:`ai_model.gpu.precision`), NOT fp8 tensor-core execution.
* ``target_arch`` (e.g. ``"sm_102"``) records the architecture whose behaviour an
  op *models*. It is a label of intent, never a claim that this box is that GPU.
* ``is_hardware_execution`` is the single source of truth for "does this actually
  run on the named silicon". It is ``False`` for everything here. The graph
  scheduler refuses to run a spec that claims ``True`` unless a real device
  backend is present, so a mislabelled op fails loudly instead of masquerading.

This is why the same name that :class:`~ai_model.gpu.torch_backend.DigitalGPUBackend`
refuses as a *bare kernel* (where it would silently be plain numpy) is allowed
*here*: the spec makes the modelling explicit and machine-checkable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

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


# ── the initial opcode set ────────────────────────────────────────────────────
register(_spec(
    name="gemm", version=1,
    inputs=("A", "B"),
    input_shapes={"A": ("M", "K"), "B": ("K", "N")},
    output_shapes={"C": ("M", "N")},
    dtypes={"A": "fp32", "B": "fp32", "C": "fp32"},
    numeric_profile="fp32_strict", deterministic=True,
    doc="Tiled matrix multiply C = A @ B (full fp32).",
))

register(_spec(
    name="add", version=1,
    inputs=("A", "B"),
    input_shapes={"A": ("*",), "B": ("*",)},
    output_shapes={"C": ("*",)},
    dtypes={"A": "fp32", "B": "fp32", "C": "fp32"},
    numeric_profile="fp32_strict", deterministic=True,
    doc="Elementwise C = A + B.",
))

register(_spec(
    name="softmax", version=1,
    inputs=("X",),
    input_shapes={"X": ("*",)},
    output_shapes={"Y": ("*",)},
    dtypes={"X": "fp32", "Y": "fp32"},
    numeric_profile="fp32_strict", deterministic=True,
    doc="Numerically-stable softmax over the last axis.",
))

register(_spec(
    name="attention", version=1,
    inputs=("Q", "K", "V"),
    input_shapes={"Q": ("B", "T", "D"), "K": ("B", "T", "D"), "V": ("B", "T", "D")},
    output_shapes={"O": ("B", "T", "D")},
    dtypes={"Q": "fp32", "K": "fp32", "V": "fp32", "O": "fp32"},
    numeric_profile="fp32_strict", deterministic=True,
    doc="Scaled-dot-product attention (full fp32).",
))

register(_spec(
    name="conv2d", version=1,
    inputs=("X", "W"),
    input_shapes={"X": ("N", "C", "H", "W"), "W": ("F", "C", "KH", "KW")},
    output_shapes={"Y": ("N", "F", "OH", "OW")},
    dtypes={"X": "fp32", "W": "fp32", "Y": "fp32"},
    numeric_profile="fp32_strict", deterministic=True,
    doc="2D convolution via im2col + gemm (reference).",
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
))
