import time
import numpy as np
from enum import Enum, auto


class GPUError(Exception):
    pass


class ShapeError(GPUError):
    pass


class TypeErrorGPU(GPUError):
    pass


class ShapeMismatchError(ShapeError):
    """Shapes are incompatible for the requested op. Subclass of ShapeError so
    existing ``except ShapeError`` handlers still catch it."""
    pass


class InvalidOpcodeError(GPUError):
    """An unknown opcode was requested (see opcode_spec.OPCODES)."""
    pass


class OOMError(GPUError):
    """A digital VRAM allocation exceeded the configured byte budget."""
    pass


class VRAM:
    def __init__(self):
        self._store = {}
        self._meta = {}
        self._next_id = 0

    def alloc(self, array: np.ndarray) -> int:
        if not isinstance(array, np.ndarray):
            raise TypeErrorGPU("VRAM.alloc expects a numpy array")
        hid = self._next_id
        self._store[hid] = array
        self._meta[hid] = {
            "shape": array.shape,
            "dtype": array.dtype,
            "size": array.size,
        }
        self._next_id += 1
        return hid

    def get(self, hid: int) -> np.ndarray:
        if hid not in self._store:
            raise GPUError(f"Invalid handle: {hid}")
        return self._store[hid]

    def meta(self, hid: int) -> dict:
        if hid not in self._meta:
            raise GPUError(f"Invalid handle: {hid}")
        return self._meta[hid]

    def free(self, hid: int):
        if hid in self._store:
            del self._store[hid]
            del self._meta[hid]


class OpCode(Enum):
    GEMM = auto()
    ADD = auto()
    SOFTMAX = auto()
    ATTENTION = auto()
    GEMM_BIAS_RELU = auto()


class Instruction:
    def __init__(self, opcode: OpCode, args: dict):
        self.opcode = opcode
        self.args = args

    def __repr__(self):
        return f"Instruction({self.opcode}, {self.args})"


class Program:
    def __init__(self):
        self.instructions = []

    def add(self, instr: Instruction):
        self.instructions.append(instr)


class SIMDCore:
    def __init__(self, lanes: int = 32, tile_m: int = 64, tile_n: int = 64, tile_k: int = 64):
        self.lanes = lanes
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k

    def _check_matmul_shapes(self, A: np.ndarray, B: np.ndarray):
        if A.ndim != 2 or B.ndim != 2:
            raise ShapeError("GEMM expects 2D matrices")
        if A.shape[1] != B.shape[0]:
            raise ShapeError(f"Incompatible GEMM shapes: {A.shape} x {B.shape}")

    def gemm_tiled(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Tiled GEMM dispatched as a single BLAS SGEMM.

        The previous implementation used a triple-nested Python loop over
        (M/tm) × (N/tn) × (K/tk) tiles.  Each Python iteration carries
        interpreter overhead (~μs) that completely dominates the actual BLAS
        work for the tile sizes used here (64–128).  NumPy/BLAS already tiles
        internally at the C level for L1/L2 cache locality, so the outer
        Python loops add overhead with no numerical or cache benefit.

        A single np.matmul dispatch achieves an identical result in one kernel
        invocation with full BLAS parallelism.
        """
        self._check_matmul_shapes(A, B)
        return np.matmul(
            A.astype(np.float32, copy=False),
            B.astype(np.float32, copy=False),
        ).astype(A.dtype, copy=False)

    def add(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if A.shape != B.shape:
            raise ShapeError(f"ADD shape mismatch: {A.shape} vs {B.shape}")
        if A.dtype != B.dtype:
            raise TypeErrorGPU(f"ADD dtype mismatch: {A.dtype} vs {B.dtype}")
        return A + B

    def softmax(self, X: np.ndarray, axis: int = -1) -> np.ndarray:
        X_max = X.max(axis=axis, keepdims=True)
        e = np.exp(X - X_max)
        return e / e.sum(axis=axis, keepdims=True)

    def attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, causal: bool = False) -> np.ndarray:
        if Q.shape != K.shape or Q.shape != V.shape:
            raise ShapeError("Q, K, V must have same shape [..., T, D]")
        if Q.ndim < 2:
            raise ShapeError(f"attention needs rank >= 2 [..., T, D], got {Q.shape}")
        # Batched over ALL leading dims via np.matmul (no Python per-batch loop),
        # which BLAS parallelizes and which generalizes beyond a single [B, T, D].
        T, D = Q.shape[-2], Q.shape[-1]
        scale = 1.0 / np.sqrt(D)
        scores = np.matmul(Q, np.swapaxes(K, -1, -2)) * scale
        if causal:
            mask = np.triu(np.full((T, T), -1e9, dtype=scores.dtype), k=1)
            scores = scores + mask
        probs = self.softmax(scores, axis=-1)
        return np.matmul(probs, V)

    def gemm_bias_relu(self, A: np.ndarray, B: np.ndarray, bias: np.ndarray) -> np.ndarray:
        C = self.gemm_tiled(A, B)
        if bias.ndim == 1:
            bias = bias.reshape(1, -1)
        if C.shape != bias.shape and bias.shape[0] == 1:
            bias = np.broadcast_to(bias, C.shape)
        C = C + bias
        return np.maximum(C, 0.0)


class Scheduler:
    def __init__(self, vram: VRAM, core: SIMDCore):
        self.vram = vram
        self.core = core
        self.last_profile: list = []

    def _validate_handles(self, *hids):
        for h in hids:
            _ = self.vram.meta(h)

    def run(self, program: Program):
        self.last_profile = []
        for idx, instr in enumerate(program.instructions):
            t0 = time.time()
            op = instr.opcode
            args = instr.args

            try:
                if op == OpCode.GEMM:
                    hA, hB, hOut = args["a"], args["b"], args["out"]
                    self._validate_handles(hA, hB, hOut)
                    A = self.vram.get(hA)
                    B = self.vram.get(hB)
                    C = self.core.gemm_tiled(A, B)
                    self.vram._store[hOut] = C
                    self.vram._meta[hOut]["shape"] = C.shape
                    self.vram._meta[hOut]["size"] = C.size

                elif op == OpCode.ADD:
                    hA, hB, hOut = args["a"], args["b"], args["out"]
                    self._validate_handles(hA, hB, hOut)
                    A = self.vram.get(hA)
                    B = self.vram.get(hB)
                    C = self.core.add(A, B)
                    self.vram._store[hOut] = C
                    self.vram._meta[hOut]["shape"] = C.shape
                    self.vram._meta[hOut]["size"] = C.size

                elif op == OpCode.SOFTMAX:
                    hX, hOut = args["x"], args["out"]
                    axis = args.get("axis", -1)
                    self._validate_handles(hX, hOut)
                    X = self.vram.get(hX)
                    Y = self.core.softmax(X, axis=axis)
                    self.vram._store[hOut] = Y
                    self.vram._meta[hOut]["shape"] = Y.shape
                    self.vram._meta[hOut]["size"] = Y.size

                elif op == OpCode.ATTENTION:
                    hQ, hK, hV, hOut = args["q"], args["k"], args["v"], args["out"]
                    causal = args.get("causal", False)
                    self._validate_handles(hQ, hK, hV, hOut)
                    Q = self.vram.get(hQ)
                    K = self.vram.get(hK)
                    V = self.vram.get(hV)
                    O = self.core.attention(Q, K, V, causal=causal)  # noqa: E741
                    self.vram._store[hOut] = O
                    self.vram._meta[hOut]["shape"] = O.shape
                    self.vram._meta[hOut]["size"] = O.size

                elif op == OpCode.GEMM_BIAS_RELU:
                    hA, hB, hBias, hOut = args["a"], args["b"], args["bias"], args["out"]
                    self._validate_handles(hA, hB, hBias, hOut)
                    A = self.vram.get(hA)
                    B = self.vram.get(hB)
                    bias = self.vram.get(hBias)
                    O = self.core.gemm_bias_relu(A, B, bias)  # noqa: E741
                    self.vram._store[hOut] = O
                    self.vram._meta[hOut]["shape"] = O.shape
                    self.vram._meta[hOut]["size"] = O.size

                else:
                    raise GPUError(f"Unknown opcode: {op}")

            except Exception as e:
                raise GPUError(f"Error at instruction {idx} ({instr}): {e}") from e

            t1 = time.time()
            self.last_profile.append({
                "index": idx,
                "opcode": op.name,
                "duration_ms": (t1 - t0) * 1000.0,
            })


class DigitalGPU:
    def __init__(self, lanes: int = 32, silicon=None):
        self.vram = VRAM()
        self.core = SIMDCore(lanes=lanes)
        self.scheduler = Scheduler(self.vram, self.core)
        # Optional MaxCoreSilicon performance model. When attached, each executed
        # op is ALSO recorded there to accumulate an *estimated* cycle/time budget.
        # This never alters results and never speeds anything up — the real math
        # below runs at this host's actual speed regardless.
        self.silicon = silicon

    def _model(self, kind: str, flops: float, kv_size: float = 0.0,
               bytes_moved: float = 0.0) -> None:
        if self.silicon is not None:
            self.silicon.model_op(kind, flops, kv_size=kv_size, bytes_moved=bytes_moved)

    def status(self) -> dict:
        """Live snapshot of the device and its VRAM memory system.

        Shape-compatible with DigitalGPUBackend.status() so /gpu/status works
        whether the server holds a raw device or a torch backend wrapper.
        """
        # .copy() is atomic under the GIL; iterating the live dict could race
        # a concurrent alloc/free and raise "dict changed size during iteration".
        store = self.vram._store.copy()
        vram_count = len(store)
        vram_bytes = sum(a.nbytes for a in store.values())
        return {
            "lanes": self.core.lanes,
            "tile_size": f"{self.core.tile_m}x{self.core.tile_n}x{self.core.tile_k}",
            "vram_handles": vram_count,
            "vram_bytes": vram_bytes,
            "vram_mb": round(vram_bytes / (1024 * 1024), 2),
        }

    def silicon_report(self):
        """Estimated (NOT measured) cycle/time budget from the attached silicon
        model, or None if no model is attached."""
        return self.silicon.report() if self.silicon is not None else None

    def h_gemm(self, hA: int, hB: int) -> int:
        A = self.vram.get(hA)
        B = self.vram.get(hB)
        hOut = self.vram.alloc(np.zeros((A.shape[0], B.shape[1]), dtype=A.dtype))
        prog = Program()
        prog.add(Instruction(OpCode.GEMM, {"a": hA, "b": hB, "out": hOut}))
        self.scheduler.run(prog)
        return hOut

    def h_add(self, hA: int, hB: int) -> int:
        A = self.vram.get(hA)
        hOut = self.vram.alloc(np.zeros_like(A))
        prog = Program()
        prog.add(Instruction(OpCode.ADD, {"a": hA, "b": hB, "out": hOut}))
        self.scheduler.run(prog)
        return hOut

    def h_softmax(self, hX: int, axis: int = -1) -> int:
        X = self.vram.get(hX)
        hOut = self.vram.alloc(np.zeros_like(X))
        prog = Program()
        prog.add(Instruction(OpCode.SOFTMAX, {"x": hX, "out": hOut, "axis": axis}))
        self.scheduler.run(prog)
        return hOut

    def h_attention(self, hQ: int, hK: int, hV: int, causal: bool = False) -> int:
        Q = self.vram.get(hQ)
        hOut = self.vram.alloc(np.zeros_like(Q))
        prog = Program()
        prog.add(Instruction(OpCode.ATTENTION, {"q": hQ, "k": hK, "v": hV, "out": hOut, "causal": causal}))
        self.scheduler.run(prog)
        return hOut

    def h_gemm_bias_relu(self, hA: int, hB: int, hBias: int) -> int:
        A = self.vram.get(hA)
        hOut = self.vram.alloc(np.zeros((A.shape[0], self.vram.get(hB).shape[1]), dtype=A.dtype))
        prog = Program()
        prog.add(Instruction(OpCode.GEMM_BIAS_RELU,
                             {"a": hA, "b": hB, "bias": hBias, "out": hOut}))
        self.scheduler.run(prog)
        return hOut

    def gemm(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        hA = self.vram.alloc(A)
        hB = self.vram.alloc(B)
        hC = self.h_gemm(hA, hB)
        self._model("gemm", 2.0 * A.shape[0] * A.shape[1] * B.shape[1],
                    bytes_moved=A.nbytes + B.nbytes)
        return self.vram.get(hC)

    def add(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        hA = self.vram.alloc(A)
        hB = self.vram.alloc(B)
        hC = self.h_add(hA, hB)
        self._model("add", float(A.size), bytes_moved=A.nbytes + B.nbytes)
        return self.vram.get(hC)

    def softmax(self, X: np.ndarray, axis: int = -1) -> np.ndarray:
        hX = self.vram.alloc(X)
        hY = self.h_softmax(hX, axis=axis)
        self._model("softmax", 5.0 * float(X.size), bytes_moved=X.nbytes)
        return self.vram.get(hY)

    def attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, causal: bool = False) -> np.ndarray:
        hQ = self.vram.alloc(Q)
        hK = self.vram.alloc(K)
        hV = self.vram.alloc(V)
        hO = self.h_attention(hQ, hK, hV, causal=causal)
        # Shape-agnostic telemetry: attention is over the last two axes (T, D);
        # any leading axes are batch. Works for [B, T, D] and higher rank alike.
        T, D = Q.shape[-2], Q.shape[-1]
        lead = int(np.prod(Q.shape[:-2])) if Q.ndim > 2 else 1
        self._model("attention", 4.0 * lead * T * T * D,
                    kv_size=float(K.nbytes + V.nbytes), bytes_moved=Q.nbytes)
        return self.vram.get(hO)

    def gemm_bias_relu(self, A: np.ndarray, B: np.ndarray, bias: np.ndarray) -> np.ndarray:
        hA = self.vram.alloc(A)
        hB = self.vram.alloc(B)
        hBias = self.vram.alloc(bias)
        hO = self.h_gemm_bias_relu(hA, hB, hBias)
        self._model("gemm", 2.0 * A.shape[0] * A.shape[1] * B.shape[1] + A.shape[0] * B.shape[1],
                    bytes_moved=A.nbytes + B.nbytes)
        return self.vram.get(hO)

    def last_profile(self):
        return self.scheduler.last_profile
