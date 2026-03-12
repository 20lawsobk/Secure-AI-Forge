import time
import numpy as np
from enum import Enum, auto


class GPUError(Exception):
    pass


class ShapeError(GPUError):
    pass


class TypeErrorGPU(GPUError):
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
        self._check_matmul_shapes(A, B)
        M, K = A.shape
        K2, N = B.shape
        C = np.zeros((M, N), dtype=A.dtype)

        tm, tn, tk = self.tile_m, self.tile_n, self.tile_k

        for i in range(0, M, tm):
            i_end = min(i + tm, M)
            for j in range(0, N, tn):
                j_end = min(j + tn, N)
                for k in range(0, K, tk):
                    k_end = min(k + tk, K)
                    C[i:i_end, j:j_end] += A[i:i_end, k:k_end] @ B[k:k_end, j:j_end]
        return C

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
            raise ShapeError("Q, K, V must have same shape [B, T, D]")
        B, T, D = Q.shape
        out = np.zeros_like(Q)
        scale = 1.0 / np.sqrt(D)

        if causal:
            mask = np.triu(np.full((T, T), -1e9, dtype=Q.dtype), k=1)
        else:
            mask = None

        for b in range(B):
            Qb = Q[b]
            Kb = K[b]
            Vb = V[b]
            scores = (Qb @ Kb.T) * scale
            if mask is not None:
                scores = scores + mask
            probs = self.softmax(scores, axis=-1)
            out[b] = probs @ Vb
        return out

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
        self.last_profile = []

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
                    O = self.core.attention(Q, K, V, causal=causal)
                    self.vram._store[hOut] = O
                    self.vram._meta[hOut]["shape"] = O.shape
                    self.vram._meta[hOut]["size"] = O.size

                elif op == OpCode.GEMM_BIAS_RELU:
                    hA, hB, hBias, hOut = args["a"], args["b"], args["bias"], args["out"]
                    self._validate_handles(hA, hB, hBias, hOut)
                    A = self.vram.get(hA)
                    B = self.vram.get(hB)
                    bias = self.vram.get(hBias)
                    O = self.core.gemm_bias_relu(A, B, bias)
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
    def __init__(self, lanes: int = 32):
        self.vram = VRAM()
        self.core = SIMDCore(lanes=lanes)
        self.scheduler = Scheduler(self.vram, self.core)

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
        return self.vram.get(hC)

    def add(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        hA = self.vram.alloc(A)
        hB = self.vram.alloc(B)
        hC = self.h_add(hA, hB)
        return self.vram.get(hC)

    def softmax(self, X: np.ndarray, axis: int = -1) -> np.ndarray:
        hX = self.vram.alloc(X)
        hY = self.h_softmax(hX, axis=axis)
        return self.vram.get(hY)

    def attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, causal: bool = False) -> np.ndarray:
        hQ = self.vram.alloc(Q)
        hK = self.vram.alloc(K)
        hV = self.vram.alloc(V)
        hO = self.h_attention(hQ, hK, hV, causal=causal)
        return self.vram.get(hO)

    def gemm_bias_relu(self, A: np.ndarray, B: np.ndarray, bias: np.ndarray) -> np.ndarray:
        hA = self.vram.alloc(A)
        hB = self.vram.alloc(B)
        hBias = self.vram.alloc(bias)
        hO = self.h_gemm_bias_relu(hA, hB, hBias)
        return self.vram.get(hO)

    def last_profile(self):
        return self.scheduler.last_profile
