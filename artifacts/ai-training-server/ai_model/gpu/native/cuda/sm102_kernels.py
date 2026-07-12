"""SM102 kernel implementations on the digital GPU.

Each function here is the digital-GPU equivalent of a .cu kernel in this
directory.  They are registered in the CUDANvcc kernel registry so that
compiling any of the SM102 source files returns callables backed by these
implementations.

Execution model
───────────────
Rather than interpreting the CUDA thread loop one thread at a time, each
implementation is *vectorised over the full block/grid* using the digital
GPU's SIMDCore / HyperSIMDCore — the same way hardware executes warps in
lock-step.  The CUDARuntime context (grid/block dims, shared memory) is
passed in but the heavy math always flows through the GPU object.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ai_model.gpu.native.cuda.runtime import CUDARuntime, Dim3

# Precision helpers — precision.py already implements OCP fp8
try:
    from ai_model.gpu.precision import quantize as _prec_quantize  # type: ignore
    def _to_fp8_e4m3(arr: np.ndarray) -> np.ndarray:
        return _prec_quantize(arr.astype(np.float32), "fp8_e4m3")
    def _to_fp8_e5m2(arr: np.ndarray) -> np.ndarray:
        return _prec_quantize(arr.astype(np.float32), "fp8_e5m2")
except Exception:
    def _to_fp8_e4m3(arr): return arr.astype(np.float32)   # graceful fallback
    def _to_fp8_e5m2(arr): return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# flash_attn_sm102  —  FP8 flash attention (e4m3 or e5m2)
#
# Maps to: flashattn_sm102.cu  /  flash_attn_sm102_kernel<FP8_FORMAT>
#
# Signature match (CUDA):
#   flash_attn_sm102_kernel<FP8_E4M3>(Q, K, V, O, B, H, T, D, causal)
#
# Digital GPU execution:
#   1. Quantise Q/K/V to the requested fp8 grid (precision.py)
#   2. Run HyperSIMDCore.flash_attention (online softmax, block_size=32)
#      for every (batch, head) independently — matches the per-bh block
#      launch in the .cu kernel
#   3. Cast output to fp16 (matches __half* O in the kernel)
# ---------------------------------------------------------------------------

def flash_attn_sm102(
    rt: CUDARuntime,
    Q: np.ndarray,          # [B, H, T, D]  float32 on entry
    K: np.ndarray,
    V: np.ndarray,
    O: np.ndarray,          # [B, H, T, D]  fp16 output, written in-place
    causal: bool = False,
    fp8_format: str = "e4m3",   # "e4m3" | "e5m2"
    _gpu=None,              # HyperGPU / HyperSIMDCore instance (injected by nvcc)
) -> None:
    """Digital-GPU execution of flash_attn_sm102_kernel."""
    to_fp8 = _to_fp8_e4m3 if fp8_format == "e4m3" else _to_fp8_e5m2

    B, H, T, D = Q.shape

    # Quantise to fp8 (models the fp8 register file in hardware)
    Q8 = to_fp8(Q)
    K8 = to_fp8(K)
    V8 = to_fp8(V)

    core = _gpu.core if _gpu is not None else None

    for b in range(B):
        # blockIdx.x = b * H + h  — each block handles one (batch, head)
        for h in range(H):
            bx = b * H + h
            rt.block_idx = Dim3(bx, 0, 0)

            q = Q8[b, h]   # [T, D]
            k = K8[b, h]
            v = V8[b, h]

            if core is not None and hasattr(core, "flash_attention"):
                # Route through HyperSIMDCore — tiled online softmax
                out_bh = core.flash_attention(
                    q[None], k[None], v[None], causal=causal
                )[0]                    # [T, D]
            else:
                # Pure-numpy fallback: standard scaled dot-product attention
                scale  = 1.0 / np.sqrt(D)
                scores = (q @ k.T) * scale          # [T, T]
                if causal:
                    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
                    scores[mask] = -np.inf
                scores -= scores.max(axis=-1, keepdims=True)
                weights = np.exp(scores)
                weights /= weights.sum(axis=-1, keepdims=True)
                out_bh  = weights @ v               # [T, D]

            O[b, h] = out_bh.astype(np.float16)

    rt.syncthreads()


# ---------------------------------------------------------------------------
# im2col_sm102  —  scatter input patches to column buffer
#
# Maps to: conv_sm102.cu  /  im2col_sm102_kernel
#
# Signature match (CUDA):
#   im2col_sm102_kernel(x, cols, N, C, H, W, K, stride, padding, out_h, out_w)
#
# Digital GPU: stride_tricks zero-copy view (matches HyperSIMDCore.conv2d
# internal im2col), mirrors what the hardware kernel does per thread block.
# ---------------------------------------------------------------------------

def im2col_sm102(
    rt: CUDARuntime,
    x: np.ndarray,          # [N, C, H, W]  float16
    cols: np.ndarray,       # [N, C*K*K, out_h*out_w]  output, written in-place
    stride: int = 1,
    padding: int = 0,
    _gpu=None,
) -> None:
    """Digital-GPU execution of im2col_sm102_kernel."""
    import numpy.lib.stride_tricks as _st

    N, C, H, W = x.shape
    _, CKK, OH_OW = cols.shape
    K = int(round((CKK / C) ** 0.5))
    out_h = int(round(OH_OW ** 0.5))           # assumes square output
    out_w = OH_OW // out_h

    # Pad input if needed — blockIdx.x = n, blockIdx.y = c in the .cu kernel
    if padding > 0:
        x_pad = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)),
                       constant_values=0).astype(x.dtype)
    else:
        x_pad = x

    # Vectorised im2col via stride_tricks (zero-copy patch view)
    _, _, Hp, Wp = x_pad.shape
    shape    = (N, C, out_h, out_w, K, K)
    strides  = (x_pad.strides[0], x_pad.strides[1],
                x_pad.strides[2] * stride, x_pad.strides[3] * stride,
                x_pad.strides[2], x_pad.strides[3])
    patches  = _st.as_strided(x_pad, shape=shape, strides=strides)
    # [N, C, out_h, out_w, K, K] → [N, C*K*K, out_h*out_w]
    cols[:] = patches.reshape(N, C * K * K, out_h * out_w).astype(x.dtype)

    rt.syncthreads()


# ---------------------------------------------------------------------------
# conv_wmma_sm102  —  WMMA tensor-core GEMM over the im2col column buffer
#
# Maps to: conv_sm102.cu  /  conv_wmma_sm102_kernel
#
# Each warp owns a 16×16 output tile (wmma::fragment<16,16,16>).
# Digital GPU: TensorCoreUnit.mixed_precision_matmul with fp16 accumulation.
# ---------------------------------------------------------------------------

def conv_wmma_sm102(
    rt: CUDARuntime,
    A: np.ndarray,      # [M, K]  half — reshaped cols
    B: np.ndarray,      # [N, K]  half — weight matrix (col-major in wmma)
    C: np.ndarray,      # [M, N]  half — output, written in-place
    _gpu=None,
) -> None:
    """Digital-GPU execution of conv_wmma_sm102_kernel."""
    M, K = A.shape
    N    = B.shape[0]

    tile = 16   # wmma fragment size matches kernel

    core = _gpu.core if _gpu is not None else None

    for wm in range(0, M, tile):
        # warpM tiles — each corresponds to one warp's 16-row slice
        for wn in range(0, N, tile):
            # warpN tiles — blockIdx.y * blockDim.y + threadIdx.y in the kernel
            rt.block_idx = Dim3(wm // tile, wn // tile, 0)

            A_tile = A[wm:wm+tile, :K].astype(np.float16)
            B_tile = B[wn:wn+tile, :K].astype(np.float16)   # col-major in hw

            if core is not None and hasattr(core, "tensor_cores"):
                tc = core.tensor_cores[0]
                # mixed_precision_matmul: fp16 inputs, fp32 accumulate → fp16 out
                out_tile = tc.mixed_precision_matmul(
                    A_tile.astype(np.float32),
                    B_tile.T.astype(np.float32),  # B is col-major → transpose
                ).astype(np.float16)
            else:
                # Fallback: fp16 matmul, fp32 accumulate (mirrors hardware)
                out_tile = (A_tile.astype(np.float32) @
                            B_tile.T.astype(np.float32)).astype(np.float16)

            C[wm:wm+tile, wn:wn+tile] = out_tile

    rt.syncthreads()


# ---------------------------------------------------------------------------
# reduction_current_sm102  —  shared-memory tree reduction dot product
#
# Maps to: reduction_sm102.cu  /  reduction_current_sm102_kernel
# ---------------------------------------------------------------------------

def reduction_current_sm102(
    rt: CUDARuntime,
    x: np.ndarray,      # [n]
    y: np.ndarray,      # [n]
    out: np.ndarray,    # [1]  accumulated in-place
    _gpu=None,
) -> None:
    """Shared-memory tree reduction — digital-GPU execution."""
    n         = len(x)
    block_sz  = rt.block_dim.x   # 256 in the .cu kernel
    n_blocks  = rt.grid_dim.x

    shm = rt.shared(block_sz)

    for bx in range(n_blocks):
        rt.block_idx = Dim3(bx, 0, 0)
        base = bx * block_sz
        # Load stripe into shared memory (sdata[tid] in the kernel)
        for tid in range(block_sz):
            i = base + tid
            shm[tid] = float(x[i] * y[i]) if i < n else 0.0

        rt.syncthreads()

        # Binary-tree reduction in shared memory
        s = block_sz // 2
        while s > 0:
            for tid in range(s):
                shm[tid] += shm[tid + s]
            rt.syncthreads()
            s >>= 1

        # tid==0 does atomicAdd
        rt.atomic_add(out, 0, shm[0])


# ---------------------------------------------------------------------------
# reduction_redesigned_sm102  —  warp-shuffle grid-stride reduction
#
# Maps to: reduction_sm102.cu  /  reduction_redesigned_sm102_kernel
# Preferred path on SM102: ~32× fewer atomics than variant 1.
# ---------------------------------------------------------------------------

def reduction_redesigned_sm102(
    rt: CUDARuntime,
    x: np.ndarray,      # [n]
    y: np.ndarray,      # [n]
    out: np.ndarray,    # [1]  accumulated in-place
    _gpu=None,
) -> None:
    """Warp-shuffle grid-stride reduction — digital-GPU execution."""
    n          = len(x)
    block_sz   = rt.block_dim.x
    n_blocks   = rt.grid_dim.x
    stride     = block_sz * n_blocks

    # Vectorise the grid-stride loop over all 32 lanes simultaneously.
    # In hardware each warp does: for i in range(idx, n, stride): sum += x*y
    # On the digital GPU we process lane_count elements in parallel.
    lane_count = rt.WARP_SIZE

    for bx in range(n_blocks):
        rt.block_idx = Dim3(bx, 0, 0)

        for warp_base in range(0, block_sz, lane_count):
            # Lane indices for this warp within the grid
            lane_ids = np.arange(lane_count)
            idxs     = bx * block_sz + warp_base + lane_ids   # threadIdx.x + block offset

            # Grid-stride accumulation across all warps
            sums = np.zeros(lane_count, dtype=np.float32)
            for base in range(0, n, stride):
                active = (idxs + base) < n
                sums[active] += (x[(idxs + base)[active]] *
                                 y[(idxs + base)[active]])

            # Warp-level butterfly reduction (__shfl_down_sync)
            reduced = rt.warp_reduce_sum(sums)

            # Lane 0 of each warp calls atomicAdd
            rt.atomic_add(out, 0, float(reduced[0]))
