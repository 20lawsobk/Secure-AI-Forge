// conv_sm102.cu — im2col + WMMA (Tensor Core) convolution for SM102.
//
// Two kernels:
//   im2col_sm102_kernel    — scatter input patches into column buffer (fp16)
//   conv_wmma_sm102_kernel — GEMM over the column buffer via nvcuda::wmma
//                            16×16×16 tiles, fp16 inputs, fp16 accumulate
//
// Usage: call im2col first to produce `cols`, then conv_wmma to compute
//   C[M×N] = cols[M×K] × W[K×N]
// where M = N·out_h·out_w, K = C·kH·kW, N = out_channels.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define WARP_SIZE 32

// ---------------------------------------------------------------------------
// im2col — one thread per (batch, channel, output-pixel)
// ---------------------------------------------------------------------------
__global__ void im2col_sm102_kernel(
    const __half* __restrict__ x,
    __half*       __restrict__ cols,
    int N, int C, int H, int W,
    int K, int stride, int padding,
    int out_h, int out_w
) {
    int n   = blockIdx.x;
    int c   = blockIdx.y;
    int idx = threadIdx.x + blockIdx.z * blockDim.x;
    int total = out_h * out_w;
    if (idx >= total) return;

    int oh = idx / out_w;
    int ow = idx % out_w;
    int h_start = oh * stride - padding;
    int w_start = ow * stride - padding;

    for (int kh = 0; kh < K; ++kh) {
        for (int kw = 0; kw < K; ++kw) {
            int hi     = h_start + kh;
            int wi     = w_start + kw;
            int col_c  = c * K * K + kh * K + kw;
            int col_idx = ((n * (C * K * K)) + col_c) * (out_h * out_w) + idx;
            if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                int x_idx = ((n * C + c) * H + hi) * W + wi;
                cols[col_idx] = x[x_idx];
            } else {
                cols[col_idx] = __float2half(0.0f);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// WMMA GEMM — each warp owns a 16×16 output tile
// Launch: blockDim = {128, 4, 1}, each block handles 4×4 warp tiles
// ---------------------------------------------------------------------------
__global__ void conv_wmma_sm102_kernel(
    const __half* __restrict__ A,   // M × K
    const __half* __restrict__ B,   // N × K  (col_major in wmma sense)
    __half*       __restrict__ C,   // M × N
    int M, int N, int K
) {
    int warp_M = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int warp_N =  blockIdx.y * blockDim.y + threadIdx.y;

    if (warp_M * 16 >= M || warp_N * 16 >= N) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, __half>               c_frag;

    wmma::fill_fragment(c_frag, __float2half(0.0f));

    for (int k = 0; k < K; k += 16) {
        const __half* a_tile = A + (warp_M * 16) * K + k;
        const __half* b_tile = B + (warp_N * 16) * K + k;
        wmma::load_matrix_sync(a_frag, a_tile, K);
        wmma::load_matrix_sync(b_frag, b_tile, K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    __half* c_tile = C + (warp_M * 16) * N + warp_N * 16;
    wmma::store_matrix_sync(c_tile, c_frag, N, wmma::mem_row_major);
}
