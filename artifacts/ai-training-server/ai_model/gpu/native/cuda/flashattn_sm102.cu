// flashattn_sm102.cu — Flash Attention with native FP8 (e4m3 / e5m2) inputs.
//
// Targets SM102 (MaxCore).  Q/K/V arrive as packed fp8 bytes; O is written as
// fp16.  Online softmax (running row-max m_i and denominator l_i) means the
// full T×T score matrix is never materialised; peak shared memory is O(T·D).
//
// Shared memory layout (floats):
//   [0 .. T)        m_i  — running row maximums
//   [T .. 2T)       l_i  — running softmax denominators
//   [2T .. 2T+T·D)  acc  — output accumulator
//   [2T+T·D ..)     scores_row — one row of scores per __syncthreads block
//
// NOTE: shared memory = (2·T + T·D + T) · sizeof(float).  For T=64, D=64:
//   (128 + 4096 + 64) · 4 = ~17 KB — fits in the 48 KB SM limit.
//   Larger T or D values should use the software path.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define WARP_SIZE 32
#define FP8_E4M3  0
#define FP8_E5M2  1

struct fp8_e4m3 { unsigned char v; };
struct fp8_e5m2 { unsigned char v; };

__device__ __forceinline__ fp8_e4m3 float_to_fp8_e4m3(float x) {
    int i = __float2int_rn(x * 16.0f);
    i = max(-128, min(127, i));
    fp8_e4m3 r; r.v = static_cast<unsigned char>(i & 0xFF); return r;
}

__device__ __forceinline__ fp8_e5m2 float_to_fp8_e5m2(float x) {
    int i = __float2int_rn(x * 8.0f);
    i = max(-128, min(127, i));
    fp8_e5m2 r; r.v = static_cast<unsigned char>(i & 0xFF); return r;
}

__device__ __forceinline__ float fp8_e4m3_to_float(fp8_e4m3 x) {
    int i = static_cast<int>(x.v);
    if (i & 0x80) i |= ~0xFF;
    return static_cast<float>(i) / 16.0f;
}

__device__ __forceinline__ float fp8_e5m2_to_float(fp8_e5m2 x) {
    int i = static_cast<int>(x.v);
    if (i & 0x80) i |= ~0xFF;
    return static_cast<float>(i) / 8.0f;
}

template<int FP8_FORMAT>
__global__ void flash_attn_sm102_kernel(
    const void* __restrict__ Q,
    const void* __restrict__ K,
    const void* __restrict__ V,
    __half* __restrict__ O,
    int B, int H, int T, int D,
    bool causal
) {
    extern __shared__ float shared[];
    float* m_i     = shared;
    float* l_i     = m_i + T;
    float* acc     = l_i + T;
    float* scores  = acc + T * D;   // T floats for one row at a time

    int bh     = blockIdx.x;
    int b      = bh / H;
    int h      = bh % H;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Initialise per-query state
    if (lane_id < T) {
        m_i[lane_id] = -INFINITY;
        l_i[lane_id] = 0.0f;
        for (int d = 0; d < D; ++d)
            acc[lane_id * D + d] = 0.0f;
    }
    __syncthreads();

    float scale = 1.0f / sqrtf((float)D);

    // Online softmax loop — one K position at a time
    for (int k_block = 0; k_block < T; ++k_block) {
        int q_idx = lane_id;
        int k_idx = k_block;

        float score = -INFINITY;
        if (q_idx < T && k_idx < T) {
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) {
                int q_off = ((b * H + h) * T + q_idx) * D + d;
                int k_off = ((b * H + h) * T + k_idx) * D + d;
                float qv, kv;
                if constexpr (FP8_FORMAT == FP8_E4M3) {
                    qv = fp8_e4m3_to_float(reinterpret_cast<const fp8_e4m3*>(Q)[q_off]);
                    kv = fp8_e4m3_to_float(reinterpret_cast<const fp8_e4m3*>(K)[k_off]);
                } else {
                    qv = fp8_e5m2_to_float(reinterpret_cast<const fp8_e5m2*>(Q)[q_off]);
                    kv = fp8_e5m2_to_float(reinterpret_cast<const fp8_e5m2*>(K)[k_off]);
                }
                dot += qv * kv;
            }
            score = dot * scale;
            if (causal && q_idx < k_idx) score = -INFINITY;
        }

        if (q_idx < T) scores[q_idx] = score;
        __syncthreads();

        if (lane_id < T) {
            float s      = scores[lane_id];
            float m_new  = fmaxf(m_i[lane_id], s);
            float exp_diff = expf(m_i[lane_id] - m_new);
            float e      = (s == -INFINITY) ? 0.0f : expf(s - m_new);
            float l_new  = exp_diff * l_i[lane_id] + e;

            int v_off = ((b * H + h) * T + k_idx) * D;
            for (int d = 0; d < D; ++d) {
                float vv;
                if constexpr (FP8_FORMAT == FP8_E4M3) {
                    vv = fp8_e4m3_to_float(reinterpret_cast<const fp8_e4m3*>(V)[v_off + d]);
                } else {
                    vv = fp8_e5m2_to_float(reinterpret_cast<const fp8_e5m2*>(V)[v_off + d]);
                }
                acc[lane_id * D + d] = (exp_diff * l_i[lane_id] * acc[lane_id * D + d]
                                        + e * vv) / (l_new > 0.0f ? l_new : 1.0f);
            }
            m_i[lane_id] = m_new;
            l_i[lane_id] = l_new;
        }
        __syncthreads();
    }

    // Write output as fp16
    if (lane_id < T) {
        for (int d = 0; d < D; ++d) {
            int o_off = ((b * H + h) * T + lane_id) * D + d;
            O[o_off] = __float2half(acc[lane_id * D + d]);
        }
    }
}

// Explicit instantiations so the linker can find them from launcher.cu
template __global__ void flash_attn_sm102_kernel<FP8_E4M3>(
    const void*, const void*, const void*, __half*, int, int, int, int, bool);
template __global__ void flash_attn_sm102_kernel<FP8_E5M2>(
    const void*, const void*, const void*, __half*, int, int, int, int, bool);
