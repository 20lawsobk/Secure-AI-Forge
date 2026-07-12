// reduction_sm102.cu — Dot-product reductions for SM102.
//
// Two implementations for the same contract  sum(x[i] * y[i], i=0..n-1):
//
//   reduction_current_sm102_kernel
//     Classic shared-memory binary-tree reduction.  Each block reduces its
//     stripe in shared memory, then one thread per block does an atomicAdd.
//     Good baseline, predictable behaviour.
//
//   reduction_redesigned_sm102_kernel
//     Warp-level reduction via __shfl_down_sync (no shared memory needed for
//     the reduction itself).  Grid-stride loop so one launch handles any n.
//     Only the warp lane-0 of each warp calls atomicAdd — ~32× fewer atomic
//     operations than the naive per-thread version.

#include <cuda_runtime.h>

#define WARP_SIZE 32

// ---------------------------------------------------------------------------
// Warp-level horizontal sum
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ---------------------------------------------------------------------------
// Variant 1: shared-memory tree reduction
// Launch: blockDim.x = 256, gridDim.x = ceil(n / 256)
// ---------------------------------------------------------------------------
__global__ void reduction_current_sm102_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float*       __restrict__ out,
    int n
) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? x[i] * y[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(out, sdata[0]);
}

// ---------------------------------------------------------------------------
// Variant 2: warp-shuffle grid-stride reduction  (preferred on SM102)
// Launch: blockDim.x = 256, gridDim.x = min(1024, ceil(n / 256))
// ---------------------------------------------------------------------------
__global__ void reduction_redesigned_sm102_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float*       __restrict__ out,
    int n
) {
    float sum = 0.0f;
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
        sum += x[i] * y[i];

    sum = warp_reduce_sum(sum);

    // Only lane 0 of each warp writes — 32× fewer atomics vs. per-thread
    if ((threadIdx.x & (WARP_SIZE - 1)) == 0)
        atomicAdd(out, sum);
}
