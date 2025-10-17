// slime/kernels/utils.cu - Warp primitives and reductions
#ifndef UTILS_CU
#define UTILS_CU
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Warp-level primitives
constexpr int WARP_SIZE = 32;

// Warp reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    unsigned mask = __activemask();
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

// Warp reduction for max
__device__ __forceinline__ float warp_reduce_max(float val) {
    unsigned mask = __activemask();
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(mask, val, offset));
    }
    return val;
}

// Warp reduction for min
__device__ __forceinline__ float warp_reduce_min(float val) {
    unsigned mask = __activemask();
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(mask, val, offset));
    }
    return val;
}

// Block-level reduction
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0;
    if (wid == 0) val = warp_reduce_sum(val);

    return val;
}

// Warp vote functions
__device__ __forceinline__ int warp_vote_all(int predicate) {
    unsigned mask = __activemask();
    return __all_sync(mask, predicate);
}

__device__ __forceinline__ int warp_vote_any(int predicate) {
    unsigned mask = __activemask();
    return __any_sync(mask, predicate);
}

__device__ __forceinline__ int warp_vote_ballot(int predicate) {
    unsigned mask = __activemask();
    return __ballot_sync(mask, predicate);
}

// Warp-level scan (prefix sum)
__device__ __forceinline__ float warp_scan_sum(float val) {
    unsigned mask = __activemask();
    int lane = threadIdx.x % 32;

    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        float n = __shfl_up_sync(mask, val, offset);
        if (lane >= offset) val += n;
    }

    return val;
}

// Hash function for deterministic randomness
__device__ __forceinline__ unsigned int jenkins_hash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// Content-addressable hash for deduplication
__device__ unsigned long long content_hash(float* data, int size) {
    unsigned long long hash = 0;
    for (int i = 0; i < size; i++) {
        unsigned int bits = __float_as_uint(data[i]);
        hash += jenkins_hash(bits + i);
        hash = (hash << 7) | (hash >> 57);  // Rotate
    }
    return hash;
}

#endif // UTILS_CU
