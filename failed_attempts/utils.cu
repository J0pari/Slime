#include <cuda_runtime.h>
#include <cuda.h>

__device__ float warp_reduce_sum_f32(float val) {
    unsigned mask = __activemask();
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__device__ float warp_reduce_max_f32(float val) {
    unsigned mask = __activemask();
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(mask, val, offset));
    }
    return val;
}

__device__ float warp_reduce_min_f32(float val) {
    unsigned mask = __activemask();
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(mask, val, offset));
    }
    return val;
}

__device__ int warp_reduce_sum_i32(int val) {
    unsigned mask = __activemask();
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__device__ float warp_scan_sum(float val) {
    int lane = threadIdx.x % 32;
    #pragma unroll
    for (int i = 1; i <= 16; i *= 2) {
        float n = __shfl_up_sync(0xFFFFFFFF, val, i);
        if (lane >= i) val += n;
    }
    return val;
}

__device__ unsigned int warp_ballot(bool predicate) {
    return __ballot_sync(0xFFFFFFFF, predicate);
}

__device__ bool warp_all(bool predicate) {
    return __all_sync(0xFFFFFFFF, predicate);
}

__device__ bool warp_any(bool predicate) {
    return __any_sync(0xFFFFFFFF, predicate);
}

__device__ float warp_broadcast(float val, int src_lane) {
    return __shfl_sync(0xFFFFFFFF, val, src_lane);
}

__device__ void block_reduce_sum_f32(float* sdata, int tid) {
    __syncthreads();
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        sdata[tid] = warp_reduce_sum_f32(sdata[tid]);
    }
}

__device__ void block_reduce_max_f32(float* sdata, int tid) {
    __syncthreads();
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid < 32) {
        sdata[tid] = warp_reduce_max_f32(sdata[tid]);
    }
}

__global__ void grid_stride_sum(float* data, float* result, int n) {
    int tid = threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + tid;

    extern __shared__ float sdata[];
    float sum = 0.0f;

    for (int i = idx; i < n; i += stride) {
        sum += data[i];
    }

    sdata[tid] = sum;
    block_reduce_sum_f32(sdata, tid);

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

__global__ void coalesced_copy(float* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

__global__ void aligned_alloc_init(void* ptr, size_t size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        size_t aligned = ((size + 127) / 128) * 128;
        cudaMemset(ptr, 0, aligned);
    }
}