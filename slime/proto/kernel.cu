#pragma once
#include <cuda_runtime.h>
#include <cuda.h>

template<typename T>
struct ComputeKernel {
    T* data;
    size_t size;
    int depth;
};

template<typename T>
__device__ T warp_reduce_sum(T val) {
    unsigned mask = 0xFFFFFFFF;
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

template<typename T>
__device__ T warp_reduce_max(T val) {
    unsigned mask = 0xFFFFFFFF;
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(mask, val, offset));
    }
    return val;
}

template<typename T>
__device__ void block_reduce_sum(T* shared_data, int tid) {
    __syncthreads();
    if (tid < 256) shared_data[tid] += shared_data[tid + 256];
    __syncthreads();
    if (tid < 128) shared_data[tid] += shared_data[tid + 128];
    __syncthreads();
    if (tid < 64) shared_data[tid] += shared_data[tid + 64];
    __syncthreads();
    if (tid < 32) {
        shared_data[tid] = warp_reduce_sum(shared_data[tid]);
    }
}

template<typename T, typename Op>
__global__ void map_kernel(T* input, T* output, size_t n, Op op) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = op(input[idx]);
    }
}

template<typename T, typename Op>
__global__ void reduce_kernel(T* input, T* output, size_t n, Op op) {
    extern __shared__ T shared[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    shared[tid] = (idx < n) ? input[idx] : op.identity();
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            shared[tid] = op(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid < 32) {
        T val = shared[tid];
        val = op(val, __shfl_down_sync(0xFFFFFFFF, val, 16));
        val = op(val, __shfl_down_sync(0xFFFFFFFF, val, 8));
        val = op(val, __shfl_down_sync(0xFFFFFFFF, val, 4));
        val = op(val, __shfl_down_sync(0xFFFFFFFF, val, 2));
        val = op(val, __shfl_down_sync(0xFFFFFFFF, val, 1));
        
        if (tid == 0) {
            output[blockIdx.x] = val;
        }
    }
}

template<typename T>
__global__ void spawn_kernel_tree(ComputeKernel<T>* kernels, int count, int max_depth) {
    if (kernels[0].depth >= max_depth) return;
    
    int tid = threadIdx.x;
    if (tid < count && tid == 0) {
        for (int i = 0; i < count; i++) {
            ComputeKernel<T> child = kernels[i];
            child.depth++;
            
            dim3 grid((child.size + 255) / 256);
            dim3 block(256);
            
            switch(child.depth) {
                case 1:
                    pseudopod_forward_kernel<<<grid, block>>>(child.data, child.size);
                    break;
                case 2:
                    ca_update_kernel<<<grid, block>>>(child.data, child.size);
                    break;
                case 3:
                    fitness_kernel<<<grid, block>>>(child.data, child.size);
                    break;
                case 4:
                    effective_rank_kernel<<<grid, block>>>(child.data, child.size);
                    coherence_kernel<<<grid, block>>>(child.data + child.size/2, child.size/2);
                    break;
            }
        }
        cudaDeviceSynchronize();
    }
}

#define LAUNCH_CHILD(kernel, grid, block, shmem, stream) \
    if (threadIdx.x == 0 && blockIdx.x == 0) { \
        kernel<<<grid, block, shmem, stream>>>(); \
    }

#define SYNC_CHILDREN() cudaDeviceSynchronize()

__device__ inline unsigned int get_smid() {
    unsigned int smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

__device__ inline unsigned int get_warpid() {
    unsigned int warpid;
    asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
    return warpid;
}

__device__ inline unsigned int get_nsmid() {
    unsigned int nsmid;
    asm volatile("mov.u32 %0, %%nsmid;" : "=r"(nsmid));
    return nsmid;
}