#ifndef KERNEL_TRACE_CU
#define KERNEL_TRACE_CU

#include <cuda_runtime.h>
#include <stdio.h>

struct KernelLaunchInfo {
    const char* kernel_name;
    const char* file;
    int line;
    unsigned int grid_x, grid_y, grid_z;
    unsigned int block_x, block_y, block_z;
    size_t shared_mem;
};

__device__ KernelLaunchInfo g_last_launch;

template<typename KernelFunc, typename... Args>
inline void traced_kernel_launch(
    KernelFunc kernel,
    const char* kernel_name,
    const char* file,
    int line,
    dim3 gridDim,
    dim3 blockDim,
    size_t sharedMem,
    cudaStream_t stream,
    Args&&... args
) {
    KernelLaunchInfo info;
    info.kernel_name = kernel_name;
    info.file = file;
    info.line = line;
    info.grid_x = gridDim.x;
    info.grid_y = gridDim.y;
    info.grid_z = gridDim.z;
    info.block_x = blockDim.x;
    info.block_y = blockDim.y;
    info.block_z = blockDim.z;
    info.shared_mem = sharedMem;

    cudaMemcpyToSymbol(g_last_launch, &info, sizeof(KernelLaunchInfo));

    kernel<<<gridDim, blockDim, sharedMem, stream>>>(std::forward<Args>(args)...);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        exit(1);
    }
}

#define LAUNCH_KERNEL(kernel, gridDim, blockDim, sharedMem, streamId, ...) \
    traced_kernel_launch(kernel, #kernel, __FILE__, __LINE__, gridDim, blockDim, sharedMem, streamId, ##__VA_ARGS__)

#define SYNC_CHECK(msg) \
    do { \
        cudaError_t err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            KernelLaunchInfo info; \
            cudaMemcpyFromSymbol(&info, g_last_launch, sizeof(KernelLaunchInfo)); \
\
\
\
\
            exit(1); \
        } \
    } while(0)

__global__ void kernel_trace_init() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
    }
}

inline void init_kernel_trace() {
    kernel_trace_init<<<1, 1>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        exit(1);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        exit(1);
    }

    size_t stack_size = 8192;
    cudaDeviceSetLimit(cudaLimitStackSize, stack_size);

    size_t malloc_heap = 128 * 1024 * 1024;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, malloc_heap);

    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 4);

    cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 32768);
}

#endif
