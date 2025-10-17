// proto/kernel.cu - Protocol for GPU kernels
#pragma once
#include <cuda_runtime.h>

// All GPU operations must implement this protocol
struct KernelProtocol {
    dim3 grid_size;
    dim3 block_size;
    size_t shared_memory;
    cudaStream_t stream;

    virtual void launch() = 0;
    virtual size_t memory_required() const = 0;
    virtual float theoretical_occupancy() const = 0;
};