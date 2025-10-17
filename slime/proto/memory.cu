// proto/memory.cu - Protocol for temporal memory with decay
#pragma once
#include <cuda_runtime.h>

struct MemoryProtocol {
    float* memories_d;
    float* decay_factors_d;
    int capacity;
    float decay_rate;

    virtual void store(float* data, int size) = 0;
    virtual void recall(float* output, float* weights) = 0;
    virtual void apply_decay(float timestep) = 0;
    virtual size_t memory_bytes() const = 0;
};