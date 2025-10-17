#pragma once
#include <cuda_runtime.h>
#include <cuda/std/atomic>

struct MemoryProtocol {
    virtual __device__ void store(float* data, float weight) = 0;
    virtual __device__ float* recall() = 0;
    virtual __device__ void clear() = 0;
    virtual __device__ size_t capacity() const = 0;
};

struct TubeMemory : MemoryProtocol {
    float* buffer;
    float* weights;
    cuda::std::atomic<int> write_pos;
    int size;
    float decay;

    __device__ void store(float* data, float weight) override {
        int pos = write_pos.fetch_add(1) % size;
        buffer[pos] = *data;
        weights[pos] = weight;

        for (int i = 0; i < size; i++) {
            weights[i] *= decay;
        }
    }

    __device__ float* recall() override {
        __shared__ float result;
        result = 0.0f;
        float total_weight = 0.0f;

        for (int i = 0; i < size; i++) {
            if (weights[i] > 0.01f) {
                result += buffer[i] * weights[i];
                total_weight += weights[i];
            }
        }

        if (total_weight > 0) {
            result /= total_weight;
            return &result;
        }
        return nullptr;
    }

    __device__ void clear() override {
        for (int i = threadIdx.x; i < size; i += blockDim.x) {
            weights[i] = 0.0f;
        }
        if (threadIdx.x == 0) {
            write_pos.store(0);
        }
    }

    __device__ size_t capacity() const override {
        return size * sizeof(float);
    }
};