#pragma once
#include <cuda_runtime.h>
#include <cuda/std/atomic>

struct ComponentProtocol {
    float fitness;
    virtual __device__ void reset() = 0;
    virtual __host__ void serialize(void* buffer) = 0;
    virtual __host__ void deserialize(const void* buffer) = 0;
};

struct PooledComponent : ComponentProtocol {
    float* genome;
    float coherence;
    float effective_rank;
    int genome_size;
    int generation;

    __device__ void reset() override {
        for (int i = threadIdx.x; i < genome_size; i += blockDim.x) {
            genome[i] = 0.0f;
        }
        coherence = 0.0f;
        effective_rank = 1.0f;
        fitness = 0.0f;
    }

    __host__ void serialize(void* buffer) override {
        char* ptr = (char*)buffer;

        memcpy(ptr, &fitness, sizeof(float));
        ptr += sizeof(float);

        memcpy(ptr, &coherence, sizeof(float));
        ptr += sizeof(float);

        memcpy(ptr, &effective_rank, sizeof(float));
        ptr += sizeof(float);

        memcpy(ptr, &generation, sizeof(int));
        ptr += sizeof(int);

        memcpy(ptr, &genome_size, sizeof(int));
        ptr += sizeof(int);

        cudaMemcpy(ptr, genome, genome_size * sizeof(float), cudaMemcpyDeviceToHost);
    }

    __host__ void deserialize(const void* buffer) override {
        const char* ptr = (const char*)buffer;

        memcpy(&fitness, ptr, sizeof(float));
        ptr += sizeof(float);

        memcpy(&coherence, ptr, sizeof(float));
        ptr += sizeof(float);

        memcpy(&effective_rank, ptr, sizeof(float));
        ptr += sizeof(float);

        memcpy(&generation, ptr, sizeof(int));
        ptr += sizeof(int);

        memcpy(&genome_size, ptr, sizeof(int));
        ptr += sizeof(int);

        cudaMalloc(&genome, genome_size * sizeof(float));
        cudaMemcpy(genome, ptr, genome_size * sizeof(float), cudaMemcpyHostToDevice);
    }
};