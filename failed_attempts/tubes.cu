#include <cuda_runtime.h>
#include <cuda/std/atomic>

constexpr int TUBE_CAPACITY = 1024;
constexpr float DECAY_RATE = 0.95f;

struct TubeNetwork {
    float* memory_buffer;
    float* decay_weights;
    cuda::std::atomic<int> write_position;
    int capacity;
    float decay_constant;
};

__global__ void store_memory_kernel(TubeNetwork* network, float* data, float weight, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) {
        int pos = network->write_position.fetch_add(1) % network->capacity;

        for (int i = 0; i < size; i++) {
            network->memory_buffer[pos * size + i] = data[i];
        }

        network->decay_weights[pos] = weight;
    }

    for (int i = tid; i < network->capacity; i += blockDim.x * gridDim.x) {
        network->decay_weights[i] *= network->decay_constant;
    }
}

__global__ void recall_memory_kernel(TubeNetwork* network, float* output, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size) {
        float weighted_sum = 0.0f;
        float total_weight = 0.0f;

        for (int i = 0; i < network->capacity; i++) {
            float weight = network->decay_weights[i];
            if (weight > 0.01f) {
                weighted_sum += network->memory_buffer[i * size + tid] * weight;
                total_weight += weight;
            }
        }

        output[tid] = (total_weight > 0) ? weighted_sum / total_weight : 0.0f;
    }
}

__global__ void apply_decay_kernel(TubeNetwork* network) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < network->capacity) {
        network->decay_weights[idx] *= network->decay_constant;

        if (network->decay_weights[idx] < 0.001f) {
            network->decay_weights[idx] = 0.0f;
        }
    }
}

__global__ void clear_network_kernel(TubeNetwork* network) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < network->capacity) {
        network->decay_weights[idx] = 0.0f;
    }

    if (idx == 0) {
        network->write_position.store(0);
    }
}