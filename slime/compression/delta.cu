
#ifndef DELTA_CU
#define DELTA_CU

#include "../config/config.cu"
#include "../utils/cuda_primitives.cuh"
#include "../memory/genome_ops.cuh"
#include <cuda_runtime.h>

struct DIRESAWeights;
__device__ void diresa_encode(const float* features, float* latent, const DIRESAWeights* weights);
__device__ void diresa_decode(const float* latent, float* reconstructed, const DIRESAWeights* weights);

__device__ void compute_genome_deltas(
    float* child_genome,
    float* parent_genome,
    uint16_t* delta_indices,
    float* delta_values,
    uint16_t* num_deltas,
    uint16_t max_deltas,
    uint64_t child_genome_hash
) {
    int threshold_slot = derive_param_slot(child_genome_hash, "delta_threshold");
    float delta_threshold = (child_genome[threshold_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE * 0.01f;

    for (int j = 0; j < GENOME_SIZE; j++) {
        float diff = child_genome[j] - parent_genome[j];
        if (fabsf(diff) > delta_threshold) {
            int delta_idx = atomicAdd((unsigned int*)num_deltas, 1);
            if (delta_idx < max_deltas) {
                delta_indices[delta_idx] = j;
                delta_values[delta_idx] = diff;
            }
        }
    }
}

__global__ void compute_delta_kernel(float* child_genome, float* parent_genome, uint16_t* delta_indices, float* delta_values, uint16_t* num_deltas, int genome_length, uint64_t child_genome_hash) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= genome_length) return;

    int threshold_slot = derive_param_slot(child_genome_hash, "delta_threshold");
    float delta_threshold = (child_genome[threshold_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE * 0.01f;

    float diff = child_genome[tid] - parent_genome[tid];
    if (fabsf(diff) > delta_threshold) {
        int idx = atomicAdd((unsigned int*)num_deltas, 1);
        if (idx < genome_length) {
            delta_indices[idx] = tid;
            delta_values[idx] = diff;
        }
    }
}

__device__ void reconstruct_from_delta(float* parent_genome, uint16_t* delta_indices, float* delta_values, uint16_t num_deltas, float* output_genome, int genome_length) {
    DEVICE_VALIDATE_PTR(parent_genome);
    DEVICE_VALIDATE_PTR(output_genome);
    DEVICE_VALIDATE_RANGE(genome_length, 1, GENOME_SIZE);

    for (int i = 0; i < genome_length; i++) {
        output_genome[i] = parent_genome[i];
        DEVICE_VALIDATE_FINITE(output_genome[i]);
    }
    for (int i = 0; i < num_deltas; i++) {
        int idx = delta_indices[i];
        DEVICE_FATAL_IF(idx >= genome_length, "reconstruct_from_delta: delta index out of bounds");
        DEVICE_VALIDATE_GENOME_SLOT(idx);
        DEVICE_VALIDATE_FINITE(delta_values[i]);
        output_genome[idx] += delta_values[i];
        DEVICE_VALIDATE_FINITE(output_genome[idx]);
    }
}

__device__ void reconstruct_genome_from_archive(
    uint64_t parent_hash,
    GPUElite* archive,
    int archive_size,
    uint16_t* delta_indices,
    float* delta_values,
    uint16_t num_deltas,
    uint16_t max_deltas,
    float* output_genome,
    int genome_length,
    float* parent_genome_workspace,
    const DIRESAWeights* weights
) {
    DEVICE_VALIDATE_PTR(archive);
    DEVICE_VALIDATE_PTR(output_genome);
    DEVICE_VALIDATE_PTR(parent_genome_workspace);
    DEVICE_VALIDATE_PTR(weights);
    DEVICE_VALIDATE_RANGE(archive_size, 0, MAX_ARCHIVE_SIZE);
    DEVICE_VALIDATE_RANGE(genome_length, 1, GENOME_SIZE);
    DEVICE_VALIDATE_RANGE(num_deltas, 0, max_deltas);

    DEVICE_FATAL_IF(parent_hash == 0, "reconstruct_genome_from_archive: parent_hash is 0 (invalid)");

    if (parent_hash == UINT64_MAX) {
        for (int i = 0; i < genome_length; i++) {
            parent_genome_workspace[i] = 0.0f;
        }
        reconstruct_from_delta(parent_genome_workspace, delta_indices, delta_values, num_deltas, output_genome, genome_length);
        return;
    }

    DEVICE_VALIDATE_PTR(archive->hash_table_keys);
    DEVICE_VALIDATE_PTR(archive->hash_table_values);

    int parent_idx = hash_table_lookup(
        archive->hash_table_keys,
        archive->hash_table_values,
        parent_hash
    );

    DEVICE_FATAL_IF(parent_idx < 0, "reconstruct_genome_from_archive: parent evicted from archive - cannot reconstruct");
    DEVICE_VALIDATE_ARCHIVE_IDX(parent_idx, archive_size);
    DEVICE_FATAL_IF(archive->latent_genome == nullptr, "reconstruct_genome_from_archive: archive latent_genome is null");

    diresa_decode(&archive->latent_genome[parent_idx * GENOME_LATENT_DIM_MAX], parent_genome_workspace, weights);
    reconstruct_from_delta(parent_genome_workspace, delta_indices, delta_values, num_deltas, output_genome, genome_length);
}

#endif
