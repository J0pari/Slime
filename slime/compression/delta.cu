
#ifndef DELTA_CU
#define DELTA_CU

#include "../config/config.cu"
#include "../utils/tile_ops.cuh"
#include <cuda_runtime.h>

// Forward declarations for DIRESA functions (defined in diresa.cu, included by organism.cu)
struct DIRESAWeights;
__device__ void diresa_encode(const float* features, float* latent, const DIRESAWeights* weights);
__device__ void diresa_decode(const float* latent, float* reconstructed, const DIRESAWeights* weights);

struct DeltaGenome {
    uint32_t parent_hash;
    uint64_t child_genome_hash;
    uint16_t num_deltas;
    uint16_t* delta_indices;
    float* delta_values;
};

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
    for (int i = 0; i < genome_length; i++) {
        output_genome[i] = parent_genome[i];
    }
    for (int i = 0; i < num_deltas; i++) {
        int idx = delta_indices[i];
        if (idx < genome_length) {
            output_genome[idx] += delta_values[i];
        }
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
    if (parent_hash == 0) {
        for (int i = 0; i < genome_length; i++) {
            output_genome[i] = 0.0f;
        }
        for (int i = 0; i < num_deltas && i < max_deltas; i++) {
            uint16_t idx = delta_indices[i];
            if (idx < genome_length) {
                output_genome[idx] = delta_values[i];
            }
        }
    } else {
        int parent_idx = -1;
        for (int i = 0; i < archive_size; i++) {
            if (archive->genome_hash[i] == parent_hash) {
                parent_idx = i;
                break;
            }
        }

        if (parent_idx >= 0 && archive->latent_genome) {
            diresa_decode(&archive->latent_genome[parent_idx * GENOME_LATENT_DIM_MAX], parent_genome_workspace, weights);
            CooperativeSync::sync_warp();
            reconstruct_from_delta(parent_genome_workspace, delta_indices, delta_values, num_deltas, output_genome, genome_length);
        } else {
            for (int i = 0; i < genome_length; i++) {
                output_genome[i] = 0.0f;
            }
            for (int i = 0; i < num_deltas && i < max_deltas; i++) {
                uint16_t idx = delta_indices[i];
                if (idx < genome_length) {
                    output_genome[idx] = delta_values[i];
                }
            }
        }
    }
}

#endif
