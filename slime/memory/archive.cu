
#ifndef ARCHIVE_CU
#define ARCHIVE_CU
#include "../config/config.cu"
#include "../utils/cuda_primitives.cuh"
#include "behavioral_ops.cuh"
#include "genome_ops.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>

struct GPUElite {
    float* fitness;
    float* coherence;
    float* effective_rank;
    uint64_t* genome_hash;
    uint32_t* parent_ids;
    uint16_t* generation;
    float* hw_coords;
    float* task_coords;
    float* gen_coords;
    float* latent_genome;
    float* hardware_features;
    float* task_performance;
    float* per_class_accuracy;
    int hw_dim;
    int task_dim;
    int gen_dim;
};

struct VoronoiCell {
    float* hw_centroid;
    float* task_centroid;
    float* gen_centroid;
    float radius;
    int density;
    int density_prev;
    float density_fluctuation;
    int best_elite_idx;
    float quality_threshold;
};

__constant__ uint16_t d_generation_counter;

__device__ __forceinline__ float elite_to_cell_distance_sq(
    const GPUElite* archive, int elite_idx,
    const VoronoiCell* cell, int hw_dim, int task_dim, int gen_dim
) {
    return compute_three_axis_distance_sq(
        &archive->hw_coords[elite_idx * hw_dim],
        &archive->task_coords[elite_idx * task_dim],
        &archive->gen_coords[elite_idx * gen_dim],
        cell->hw_centroid, cell->task_centroid, cell->gen_centroid,
        hw_dim, task_dim, gen_dim
    );
}

__device__ uint64_t gpu_sha256(float* genome, uint32_t size) {

    uint64_t hash = 0x9e3779b97f4a7c15ULL;

    for (uint32_t i = 0; i < size; i++) {
        uint32_t bits = __float_as_uint(genome[i]);

        hash ^= bits + 0x9e3779b9 + (hash << JENKINS_FINAL_SHIFT_A) + (hash >> 2);
        hash += (hash << 3);
        hash ^= (hash >> JENKINS_FINAL_SHIFT_B);
        hash += (hash << (WMMA_TILE_DIM - 1));

        hash *= 0xc4ceb9fe1a85ec53ULL;
        hash ^= hash >> HASH_FINALIZER_SHIFT;
    }

    return hash;
}

__device__ uint16_t get_generation() {
    return d_generation_counter;
}

__global__ void create_elite_kernel(
    GPUElite* __restrict__ archive,
    int elite_idx,
    float* __restrict__ genome,
    float fitness_val,
    float coherence_val,
    float effective_rank_val,
    uint32_t genome_size
) {
    if (threadIdx.x == 0) {
        archive->fitness[elite_idx] = fitness_val;
        archive->coherence[elite_idx] = coherence_val;
        archive->effective_rank[elite_idx] = effective_rank_val;
        archive->genome_hash[elite_idx] = gpu_sha256(genome, genome_size);
        archive->generation[elite_idx] = get_generation();
    }
}


__global__ void update_voronoi_density_kernel(
    VoronoiCell* __restrict__ cells,
    GPUElite* __restrict__ archive,
    int num_elites,
    int num_cells,
    int behavioral_dim_total,
    const float* genome,
    const float* gradients,
    uint64_t genome_hash,
    float ctx_metabolic,
    float ctx_stress,
    float ctx_morphogen,
    float ctx_complexity,
    float ctx_niche,
    float ctx_learning,
    float ctx_performance
) {
    if (num_cells <= 0) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("FATAL [update_voronoi_density]: num_cells=%d\n", num_cells);
        }
        return;
    }
    int cell_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_id >= num_cells) return;

    VoronoiCell* cell = &cells[cell_id];
    cell->density = 0;
    cell->best_elite_idx = -1;
    float best_fitness = -1.0f;

    int hw_dim = archive->hw_dim;
    int task_dim = archive->task_dim;
    int gen_dim = archive->gen_dim;

    for (int i = 0; i < num_elites; i++) {
        float dist_sq = 0.0f;
        for (int d = 0; d < hw_dim; d++) {
            float diff = archive->hw_coords[i * hw_dim + d] - cell->hw_centroid[d];
            dist_sq += diff * diff;
        }
        for (int d = 0; d < task_dim; d++) {
            float diff = archive->task_coords[i * task_dim + d] - cell->task_centroid[d];
            dist_sq += diff * diff;
        }
        for (int d = 0; d < gen_dim; d++) {
            float diff = archive->gen_coords[i * gen_dim + d] - cell->gen_centroid[d];
            dist_sq += diff * diff;
        }

        if (sqrtf(dist_sq) < cell->radius) {
            cell->density++;
            if (archive->fitness[i] > best_fitness) {
                best_fitness = archive->fitness[i];
                cell->best_elite_idx = i;
            }
        }
    }

    __shared__ float shared_densities[256];
    shared_densities[threadIdx.x] = (cell_id < num_cells) ? (float)cell->density : 0.0f;
    __syncthreads();

    if (threadIdx.x == 0) {
        float density_mean = 0.0f;
        for (int i = 0; i < blockDim.x && (blockIdx.x * blockDim.x + i) < num_cells; i++) {
            density_mean += shared_densities[i];
        }
        density_mean /= num_cells;

        float density_variance = 0.0f;
        for (int i = 0; i < blockDim.x && (blockIdx.x * blockDim.x + i) < num_cells; i++) {
            float diff = shared_densities[i] - density_mean;
            density_variance += diff * diff;
        }
        density_variance /= num_cells;

        int correlation_exponent_slot = derive_param_slot(genome_hash, "voronoi_correlation_exponent");
        float correlation_exponent = genome_to_param(
            genome, gradients, correlation_exponent_slot,
            ctx_metabolic, ctx_stress, ctx_morphogen,
            ctx_complexity, ctx_niche, ctx_learning, ctx_performance,
            VORONOI_CORRELATION_EXPONENT_MIN, VORONOI_CORRELATION_EXPONENT_MAX
        );

        for (int i = 0; i < blockDim.x && (blockIdx.x * blockDim.x + i) < num_cells; i++) {
            int idx = blockIdx.x * blockDim.x + i;

            if (density_mean <= 0.0f) {
                printf("FATAL [archive]: density_mean=%f\n", density_mean);
                return;
            }
            cells[idx].density_fluctuation = fabsf((float)cells[idx].density - (float)cells[idx].density_prev) / density_mean;
            cells[idx].density_prev = cells[idx].density;

            cells[idx].radius = powf(fmaxf(cells[idx].density_fluctuation, safe_epsilon(1.0f)), -correlation_exponent);
        }
    }
}

__global__ void insert_elite_kernel(
    GPUElite* __restrict__ archive,
    int* __restrict__ archive_size,
    float fitness_val,
    float coherence_val,
    float effective_rank_val,
    uint64_t genome_hash_val,
    uint32_t parent_id_0,
    uint32_t parent_id_1,
    uint16_t generation_val,
    float* hw_coords_new,
    float* task_coords_new,
    float* gen_coords_new,
    float task_performance_val,
    float* per_class_accuracy_new,
    int num_classes,
    VoronoiCell* __restrict__ cells,
    int num_cells
) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    bool is_duplicate = false;

    if (tid < *archive_size) {
        if (archive->genome_hash[tid] == genome_hash_val) {
            is_duplicate = true;
        }
    }

    __shared__ bool block_has_duplicate;
    if (threadIdx.x == 0) block_has_duplicate = false;
    __syncthreads();

    if (is_duplicate) {
        block_has_duplicate = true;
    }
    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (!block_has_duplicate) {
            int idx = atomicAdd(archive_size, 1);
            if (idx < MAX_ARCHIVE_SIZE) {
            int hw_dim = archive->hw_dim;
            int task_dim = archive->task_dim;
            int gen_dim = archive->gen_dim;

            archive->fitness[idx] = fitness_val;
            archive->coherence[idx] = coherence_val;
            archive->effective_rank[idx] = effective_rank_val;
            archive->genome_hash[idx] = genome_hash_val;
            archive->parent_ids[idx * PARENT_COUNT] = parent_id_0;
            archive->parent_ids[idx * PARENT_COUNT + 1] = parent_id_1;
            archive->generation[idx] = generation_val;

            int hw_base = idx * hw_dim;
            int task_base = idx * task_dim;
            int gen_base = idx * gen_dim;

            for (int d = 0; d < hw_dim; d++) {
                archive->hw_coords[hw_base + d] = hw_coords_new[d];
            }
            for (int d = 0; d < task_dim; d++) {
                archive->task_coords[task_base + d] = task_coords_new[d];
            }
            for (int d = 0; d < gen_dim; d++) {
                archive->gen_coords[gen_base + d] = gen_coords_new[d];
            }

            archive->task_performance[idx] = task_performance_val;
            for (int c = 0; c < num_classes; c++) {
                archive->per_class_accuracy[idx * num_classes + c] = per_class_accuracy_new[c];
            }

            float min_dist = 1e9f;
            int closest_cell = 0;
            for (int c = 0; c < num_cells; c++) {
                float dist_sq = elite_to_cell_distance_sq(archive, idx, &cells[c], hw_dim, task_dim, gen_dim);
                if (dist_sq < min_dist) {
                    min_dist = dist_sq;
                    closest_cell = c;
                }
            }

            atomicAdd(&cells[closest_cell].density, 1);
            cells[closest_cell].best_elite_idx = idx;
            }
        }
    }
}

__global__ void adapt_embedding_dim_kernel(
    float* __restrict__ reconstruction_error,
    int* __restrict__ embedding_dim,
    int current_dim,
    float error_threshold,
    int behavioral_dim_max
) {
    float local_error = reconstruction_error[threadIdx.x];

    float total_error = BlockReduce<BLOCK_SIZE>::sum(local_error);

    __shared__ float avg_error;
    if (threadIdx.x == 0) {
        avg_error = total_error / blockDim.x;

        if (avg_error > error_threshold && current_dim < behavioral_dim_max) {
            *embedding_dim = current_dim + 1;
        } else if (avg_error < error_threshold * 0.5f && current_dim > 2) {
            *embedding_dim = current_dim - 1;
        }
    }
}

__global__ void init_voronoi_cells_kernel(
    VoronoiCell* cells,
    int num_cells,
    int hw_dim,
    int task_dim,
    int gen_dim,
    unsigned int seed
) {
    int cell_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_id >= num_cells) return;

    curandState_t state;
    curand_init(seed, cell_id, 0, &state);

    VoronoiCell* cell = &cells[cell_id];

    for (int d = 0; d < hw_dim; d++) {
        cell->hw_centroid[d] = validated_curand_normal(&state, "init_voronoi_hw", cell_id * hw_dim + d) * 0.5f;
    }
    for (int d = 0; d < task_dim; d++) {
        cell->task_centroid[d] = validated_curand_normal(&state, "init_voronoi_task", cell_id * task_dim + d) * 0.5f;
    }
    for (int d = 0; d < gen_dim; d++) {
        cell->gen_centroid[d] = validated_curand_normal(&state, "init_voronoi_gen", cell_id * gen_dim + d) * 0.5f;
    }

    int total_dims = hw_dim + task_dim + gen_dim;
    float typical_spacing = powf((float)num_cells, -1.0f / total_dims);
    cell->radius = typical_spacing * 2.0f;

    cell->density = 0;
    cell->best_elite_idx = -1;
    cell->quality_threshold = 0.0f;

    if (cell_id == 0) {
        printf("[voronoi] cells=%p num=%d dims=%d radius=%f\n", cells, num_cells, total_dims, typical_spacing * 2.0f);
    }
}

// Streaming genome element reconstruction - implementation here requires GPUElite definition

#endif
