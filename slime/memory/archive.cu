
#ifndef ARCHIVE_CU
#define ARCHIVE_CU
#include "../config/config.cu"
#include "../utils/cuda_primitives.cuh"
#include "behavioral_ops.cuh"
#include "genome_ops.cuh"
#include "tubes.cu"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <stdint.h>

constexpr int GENOME_HASH_TABLE_SIZE = 16384;  
constexpr uint64_t HASH_TABLE_EMPTY_KEY = 0ULL;  

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

    half* weight_deltas;             
    uint32_t* weight_delta_indices;  
    uint16_t* num_weight_deltas;     

    int* archived_num_heads;         
    int* archived_channels;          
    int* archived_head_dim;          

    uint64_t* hash_table_keys;       
    int* hash_table_values;          
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


__device__ __forceinline__ int hash_table_slot(uint64_t genome_hash) {
    return (int)((genome_hash * 11400714819323198485ULL) >> 50) & (GENOME_HASH_TABLE_SIZE - 1);
}

__device__ int hash_table_lookup(
    const uint64_t* __restrict__ keys,
    const int* __restrict__ values,
    uint64_t genome_hash
) {
    if (genome_hash == HASH_TABLE_EMPTY_KEY) return -1;  

    int slot = hash_table_slot(genome_hash);
    int probes = 0;

    while (probes < GENOME_HASH_TABLE_SIZE) {
        uint64_t key = keys[slot];
        if (key == genome_hash) {
            return values[slot];  
        }
        if (key == HASH_TABLE_EMPTY_KEY) {
            return -1;  
        }
        slot = (slot + 1) & (GENOME_HASH_TABLE_SIZE - 1);
        probes++;
    }
    return -1;  
}

__device__ bool hash_table_insert(
    uint64_t* __restrict__ keys,
    int* __restrict__ values,
    uint64_t genome_hash,
    int archive_idx
) {
    DEVICE_FATAL_IF(genome_hash == HASH_TABLE_EMPTY_KEY, "hash_table_insert: empty key sentinel passed as genome_hash");

    int slot = hash_table_slot(genome_hash);
    int probes = 0;

    while (probes < GENOME_HASH_TABLE_SIZE) {
        uint64_t expected = HASH_TABLE_EMPTY_KEY;
        uint64_t old = atomicCAS((unsigned long long*)&keys[slot], expected, genome_hash);

        if (old == HASH_TABLE_EMPTY_KEY) {
            values[slot] = archive_idx;
            return true;
        }
        DEVICE_FATAL_IF(old == genome_hash, "hash_table_insert: duplicate key - genome already in table");
        slot = (slot + 1) & (GENOME_HASH_TABLE_SIZE - 1);
        probes++;
    }
    DEVICE_FATAL_IF(true, "hash_table_insert: table full after probing all slots");
    return false;
}

__device__ void hash_table_remove(
    uint64_t* __restrict__ keys,
    int* __restrict__ values,
    uint64_t genome_hash
) {
    if (genome_hash == HASH_TABLE_EMPTY_KEY) return;

    int slot = hash_table_slot(genome_hash);
    int probes = 0;

    while (probes < GENOME_HASH_TABLE_SIZE) {
        if (keys[slot] == genome_hash) {
            keys[slot] = HASH_TABLE_EMPTY_KEY;
            values[slot] = -1;
            return;
        }
        DEVICE_FATAL_IF(keys[slot] == HASH_TABLE_EMPTY_KEY, "hash_table_remove: key not found in table");
        slot = (slot + 1) & (GENOME_HASH_TABLE_SIZE - 1);
        probes++;
    }
    DEVICE_FATAL_IF(true, "hash_table_remove: exhausted probe limit");
}

__global__ void init_hash_table_kernel(
    uint64_t* keys,
    int* values,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        keys[idx] = HASH_TABLE_EMPTY_KEY;
        values[idx] = -1;
    }
}

__global__ void rebuild_hash_table_kernel(
    GPUElite* archive,
    int archive_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < archive_size) {
        uint64_t genome_hash = archive->genome_hash[idx];
        if (genome_hash != HASH_TABLE_EMPTY_KEY) {
            hash_table_insert(archive->hash_table_keys, archive->hash_table_values, genome_hash, idx);
        }
    }
}

__constant__ uint16_t d_generation_counter;

__device__ __forceinline__ float elite_to_cell_distance_sq(
    const GPUElite* archive, int elite_idx,
    const VoronoiCell* cell, int hw_dim, int task_dim, int gen_dim
) {
    DEVICE_FATAL_IF(hw_dim <= 0 || hw_dim > BEHAVIORAL_DIM_MAX, "elite_to_cell_distance_sq: hw_dim invalid");
    DEVICE_FATAL_IF(task_dim <= 0 || task_dim > BEHAVIORAL_DIM_MAX, "elite_to_cell_distance_sq: task_dim invalid");
    DEVICE_FATAL_IF(gen_dim <= 0 || gen_dim > BEHAVIORAL_DIM_MAX, "elite_to_cell_distance_sq: gen_dim invalid");
    DEVICE_FATAL_IF(elite_idx < 0 || elite_idx >= MAX_ARCHIVE_SIZE, "elite_to_cell_distance_sq: elite_idx out of bounds");
    return compute_three_axis_distance_sq(
        &archive->hw_coords[elite_idx * hw_dim],
        &archive->task_coords[elite_idx * task_dim],
        &archive->gen_coords[elite_idx * gen_dim],
        cell->hw_centroid, cell->task_centroid, cell->gen_centroid,
        hw_dim, task_dim, gen_dim
    );
}

__device__ uint64_t gpu_sha256(float* genome, uint32_t size) {

    uint64_t hash = XORSHIFT_GOLDEN_RATIO_A;

    for (uint32_t i = 0; i < size; i++) {
        uint32_t bits = __float_as_uint(genome[i]);

        hash ^= bits + HASH_GOLDEN_RATIO_32 + (hash << JENKINS_FINAL_SHIFT_A) + (hash >> 2);
        hash += (hash << 3);
        hash ^= (hash >> JENKINS_FINAL_SHIFT_B);
        hash += (hash << (WMMA_TILE_DIM - 1));

        hash *= HASH_MIX_CONSTANT_B;
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

    DEVICE_FATAL_IF(hw_dim <= 0 || hw_dim > BEHAVIORAL_DIM_MAX, "update_voronoi: hw_dim invalid");
    DEVICE_FATAL_IF(task_dim <= 0 || task_dim > BEHAVIORAL_DIM_MAX, "update_voronoi: task_dim invalid");
    DEVICE_FATAL_IF(gen_dim <= 0 || gen_dim > BEHAVIORAL_DIM_MAX, "update_voronoi: gen_dim invalid");

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
                return;
            }
            cells[idx].density_fluctuation = fabsf((float)cells[idx].density - (float)cells[idx].density_prev) / density_mean;
            cells[idx].density_prev = cells[idx].density;

            cells[idx].radius = powf(fmaxf(cells[idx].density_fluctuation, safe_epsilon(1.0f)), -correlation_exponent);
        }
    }
}

__device__ void insert_elite_device(
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
    int existing_idx = hash_table_lookup(
        archive->hash_table_keys,
        archive->hash_table_values,
        genome_hash_val
    );
    if (existing_idx >= 0) {
        return;  
    }

    int idx = atomicAdd(archive_size, 1);
    if (idx >= MAX_ARCHIVE_SIZE) {
        atomicSub(archive_size, 1);
        return;
    }

    bool inserted = hash_table_insert(
        archive->hash_table_keys,
        archive->hash_table_values,
        genome_hash_val,
        idx
    );
    if (!inserted) {
        atomicSub(archive_size, 1);
        return;
    }

    int hw_dim = archive->hw_dim;
    int task_dim = archive->task_dim;
    int gen_dim = archive->gen_dim;

    DEVICE_FATAL_IF(hw_dim <= 0 || hw_dim > BEHAVIORAL_DIM_MAX, "insert_elite_device: hw_dim invalid");
    DEVICE_FATAL_IF(task_dim <= 0 || task_dim > BEHAVIORAL_DIM_MAX, "insert_elite_device: task_dim invalid");
    DEVICE_FATAL_IF(gen_dim <= 0 || gen_dim > BEHAVIORAL_DIM_MAX, "insert_elite_device: gen_dim invalid");
    DEVICE_FATAL_IF(num_classes <= 0 || num_classes > NUM_CLASSES_MAX, "insert_elite_device: num_classes invalid");

    archive->fitness[idx] = fitness_val;
    archive->coherence[idx] = coherence_val;
    archive->effective_rank[idx] = effective_rank_val;
    archive->genome_hash[idx] = genome_hash_val;
    archive->parent_ids[idx * PARENT_COUNT] = parent_id_0;
    archive->parent_ids[idx * PARENT_COUNT + 1] = parent_id_1;
    archive->generation[idx] = generation_val;

    for (int d = 0; d < hw_dim; d++) {
        archive->hw_coords[idx * hw_dim + d] = hw_coords_new[d];
    }
    for (int d = 0; d < task_dim; d++) {
        archive->task_coords[idx * task_dim + d] = task_coords_new[d];
    }
    for (int d = 0; d < gen_dim; d++) {
        archive->gen_coords[idx * gen_dim + d] = gen_coords_new[d];
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
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int existing_idx = hash_table_lookup(
        archive->hash_table_keys,
        archive->hash_table_values,
        genome_hash_val
    );
    if (existing_idx >= 0) {
        return;  
    }

    int idx = atomicAdd(archive_size, 1);
    if (idx >= MAX_ARCHIVE_SIZE) {
        atomicSub(archive_size, 1);
        return;
    }

    bool inserted = hash_table_insert(
        archive->hash_table_keys,
        archive->hash_table_values,
        genome_hash_val,
        idx
    );
    if (!inserted) {
        atomicSub(archive_size, 1);
        return;
    }

    int hw_dim = archive->hw_dim;
    int task_dim = archive->task_dim;
    int gen_dim = archive->gen_dim;

    DEVICE_FATAL_IF(hw_dim <= 0 || hw_dim > BEHAVIORAL_DIM_MAX, "insert_elite_kernel: hw_dim invalid");
    DEVICE_FATAL_IF(task_dim <= 0 || task_dim > BEHAVIORAL_DIM_MAX, "insert_elite_kernel: task_dim invalid");
    DEVICE_FATAL_IF(gen_dim <= 0 || gen_dim > BEHAVIORAL_DIM_MAX, "insert_elite_kernel: gen_dim invalid");
    DEVICE_FATAL_IF(num_classes <= 0 || num_classes > NUM_CLASSES_MAX, "insert_elite_kernel: num_classes invalid");

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

    DEVICE_FATAL_IF(hw_dim <= 0 || hw_dim > BEHAVIORAL_DIM_MAX, "init_voronoi: hw_dim invalid");
    DEVICE_FATAL_IF(task_dim <= 0 || task_dim > BEHAVIORAL_DIM_MAX, "init_voronoi: task_dim invalid");
    DEVICE_FATAL_IF(gen_dim <= 0 || gen_dim > BEHAVIORAL_DIM_MAX, "init_voronoi: gen_dim invalid");

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
    }
}


__global__ void compute_voronoi_occupancy_kernel(
    VoronoiCell* voronoi_cells,
    int num_voronoi_cells,
    float* voronoi_occupancy_histogram
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < num_voronoi_cells) {
        voronoi_occupancy_histogram[tid] = (float)voronoi_cells[tid].density;
    }
}

__device__ float get_ca_xavier_scale(
    int flat_idx, int num_heads, int channels, int head_dim,
    int* out_matrix, int* out_local_idx
);

__global__ void store_elite_weight_deltas_kernel(
    GPUElite* archive,
    int elite_idx,
    const half* perception_weights,
    const half* interaction_weights,
    const half* value_weights,
    int num_heads,
    int channels,
    int head_dim,
    uint64_t genome_hash,
    const float* genome,
    int* num_deltas_out
) {
    DEVICE_FATAL_IF(archive->weight_deltas == nullptr, "archive->weight_deltas is null");
    DEVICE_FATAL_IF(num_heads <= 0 || num_heads > NUM_HEADS_MAX, "store_weight_deltas: num_heads invalid");
    DEVICE_FATAL_IF(channels <= 0 || channels > CHANNELS_MAX, "store_weight_deltas: channels invalid");
    DEVICE_FATAL_IF(head_dim <= 0 || head_dim > HEAD_DIM_MAX, "store_weight_deltas: head_dim invalid");
    DEVICE_FATAL_IF(elite_idx < 0 || elite_idx >= MAX_ARCHIVE_SIZE, "store_weight_deltas: elite_idx invalid");

    int perception_size = num_heads * channels * head_dim;
    int interaction_size = num_heads * head_dim * head_dim;
    int value_size = num_heads * head_dim * channels;
    int total_size = perception_size + interaction_size + value_size;

    int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_idx >= total_size) return;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        archive->archived_num_heads[elite_idx] = num_heads;
        archive->archived_channels[elite_idx] = channels;
        archive->archived_head_dim[elite_idx] = head_dim;
    }

    int delta_threshold_slot = derive_param_slot(genome_hash, "weight_delta_threshold");
    float delta_threshold = (genome[delta_threshold_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE * 0.01f;

    curandState_t rand_state;
    curand_init(genome_hash, flat_idx, 0, &rand_state);

    int matrix, local_idx;
    float scale = get_ca_xavier_scale(flat_idx, num_heads, channels, head_dim, &matrix, &local_idx);
    float baseline = curand_normal(&rand_state) * scale;

    float current;
    if (matrix == 0) {
        current = __half2float(perception_weights[local_idx]);
    } else if (matrix == 1) {
        current = __half2float(interaction_weights[local_idx]);
    } else if (matrix == 2) {
        current = __half2float(value_weights[local_idx]);
    } else {
        DEVICE_FATAL("invalid matrix index from get_ca_xavier_scale");
    }

    float delta = current - baseline;

    if (fabsf(delta) > delta_threshold) {
        int slot = atomicAdd(num_deltas_out, 1);
        if (slot < MAX_WEIGHT_DELTAS_PER_ELITE) {
            int base_offset = elite_idx * MAX_WEIGHT_DELTAS_PER_ELITE;
            archive->weight_delta_indices[base_offset + slot] = flat_idx;
            archive->weight_deltas[base_offset + slot] = __float2half(delta);
        }
    }
}

__global__ void finalize_weight_deltas_kernel(
    GPUElite* archive,
    int elite_idx,
    int* num_deltas_out
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int count = min(*num_deltas_out, MAX_WEIGHT_DELTAS_PER_ELITE);
        archive->num_weight_deltas[elite_idx] = count;
    }
}

__global__ void restore_elite_weights_kernel(
    const GPUElite* archive,
    int elite_idx,
    half* perception_weights,
    half* interaction_weights,
    half* value_weights
) {
    DEVICE_FATAL_IF(archive->weight_deltas == nullptr, "archive->weight_deltas is null");
    DEVICE_FATAL_IF(elite_idx < 0 || elite_idx >= MAX_ARCHIVE_SIZE, "restore_weights: elite_idx invalid");

    int num_heads = archive->archived_num_heads[elite_idx];
    int channels = archive->archived_channels[elite_idx];
    int head_dim = archive->archived_head_dim[elite_idx];
    uint64_t genome_hash = archive->genome_hash[elite_idx];

    DEVICE_FATAL_IF(num_heads <= 0 || num_heads > NUM_HEADS_MAX, "restore_weights: archived num_heads invalid");
    DEVICE_FATAL_IF(channels <= 0 || channels > CHANNELS_MAX, "restore_weights: archived channels invalid");
    DEVICE_FATAL_IF(head_dim <= 0 || head_dim > HEAD_DIM_MAX, "restore_weights: archived head_dim invalid");

    int perception_size = num_heads * channels * head_dim;
    int interaction_size = num_heads * head_dim * head_dim;
    int value_size = num_heads * head_dim * channels;
    int total_size = perception_size + interaction_size + value_size;

    int flat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_idx >= total_size) return;

    curandState_t rand_state;
    curand_init(genome_hash, flat_idx, 0, &rand_state);

    int matrix, local_idx;
    float scale = get_ca_xavier_scale(flat_idx, num_heads, channels, head_dim, &matrix, &local_idx);
    float val = curand_normal(&rand_state) * scale;

    if (matrix == 0) {
        perception_weights[local_idx] = __float2half(val);
    } else if (matrix == 1) {
        interaction_weights[local_idx] = __float2half(val);
    } else if (matrix == 2) {
        value_weights[local_idx] = __float2half(val);
    }
}

__global__ void apply_weight_deltas_kernel(
    const GPUElite* archive,
    int elite_idx,
    half* perception_weights,
    half* interaction_weights,
    half* value_weights
) {
    DEVICE_FATAL_IF(archive->weight_deltas == nullptr, "archive->weight_deltas is null");
    DEVICE_FATAL_IF(elite_idx < 0 || elite_idx >= MAX_ARCHIVE_SIZE, "apply_deltas: elite_idx invalid");

    int num_heads = archive->archived_num_heads[elite_idx];
    int channels = archive->archived_channels[elite_idx];
    int head_dim = archive->archived_head_dim[elite_idx];

    DEVICE_FATAL_IF(num_heads <= 0 || num_heads > NUM_HEADS_MAX, "apply_deltas: archived num_heads invalid");
    DEVICE_FATAL_IF(channels <= 0 || channels > CHANNELS_MAX, "apply_deltas: archived channels invalid");
    DEVICE_FATAL_IF(head_dim <= 0 || head_dim > HEAD_DIM_MAX, "apply_deltas: archived head_dim invalid");

    int num_deltas = archive->num_weight_deltas[elite_idx];
    int delta_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (delta_idx >= num_deltas) return;

    int base_offset = elite_idx * MAX_WEIGHT_DELTAS_PER_ELITE;
    uint32_t flat_idx = archive->weight_delta_indices[base_offset + delta_idx];
    half delta = archive->weight_deltas[base_offset + delta_idx];

    int matrix, local_idx;
    get_ca_xavier_scale(flat_idx, num_heads, channels, head_dim, &matrix, &local_idx);

    if (matrix == 0) {
        float current = __half2float(perception_weights[local_idx]);
        perception_weights[local_idx] = __float2half(current + __half2float(delta));
    } else if (matrix == 1) {
        float current = __half2float(interaction_weights[local_idx]);
        interaction_weights[local_idx] = __float2half(current + __half2float(delta));
    } else if (matrix == 2) {
        float current = __half2float(value_weights[local_idx]);
        value_weights[local_idx] = __float2half(current + __half2float(delta));
    }
}

__global__ void archive_to_tube_kernel(
    const GPUElite* __restrict__ archive,
    int elite_idx,
    TemporalTube* __restrict__ tube
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    DEVICE_FATAL_IF(elite_idx < 0 || elite_idx >= MAX_ARCHIVE_SIZE, "archive_to_tube: elite_idx invalid");
    DEVICE_FATAL_IF(tube == nullptr, "archive_to_tube: tube is null");
    DEVICE_FATAL_IF(tube->entries == nullptr, "archive_to_tube: tube entries is null");

    int hw_dim = archive->hw_dim;
    int task_dim = archive->task_dim;
    int gen_dim = archive->gen_dim;

    DEVICE_FATAL_IF(hw_dim <= 0 || hw_dim > BEHAVIORAL_DIM_MAX, "archive_to_tube: hw_dim invalid");
    DEVICE_FATAL_IF(task_dim <= 0 || task_dim > BEHAVIORAL_DIM_MAX, "archive_to_tube: task_dim invalid");
    DEVICE_FATAL_IF(gen_dim <= 0 || gen_dim > BEHAVIORAL_DIM_MAX, "archive_to_tube: gen_dim invalid");

    int total_dim = hw_dim + task_dim + gen_dim;
    int idx = tube->head;

    DEVICE_FATAL_IF(tube->entries[idx].data == nullptr, "archive_to_tube: entry data buffer is null");
    DEVICE_FATAL_IF(tube->entries[idx].size < total_dim, "archive_to_tube: entry size too small for behavioral dims");

    int offset = 0;
    for (int d = 0; d < hw_dim; d++) {
        tube->entries[idx].data[offset++] = archive->hw_coords[elite_idx * hw_dim + d];
    }
    for (int d = 0; d < task_dim; d++) {
        tube->entries[idx].data[offset++] = archive->task_coords[elite_idx * task_dim + d];
    }
    for (int d = 0; d < gen_dim; d++) {
        tube->entries[idx].data[offset++] = archive->gen_coords[elite_idx * gen_dim + d];
    }

    tube->entries[idx].size = total_dim;
    tube->entries[idx].timestamp = tube->global_time;
    tube->entries[idx].importance = archive->fitness[elite_idx];
    tube->entries[idx].decay_factor = 1.0f;

    tube->head = (tube->head + 1) % tube->capacity;
    if (tube->count < tube->capacity) {
        tube->count++;
    }
}

__global__ void tube_to_archive_query_kernel(
    const GPUElite* __restrict__ archive,
    int archive_size,
    const float* __restrict__ tube_recall,
    int hw_dim,
    int task_dim,
    int gen_dim,
    int* __restrict__ best_elite_idx,
    float* __restrict__ best_distance
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    DEVICE_FATAL_IF(hw_dim <= 0 || hw_dim > BEHAVIORAL_DIM_MAX, "tube_to_archive: hw_dim invalid");
    DEVICE_FATAL_IF(task_dim <= 0 || task_dim > BEHAVIORAL_DIM_MAX, "tube_to_archive: task_dim invalid");
    DEVICE_FATAL_IF(gen_dim <= 0 || gen_dim > BEHAVIORAL_DIM_MAX, "tube_to_archive: gen_dim invalid");

    int total_dim = hw_dim + task_dim + gen_dim;

    float local_best_dist = 1e9f;
    int local_best_idx = -1;

    for (int i = tid; i < archive_size; i += num_threads) {
        float dist_sq = 0.0f;

        int offset = 0;
        for (int d = 0; d < hw_dim; d++) {
            float diff = archive->hw_coords[i * hw_dim + d] - tube_recall[offset++];
            dist_sq += diff * diff;
        }
        for (int d = 0; d < task_dim; d++) {
            float diff = archive->task_coords[i * task_dim + d] - tube_recall[offset++];
            dist_sq += diff * diff;
        }
        for (int d = 0; d < gen_dim; d++) {
            float diff = archive->gen_coords[i * gen_dim + d] - tube_recall[offset++];
            dist_sq += diff * diff;
        }

        if (dist_sq < local_best_dist) {
            local_best_dist = dist_sq;
            local_best_idx = i;
        }
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float other_dist = __shfl_down_sync(0xffffffff, local_best_dist, offset);
        int other_idx = __shfl_down_sync(0xffffffff, local_best_idx, offset);
        if (other_dist < local_best_dist) {
            local_best_dist = other_dist;
            local_best_idx = other_idx;
        }
    }

    if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
        float old_dist = *best_distance;
        while (local_best_dist < old_dist) {
            float assumed = old_dist;
            old_dist = __int_as_float(atomicCAS(
                (int*)best_distance,
                __float_as_int(assumed),
                __float_as_int(local_best_dist)
            ));
            if (old_dist == assumed) {
                *best_elite_idx = local_best_idx;
                break;
            }
        }
    }
}

#endif
