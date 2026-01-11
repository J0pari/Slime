
#ifndef GENEALOGY_CU
#define GENEALOGY_CU

#include "../config/config.cu"
#include <cuda_runtime.h>

__device__ void set_parent_ids(GPUElite* archive, int elite_idx, uint32_t parent1_id, uint32_t parent2_id) {
    archive->parent_ids[elite_idx * PARENT_COUNT] = parent1_id;
    archive->parent_ids[elite_idx * PARENT_COUNT + 1] = parent2_id;
}

__device__ int find_parent_by_hash(GPUElite* archive, int archive_size, uint64_t parent_hash) {
    // O(1) lookup via hash table
    return hash_table_lookup(
        archive->hash_table_keys,
        archive->hash_table_values,
        parent_hash
    );
}

__device__ void update_genealogy_on_spawn(GPUElite* archive, int new_elite_idx, int parent1_idx, int parent2_idx) {
    archive->parent_ids[new_elite_idx * PARENT_COUNT] = (parent1_idx >= 0) ? parent1_idx : 0;
    archive->parent_ids[new_elite_idx * PARENT_COUNT + 1] = (parent2_idx >= 0) ? parent2_idx : 0;
}

#endif
