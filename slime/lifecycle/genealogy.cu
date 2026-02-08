
#ifndef GENEALOGY_CU
#define GENEALOGY_CU

#include "../config/config.cu"
#include "../debug/param_validator.cu"
#include <cuda_runtime.h>

__device__ void set_parent_ids(GPUElite* archive, int elite_idx, uint32_t parent1_id, uint32_t parent2_id) {
    DEVICE_VALIDATE_PTR(archive);
    DEVICE_VALIDATE_RANGE(elite_idx, 0, MAX_ARCHIVE_SIZE - 1);
    archive->parent_ids[elite_idx * PARENT_COUNT] = parent1_id;
    archive->parent_ids[elite_idx * PARENT_COUNT + 1] = parent2_id;
}

__device__ int find_parent_by_hash(GPUElite* archive, int archive_size, uint64_t parent_hash) {
    DEVICE_VALIDATE_PTR(archive);
    DEVICE_VALIDATE_RANGE(archive_size, 0, MAX_ARCHIVE_SIZE);
    DEVICE_VALIDATE_PTR(archive->hash_table_keys);
    DEVICE_VALIDATE_PTR(archive->hash_table_values);
    int result = hash_table_lookup(
        archive->hash_table_keys,
        archive->hash_table_values,
        parent_hash
    );
    if (result >= 0) {
        DEVICE_VALIDATE_ARCHIVE_IDX(result, archive_size);
    }
    return result;
}

__device__ void update_genealogy_on_spawn(GPUElite* archive, int new_elite_idx, int parent1_idx, int parent2_idx) {
    DEVICE_VALIDATE_PTR(archive);
    DEVICE_VALIDATE_RANGE(new_elite_idx, 0, MAX_ARCHIVE_SIZE - 1);
    // Use UINT32_MAX as sentinel for "no parent" (not 0 which is a valid index)
    archive->parent_ids[new_elite_idx * PARENT_COUNT] = (parent1_idx >= 0) ? (uint32_t)parent1_idx : UINT32_MAX;
    archive->parent_ids[new_elite_idx * PARENT_COUNT + 1] = (parent2_idx >= 0) ? (uint32_t)parent2_idx : UINT32_MAX;
}

#endif
