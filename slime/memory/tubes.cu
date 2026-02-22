
#ifndef TUBES_CU
#define TUBES_CU
#include "../config/config.cu"
#include "../core/organism.cu"
#include <cuda_runtime.h>

__device__ void store_memory_device(
    Organism* organism,
    float* data,
    int size,
    float importance
) {
    TemporalTube* tube = organism->temporal_tube;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int idx = tube->head;

        tube->entries[idx].size = size;
        tube->entries[idx].timestamp = tube->global_time;
        tube->entries[idx].importance = importance;
        tube->entries[idx].decay_factor = 1.0f;

        if (data && size > 0 && tube->entries[idx].data) {
            for (int i = 0; i < size; i++) {
                tube->entries[idx].data[i] = data[i];
            }
        }

        tube->head = (tube->head + 1) % tube->capacity;
        if (tube->count < tube->capacity) {
            tube->count++;
        }
    }
}

__device__ void recall_memory_device(
    Organism* organism,
    float* query,
    float* output,
    int query_size,
    int output_size
) {
    TemporalTube* tube = organism->temporal_tube;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < output_size) {
        float weighted_sum = 0.0f;
        float weight_total = 0.0f;

        for (int i = 0; i < tube->count; i++) {
            int entry_idx = (tube->head - tube->count + i + tube->capacity) % tube->capacity;
            MemoryEntry* entry = &tube->entries[entry_idx];

            float similarity = 0.0f;
            if (query && entry->data) {
                for (int j = 0; j < min(query_size, entry->size); j++) {
                    similarity += query[j] * entry->data[j];
                }
                similarity = tanhf(similarity / sqrtf((float)query_size));
            }

            float weight = entry->decay_factor * (0.5f + 0.5f * similarity);

            if (tid < entry->size && entry->data) {
                weighted_sum += entry->data[tid] * weight;
                weight_total += weight;
            }
        }

        if (weight_total > 0.0f) {
            output[tid] = weighted_sum / weight_total;
        }
    }
}

__device__ void init_tube_device(Organism* organism) {
    TemporalTube* tube = organism->temporal_tube;
    int capacity = organism->tube_capacity;
    float decay_rate = organism->tube_decay_rate;
    float* data_buffer = organism->history_data_buffer;
    int entry_size = organism->tube_entry_size;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    for (int idx = thread_id; idx < capacity; idx += total_threads) {
        tube->entries[idx].data = &data_buffer[idx * entry_size];
        tube->entries[idx].size = entry_size;
        tube->entries[idx].timestamp = 0.0f;
        tube->entries[idx].decay_factor = 1.0f;
        tube->entries[idx].importance = 0.0f;
    }

    if (thread_id == 0) {
        tube->capacity = capacity;
        tube->head = 0;
        tube->count = 0;
        tube->global_time = 0.0f;
        tube->decay_rate = decay_rate;
    }
}


__device__ void memory_stats_device(
    Organism* organism,
    float* avg_decay,
    float* total_importance,
    int* active_memories
) {
    TemporalTube* tube = organism->temporal_tube;
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float local_decay = 0.0f;
    float local_importance = 0.0f;
    int local_count = 0;

    if (idx < tube->count) {
        int entry_idx = (tube->head - tube->count + idx + tube->capacity) % tube->capacity;
        MemoryEntry* entry = &tube->entries[entry_idx];

        if (entry->size > 0) {
            local_decay = entry->decay_factor;
            local_importance = entry->importance;
            local_count = 1;
        }
    }

    float total_decay = BlockReduce<BLOCK_SIZE>::sum(local_decay);
    float total_imp = BlockReduce<BLOCK_SIZE>::sum(local_importance);
    int total_count = BlockReduce<BLOCK_SIZE>::sum(local_count);

    if (tid == 0) {
        atomicAdd(avg_decay, total_decay);
        atomicAdd(total_importance, total_imp);
        atomicAdd(active_memories, total_count);
    }
}

__device__ void store_navigation_history_device(Organism* organism) {
    BehavioralState* agents = organism->behavioral_agents;
    int hw_dim = organism->behavioral_dim_hw;
    int task_dim = organism->behavioral_dim_task;
    int gen_dim = organism->behavioral_dim_gen;

    int tid = threadIdx.x;

    if (tid < POOL_CAPACITY_MAX) {
        int behavioral_dim = hw_dim + task_dim + gen_dim;
        int memory_entry_size = behavioral_dim + AGENT_SPATIAL_DIMS;
        float* d_memory_data = organism->memory_data_pool + tid * memory_entry_size;

        d_memory_data[0] = agents[tid].position[0];
        d_memory_data[1] = agents[tid].position[1];
        d_memory_data[2] = agents[tid].velocity[0];
        d_memory_data[3] = agents[tid].velocity[1];

        int offset = AGENT_SPATIAL_DIMS;
        for (int i = 0; i < hw_dim; i++) {
            d_memory_data[offset++] = agents[tid].hw_coords[i];
        }
        for (int i = 0; i < task_dim; i++) {
            d_memory_data[offset++] = agents[tid].task_coords[i];
        }
        for (int i = 0; i < gen_dim; i++) {
            d_memory_data[offset++] = agents[tid].gen_coords[i];
        }

        float importance = agents[tid].exploration_noise;

        if (tid == 0) {
            printf("V:nav_hist_pre_store gen=%d\n", organism->generation);
            store_memory_device(organism, d_memory_data, memory_entry_size, importance);
            printf("V:nav_hist_post_store gen=%d\n", organism->generation);
        }
    }
}

#endif
