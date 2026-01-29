
#ifndef TUBES_CU
#define TUBES_CU
#include "../config/config.cu"
#include <cuda_runtime.h>

struct MemoryEntry {
    float* data;
    int size;
    float timestamp;
    float decay_factor;
    float importance;
};

struct TemporalTube {
    MemoryEntry* entries;
    int capacity;
    int head;
    int count;
    float global_time;
    float decay_rate;
};

__global__ void store_memory_kernel(
    TemporalTube* tube,
    float* data,
    int size,
    float importance
) {
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

__global__ void apply_decay_kernel(
    TemporalTube* tube,
    float timestep
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < tube->count) {
        int entry_idx = (tube->head - tube->count + idx + tube->capacity) % tube->capacity;
        MemoryEntry* entry = &tube->entries[entry_idx];

        float age = tube->global_time - entry->timestamp;
        entry->decay_factor = expf(-age * tube->decay_rate);

        entry->decay_factor *= (1.0f + entry->importance * 0.5f);
    }

    if (idx == 0) {
        tube->global_time += timestep;
    }
}

__global__ void recall_memory_kernel(
    TemporalTube* tube,
    float* query,
    float* output,
    int query_size,
    int output_size
) {
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

        // Temporal coherence: if no memories contribute, preserve existing output value
        if (weight_total > 0.0f) {
            output[tid] = weighted_sum / weight_total;
        }
        // else: preserve existing output[tid] (no memories to recall)
    }
}

__global__ void prune_memories_kernel(
    TemporalTube* tube,
    float decay_threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < tube->count) {
        int entry_idx = (tube->head - tube->count + idx + tube->capacity) % tube->capacity;
        MemoryEntry* entry = &tube->entries[entry_idx];

        if (entry->decay_factor < decay_threshold) {
            entry->size = 0;
        }
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int write_idx = 0;
        for (int i = 0; i < tube->count; i++) {
            int read_idx = (tube->head - tube->count + i + tube->capacity) % tube->capacity;
            if (tube->entries[read_idx].size > 0) {
                if (read_idx != write_idx) {
                    tube->entries[write_idx] = tube->entries[read_idx];
                }
                write_idx = (write_idx + 1) % tube->capacity;
            }
        }
        tube->count = write_idx;
    }
}

__global__ void consolidate_memories_kernel(
    TemporalTube* tube,
    float similarity_threshold
) {
    __shared__ bool merged[MAX_MEMORY_SIZE + BANK_PAD];

    int idx = threadIdx.x;
    if (idx < tube->count) {
        merged[idx] = false;
    }
    __syncthreads();

    if (idx < tube->count && !merged[idx]) {
        int entry_idx = (tube->head - tube->count + idx + tube->capacity) % tube->capacity;
        MemoryEntry* entry = &tube->entries[entry_idx];

        for (int j = idx + 1; j < tube->count; j++) {
            if (merged[j]) continue;

            int other_idx = (tube->head - tube->count + j + tube->capacity) % tube->capacity;
            MemoryEntry* other = &tube->entries[other_idx];

            float similarity = 0.0f;
            if (entry->data && other->data) {
                int min_size = min(entry->size, other->size);
                // Skip this pair if no valid data, continue to check other pairs
                if (min_size > 0) {
                    for (int k = 0; k < min_size; k++) {
                        similarity += entry->data[k] * other->data[k];
                    }
                    similarity /= sqrtf((float)min_size);
                }
            }

            if (similarity > similarity_threshold) {

                for (int k = 0; k < min(entry->size, other->size); k++) {
                    entry->data[k] = (entry->data[k] + other->data[k]) * 0.5f;
                }

                entry->importance = fmaxf(entry->importance, other->importance);
                entry->decay_factor = fmaxf(entry->decay_factor, other->decay_factor);

                merged[j] = true;
            }
        }
    }
}

__global__ void init_tube_kernel(
    TemporalTube* tube,
    int capacity,
    float decay_rate,
    float* data_buffer,
    int entry_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < capacity) {
        // Wire data immediately - never leave entries with null data
        tube->entries[idx].data = &data_buffer[idx * entry_size];
        tube->entries[idx].size = entry_size;
        tube->entries[idx].timestamp = 0.0f;
        tube->entries[idx].decay_factor = 1.0f;
        tube->entries[idx].importance = 0.0f;
    }

    if (idx == 0) {
        tube->capacity = capacity;
        tube->head = 0;
        tube->count = 0;
        tube->global_time = 0.0f;
        tube->decay_rate = decay_rate;
    }
}


__global__ void memory_stats_kernel(
    TemporalTube* tube,
    float* avg_decay,
    float* total_importance,
    int* active_memories
) {
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

#endif
