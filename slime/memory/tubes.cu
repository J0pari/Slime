// slime/memory/tubes.cu - Temporal memory with decay
#ifndef TUBES_CU
#define TUBES_CU
#include <cuda_runtime.h>

constexpr int MAX_MEMORY_SIZE = 1024;
constexpr int MAX_HISTORY_LENGTH = 100;
constexpr float DEFAULT_DECAY_RATE = 0.95f;

// Memory entry with timestamp
struct MemoryEntry {
    float* data;
    int size;
    float timestamp;
    float decay_factor;
    float importance;
};

// Temporal memory tube
struct TemporalTube {
    MemoryEntry* entries;
    int capacity;
    int head;  // Circular buffer head
    int count;
    float global_time;
    float decay_rate;
};

// Store memory with importance weighting
__global__ void store_memory_kernel(
    TemporalTube* tube,
    float* data,
    int size,
    float importance
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int idx = tube->head;

        // Store in circular buffer
        tube->entries[idx].size = size;
        tube->entries[idx].timestamp = tube->global_time;
        tube->entries[idx].importance = importance;
        tube->entries[idx].decay_factor = 1.0f;

        // Copy data
        if (tube->entries[idx].data && data) {
            for (int i = 0; i < size; i++) {
                tube->entries[idx].data[i] = data[i];
            }
        }

        // Update circular buffer
        tube->head = (tube->head + 1) % tube->capacity;
        if (tube->count < tube->capacity) {
            tube->count++;
        }
    }
}

// Apply exponential decay
__global__ void apply_decay_kernel(
    TemporalTube* tube,
    float timestep
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < tube->count) {
        int entry_idx = (tube->head - tube->count + idx + tube->capacity) % tube->capacity;
        MemoryEntry* entry = &tube->entries[entry_idx];

        // Exponential decay: e^(-t/τ)
        float age = tube->global_time - entry->timestamp;
        entry->decay_factor = expf(-age * tube->decay_rate);

        // Importance-weighted decay (important memories decay slower)
        entry->decay_factor *= (1.0f + entry->importance * 0.5f);
    }

    if (idx == 0) {
        tube->global_time += timestep;
    }
}

// Recall memories with weighted blending
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

        // Blend memories based on similarity and decay
        for (int i = 0; i < tube->count; i++) {
            int entry_idx = (tube->head - tube->count + i + tube->capacity) % tube->capacity;
            MemoryEntry* entry = &tube->entries[entry_idx];

            // Compute similarity to query
            float similarity = 0.0f;
            if (query && entry->data) {
                for (int j = 0; j < min(query_size, entry->size); j++) {
                    similarity += query[j] * entry->data[j];
                }
                similarity = tanhf(similarity / sqrtf((float)query_size));
            }

            // Weight by decay and similarity
            float weight = entry->decay_factor * (0.5f + 0.5f * similarity);

            if (tid < entry->size && entry->data) {
                weighted_sum += entry->data[tid] * weight;
                weight_total += weight;
            }
        }

        // Normalize
        output[tid] = weight_total > 0.0f ? weighted_sum / weight_total : 0.0f;
    }
}

// Prune old memories below threshold
__global__ void prune_memories_kernel(
    TemporalTube* tube,
    float decay_threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < tube->count) {
        int entry_idx = (tube->head - tube->count + idx + tube->capacity) % tube->capacity;
        MemoryEntry* entry = &tube->entries[entry_idx];

        // Mark for removal if below threshold
        if (entry->decay_factor < decay_threshold) {
            entry->size = 0;  // Mark as invalid
        }
    }

    // Compact valid entries (single-threaded for simplicity)
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

// Consolidate similar memories
__global__ void consolidate_memories_kernel(
    TemporalTube* tube,
    float similarity_threshold
) {
    __shared__ bool merged[MAX_MEMORY_SIZE];

    int idx = threadIdx.x;
    if (idx < tube->count) {
        merged[idx] = false;
    }
    __syncthreads();

    if (idx < tube->count && !merged[idx]) {
        int entry_idx = (tube->head - tube->count + idx + tube->capacity) % tube->capacity;
        MemoryEntry* entry = &tube->entries[entry_idx];

        // Find similar memories to merge
        for (int j = idx + 1; j < tube->count; j++) {
            if (merged[j]) continue;

            int other_idx = (tube->head - tube->count + j + tube->capacity) % tube->capacity;
            MemoryEntry* other = &tube->entries[other_idx];

            // Compute similarity
            float similarity = 0.0f;
            if (entry->data && other->data) {
                for (int k = 0; k < min(entry->size, other->size); k++) {
                    similarity += entry->data[k] * other->data[k];
                }
                similarity /= sqrtf((float)min(entry->size, other->size));
            }

            // Merge if similar
            if (similarity > similarity_threshold) {
                // Average the memories
                for (int k = 0; k < min(entry->size, other->size); k++) {
                    entry->data[k] = (entry->data[k] + other->data[k]) * 0.5f;
                }

                // Combine importance and decay
                entry->importance = fmaxf(entry->importance, other->importance);
                entry->decay_factor = fmaxf(entry->decay_factor, other->decay_factor);

                merged[j] = true;
            }
        }
    }
}

// Initialize temporal tube
__global__ void init_tube_kernel(
    TemporalTube* tube,
    int capacity,
    float decay_rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < capacity) {
        tube->entries[idx].data = nullptr;
        tube->entries[idx].size = 0;
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

// Compute memory statistics
__global__ void memory_stats_kernel(
    TemporalTube* tube,
    float* avg_decay,
    float* total_importance,
    int* active_memories
) {
    __shared__ float shared_decay[256];
    __shared__ float shared_importance[256];
    __shared__ int shared_count[256];

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

    shared_decay[tid] = local_decay;
    shared_importance[tid] = local_importance;
    shared_count[tid] = local_count;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_decay[tid] += shared_decay[tid + stride];
            shared_importance[tid] += shared_importance[tid + stride];
            shared_count[tid] += shared_count[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(avg_decay, shared_decay[0]);
        atomicAdd(total_importance, shared_importance[0]);
        atomicAdd(active_memories, shared_count[0]);
    }
}

#endif // TUBES_CU
