// slime/memory/pool.cu - Component pool management 
#pragma once
#include <cuda_runtime.h>
#include <cuda/atomic>

constexpr int MAX_POOL_SIZE = 256;
constexpr int MIN_POOL_SIZE = 8;
constexpr int GENOME_SIZE = 512;  // Exact size from blueprint

// Pool entry structure
struct PoolEntry {
    int id;
    float fitness;
    float coherence;
    float hunger;
    int age;
    bool alive;
    float genome[GENOME_SIZE];
    float gradients[GENOME_SIZE];
};

// Component pool
struct ComponentPool {
    PoolEntry* entries;
    cuda::atomic<int> active_count;
    cuda::atomic<int> total_spawned;
    cuda::atomic<int> total_culled;
    int capacity;
};

// xorshift128+ PRNG state
struct PRNGState {
    uint64_t s0;
    uint64_t s1;

    __device__ float next() {
        uint64_t x = s0;
        uint64_t const y = s1;
        s0 = y;
        x ^= x << 23;
        s1 = x ^ y ^ (x >> 17) ^ (y >> 26);
        return (s1 + y) / 18446744073709551616.0f;
    }

    __device__ float gaussian() {
        // Box-Muller transform for Gaussian distribution
        float u1 = next();
        float u2 = next();
        float r = sqrtf(-2.0f * logf(u1 + 1e-10f));
        float theta = 2.0f * 3.14159265f * u2;
        return r * cosf(theta);
    }
};

// Spawn component kernel with full mutation
__global__ void spawn_component_kernel(
    ComponentPool* pool,
    int parent_id,
    float mutation_rate
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int count = pool->active_count.load();

        if (count < pool->capacity) {
            int new_id = pool->total_spawned.fetch_add(1);

            // Find empty slot
            for (int i = 0; i < pool->capacity; i++) {
                if (!pool->entries[i].alive) {
                    pool->entries[i].id = new_id;
                    pool->entries[i].fitness = 0.0f;
                    pool->entries[i].coherence = 0.0f;
                    pool->entries[i].hunger = 0.5f;
                    pool->entries[i].age = 0;
                    pool->entries[i].alive = true;

                    // Inherit from parent with mutation
                    if (parent_id >= 0 && parent_id < pool->capacity) {
                        PoolEntry* parent = &pool->entries[parent_id];

                        // Initialize PRNG with unique seeds
                        PRNGState rng;
                        rng.s0 = new_id * 0x9e3779b97f4a7c15ULL;
                        rng.s1 = parent_id * 0xbf58476d1ce4e5b9ULL;

                        // Copy and mutate genome
                        for (int j = 0; j < GENOME_SIZE; j++) {
                            pool->entries[i].genome[j] = parent->genome[j];

                            if (rng.next() < mutation_rate) {
                                // Gaussian mutation
                                pool->entries[i].genome[j] += rng.gaussian() * 0.1f;
                                pool->entries[i].genome[j] = fmaxf(-1.0f, fminf(1.0f,
                                    pool->entries[i].genome[j]));
                            }
                        }

                        // Copy gradients with decay
                        for (int j = 0; j < GENOME_SIZE; j++) {
                            pool->entries[i].gradients[j] = parent->gradients[j] * 0.9f;
                        }
                    } else {
                        // Random initialization if no parent
                        PRNGState rng;
                        rng.s0 = new_id * 0x9e3779b97f4a7c15ULL;
                        rng.s1 = new_id * 0xbf58476d1ce4e5b9ULL;

                        for (int j = 0; j < GENOME_SIZE; j++) {
                            pool->entries[i].genome[j] = rng.next() * 2.0f - 1.0f;
                            pool->entries[i].gradients[j] = 0.0f;
                        }
                    }

                    pool->active_count.fetch_add(1);
                    break;
                }
            }
        }
    }
}

// Cull weak components based on fitness
__global__ void cull_weak_kernel(
    ComponentPool* pool,
    float threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pool->capacity) {
        PoolEntry* entry = &pool->entries[idx];

        // Cull based on fitness threshold
        if (entry->alive && entry->fitness < threshold) {
            entry->alive = false;
            pool->active_count.fetch_sub(1);
            pool->total_culled.fetch_add(1);
        }
    }
}

// Cull hungry components (high hunger = low coherence)
__global__ void cull_hungry_kernel(
    ComponentPool* pool,
    float hunger_threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pool->capacity) {
        PoolEntry* entry = &pool->entries[idx];

        // hunger = 1.0 - coherence (from blueprint)
        if (entry->alive && entry->hunger > hunger_threshold) {
            entry->alive = false;
            pool->active_count.fetch_sub(1);
            pool->total_culled.fetch_add(1);
        }
    }
}

// Age all components
__global__ void age_components_kernel(ComponentPool* pool) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pool->capacity && pool->entries[idx].alive) {
        pool->entries[idx].age++;
    }
}

// Update hunger based on coherence
__global__ void update_hunger_kernel(ComponentPool* pool) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pool->capacity && pool->entries[idx].alive) {
        // hunger = 1.0 - coherence (curiosity-driven lifecycle)
        pool->entries[idx].hunger = 1.0f - pool->entries[idx].coherence;
    }
}

// Sort pool by fitness (bitonic sort)
__global__ void sort_by_fitness_kernel(
    ComponentPool* pool,
    int stage,
    int step
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pair_distance = 1 << step;
    int block_width = pair_distance << 1;
    int block_id = tid / pair_distance;

    int left_id = block_id * block_width + (tid % pair_distance);
    int right_id = left_id + pair_distance;

    if (right_id < pool->capacity) {
        PoolEntry left = pool->entries[left_id];
        PoolEntry right = pool->entries[right_id];

        bool ascending = ((block_id >> stage) & 1) == 0;
        bool swap = ascending ? (left.fitness > right.fitness) :
                               (left.fitness < right.fitness);

        if (swap) {
            pool->entries[left_id] = right;
            pool->entries[right_id] = left;
        }
    }
}

// Select top-k components
__global__ void select_top_k_kernel(
    ComponentPool* pool,
    int* selected_indices,
    int k
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < k && tid < pool->capacity) {
        // After sorting, first k entries are best
        if (pool->entries[tid].alive) {
            selected_indices[tid] = tid;
        } else {
            selected_indices[tid] = -1;
        }
    }
}

// Compute full genome distance
__device__ float compute_genome_distance(
    const float* genome1,
    const float* genome2
) {
    float distance = 0.0f;

    // Euclidean distance across all genome dimensions
    for (int i = 0; i < GENOME_SIZE; i++) {
        float diff = genome1[i] - genome2[i];
        distance += diff * diff;
    }

    return sqrtf(distance / GENOME_SIZE);
}

// Diversity-based selection with full genome distance
__global__ void diversity_selection_kernel(
    ComponentPool* pool,
    int* selected_indices,
    int num_select
) {
    __shared__ float distances[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < pool->capacity && pool->entries[idx].alive) {
        // Compute average genome distance to others
        float avg_distance = 0.0f;
        int count = 0;

        for (int i = 0; i < pool->capacity; i++) {
            if (i != idx && pool->entries[i].alive) {
                // Full genome distance computation
                float dist = compute_genome_distance(
                    pool->entries[idx].genome,
                    pool->entries[i].genome
                );
                avg_distance += dist;
                count++;
            }
        }

        distances[tid] = count > 0 ? avg_distance / count : 0.0f;
    } else {
        distances[tid] = -1.0f;
    }

    __syncthreads();

    // Select most diverse
    if (tid == 0) {
        for (int i = 0; i < num_select && i < blockDim.x; i++) {
            float max_dist = -1.0f;
            int max_idx = -1;

            for (int j = 0; j < blockDim.x; j++) {
                if (distances[j] > max_dist) {
                    max_dist = distances[j];
                    max_idx = j;
                }
            }

            if (max_idx >= 0) {
                selected_indices[i] = blockIdx.x * blockDim.x + max_idx;
                distances[max_idx] = -1.0f;  // Mark as selected
            }
        }
    }
}

// Initialize pool
__global__ void init_pool_kernel(
    ComponentPool* pool,
    int capacity
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < capacity) {
        pool->entries[idx].id = -1;
        pool->entries[idx].alive = false;
        pool->entries[idx].fitness = 0.0f;
        pool->entries[idx].coherence = 0.0f;
        pool->entries[idx].hunger = 0.5f;
        pool->entries[idx].age = 0;

        // Initialize genome and gradients
        for (int i = 0; i < GENOME_SIZE; i++) {
            pool->entries[idx].genome[i] = 0.0f;
            pool->entries[idx].gradients[i] = 0.0f;
        }
    }

    if (idx == 0) {
        pool->capacity = capacity;
        pool->active_count = 0;
        pool->total_spawned = 0;
        pool->total_culled = 0;
    }
}

// Compute pool statistics
__global__ void compute_pool_stats_kernel(
    ComponentPool* pool,
    float* avg_fitness,
    float* avg_coherence,
    float* avg_age,
    float* genetic_diversity
) {
    __shared__ float shared_fitness[256];
    __shared__ float shared_coherence[256];
    __shared__ float shared_age[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float local_fitness = 0.0f;
    float local_coherence = 0.0f;
    float local_age = 0.0f;

    if (idx < pool->capacity && pool->entries[idx].alive) {
        local_fitness = pool->entries[idx].fitness;
        local_coherence = pool->entries[idx].coherence;
        local_age = (float)pool->entries[idx].age;
    }

    shared_fitness[tid] = local_fitness;
    shared_coherence[tid] = local_coherence;
    shared_age[tid] = local_age;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_fitness[tid] += shared_fitness[tid + stride];
            shared_coherence[tid] += shared_coherence[tid + stride];
            shared_age[tid] += shared_age[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(avg_fitness, shared_fitness[0]);
        atomicAdd(avg_coherence, shared_coherence[0]);
        atomicAdd(avg_age, shared_age[0]);
    }

    // Compute genetic diversity
    if (idx < pool->capacity && pool->entries[idx].alive) {
        float diversity = 0.0f;

        // Sample random other individuals
        PRNGState rng;
        rng.s0 = idx * 0x9e3779b97f4a7c15ULL;
        rng.s1 = pool->total_spawned * 0xbf58476d1ce4e5b9ULL;

        for (int i = 0; i < 10; i++) {
            int other_idx = (int)(rng.next() * pool->capacity) % pool->capacity;
            if (other_idx != idx && pool->entries[other_idx].alive) {
                diversity += compute_genome_distance(
                    pool->entries[idx].genome,
                    pool->entries[other_idx].genome
                );
            }
        }

        atomicAdd(genetic_diversity, diversity / 10.0f);
    }
}