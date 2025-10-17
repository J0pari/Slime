#include <cuda_runtime.h>
#include <cuda/std/atomic>

constexpr int MAX_POOL_SIZE = 256;
constexpr int MIN_POOL_SIZE = 8;

struct PoolEntry {
    int id;
    float fitness;
    float coherence;
    float hunger;
    int age;
    bool alive;
};

struct ComponentPool {
    PoolEntry* entries;
    cuda::std::atomic<int> active_count;
    cuda::std::atomic<int> total_spawned;
    cuda::std::atomic<int> total_culled;
    int capacity;
};

__global__ void spawn_component_kernel(ComponentPool* pool, int parent_id) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int count = pool->active_count.load();

        if (count < pool->capacity) {
            int new_id = pool->total_spawned.fetch_add(1);

            for (int i = 0; i < pool->capacity; i++) {
                if (!pool->entries[i].alive) {
                    pool->entries[i].id = new_id;
                    pool->entries[i].fitness = 0.0f;
                    pool->entries[i].coherence = 0.0f;
                    pool->entries[i].hunger = 0.5f;
                    pool->entries[i].age = 0;
                    pool->entries[i].alive = true;

                    pool->active_count.fetch_add(1);
                    break;
                }
            }
        }
    }
}

__global__ void cull_weak_kernel(ComponentPool* pool, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pool->capacity) {
        PoolEntry* entry = &pool->entries[idx];

        if (entry->alive && entry->fitness < threshold) {
            entry->alive = false;
            pool->active_count.fetch_sub(1);
            pool->total_culled.fetch_add(1);
        }
    }
}

__global__ void age_components_kernel(ComponentPool* pool) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < pool->capacity && pool->entries[idx].alive) {
        pool->entries[idx].age++;
    }
}

__global__ void sort_by_fitness_kernel(ComponentPool* pool) {
    extern __shared__ PoolEntry shared_entries[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < pool->capacity) {
        shared_entries[tid] = pool->entries[idx];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride && idx + stride < pool->capacity) {
            if (shared_entries[tid].fitness < shared_entries[tid + stride].fitness) {
                PoolEntry temp = shared_entries[tid];
                shared_entries[tid] = shared_entries[tid + stride];
                shared_entries[tid + stride] = temp;
            }
        }
        __syncthreads();
    }

    if (idx < pool->capacity) {
        pool->entries[idx] = shared_entries[tid];
    }
}