
#ifndef POOL_TYPES_CUH
#define POOL_TYPES_CUH

#include "../memory/genome_ops.cuh"
#include <cuda/atomic>

struct ComponentPool {
    PoolEntry* entries;
    cuda::atomic<int, cuda::thread_scope_system> active_count;
    cuda::atomic<int, cuda::thread_scope_system> total_spawned;
    cuda::atomic<int, cuda::thread_scope_system> total_culled;
    int capacity;

    int* alive_indices;
    int alive_indices_count;

    bool* alive_flags;
    float* fitness_values;
};

#endif
