
#ifndef PARALLEL_COMPACTION_CU
#define PARALLEL_COMPACTION_CU

#include "../config/config.cu"
#include "../utils/tile_ops.cuh"
#include "tubes.cu"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void mark_valid_entries_kernel(
    TemporalTube* tube,
    int* valid_flags,
    float decay_threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < tube->count) {
        int entry_idx = (tube->head - tube->count + idx + tube->capacity) % tube->capacity;
        MemoryEntry* entry = &tube->entries[entry_idx];

        valid_flags[idx] = (entry->decay_factor >= decay_threshold && entry->size > 0) ? 1 : 0;
    } else if (idx < tube->capacity) {
        valid_flags[idx] = 0;
    }
}

__global__ void exclusive_scan_kernel(
    int* input,
    int* output,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    __shared__ int warp_sums[WARP_SIZE];

    int val = (tid < N) ? input[tid] : 0;

    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        int n = warp.shfl_up(val, offset);
        if (lane >= offset) val += n;
    }

    if (lane == WARP_SIZE - 1) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        int warp_sum = (lane < (blockDim.x / WARP_SIZE)) ? warp_sums[lane] : 0;

        #pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
            int n = warp.shfl_up(warp_sum, offset);
            if (lane >= offset) warp_sum += n;
        }

        warp_sums[lane] = warp_sum;
    }
    __syncthreads();

    int warp_offset = (warp_id > 0) ? warp_sums[warp_id - 1] : 0;

    int exclusive_val = warp_offset + val - ((tid < N) ? input[tid] : 0);

    if (tid < N) {
        output[tid] = exclusive_val;
    }
}

__global__ void compact_entries_kernel(
    TemporalTube* tube,
    int* valid_flags,
    int* write_indices,
    MemoryEntry* temp_buffer,
    int old_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < old_count) {
        int entry_idx = (tube->head - tube->count + idx + tube->capacity) % tube->capacity;

        if (valid_flags[idx]) {
            int write_pos = write_indices[idx];
            temp_buffer[write_pos] = tube->entries[entry_idx];
        }
    }
}

__global__ void copy_compacted_kernel(
    TemporalTube* tube,
    MemoryEntry* temp_buffer,
    int new_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < new_count) {
        tube->entries[idx] = temp_buffer[idx];
    }

    if (idx == 0) {
        tube->count = new_count;
        tube->head = new_count % tube->capacity;
    }
}

__global__ void compact_memory_tubes_parallel_kernel(
    TemporalTube* tube,
    int* valid_flags_workspace,
    int* scan_workspace,
    MemoryEntry* temp_buffer,
    float decay_threshold
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int old_count = tube->count;

    dim3 mark_grid((tube->capacity + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 mark_block(BLOCK_SIZE);

    mark_valid_entries_kernel<<<mark_grid, mark_block>>>(
        tube,
        valid_flags_workspace,
        decay_threshold
    );

    exclusive_scan_kernel<<<mark_grid, mark_block>>>(
        valid_flags_workspace,
        scan_workspace,
        tube->capacity
    );

    compact_entries_kernel<<<mark_grid, mark_block>>>(
        tube,
        valid_flags_workspace,
        scan_workspace,
        temp_buffer,
        old_count
    );

    int new_count = 0;
    if (old_count > 0) {
        int last_valid_idx = old_count - 1;

        int last_write_idx = scan_workspace[last_valid_idx];
        int last_valid_flag = valid_flags_workspace[last_valid_idx];

        new_count = last_write_idx + last_valid_flag;
    }

    copy_compacted_kernel<<<mark_grid, mark_block>>>(
        tube,
        temp_buffer,
        new_count
    );
}

__global__ void prune_and_compact_memories_kernel(
    TemporalTube* tube,
    int* valid_flags_workspace,
    int* scan_workspace,
    MemoryEntry* temp_buffer,
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
    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        compact_memory_tubes_parallel_kernel<<<1, 1>>>(
            tube,
            valid_flags_workspace,
            scan_workspace,
            temp_buffer,
            decay_threshold
        );
    }
}

__global__ void refine_elite_kernel(
    GPUElite* elite,
    ADTape* tape,
    float* genome_buffer,
    uint8_t* elite_compressed_pool,
    uint32_t* elite_size_pool,
    int elite_idx,
    float learning_rate,
    float gradient_clip_norm
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Gradient-based optimization in DIRESA latent space
    float* latent = &elite->latent_genome[elite_idx * GENOME_LATENT_DIM_MAX];

    reset_tape_kernel<<<(VALUE_CAPACITY + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(tape);

    for (int i = 0; i < GENOME_LATENT_DIM_MAX; i++) {
        tape->value_buffer[i] = latent[i];
    }

    int fitness_op_idx = 0;
    ad_backward_kernel<<<1, BLOCK_SIZE>>>(tape, fitness_op_idx, 1.0f);

    dim3 grad_grid((GENOME_LATENT_DIM_MAX + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 grad_block(BLOCK_SIZE);

    apply_gradients_kernel<<<grad_grid, grad_block>>>(
        latent,
        tape->grad_buffer,
        GENOME_LATENT_DIM_MAX,
        learning_rate,
        gradient_clip_norm
    );
}

#endif
