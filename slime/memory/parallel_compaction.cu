
#ifndef PARALLEL_COMPACTION_CU
#define PARALLEL_COMPACTION_CU

#include "../config/config.cu"
#include "../utils/cuda_primitives.cuh"
#include "../learning/autodiff.cu"
#include "tubes.cu"
#include "genome_ops.cuh"
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

// Single-block inclusive scan (building block for multi-block)
__device__ void block_inclusive_scan(int* data, int n, int lane, int warp_id) {
    __shared__ int warp_sums[WARP_SIZE];

    int val = (threadIdx.x < n) ? data[threadIdx.x] : 0;

    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        int x = warp.shfl_up(val, offset);
        if (lane >= offset) val += x;
    }

    if (lane == WARP_SIZE - 1) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        int warp_sum = (lane < (blockDim.x / WARP_SIZE)) ? warp_sums[lane] : 0;
        #pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
            int x = warp.shfl_up(warp_sum, offset);
            if (lane >= offset) warp_sum += x;
        }
        warp_sums[lane] = warp_sum;
    }
    __syncthreads();

    int warp_offset = (warp_id > 0) ? warp_sums[warp_id - 1] : 0;
    if (threadIdx.x < n) {
        data[threadIdx.x] = warp_offset + val;
    }
}

// Phase 1: Each block does local inclusive scan, stores block total
__global__ void scan_phase1_kernel(
    int* input,
    int* output,
    int* block_sums,
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
    int inclusive_val = warp_offset + val;
    int exclusive_val = inclusive_val - ((tid < N) ? input[tid] : 0);

    if (tid < N) {
        output[tid] = exclusive_val;
    }

    // Last thread in block stores block total
    if (threadIdx.x == blockDim.x - 1) {
        block_sums[blockIdx.x] = inclusive_val;
    }
}

// Phase 3: Add block prefix to each element
__global__ void scan_phase3_kernel(
    int* output,
    int* block_prefixes,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N && blockIdx.x > 0) {
        output[tid] += block_prefixes[blockIdx.x - 1];
    }
}

// Single-kernel exclusive scan using Hillis-Steele algorithm
// Handles arrays up to MAX_MEMORY_SIZE using shared memory
__global__ void exclusive_scan_single_kernel(
    int* input,
    int* output,
    int N
) {
    __shared__ int temp[MAX_MEMORY_SIZE + BANK_PAD];

    // Load input to shared memory (all threads participate)
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        temp[i] = input[i];
    }
    __syncthreads();

    // Hillis-Steele inclusive scan
    for (int stride = 1; stride < N; stride *= 2) {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            int val = temp[i];
            if (i >= stride) {
                val += temp[i - stride];
            }
            __syncthreads();
            temp[i] = val;
            __syncthreads();
        }
    }

    // Convert to exclusive scan (shift right, first element = 0)
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        output[i] = (i == 0) ? 0 : temp[i - 1];
    }
}

// Device-side recursive multi-block exclusive scan (CDP)
// workspace must have size >= N (partitioned across recursion levels)
__global__ void exclusive_scan_recursive_kernel(
    int* input,
    int* output,
    int* workspace,
    int N,
    int block_size
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    if (N <= 0) return;

    int num_blocks = (N + block_size - 1) / block_size;

    if (num_blocks == 1) {
        // Single block case - do it inline
        scan_phase1_kernel<<<1, block_size>>>(input, output, workspace, N);
        cudaDeviceSynchronize();
        return;
    }

    // Multi-block case
    int* block_sums = workspace;
    int* next_workspace = workspace + num_blocks;

    // Phase 1: local scans, collect block totals
    scan_phase1_kernel<<<num_blocks, block_size>>>(input, output, block_sums, N);
    cudaDeviceSynchronize();

    // Phase 2: recursively scan block sums (CDP recursion)
    exclusive_scan_recursive_kernel<<<1, 1>>>(block_sums, block_sums, next_workspace, num_blocks, block_size);
    cudaDeviceSynchronize();

    // Phase 3: add block prefixes
    scan_phase3_kernel<<<num_blocks, block_size>>>(output, block_sums, N);
    cudaDeviceSynchronize();
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

// Computes new_count from scan results and copies compacted entries back to tube
__global__ void finalize_and_copy_compacted_kernel(
    TemporalTube* tube,
    int* valid_flags,
    int* scan_output,
    MemoryEntry* temp_buffer,
    int old_count
) {
    // Thread 0 computes new_count
    __shared__ int new_count;
    if (threadIdx.x == 0) {
        if (old_count > 0) {
            int last_write_idx = scan_output[old_count - 1];
            int last_valid_flag = valid_flags[old_count - 1];
            new_count = last_write_idx + last_valid_flag;
        } else {
            new_count = 0;
        }
    }
    __syncthreads();

    // All threads copy compacted entries
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < new_count) {
        tube->entries[idx] = temp_buffer[idx];
    }

    // Thread 0 updates tube metadata
    if (idx == 0) {
        tube->count = new_count;
        tube->head = new_count % tube->capacity;
    }
}

__global__ void compact_memory_tubes_parallel_kernel(
    TemporalTube* tube,
    int* valid_flags_workspace,
    int* scan_workspace,
    int* scan_recursive_workspace,
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
    cudaDeviceSynchronize();

    // Use device-side recursive scan for correct multi-block behavior
    exclusive_scan_recursive_kernel<<<1, 1>>>(
        valid_flags_workspace,
        scan_workspace,
        scan_recursive_workspace,
        tube->capacity,
        BLOCK_SIZE
    );
    cudaDeviceSynchronize();

    compact_entries_kernel<<<mark_grid, mark_block>>>(
        tube,
        valid_flags_workspace,
        scan_workspace,
        temp_buffer,
        old_count
    );
    cudaDeviceSynchronize();

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
    cudaDeviceSynchronize();
}

__global__ void prune_and_compact_memories_kernel(
    TemporalTube* tube,
    int* valid_flags_workspace,
    int* scan_workspace,
    int* scan_recursive_workspace,
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
            scan_recursive_workspace,
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

__global__ void memory_update_kernel(
    TemporalTube* tubes,
    float* fitness_history,
    int* valid_flags_workspace,
    int* scan_workspace,
    int* scan_recursive_workspace,
    MemoryEntry* temp_buffer,
    int generation,
    float* genome,
    float* gradients,
    uint64_t genome_hash,
    float ctx_metabolic,
    float ctx_stress,
    float ctx_morphogen,
    float ctx_complexity,
    float ctx_niche,
    float ctx_learning,
    float ctx_performance
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    if (!tubes || tubes->count <= 0) return;

    int decay_threshold_slot = derive_param_slot(genome_hash, "memory_decay_threshold");
    int consolidation_threshold_slot = derive_param_slot(genome_hash, "memory_consolidation_threshold");
    int flow_dt_slot = derive_param_slot(genome_hash, "memory_flow_lenia_dt");
    int compaction_interval_slot = derive_param_slot(genome_hash, "memory_compaction_interval");

    float decay_threshold = genome_to_param(
        genome, gradients, decay_threshold_slot,
        ctx_metabolic, ctx_stress, ctx_morphogen,
        ctx_complexity, ctx_niche, ctx_learning, ctx_performance,
        DECAY_THRESHOLD_MIN, DECAY_THRESHOLD_MAX
    );

    float consolidation_threshold = genome_to_param(
        genome, gradients, consolidation_threshold_slot,
        ctx_metabolic, ctx_stress, ctx_morphogen,
        ctx_complexity, ctx_niche, ctx_learning, ctx_performance,
        CONSOLIDATION_THRESHOLD_MIN, CONSOLIDATION_THRESHOLD_MAX
    );

    float flow_lenia_dt = genome_to_param(
        genome, gradients, flow_dt_slot,
        ctx_metabolic, ctx_stress, ctx_morphogen,
        ctx_complexity, ctx_niche, ctx_learning, ctx_performance,
        FLOW_LENIA_DT_MIN, FLOW_LENIA_DT_MAX
    );

    float compaction_interval_norm = genome_to_param(
        genome, gradients, compaction_interval_slot,
        ctx_metabolic, ctx_stress, ctx_morphogen,
        ctx_complexity, ctx_niche, ctx_learning, ctx_performance,
        0.0f, 1.0f
    );
    int compaction_interval = 1 + (int)(compaction_interval_norm * 15.0f);
    bool should_compact = (generation % compaction_interval == 0);

    int tube_count = tubes->count;
    int mem_blocks = (tube_count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    apply_decay_kernel<<<mem_blocks, BLOCK_SIZE>>>(tubes, flow_lenia_dt);
    cudaDeviceSynchronize();

    prune_memories_kernel<<<mem_blocks, BLOCK_SIZE>>>(tubes, decay_threshold);
    cudaDeviceSynchronize();

    int consol_threads = (tube_count < BLOCK_SIZE) ? tube_count : BLOCK_SIZE;
    consolidate_memories_kernel<<<1, consol_threads>>>(tubes, consolidation_threshold);
    cudaDeviceSynchronize();

    if (should_compact && valid_flags_workspace && scan_workspace && temp_buffer) {
        int old_count = tubes->count;

        mark_valid_entries_kernel<<<mem_blocks, BLOCK_SIZE>>>(
            tubes, valid_flags_workspace, decay_threshold
        );
        cudaDeviceSynchronize();

        exclusive_scan_single_kernel<<<1, BLOCK_SIZE>>>(
            valid_flags_workspace, scan_workspace, tubes->capacity
        );
        cudaDeviceSynchronize();

        compact_entries_kernel<<<mem_blocks, BLOCK_SIZE>>>(
            tubes, valid_flags_workspace, scan_workspace, temp_buffer, old_count
        );
        cudaDeviceSynchronize();

        finalize_and_copy_compacted_kernel<<<mem_blocks, BLOCK_SIZE>>>(
            tubes, valid_flags_workspace, scan_workspace, temp_buffer, old_count
        );
        cudaDeviceSynchronize();
    }
}

#endif
