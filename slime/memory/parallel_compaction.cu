
#ifndef PARALLEL_COMPACTION_CU
#define PARALLEL_COMPACTION_CU

#include "../config/config.cu"
#include "../core/organism.cu"
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

    if (threadIdx.x == blockDim.x - 1) {
        block_sums[blockIdx.x] = inclusive_val;
    }
}

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

__global__ void exclusive_scan_single_kernel(
    int* input,
    int* output,
    int N
) {
    __shared__ int temp[MAX_MEMORY_SIZE + BANK_PAD];

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        temp[i] = input[i];
    }
    __syncthreads();

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

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        output[i] = (i == 0) ? 0 : temp[i - 1];
    }
}

__global__ void exclusive_scan_coop_kernel(
    int* input,
    int* output,
    int* block_sums,
    int N
) {
    cg::grid_group grid = cg::this_grid();
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

    if (threadIdx.x == blockDim.x - 1) {
        block_sums[blockIdx.x] = inclusive_val;
    }

    grid.sync();

    if (blockIdx.x == 0) {
        __shared__ int bsum_shared[WARP_SIZE];
        int num_blocks = gridDim.x;

        int bval = (threadIdx.x < num_blocks) ? block_sums[threadIdx.x] : 0;

        #pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
            int n = warp.shfl_up(bval, offset);
            if (lane >= offset) bval += n;
        }

        if (lane == WARP_SIZE - 1) {
            bsum_shared[warp_id] = bval;
        }
        __syncthreads();

        if (warp_id == 0) {
            int ws = (lane < (blockDim.x / WARP_SIZE)) ? bsum_shared[lane] : 0;
            #pragma unroll
            for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
                int n = warp.shfl_up(ws, offset);
                if (lane >= offset) ws += n;
            }
            bsum_shared[lane] = ws;
        }
        __syncthreads();

        int bprefix = (warp_id > 0) ? bsum_shared[warp_id - 1] : 0;
        int b_inclusive = bprefix + bval;
        int b_exclusive = b_inclusive - ((threadIdx.x < num_blocks) ? block_sums[threadIdx.x] : 0);

        if (threadIdx.x < num_blocks) {
            block_sums[threadIdx.x] = b_exclusive;
        }
    }

    grid.sync();

    if (tid < N && blockIdx.x > 0) {
        output[tid] += block_sums[blockIdx.x];
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

__global__ void finalize_and_copy_compacted_kernel(
    TemporalTube* tube,
    int* valid_flags,
    int* scan_output,
    MemoryEntry* temp_buffer,
    int old_count
) {
    __shared__ int new_count;
    if (threadIdx.x == 0) {
        DEVICE_FATAL_IF(old_count <= 0, "stream_compaction_kernel: old_count non-positive");
        int last_write_idx = scan_output[old_count - 1];
        int last_valid_flag = valid_flags[old_count - 1];
        new_count = last_write_idx + last_valid_flag;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < new_count) {
        tube->entries[idx] = temp_buffer[idx];
    }

    if (idx == 0) {
        tube->count = new_count;
        tube->head = new_count % tube->capacity;
    }
}

__global__ void compact_memory_tubes_coop_kernel(
    TemporalTube* tube,
    int* valid_flags,
    int* scan_output,
    int* block_sums,
    MemoryEntry* temp_buffer,
    float decay_threshold
) {
    cg::grid_group grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    __shared__ int shared_old_count;
    __shared__ int shared_new_count;
    __shared__ int warp_sums[WARP_SIZE];

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        shared_old_count = tube->count;
    }
    grid.sync();

    int old_count = tube->count;
    int capacity = tube->capacity;

    if (tid < capacity) {
        int entry_idx = (tube->head - old_count + tid + capacity) % capacity;
        MemoryEntry* entry = &tube->entries[entry_idx];
        valid_flags[tid] = (tid < old_count && entry->decay_factor >= decay_threshold && entry->size > 0) ? 1 : 0;
    } else if (tid < capacity) {
        valid_flags[tid] = 0;
    }

    grid.sync();

    int val = (tid < capacity) ? valid_flags[tid] : 0;

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
    int exclusive_val = inclusive_val - ((tid < capacity) ? valid_flags[tid] : 0);

    if (tid < capacity) {
        scan_output[tid] = exclusive_val;
    }

    if (threadIdx.x == blockDim.x - 1) {
        block_sums[blockIdx.x] = inclusive_val;
    }

    grid.sync();

    if (blockIdx.x == 0) {
        __shared__ int bsum_shared[WARP_SIZE];
        int num_blocks = gridDim.x;

        int bval = (threadIdx.x < num_blocks) ? block_sums[threadIdx.x] : 0;

        #pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
            int n = warp.shfl_up(bval, offset);
            if (lane >= offset) bval += n;
        }

        if (lane == WARP_SIZE - 1) {
            bsum_shared[warp_id] = bval;
        }
        __syncthreads();

        if (warp_id == 0) {
            int ws = (lane < (blockDim.x / WARP_SIZE)) ? bsum_shared[lane] : 0;
            #pragma unroll
            for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
                int n = warp.shfl_up(ws, offset);
                if (lane >= offset) ws += n;
            }
            bsum_shared[lane] = ws;
        }
        __syncthreads();

        int bprefix = (warp_id > 0) ? bsum_shared[warp_id - 1] : 0;
        int b_inclusive = bprefix + bval;
        int b_exclusive = b_inclusive - ((threadIdx.x < num_blocks) ? block_sums[threadIdx.x] : 0);

        if (threadIdx.x < num_blocks) {
            block_sums[threadIdx.x] = b_exclusive;
        }
    }

    grid.sync();

    if (tid < capacity && blockIdx.x > 0) {
        scan_output[tid] += block_sums[blockIdx.x];
    }

    grid.sync();

    if (tid < old_count && valid_flags[tid]) {
        int entry_idx = (tube->head - old_count + tid + capacity) % capacity;
        int write_pos = scan_output[tid];
        temp_buffer[write_pos] = tube->entries[entry_idx];
    }

    grid.sync();

    if (tid == 0) {
        DEVICE_FATAL_IF(old_count <= 0, "memory_compaction: old_count non-positive");
        shared_new_count = scan_output[old_count - 1] + valid_flags[old_count - 1];
    }

    grid.sync();

    int new_count = shared_new_count;

    if (tid < new_count) {
        tube->entries[tid] = temp_buffer[tid];
    }

    if (tid == 0) {
        tube->count = new_count;
        tube->head = new_count % capacity;
    }
}

__global__ void prune_and_compact_coop_kernel(
    TemporalTube* tube,
    int* valid_flags,
    int* scan_output,
    int* block_sums,
    MemoryEntry* temp_buffer,
    float decay_threshold
) {
    cg::grid_group grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    int old_count = tube->count;
    int capacity = tube->capacity;

    __shared__ int shared_new_count;
    __shared__ int warp_sums[WARP_SIZE];

    if (tid < old_count) {
        int entry_idx = (tube->head - old_count + tid + capacity) % capacity;
        MemoryEntry* entry = &tube->entries[entry_idx];
        if (entry->decay_factor < decay_threshold) {
            entry->size = 0;
        }
    }

    grid.sync();

    int is_valid = 0;
    if (tid < old_count) {
        int entry_idx = (tube->head - old_count + tid + capacity) % capacity;
        MemoryEntry* entry = &tube->entries[entry_idx];
        is_valid = (entry->decay_factor >= decay_threshold && entry->size > 0) ? 1 : 0;
    }
    if (tid < capacity) {
        valid_flags[tid] = is_valid;
    }

    grid.sync();

    int val = (tid < capacity) ? valid_flags[tid] : 0;

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
    int exclusive_val = inclusive_val - ((tid < capacity) ? valid_flags[tid] : 0);

    if (tid < capacity) {
        scan_output[tid] = exclusive_val;
    }

    if (threadIdx.x == blockDim.x - 1) {
        block_sums[blockIdx.x] = inclusive_val;
    }

    grid.sync();

    if (blockIdx.x == 0) {
        __shared__ int bsum_shared[WARP_SIZE];
        int num_blocks = gridDim.x;

        int bval = (threadIdx.x < num_blocks) ? block_sums[threadIdx.x] : 0;

        #pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
            int n = warp.shfl_up(bval, offset);
            if (lane >= offset) bval += n;
        }

        if (lane == WARP_SIZE - 1) {
            bsum_shared[warp_id] = bval;
        }
        __syncthreads();

        if (warp_id == 0) {
            int ws = (lane < (blockDim.x / WARP_SIZE)) ? bsum_shared[lane] : 0;
            #pragma unroll
            for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
                int n = warp.shfl_up(ws, offset);
                if (lane >= offset) ws += n;
            }
            bsum_shared[lane] = ws;
        }
        __syncthreads();

        int bprefix = (warp_id > 0) ? bsum_shared[warp_id - 1] : 0;
        int b_inclusive = bprefix + bval;
        int b_exclusive = b_inclusive - ((threadIdx.x < num_blocks) ? block_sums[threadIdx.x] : 0);

        if (threadIdx.x < num_blocks) {
            block_sums[threadIdx.x] = b_exclusive;
        }
    }

    grid.sync();

    if (tid < capacity && blockIdx.x > 0) {
        scan_output[tid] += block_sums[blockIdx.x];
    }

    grid.sync();

    if (tid < old_count && valid_flags[tid]) {
        int entry_idx = (tube->head - old_count + tid + capacity) % capacity;
        int write_pos = scan_output[tid];
        temp_buffer[write_pos] = tube->entries[entry_idx];
    }

    grid.sync();

    if (tid == 0) {
        DEVICE_FATAL_IF(old_count <= 0, "memory_compaction: old_count non-positive");
        shared_new_count = scan_output[old_count - 1] + valid_flags[old_count - 1];
    }

    grid.sync();

    int new_count = shared_new_count;

    if (tid < new_count) {
        tube->entries[tid] = temp_buffer[tid];
    }

    if (tid == 0) {
        tube->count = new_count;
        tube->head = new_count % capacity;
    }
}

__global__ void refine_elite_coop_kernel(
    GPUElite* elite,
    ADTape* tape,
    int elite_idx,
    float learning_rate,
    float gradient_clip_norm
) {
    cg::grid_group grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float* latent = &elite->latent_genome[elite_idx * GENOME_LATENT_DIM_MAX];

    if (blockIdx.x == 0) {
        if (tape->current_size > 0 && threadIdx.x == 0) {
            int output_idx = tape->entries[tape->current_size - 1].output_idx;
            tape->grad_buffer[output_idx] = 1.0f;
        }
        __syncthreads();

        for (int op_idx = tape->current_size - 1; op_idx >= 0; op_idx--) {
            TapeEntry* entry = &tape->entries[op_idx];
            float grad_out = tape->grad_buffer[entry->output_idx];

            if (threadIdx.x == 0 && grad_out != 0.0f) {
                switch (entry->op) {
                    case OP_ADD:
                        atomicAdd(&tape->grad_buffer[entry->input1_idx], grad_out);
                        atomicAdd(&tape->grad_buffer[entry->input2_idx], grad_out);
                        break;
                    case OP_MUL:
                        atomicAdd(&tape->grad_buffer[entry->input1_idx], grad_out * tape->value_buffer[entry->input2_idx]);
                        atomicAdd(&tape->grad_buffer[entry->input2_idx], grad_out * tape->value_buffer[entry->input1_idx]);
                        break;
                    case OP_RELU:
                        if (entry->aux_data > 0.0f) {
                            atomicAdd(&tape->grad_buffer[entry->input1_idx], grad_out);
                        }
                        break;
                    case OP_EXP:
                        atomicAdd(&tape->grad_buffer[entry->input1_idx], grad_out * tape->value_buffer[entry->output_idx]);
                        break;
                    case OP_LOG: {
                        float input_val = entry->aux_data;
                        if (input_val > 1e-8f) {
                            atomicAdd(&tape->grad_buffer[entry->input1_idx], grad_out / input_val);
                        }
                        break;
                    }
                    case OP_TANH: {
                        float tanh_val = entry->aux_data;
                        atomicAdd(&tape->grad_buffer[entry->input1_idx], grad_out * (1.0f - tanh_val * tanh_val));
                        break;
                    }
                    case OP_SQRT:
                        if (entry->aux_data > 1e-8f) {
                            atomicAdd(&tape->grad_buffer[entry->input1_idx], grad_out / (2.0f * entry->aux_data));
                        }
                        break;
                    case OP_SIN:
                        atomicAdd(&tape->grad_buffer[entry->input1_idx], grad_out * cosf(entry->aux_data));
                        break;
                    case OP_COS:
                        atomicAdd(&tape->grad_buffer[entry->input1_idx], -grad_out * sinf(entry->aux_data));
                        break;
                    default:
                        break;
                }
            }
            __syncthreads();
        }
    }

    grid.sync();

    if (tid < GENOME_LATENT_DIM_MAX) {
        float grad = tape->grad_buffer[tid];
        float grad_norm = fabsf(grad);
        if (grad_norm > gradient_clip_norm) {
            grad = grad * (gradient_clip_norm / grad_norm);
        }
        latent[tid] -= learning_rate * grad;
    }
}

struct MemoryUpdateParams {
    float decay_threshold;
    float consolidation_threshold;
    float flow_lenia_dt;
    float fitness_trend;
    int old_count;
    int new_count;
};

__global__ void memory_update_params_kernel(
    MemoryUpdateParams* params,
    TemporalTube* tubes,
    float* fitness_history,
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

    DEVICE_FATAL_IF(tubes == nullptr, "memory_update_params_kernel: tubes is null");
    DEVICE_FATAL_IF(tubes->count <= 0, "memory_update_params_kernel: tubes->count non-positive");
    DEVICE_FATAL_IF(generation < 1, "memory_update_params_kernel: generation < 1 - no previous data exists for fitness_trend");
    DEVICE_FATAL_IF(fitness_history == nullptr, "memory_update_params_kernel: fitness_history is null");

    int decay_threshold_slot = GenomeParamTable::memory_decay_threshold;
    int consolidation_threshold_slot = GenomeParamTable::memory_consolidation_threshold;
    int flow_dt_slot = GenomeParamTable::memory_flow_lenia_dt;

    params->decay_threshold = genome_to_param(
        genome, gradients, decay_threshold_slot,
        ctx_metabolic, ctx_stress, ctx_morphogen,
        ctx_complexity, ctx_niche, ctx_learning, ctx_performance,
        DECAY_THRESHOLD_MIN, DECAY_THRESHOLD_MAX
    );

    params->consolidation_threshold = genome_to_param(
        genome, gradients, consolidation_threshold_slot,
        ctx_metabolic, ctx_stress, ctx_morphogen,
        ctx_complexity, ctx_niche, ctx_learning, ctx_performance,
        CONSOLIDATION_THRESHOLD_MIN, CONSOLIDATION_THRESHOLD_MAX
    );

    params->flow_lenia_dt = genome_to_param(
        genome, gradients, flow_dt_slot,
        ctx_metabolic, ctx_stress, ctx_morphogen,
        ctx_complexity, ctx_niche, ctx_learning, ctx_performance,
        FLOW_LENIA_DT_MIN, FLOW_LENIA_DT_MAX
    );

    params->old_count = tubes->count;

    int curr_gen_offset = (generation % 2) * POOL_CAPACITY_MAX;
    int prev_gen_offset = ((generation - 1) % 2) * POOL_CAPACITY_MAX;
    float curr_fitness = fitness_history[curr_gen_offset];
    float prev_fitness = fitness_history[prev_gen_offset];
    params->fitness_trend = curr_fitness - prev_fitness;
}

__global__ void memory_decay_kernel(
    TemporalTube* tubes,
    MemoryUpdateParams* params
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    DEVICE_FATAL_IF(tubes == nullptr, "memory_decay_kernel: tubes is null");
    DEVICE_FATAL_IF(params->old_count <= 0, "memory_decay_kernel: old_count non-positive");

    int tube_count = params->old_count;
    int capacity = tubes->capacity;

    if (tid < tube_count) {
        int entry_idx = (tubes->head - tube_count + tid + capacity) % capacity;
        MemoryEntry* entry = &tubes->entries[entry_idx];
        entry->decay_factor *= expf(-params->flow_lenia_dt);
        float importance_delta = params->fitness_trend * 0.1f;
        entry->importance = clamp(entry->importance + importance_delta, 0.0f, 1.0f);
    }
}

__global__ void memory_prune_kernel(
    TemporalTube* tubes,
    MemoryUpdateParams* params
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    DEVICE_FATAL_IF(tubes == nullptr, "memory_prune_kernel: tubes is null");
    DEVICE_FATAL_IF(params->old_count <= 0, "memory_prune_kernel: old_count non-positive");

    int tube_count = params->old_count;
    int capacity = tubes->capacity;

    if (tid < tube_count) {
        int entry_idx = (tubes->head - tube_count + tid + capacity) % capacity;
        MemoryEntry* entry = &tubes->entries[entry_idx];
        if (entry->decay_factor < params->decay_threshold) {
            entry->size = 0;
        }
    }
}

__global__ void memory_consolidate_kernel(
    TemporalTube* tubes,
    MemoryUpdateParams* params
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    DEVICE_FATAL_IF(tubes == nullptr, "memory_consolidate_kernel: tubes is null");
    DEVICE_FATAL_IF(params->old_count <= 0, "memory_consolidate_kernel: old_count non-positive");

    int tube_count = params->old_count;
    int capacity = tubes->capacity;
    float consolidation_threshold = params->consolidation_threshold;

    for (int i = 0; i < tube_count; i++) {
        int entry_i = (tubes->head - tube_count + i + capacity) % capacity;
        MemoryEntry* mi = &tubes->entries[entry_i];
        if (mi->size == 0) continue;

        for (int j = i + 1; j < tube_count; j++) {
            int entry_j = (tubes->head - tube_count + j + capacity) % capacity;
            MemoryEntry* mj = &tubes->entries[entry_j];
            if (mj->size == 0) continue;

            int min_size = (mi->size < mj->size) ? mi->size : mj->size;
            if (min_size > 0) {
                float similarity = 0.0f;
                for (int k = 0; k < min_size; k++) {
                    float diff = mi->data[k] - mj->data[k];
                    similarity += diff * diff;
                }
                similarity = 1.0f - sqrtf(similarity / min_size);

                if (similarity > consolidation_threshold) {
                    for (int k = 0; k < min_size; k++) {
                        mi->data[k] = 0.5f * (mi->data[k] + mj->data[k]);
                    }
                    mi->decay_factor = fmaxf(mi->decay_factor, mj->decay_factor);
                    mj->size = 0;
                }
            }
        }
    }
}

__global__ void memory_mark_valid_kernel(
    TemporalTube* tubes,
    MemoryUpdateParams* params,
    int* valid_flags
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    DEVICE_FATAL_IF(tubes == nullptr, "memory_mark_valid_kernel: tubes is null");
    DEVICE_FATAL_IF(valid_flags == nullptr, "memory_mark_valid_kernel: valid_flags is null");

    int old_count = params->old_count;
    int capacity = tubes->capacity;

    int is_valid = 0;
    if (tid < old_count) {
        int entry_idx = (tubes->head - old_count + tid + capacity) % capacity;
        MemoryEntry* entry = &tubes->entries[entry_idx];
        is_valid = (entry->decay_factor >= params->decay_threshold && entry->size > 0) ? 1 : 0;
    }
    if (tid < capacity) {
        valid_flags[tid] = is_valid;
    }
}

__global__ void memory_scan_kernel(
    int* valid_flags,
    int* scan_output,
    int* block_sums,
    int capacity
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    __shared__ int warp_sums[WARP_SIZE];

    int val = (tid < capacity) ? valid_flags[tid] : 0;

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
    int exclusive_val = inclusive_val - ((tid < capacity) ? valid_flags[tid] : 0);

    if (tid < capacity) {
        scan_output[tid] = exclusive_val;
    }

    if (threadIdx.x == blockDim.x - 1) {
        block_sums[blockIdx.x] = inclusive_val;
    }
}

__global__ void memory_scan_block_sums_kernel(
    int* block_sums,
    int num_blocks
) {
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    __shared__ int bsum_shared[WARP_SIZE];

    int bval = (threadIdx.x < num_blocks) ? block_sums[threadIdx.x] : 0;

    auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        int n = warp.shfl_up(bval, offset);
        if (lane >= offset) bval += n;
    }

    if (lane == WARP_SIZE - 1) {
        bsum_shared[warp_id] = bval;
    }
    __syncthreads();

    if (warp_id == 0) {
        int ws = (lane < (blockDim.x / WARP_SIZE)) ? bsum_shared[lane] : 0;
        #pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
            int n = warp.shfl_up(ws, offset);
            if (lane >= offset) ws += n;
        }
        bsum_shared[lane] = ws;
    }
    __syncthreads();

    int bprefix = (warp_id > 0) ? bsum_shared[warp_id - 1] : 0;
    int b_inclusive = bprefix + bval;
    int b_exclusive = b_inclusive - ((threadIdx.x < num_blocks) ? block_sums[threadIdx.x] : 0);

    if (threadIdx.x < num_blocks) {
        block_sums[threadIdx.x] = b_exclusive;
    }
}

__global__ void memory_add_block_offsets_kernel(
    int* scan_output,
    int* block_sums,
    int capacity
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < capacity && blockIdx.x > 0) {
        scan_output[tid] += block_sums[blockIdx.x];
    }
}

__global__ void memory_compact_kernel(
    TemporalTube* tubes,
    MemoryUpdateParams* params,
    int* valid_flags,
    int* scan_output,
    MemoryEntry* temp_buffer
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    DEVICE_FATAL_IF(tubes == nullptr, "memory_compact_kernel: tubes is null");
    DEVICE_FATAL_IF(valid_flags == nullptr, "memory_compact_kernel: valid_flags is null");
    DEVICE_FATAL_IF(scan_output == nullptr, "memory_compact_kernel: scan_output is null");
    DEVICE_FATAL_IF(temp_buffer == nullptr, "memory_compact_kernel: temp_buffer is null");

    int old_count = params->old_count;
    int capacity = tubes->capacity;

    if (tid < old_count && valid_flags[tid]) {
        int entry_idx = (tubes->head - old_count + tid + capacity) % capacity;
        int write_pos = scan_output[tid];
        temp_buffer[write_pos] = tubes->entries[entry_idx];
    }
}

__global__ void memory_finalize_kernel(
    TemporalTube* tubes,
    MemoryUpdateParams* params,
    int* valid_flags,
    int* scan_output,
    MemoryEntry* temp_buffer
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    DEVICE_FATAL_IF(tubes == nullptr, "memory_finalize_kernel: tubes is null");
    DEVICE_FATAL_IF(valid_flags == nullptr, "memory_finalize_kernel: valid_flags is null");
    DEVICE_FATAL_IF(scan_output == nullptr, "memory_finalize_kernel: scan_output is null");

    int old_count = params->old_count;
    int new_count = params->new_count;
    int capacity = tubes->capacity;

    if (tid == 0) {
        DEVICE_FATAL_IF(old_count <= 0, "memory_finalize_kernel: old_count non-positive");
        params->new_count = scan_output[old_count - 1] + valid_flags[old_count - 1];
    }
}

__global__ void memory_copy_back_kernel(
    TemporalTube* tubes,
    MemoryUpdateParams* params,
    MemoryEntry* temp_buffer
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    DEVICE_FATAL_IF(tubes == nullptr, "memory_copy_back_kernel: tubes is null");
    DEVICE_FATAL_IF(temp_buffer == nullptr, "memory_copy_back_kernel: temp_buffer is null");

    int new_count = params->new_count;
    int capacity = tubes->capacity;

    if (tid < new_count) {
        tubes->entries[tid] = temp_buffer[tid];
    }

    if (tid == 0) {
        tubes->count = new_count;
        tubes->head = new_count % capacity;
    }
}

__device__ void memory_update_cdp(
    TemporalTube* tubes,
    MemoryUpdateParams* params,
    float* fitness_history,
    int* valid_flags,
    int* scan_output,
    int* block_sums,
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
    DEVICE_FATAL_IF(tubes == nullptr, "memory_update_cdp: tubes is null");
    DEVICE_FATAL_IF(params == nullptr, "memory_update_cdp: params is null");
    DEVICE_FATAL_IF(fitness_history == nullptr, "memory_update_cdp: fitness_history is null");
    DEVICE_FATAL_IF(valid_flags == nullptr, "memory_update_cdp: valid_flags is null");
    DEVICE_FATAL_IF(scan_output == nullptr, "memory_update_cdp: scan_output is null");
    DEVICE_FATAL_IF(block_sums == nullptr, "memory_update_cdp: block_sums is null");
    DEVICE_FATAL_IF(temp_buffer == nullptr, "memory_update_cdp: temp_buffer is null");
    DEVICE_FATAL_IF(genome == nullptr, "memory_update_cdp: genome is null");
    DEVICE_FATAL_IF(gradients == nullptr, "memory_update_cdp: gradients is null");

    memory_update_params_kernel<<<1, 1>>>(
        params, tubes, fitness_history, generation,
        genome, gradients, genome_hash,
        ctx_metabolic, ctx_stress, ctx_morphogen,
        ctx_complexity, ctx_niche, ctx_learning, ctx_performance
    );
    cudaDeviceSynchronize();

    DEVICE_FATAL_IF(params->old_count <= 0, "memory_update_cdp: old_count non-positive after params kernel");

    int tube_count = params->old_count;
    int capacity = tubes->capacity;
    int num_blocks = (tube_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int cap_blocks = (capacity + BLOCK_SIZE - 1) / BLOCK_SIZE;

    memory_decay_kernel<<<num_blocks, BLOCK_SIZE>>>(tubes, params);
    cudaDeviceSynchronize();

    memory_prune_kernel<<<num_blocks, BLOCK_SIZE>>>(tubes, params);
    cudaDeviceSynchronize();

    memory_consolidate_kernel<<<1, 1>>>(tubes, params);
    cudaDeviceSynchronize();

    memory_mark_valid_kernel<<<cap_blocks, BLOCK_SIZE>>>(tubes, params, valid_flags);
    cudaDeviceSynchronize();

    memory_scan_kernel<<<cap_blocks, BLOCK_SIZE>>>(valid_flags, scan_output, block_sums, capacity);
    cudaDeviceSynchronize();

    memory_scan_block_sums_kernel<<<1, BLOCK_SIZE>>>(block_sums, cap_blocks);
    cudaDeviceSynchronize();

    memory_add_block_offsets_kernel<<<cap_blocks, BLOCK_SIZE>>>(scan_output, block_sums, capacity);
    cudaDeviceSynchronize();

    memory_compact_kernel<<<cap_blocks, BLOCK_SIZE>>>(tubes, params, valid_flags, scan_output, temp_buffer);
    cudaDeviceSynchronize();

    memory_finalize_kernel<<<1, 1>>>(tubes, params, valid_flags, scan_output, temp_buffer);
    cudaDeviceSynchronize();

    memory_copy_back_kernel<<<cap_blocks, BLOCK_SIZE>>>(tubes, params, temp_buffer);
    cudaDeviceSynchronize();
}

#endif
