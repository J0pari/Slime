
#ifndef HARDWARE_GEOMETRY_CU
#define HARDWARE_GEOMETRY_CU
#include "../config/config.cu"
#include "../core/organism.cu"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

__device__ float g_gpu_peak_memory_bandwidth = 0.0f;

__device__ void reset_trace_buffer_device(Organism* organism) {
    int alive_count = organism->pool->alive_indices_count;
    int tid = threadIdx.x;
    for (int compact = 0; compact < alive_count; compact++) {
        int entry_idx = organism->pool->alive_indices[compact];
        TraceBuffer* tb = &organism->ca_state_pool[entry_idx].trace;
        if (tid == 0) {
            tb->current_idx = 0;
        }
        for (int i = tid; i < tb->capacity; i += blockDim.x) {
            ExecutionTrace* t = &tb->traces[i];
            t->active_warps = 0;
            t->divergent_branches = 0;
            t->total_branches = 0;
            t->global_loads = 0;
            t->global_stores = 0;
            t->l2_transactions = 0;
            t->dram_transactions = 0;
            t->shared_loads = 0;
            t->shared_stores = 0;
            t->bank_conflicts = 0;
            t->inst_executed = 0;
            t->inst_issued = 0;
            t->cycles_elapsed = 0;
            t->cycle_start = ULLONG_MAX;
            t->tensor_core_cycles = 0;
        }
    }
}

__device__ void init_trace_buffer_device(Organism* organism, int capacity) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pool_cap = organism->pool->capacity;

    for (int entry_idx = 0; entry_idx < pool_cap; entry_idx++) {
        TraceBuffer* buffer = &organism->ca_state_pool[entry_idx].trace;

        if (tid == 0) {
            buffer->capacity = capacity;
            buffer->current_idx = 0;
        }

        if (tid < capacity) {
            ExecutionTrace* trace = &buffer->traces[tid];
            trace->active_warps = 0;
            trace->divergent_branches = 0;
            trace->total_branches = 0;
            trace->global_loads = 0;
            trace->global_stores = 0;
            trace->l2_transactions = 0;
            trace->dram_transactions = 0;
            trace->shared_loads = 0;
            trace->shared_stores = 0;
            trace->bank_conflicts = 0;
            trace->inst_executed = 0;
            trace->inst_issued = 0;
            trace->cycles_elapsed = 0;
            trace->cycle_start = ULLONG_MAX;
            trace->tensor_core_cycles = 0;
            trace->sm_occupancy = NAN;
            trace->achieved_bandwidth = NAN;
            trace->peak_bandwidth = NAN;
        }
    }
}

__device__ void record_warp_metrics(ExecutionTrace* trace, int warp_id) {

    unsigned int active_mask = __activemask();
    int active_count = __popc(active_mask);

    atomicAdd((unsigned long long*)&trace->active_warps, 1ULL);

    if (active_count < WARP_SIZE) {
        atomicAdd((unsigned long long*)&trace->divergent_branches, 1ULL);
    }
    atomicAdd((unsigned long long*)&trace->total_branches, 1ULL);

    // Use active thread count as proxy for instructions executed
    atomicAdd((unsigned long long*)&trace->inst_executed, (unsigned long long)active_count);
    atomicAdd((unsigned long long*)&trace->inst_issued, 1ULL);
}

__device__ void record_memory_access(ExecutionTrace* trace, void* address, bool is_load) {

    if (is_load) {
        atomicAdd((unsigned long long*)&trace->global_loads, 1ULL);
    } else {
        atomicAdd((unsigned long long*)&trace->global_stores, 1ULL);
    }

    unsigned long long addr = (unsigned long long)address;
    int lane_id = threadIdx.x % WARP_SIZE;

    unsigned long long first_addr = __shfl_sync(0xFFFFFFFF, addr, 0);
    unsigned long long expected_addr = first_addr + lane_id * sizeof(float);

    if (addr != expected_addr) {
        atomicAdd((unsigned long long*)&trace->l2_transactions, 1ULL);
    }
}

__device__ void record_shared_memory_access(ExecutionTrace* trace, bool is_store, bool had_conflict) {
    if (is_store) {
        atomicAdd((unsigned long long*)&trace->shared_stores, 1ULL);
    } else {
        atomicAdd((unsigned long long*)&trace->shared_loads, 1ULL);
    }
    if (had_conflict) {
        atomicAdd((unsigned long long*)&trace->bank_conflicts, 1ULL);
    }
}

__device__ void compute_hardware_geometry(ExecutionTrace* trace, HardwareGeometry* geom) {

    DEVICE_FATAL_IF(trace->total_branches == 0, "compute_hardware_geometry: no branches recorded");
    float divergence_rate = (float)trace->divergent_branches / trace->total_branches;
    geom->warp_divergence_entropy = -divergence_rate * safe_log(divergence_rate)
                                   - (1.0f - divergence_rate) * safe_log(1.0f - divergence_rate);
    geom->warp_convergence_rate = 1.0f - divergence_rate;
    geom->active_thread_fraction = (float)trace->active_warps / (trace->total_branches * (float)WARP_SIZE);

    unsigned long long total_mem_ops = trace->global_loads + trace->global_stores;
    DEVICE_FATAL_IF(total_mem_ops == 0, "compute_hardware_geometry: no memory operations recorded");
    geom->memory_coalescing_efficiency = 1.0f - fminf(1.0f, (float)trace->l2_transactions / total_mem_ops);

    DEVICE_FATAL_IF(trace->l2_transactions == 0, "compute_hardware_geometry: no L2 transactions recorded");
    geom->cache_line_utilization = 1.0f - (float)trace->dram_transactions / trace->l2_transactions;

    unsigned long long shared_ops = trace->shared_loads + trace->shared_stores;
    DEVICE_FATAL_IF(shared_ops == 0, "compute_hardware_geometry: no shared memory operations recorded");
    geom->memory_divergence_spread = (float)trace->bank_conflicts / shared_ops;
    geom->bank_conflict_density = geom->memory_divergence_spread;

    DEVICE_FATAL_IF(trace->cycles_elapsed == 0, "compute_hardware_geometry: no cycles elapsed");
    geom->tensor_core_usage = (float)trace->tensor_core_cycles / trace->cycles_elapsed;
    geom->tensor_memory_bandwidth = (float)(trace->global_loads + trace->global_stores) * sizeof(float) / trace->cycles_elapsed;
    geom->instruction_throughput = (float)trace->inst_executed / trace->cycles_elapsed;

    DEVICE_FATAL_IF(trace->inst_issued == 0, "compute_hardware_geometry: no instructions issued");
    geom->pipeline_stall_fraction = 1.0f - (float)trace->inst_executed / trace->inst_issued;

    float mean_occupancy = (float)trace->active_warps / trace->total_branches;
    float variance = divergence_rate * (1.0f - divergence_rate);
    variance += OCCUPANCY_VARIANCE_WEIGHT * (1.0f - mean_occupancy);
    geom->occupancy_variance = variance;

    unsigned long long total_bytes = (trace->global_loads + trace->global_stores) * 4ULL;
    geom->arithmetic_intensity = (float)trace->inst_executed / total_bytes;

    DEVICE_FATAL_IF(trace->peak_bandwidth <= 0.0f, "compute_hardware_geometry: peak_bandwidth not set");
    geom->memory_bandwidth_saturation = trace->achieved_bandwidth / trace->peak_bandwidth;
}

__device__ void aggregate_hardware_geometry_device(Organism* organism) {
    HardwareGeometry* output_geom = organism->hardware_geom;
    ComponentPool* pool = organism->pool;
    MultiHeadCAState* ca_state_pool = organism->ca_state_pool;

    __shared__ alignas(128) ExecutionTrace aggregate_trace;

    int tid = threadIdx.x;

    if (tid == 0) {
        aggregate_trace.active_warps = 0;
        aggregate_trace.divergent_branches = 0;
        aggregate_trace.total_branches = 0;
        aggregate_trace.global_loads = 0;
        aggregate_trace.global_stores = 0;
        aggregate_trace.l2_transactions = 0;
        aggregate_trace.dram_transactions = 0;
        aggregate_trace.shared_loads = 0;
        aggregate_trace.shared_stores = 0;
        aggregate_trace.bank_conflicts = 0;
        aggregate_trace.inst_executed = 0;
        aggregate_trace.inst_issued = 0;
        aggregate_trace.cycles_elapsed = 0;
        aggregate_trace.tensor_core_cycles = 0;
        aggregate_trace.sm_occupancy = 0.0f;
        aggregate_trace.achieved_bandwidth = 0.0f;
        aggregate_trace.peak_bandwidth = 0.0f;
    }
    cg::this_grid().sync();

    // Aggregate traces from all alive entries' per-entry trace buffers
    int alive_count = pool->alive_indices_count;
    for (int compact = 0; compact < alive_count; compact++) {
        int entry_idx = pool->alive_indices[compact];
        TraceBuffer* buffer = &ca_state_pool[entry_idx].trace;
        int trace_count = buffer->current_idx;
        for (int i = tid; i < trace_count; i += blockDim.x) {
            ExecutionTrace* trace = &buffer->traces[i];

            atomicAdd((unsigned long long*)&aggregate_trace.active_warps, trace->active_warps);
            atomicAdd((unsigned long long*)&aggregate_trace.divergent_branches, trace->divergent_branches);
            atomicAdd((unsigned long long*)&aggregate_trace.total_branches, trace->total_branches);
            atomicAdd((unsigned long long*)&aggregate_trace.global_loads, trace->global_loads);
            atomicAdd((unsigned long long*)&aggregate_trace.global_stores, trace->global_stores);
            atomicAdd((unsigned long long*)&aggregate_trace.l2_transactions, trace->l2_transactions);
            atomicAdd((unsigned long long*)&aggregate_trace.dram_transactions, trace->dram_transactions);
            atomicAdd((unsigned long long*)&aggregate_trace.shared_loads, trace->shared_loads);
            atomicAdd((unsigned long long*)&aggregate_trace.shared_stores, trace->shared_stores);
            atomicAdd((unsigned long long*)&aggregate_trace.bank_conflicts, trace->bank_conflicts);
            atomicAdd((unsigned long long*)&aggregate_trace.inst_executed, trace->inst_executed);
            atomicAdd((unsigned long long*)&aggregate_trace.inst_issued, trace->inst_issued);
            atomicAdd((unsigned long long*)&aggregate_trace.cycles_elapsed, trace->cycles_elapsed);
            atomicAdd((unsigned long long*)&aggregate_trace.tensor_core_cycles, trace->tensor_core_cycles);
        }
    }
    cg::this_grid().sync();

    if (tid == 0) {
        if (aggregate_trace.total_branches > 0) {
            aggregate_trace.peak_bandwidth = g_gpu_peak_memory_bandwidth;
            if (aggregate_trace.cycles_elapsed > 0) {
                float total_bytes = (float)(aggregate_trace.global_loads + aggregate_trace.global_stores) * sizeof(float);
                aggregate_trace.achieved_bandwidth = total_bytes / (float)aggregate_trace.cycles_elapsed;
            }
            compute_hardware_geometry(&aggregate_trace, output_geom);
        }
    }
}

__device__ void extract_hardware_features(HardwareGeometry* geom, float* features) {
    features[0] = geom->warp_divergence_entropy;
    features[1] = geom->warp_convergence_rate;
    features[2] = geom->active_thread_fraction;
    features[3] = geom->memory_coalescing_efficiency;
    features[4] = geom->cache_line_utilization;
    features[FEATURE_MEMORY_DIVERGENCE_SPREAD] = geom->memory_divergence_spread;
    features[FEATURE_BANK_CONFLICT_DENSITY] = geom->bank_conflict_density;
    features[FEATURE_TENSOR_CORE_USAGE] = geom->tensor_core_usage;
    features[FEATURE_TENSOR_MEMORY_BANDWIDTH] = geom->tensor_memory_bandwidth;
    features[FEATURE_INSTRUCTION_THROUGHPUT] = geom->instruction_throughput;
    features[FEATURE_PIPELINE_STALL_FRACTION] = geom->pipeline_stall_fraction;
    features[FEATURE_OCCUPANCY_VARIANCE] = geom->occupancy_variance;
    features[FEATURE_ARITHMETIC_INTENSITY] = geom->arithmetic_intensity;
    features[FEATURE_MEMORY_BANDWIDTH_SATURATION] = geom->memory_bandwidth_saturation;
    features[FEATURE_INTERACTION_TERM] = geom->warp_divergence_entropy * geom->memory_coalescing_efficiency;
}

#endif
