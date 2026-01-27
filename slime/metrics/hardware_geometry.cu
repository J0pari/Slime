
#ifndef HARDWARE_GEOMETRY_CU
#define HARDWARE_GEOMETRY_CU
#include "../config/config.cu"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

struct HardwareGeometry {

    float warp_divergence_entropy;
    float warp_convergence_rate;
    float active_thread_fraction;

    float memory_coalescing_efficiency;
    float cache_line_utilization;
    float memory_divergence_spread;
    float bank_conflict_density;

    float tensor_core_usage;
    float tensor_memory_bandwidth;

    float instruction_throughput;
    float pipeline_stall_fraction;
    float occupancy_variance;

    float arithmetic_intensity;
    float memory_bandwidth_saturation;
};

struct ExecutionTrace {

    unsigned long long active_warps;
    unsigned long long divergent_branches;
    unsigned long long total_branches;

    unsigned long long global_loads;
    unsigned long long global_stores;
    unsigned long long l2_transactions;
    unsigned long long dram_transactions;
    unsigned long long shared_loads;
    unsigned long long shared_stores;
    unsigned long long bank_conflicts;

    unsigned long long inst_executed;
    unsigned long long inst_issued;
    unsigned long long cycles_elapsed;
    unsigned long long tensor_core_cycles;

    float sm_occupancy;
    float achieved_bandwidth;
    float peak_bandwidth;
};

struct TraceBuffer {
    ExecutionTrace* traces;
    int capacity;
    int current_idx;
};

__global__ void init_trace_buffer_kernel(TraceBuffer* buffer, int capacity) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid == 0) {
        buffer->capacity = capacity;
        buffer->current_idx = 0;
    }
    
    // Initialize all trace entries to zero
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
        trace->tensor_core_cycles = 0;
        trace->sm_occupancy = NAN;
        trace->achieved_bandwidth = NAN;
        trace->peak_bandwidth = NAN;
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

__device__ void compute_hardware_geometry(ExecutionTrace* trace, HardwareGeometry* geom) {

    if (trace->total_branches > 0) {
        float divergence_rate = (float)trace->divergent_branches / trace->total_branches;
        geom->warp_divergence_entropy = -divergence_rate * safe_log(divergence_rate)
                                       - (1.0f - divergence_rate) * safe_log(1.0f - divergence_rate);
        geom->warp_convergence_rate = 1.0f - divergence_rate;
        geom->active_thread_fraction = (float)trace->active_warps / (trace->total_branches * (float)WARP_SIZE);
    } else {
        geom->warp_divergence_entropy = NAN;
        geom->warp_convergence_rate = PERFECT_EFFICIENCY;
        geom->active_thread_fraction = PERFECT_EFFICIENCY;
    }

    unsigned long long total_mem_ops = trace->global_loads + trace->global_stores;
    if (total_mem_ops > 0) {

        geom->memory_coalescing_efficiency = PERFECT_COALESCING - fminf(PERFECT_COALESCING, (float)trace->l2_transactions / total_mem_ops);

        if (trace->l2_transactions > 0) {
            geom->cache_line_utilization = PERFECT_CACHE_UTILIZATION - (float)trace->dram_transactions / trace->l2_transactions;
        } else {
            geom->cache_line_utilization = PERFECT_CACHE_UTILIZATION;
        }

        if (trace->shared_loads + trace->shared_stores > 0) {
            geom->memory_divergence_spread = (float)trace->bank_conflicts / (trace->shared_loads + trace->shared_stores);
            geom->bank_conflict_density = geom->memory_divergence_spread;
        } else {
            geom->memory_divergence_spread = NAN;
            geom->bank_conflict_density = NAN;
        }
    } else {
        geom->memory_coalescing_efficiency = PERFECT_COALESCING;
        geom->cache_line_utilization = PERFECT_CACHE_UTILIZATION;
        geom->memory_divergence_spread = NAN;
        geom->bank_conflict_density = NAN;
    }

    if (trace->cycles_elapsed > 0) {
        geom->tensor_core_usage = (float)trace->tensor_core_cycles / trace->cycles_elapsed;

        geom->tensor_memory_bandwidth = (float)(trace->global_loads + trace->global_stores) * sizeof(float) / trace->cycles_elapsed;
    } else {
        geom->tensor_core_usage = NAN;
        geom->tensor_memory_bandwidth = NAN;
    }

    if (trace->cycles_elapsed > 0) {
        geom->instruction_throughput = (float)trace->inst_executed / trace->cycles_elapsed;

        if (trace->inst_issued > 0) {
            geom->pipeline_stall_fraction = 1.0f - (float)trace->inst_executed / trace->inst_issued;
        } else {
            geom->pipeline_stall_fraction = NAN;
        }

        float mean_occupancy = (float)trace->active_warps / max(trace->total_branches, 1ULL);
        float variance = 0.0f;

        if (trace->total_branches > 0) {
            float divergence_rate = (float)trace->divergent_branches / trace->total_branches;
            variance = divergence_rate * (PERFECT_EFFICIENCY - divergence_rate);
            variance += OCCUPANCY_VARIANCE_WEIGHT * (PERFECT_EFFICIENCY - mean_occupancy);
        }

        geom->occupancy_variance = variance;
    } else {
        geom->instruction_throughput = NAN;
        geom->pipeline_stall_fraction = NAN;
        geom->occupancy_variance = NAN;
    }

    unsigned long long total_bytes = (trace->global_loads + trace->global_stores) * 4ULL;
    if (total_bytes > 0) {

        geom->arithmetic_intensity = (float)trace->inst_executed / total_bytes;

        if (trace->peak_bandwidth > 0.0f) {
            geom->memory_bandwidth_saturation = trace->achieved_bandwidth / trace->peak_bandwidth;
        } else {
            geom->memory_bandwidth_saturation = NAN;
        }
    } else {
        geom->arithmetic_intensity = NAN;
        geom->memory_bandwidth_saturation = NAN;
    }
}

__global__ void aggregate_hardware_geometry_kernel(
    TraceBuffer* buffer,
    HardwareGeometry* output_geom
) {
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
    __syncthreads();

    for (int i = tid; i < buffer->current_idx; i += blockDim.x) {
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
    __syncthreads();

    if (tid == 0) {
        compute_hardware_geometry(&aggregate_trace, output_geom);
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
