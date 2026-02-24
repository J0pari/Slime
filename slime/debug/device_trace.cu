#ifndef DEVICE_TRACE_CU
#define DEVICE_TRACE_CU

#include "../config/config.cu"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

// Device-side trace buffer for debugging within the persistent kernel
struct DeviceTraceEntry {
    const char* location;
    int block_id;
    int thread_id;
    int value1;
    int value2;
    float fvalue;
};

// Global trace state
__device__ int g_trace_write_idx = 0;
__device__ DeviceTraceEntry g_trace_buffer[1024];
__device__ int g_trace_enabled = 1;

// Unified memory checkpoint counters - readable from host even during hang
// These use __managed__ so host can read while kernel runs
__managed__ int g_checkpoint_blocks_reached = 0;  // How many blocks reached current checkpoint
__managed__ int g_checkpoint_id = 0;              // Current checkpoint ID
__managed__ int g_checkpoint_per_block[64];       // Per-block progress (up to 64 blocks)

// Mark that this block reached a checkpoint - visible to host immediately
// Call from thread 0 of each block only
__device__ void mark_checkpoint(int checkpoint_id) {
    if (threadIdx.x == 0) {
        g_checkpoint_per_block[blockIdx.x] = checkpoint_id;
        atomicAdd(&g_checkpoint_blocks_reached, 1);
        g_checkpoint_id = checkpoint_id;
    }
}

// Reset checkpoint counter - call from single thread before starting
__device__ void reset_checkpoint_counter() {
    g_checkpoint_blocks_reached = 0;
}

// Trace a checkpoint with optional values
__device__ void trace_checkpoint(const char* location, int val1 = 0, int val2 = 0, float fval = 0.0f) {
    if (g_trace_enabled) {
        int idx = atomicAdd(&g_trace_write_idx, 1);
        if (idx < 1024) {
            g_trace_buffer[idx].location = location;
            g_trace_buffer[idx].block_id = blockIdx.x;
            g_trace_buffer[idx].thread_id = threadIdx.x;
            g_trace_buffer[idx].value1 = val1;
            g_trace_buffer[idx].value2 = val2;
            g_trace_buffer[idx].fvalue = fval;
        }
    }
}

// Print trace from thread 0 of block 0
__device__ void trace_print(const char* msg) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("T: %s\n", msg);
    }
}

// Print trace with integer value
__device__ void trace_print_int(const char* msg, int val) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("T: %s = %d\n", msg, val);
    }
}

// Print trace with float value
__device__ void trace_print_float(const char* msg, float val) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("T: %s = %f\n", msg, val);
    }
}

// Print trace with pointer value
__device__ void trace_print_ptr(const char* msg, void* ptr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("T: %s = %p\n", msg, ptr);
    }
}

// Verbose trace - prints from specific block
__device__ void trace_verbose(const char* msg, int block_filter = 0) {
    if (threadIdx.x == 0 && blockIdx.x == block_filter) {
        printf("V:%s\n", msg);
    }
}

// Grid sync with trace
__device__ void trace_grid_sync(const char* phase_name) {
    cg::grid_group grid = cg::this_grid();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("SYNC_BEGIN: %s\n", phase_name);
    }
    grid.sync();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("SYNC_END: %s\n", phase_name);
    }
}

// Block sync with trace (for within-block synchronization)
__device__ void trace_block_sync(const char* phase_name) {
    if (threadIdx.x == 0) {
        printf("BLOCK_SYNC: blk=%d %s\n", blockIdx.x, phase_name);
    }
    __syncthreads();
}

// Assert with trace - prints error and location if condition fails
__device__ void trace_assert(bool condition, const char* msg, int val1 = 0, int val2 = 0) {
    if (!condition) {
        printf("ASSERT_FAIL: blk=%d thr=%d %s v1=%d v2=%d\n",
               blockIdx.x, threadIdx.x, msg, val1, val2);
    }
}

// Check pointer validity
__device__ void trace_check_ptr(void* ptr, const char* name) {
    if (ptr == nullptr && threadIdx.x == 0 && blockIdx.x == 0) {
        printf("NULL_PTR: %s\n", name);
    }
}

// Phase entry/exit markers for debugging control flow
__device__ void trace_phase_enter(const char* phase) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf(">>> ENTER: %s\n", phase);
    }
}

__device__ void trace_phase_exit(const char* phase) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("<<< EXIT: %s\n", phase);
    }
}

// Memory validation trace
__device__ void trace_memory_access(const char* op, void* ptr, size_t size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MEM_%s: ptr=%p size=%zu\n", op, ptr, size);
    }
}

#endif
