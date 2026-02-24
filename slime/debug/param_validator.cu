#ifndef PARAM_VALIDATOR_CU
#define PARAM_VALIDATOR_CU

#include "../config/config.cu"
#include <cuda_runtime.h>
#include <stdio.h>


__device__ __forceinline__ bool validate_pointer(const char* name, void* ptr, bool must_be_device, const char* file, int line) {
    if (ptr == nullptr) {
        return false;
    }

    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);

    if ((addr & 0x3) != 0) {
        return false;
    }

    if (addr == UINTPTR_MAX || addr == 0xDEADBEEF || addr == 0xFEEDFACE ||
        addr == 0xBADDCAFE || addr == 0xDEADC0DE) {
        return false;
    }

    uint32_t addr_low = static_cast<uint32_t>(addr);
    uint32_t addr_high = static_cast<uint32_t>(addr >> 32);

    if ((addr_low == 0xFFFFFFFF && addr_high == 0xFFFFFFFF) ||
        (addr_low == 0x00000000 && addr_high == 0x00000000)) {
        return false;
    }

    if (__isGlobal(ptr) == 0 && __isShared(ptr) == 0 && __isConstant(ptr) == 0) {
        if (must_be_device) {
            return false;
        }
    }

    if (__isGlobal(ptr) != 0) {
        volatile char test_byte = *reinterpret_cast<volatile char*>(ptr);
        (void)test_byte;
    }

    return true;
}


template<typename T>
__device__ __forceinline__ void print_struct_layout(const char* name) {
}


__device__ __forceinline__ bool validate_int_range(const char* name, int value, int min, int max, const char* file, int line) {
    if (value < min || value > max) {
        return false;
    }
    return true;
}


inline bool validate_launch_config(dim3 grid, dim3 block, size_t shared_mem, const char* kernel_name, const char* file, int line) {
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    bool valid = true;

    if (block.x * block.y * block.z > (unsigned)prop.maxThreadsPerBlock) {
        valid = false;
    }

    if (grid.x > (unsigned)prop.maxGridSize[0] ||
        grid.y > (unsigned)prop.maxGridSize[1] ||
        grid.z > (unsigned)prop.maxGridSize[2]) {
        valid = false;
    }

    if (shared_mem > prop.sharedMemPerBlock) {
        valid = false;
    }

    if (valid) {
    }

    return valid;
}

#define VALIDATE_DEVICE_PTR(ptr) \
    validate_pointer(#ptr, (void*)(ptr), true, __FILE__, __LINE__)

#define VALIDATE_HOST_PTR(ptr) \
    validate_pointer(#ptr, (void*)(ptr), false, __FILE__, __LINE__)

#define VALIDATE_INT_RANGE(val, min, max) \
    validate_int_range(#val, val, min, max, __FILE__, __LINE__)

#define VALIDATE_LAUNCH_CONFIG(grid, block, shared, kernel) \
    validate_launch_config(grid, block, shared, #kernel, __FILE__, __LINE__)

#define PRINT_STRUCT(type) \
    print_struct_layout<type>(#type)

#define BEGIN_KERNEL_VALIDATION(kernel_name) \
    do { \
\
\
        cudaError_t _pending_err = cudaGetLastError(); \
        if (_pending_err != cudaSuccess) { \
\
        } \
    } while(0)

#define END_KERNEL_VALIDATION() \
    do { \
\
    } while(0)

#ifdef __CUDACC__

__device__ __forceinline__ void device_validate_ptr(const void* ptr, const char* name) {
    if (ptr == nullptr) {
        printf("DEVICE_FATAL: %s is null\n", name);
    }
}

__device__ __forceinline__ void device_validate_range(int val, int min_val, int max_val, const char* name) {
    if (val < min_val || val > max_val) {
        printf("DEVICE_FATAL: %s=%d out of range [%d, %d]\n", name, val, min_val, max_val);
    }
}

__device__ __forceinline__ void device_validate_finite(float val, const char* name) {
    if (!isfinite(val)) {
        printf("DEVICE_FATAL: %s is not finite (val=%f)\n", name, val);
    }
}

__device__ __forceinline__ void device_validate_alignment(const void* ptr, size_t alignment, const char* name) {
    if (((uintptr_t)ptr % alignment) != 0) {
        printf("DEVICE_FATAL: %s not aligned to %zu (addr=%p)\n", name, alignment, ptr);
    }
}

__device__ __forceinline__ void device_validate_warp_uniform(int val, const char* name) {
    unsigned mask = __activemask();
    int first_val = __shfl_sync(mask, val, 0);
    if (val != first_val) {
        printf("DEVICE_FATAL: %s not warp-uniform (lane0=%d, this=%d)\n", name, first_val, val);
    }
}

__device__ __forceinline__ void device_validate_coalesced_access(const void* ptr, size_t stride, const char* name) {
    unsigned mask = __activemask();
    int lane = threadIdx.x % warpSize;
    uintptr_t addr = (uintptr_t)ptr + lane * stride;
    uintptr_t base = __shfl_sync(mask, addr, 0);
    uintptr_t expected = base + lane * stride;
    if (addr != expected) {
        printf("DEVICE_FATAL: %s access not coalesced (expected=%p, got=%p)\n", name, (void*)expected, (void*)addr);
    }
}

__device__ __forceinline__ void device_validate_grid_bounds(int idx, int grid_size, const char* name) {
    if (idx < 0 || idx >= grid_size) {
        printf("DEVICE_FATAL: %s=%d out of grid bounds [0, %d)\n", name, idx, grid_size);
    }
}

__device__ __forceinline__ void device_validate_capacity(int count, int capacity, const char* count_name, const char* capacity_name) {
    if (count > capacity) {
        printf("DEVICE_FATAL: %s=%d exceeds %s=%d\n", count_name, count, capacity_name, capacity);
    }
}

__device__ __forceinline__ void device_validate_probability(float p, const char* name) {
    if (p < 0.0f || p > 1.0f || !isfinite(p)) {
        printf("DEVICE_FATAL: %s=%f is not a valid probability [0,1]\n", name, p);
    }
}

__device__ __forceinline__ void device_validate_normalized(float val, const char* name) {
    if (val < -1.0f || val > 1.0f || !isfinite(val)) {
        printf("DEVICE_FATAL: %s=%f is not normalized [-1,1]\n", name, val);
    }
}

__device__ __forceinline__ void device_validate_positive_definite(float val, const char* name) {
    if (val <= 0.0f || !isfinite(val)) {
        printf("DEVICE_FATAL: %s=%f must be positive definite\n", name, val);
    }
}

__device__ __forceinline__ void device_validate_tensor_layout(const void* ptr, int rows, int cols, const char* name) {
    if (ptr == nullptr) {
        printf("DEVICE_FATAL: %s tensor is null\n", name);
    }
    if (rows <= 0 || cols <= 0) {
        printf("DEVICE_FATAL: %s has invalid dimensions rows=%d cols=%d\n", name, rows, cols);
    }
    if (((uintptr_t)ptr % 16) != 0) {
        printf("DEVICE_FATAL: %s tensor not 16-byte aligned (addr=%p)\n", name, ptr);
    }
    if ((cols % 4) != 0) {
        printf("DEVICE_FATAL: %s tensor cols=%d not aligned to 4 for vectorized access\n", name, cols);
    }
}

__device__ __forceinline__ void device_validate_gradient_magnitude(float grad, float max_norm, const char* name) {
    if (!isfinite(grad)) {
        printf("DEVICE_FATAL: %s gradient is not finite (val=%f)\n", name, grad);
    }
    float abs_grad = fabsf(grad);
    if (abs_grad > max_norm) {
        printf("DEVICE_FATAL: %s gradient magnitude %f exceeds max_norm %f\n", name, abs_grad, max_norm);
    }
}

__device__ __forceinline__ void device_validate_memory_alignment(const void* ptr, size_t alignment, const char* name) {
    if (ptr == nullptr) {
        printf("DEVICE_FATAL: %s is null\n", name);
    }
    uintptr_t addr = (uintptr_t)ptr;
    if ((addr % alignment) != 0) {
        printf("DEVICE_FATAL: %s (addr=%p) not aligned to %zu bytes\n", name, ptr, alignment);
    }
    unsigned mask = __activemask();
    uintptr_t lane0_addr = __shfl_sync(mask, addr, 0);
    int lane = threadIdx.x % warpSize;
    if (lane > 0) {
        uintptr_t expected_offset = lane * alignment;
        uintptr_t actual_offset = addr - lane0_addr;
        if (actual_offset != expected_offset && actual_offset != 0) {
            printf("DEVICE_FATAL: %s warp access pattern non-contiguous lane=%d\n", name, lane);
        }
    }
}

__device__ __forceinline__ void device_validate_shared_memory_bounds(int offset, int max_size, const char* name) {
    if (offset < 0) {
        printf("DEVICE_FATAL: %s shared memory offset %d is negative\n", name, offset);
    }
    if (offset >= max_size) {
        printf("DEVICE_FATAL: %s shared memory offset %d exceeds capacity %d\n", name, offset, max_size);
    }
    unsigned mask = __activemask();
    int max_in_warp = offset;
    for (int delta = 16; delta > 0; delta /= 2) {
        int other = __shfl_down_sync(mask, max_in_warp, delta);
        max_in_warp = max(max_in_warp, other);
    }
    max_in_warp = __shfl_sync(mask, max_in_warp, 0);
    if (max_in_warp >= max_size) {
        printf("DEVICE_FATAL: %s warp max shared offset %d exceeds capacity %d\n", name, max_in_warp, max_size);
    }
}

__device__ __forceinline__ void device_validate_grid_coordinates(int x, int y, int grid_size, const char* name) {
    if (x < 0 || x >= grid_size) {
        printf("DEVICE_FATAL: %s x=%d out of grid [0, %d)\n", name, x, grid_size);
    }
    if (y < 0 || y >= grid_size) {
        printf("DEVICE_FATAL: %s y=%d out of grid [0, %d)\n", name, y, grid_size);
    }
    int linear_idx = y * grid_size + x;
    if (linear_idx < 0 || linear_idx >= grid_size * grid_size) {
        printf("DEVICE_FATAL: %s linear index %d overflow\n", name, linear_idx);
    }
}

__device__ __forceinline__ void device_validate_behavioral_dimension(int idx, const char* name) {
    extern __constant__ int d_behavioral_dim;
    if (idx < 0) {
        printf("DEVICE_FATAL: %s behavioral dimension index %d is negative\n", name, idx);
    }
    #ifdef BEHAVIORAL_DIM
    if (idx >= BEHAVIORAL_DIM) {
        printf("DEVICE_FATAL: %s behavioral dimension index %d >= BEHAVIORAL_DIM=%d\n", name, idx, BEHAVIORAL_DIM);
    }
    #endif
}

__device__ __forceinline__ void device_validate_archive_index(int idx, int archive_size, const char* name) {
    if (idx < 0) {
        printf("DEVICE_FATAL: %s archive index %d is negative\n", name, idx);
    }
    if (idx >= archive_size) {
        printf("DEVICE_FATAL: %s archive index %d >= archive_size=%d\n", name, idx, archive_size);
    }
    #ifdef MAX_ARCHIVE_SIZE
    if (archive_size > MAX_ARCHIVE_SIZE) {
        printf("DEVICE_FATAL: %s archive_size=%d exceeds MAX_ARCHIVE_SIZE=%d\n", name, archive_size, MAX_ARCHIVE_SIZE);
    }
    #endif
}

__device__ __forceinline__ void device_validate_genome_slot(int slot, const char* name) {
    if (slot < 0) {
        printf("DEVICE_FATAL: %s genome slot %d is negative\n", name, slot);
    }
    #ifdef GENOME_SIZE
    if (slot >= GENOME_SIZE) {
        printf("DEVICE_FATAL: %s genome slot %d >= GENOME_SIZE=%d\n", name, slot, GENOME_SIZE);
    }
    #endif
}

__device__ __forceinline__ void device_validate_hardware_counter(unsigned long long val, unsigned long long min_plausible, unsigned long long max_plausible, const char* name) {
    if (val == 0 && min_plausible > 0) {
        printf("DEVICE_FATAL: %s hardware counter is zero but min_plausible=%llu\n", name, min_plausible);
    }
    if (val < min_plausible) {
        printf("DEVICE_FATAL: %s hardware counter %llu < min_plausible=%llu\n", name, val, min_plausible);
    }
    if (val > max_plausible) {
        printf("DEVICE_FATAL: %s hardware counter %llu > max_plausible=%llu (implausible value suggests corruption)\n", name, val, max_plausible);
    }
}

__device__ __forceinline__ void device_validate_fitness_components(float fitness, float coherence, float rank, const char* name) {
    if (!isfinite(fitness)) {
        printf("DEVICE_FATAL: %s fitness=%f is not finite\n", name, fitness);
    }
    if (!isfinite(coherence)) {
        printf("DEVICE_FATAL: %s coherence=%f is not finite\n", name, coherence);
    }
    if (!isfinite(rank)) {
        printf("DEVICE_FATAL: %s rank=%f is not finite\n", name, rank);
    }
    if (coherence < 0.0f || coherence > 1.0f) {
        printf("DEVICE_FATAL: %s coherence=%f not in [0,1]\n", name, coherence);
    }
    if (rank < 0.0f) {
        printf("DEVICE_FATAL: %s rank=%f is negative\n", name, rank);
    }
    float consistency = fitness * coherence;
    if (!isfinite(consistency)) {
        printf("DEVICE_FATAL: %s fitness*coherence product not finite\n", name);
    }
}

__device__ __forceinline__ void device_validate_warp_divergence_acceptable(unsigned int ballot_mask, int max_divergent_lanes, const char* name) {
    int active_lanes = __popc(ballot_mask);
    int inactive_lanes = warpSize - active_lanes;
    if (inactive_lanes > max_divergent_lanes) {
        printf("DEVICE_FATAL: %s warp divergence %d inactive lanes exceeds max=%d (ballot=0x%08x)\n",
               name, inactive_lanes, max_divergent_lanes, ballot_mask);
    }
    unsigned expected_mask = __activemask();
    if ((ballot_mask & expected_mask) != ballot_mask) {
        printf("DEVICE_FATAL: %s ballot_mask 0x%08x has lanes outside activemask 0x%08x\n",
               name, ballot_mask, expected_mask);
    }
}

__device__ __forceinline__ void device_validate_causality_chain(int current_gen, int dependency_gen, const char* current_name, const char* dependency_name) {
    if (dependency_gen >= current_gen) {
        printf("DEVICE_FATAL: Causality violation: %s (gen=%d) depends on %s (gen=%d) which is not from past\n",
               current_name, current_gen, dependency_name, dependency_gen);
    }
    if (dependency_gen < 0) {
        printf("DEVICE_FATAL: %s has invalid dependency generation %d\n", dependency_name, dependency_gen);
    }
}

__device__ __forceinline__ void device_validate_data_flow_origin(const void* ptr, bool must_be_from_dataloader, const char* name) {
    if (ptr == nullptr) {
        printf("DEVICE_FATAL: %s is null - data flow broken\n", name);
    }
    if (must_be_from_dataloader) {
        uintptr_t addr = (uintptr_t)ptr;
        if ((addr & 0xFFFF000000000000ULL) != 0) {
            printf("DEVICE_FATAL: %s address %p looks corrupted - not from valid allocation\n", name, ptr);
        }
    }
}

__device__ __forceinline__ void device_validate_epigenetic_bounds(float val, float genome_min, float genome_max, const char* name) {
    if (!isfinite(val)) {
        printf("DEVICE_FATAL: %s epigenetic value %f not finite\n", name, val);
    }
    float epi_min = genome_min * 0.5f;
    float epi_max = 2.0f + genome_max;
    if (val < epi_min || val > epi_max) {
        printf("DEVICE_FATAL: %s epigenetic value %f outside genome-derived bounds [%f, %f]\n",
               name, val, epi_min, epi_max);
    }
}

__device__ __forceinline__ void device_validate_tensor_core_alignment(const void* ptr, int m, int n, int k, const char* name) {
    if (ptr == nullptr) {
        printf("DEVICE_FATAL: %s tensor core operand is null\n", name);
    }
    if ((m % 16) != 0 || (n % 16) != 0 || (k % 16) != 0) {
        printf("DEVICE_FATAL: %s tensor core dims m=%d n=%d k=%d not aligned to 16\n", name, m, n, k);
    }
    if (((uintptr_t)ptr % 256) != 0) {
        printf("DEVICE_FATAL: %s tensor core ptr not 256-byte aligned\n", name);
    }
}

__device__ __forceinline__ void device_validate_flow_lenia_state(float concentration, float velocity_x, float velocity_y, const char* name) {
    if (!isfinite(concentration) || concentration < 0.0f) {
        printf("DEVICE_FATAL: %s flow lenia concentration %f invalid\n", name, concentration);
    }
    if (!isfinite(velocity_x) || !isfinite(velocity_y)) {
        printf("DEVICE_FATAL: %s flow lenia velocity (%f, %f) not finite\n", name, velocity_x, velocity_y);
    }
    float speed_sq = velocity_x * velocity_x + velocity_y * velocity_y;
    if (speed_sq > 1e6f) {
        printf("DEVICE_FATAL: %s flow lenia speed^2 %f implausibly high\n", name, speed_sq);
    }
}

#define DEVICE_VALIDATE_PTR(ptr) device_validate_ptr((ptr), #ptr)
#define DEVICE_VALIDATE_RANGE(val, min, max) device_validate_range((val), (min), (max), #val)
#define DEVICE_VALIDATE_FINITE(val) device_validate_finite((val), #val)
#define DEVICE_VALIDATE_ALIGNMENT(ptr, align) device_validate_alignment((ptr), (align), #ptr)
#define DEVICE_VALIDATE_WARP_UNIFORM(val) device_validate_warp_uniform((val), #val)
#define DEVICE_VALIDATE_COALESCED(ptr, stride) device_validate_coalesced_access((ptr), (stride), #ptr)
#define DEVICE_VALIDATE_GRID_BOUNDS(idx, size) device_validate_grid_bounds((idx), (size), #idx)
#define DEVICE_VALIDATE_CAPACITY(count, cap) device_validate_capacity((count), (cap), #count, #cap)
#define DEVICE_VALIDATE_PROBABILITY(p) device_validate_probability((p), #p)
#define DEVICE_VALIDATE_NORMALIZED(val) device_validate_normalized((val), #val)
#define DEVICE_VALIDATE_POSITIVE_DEFINITE(val) device_validate_positive_definite((val), #val)
#define DEVICE_VALIDATE_TENSOR_LAYOUT(ptr, rows, cols) device_validate_tensor_layout((ptr), (rows), (cols), #ptr)
#define DEVICE_VALIDATE_GRADIENT_MAGNITUDE(grad, max_norm) device_validate_gradient_magnitude((grad), (max_norm), #grad)
#define DEVICE_VALIDATE_MEMORY_ALIGNMENT(ptr, align) device_validate_memory_alignment((ptr), (align), #ptr)
#define DEVICE_VALIDATE_SHARED_BOUNDS(offset, max_size) device_validate_shared_memory_bounds((offset), (max_size), #offset)
#define DEVICE_VALIDATE_GRID_COORDS(x, y, size) device_validate_grid_coordinates((x), (y), (size), "grid_coords")
#define DEVICE_VALIDATE_BEHAVIORAL_DIM(idx) device_validate_behavioral_dimension((idx), #idx)
#define DEVICE_VALIDATE_ARCHIVE_IDX(idx, size) device_validate_archive_index((idx), (size), #idx)
#define DEVICE_VALIDATE_GENOME_SLOT(slot) device_validate_genome_slot((slot), #slot)
#define DEVICE_VALIDATE_HW_COUNTER(val, min_p, max_p) device_validate_hardware_counter((val), (min_p), (max_p), #val)
#define DEVICE_VALIDATE_FITNESS(f, c, r) device_validate_fitness_components((f), (c), (r), "fitness_components")
#define DEVICE_VALIDATE_DIVERGENCE(mask, max_div) device_validate_warp_divergence_acceptable((mask), (max_div), "warp_divergence")
#define DEVICE_VALIDATE_CAUSALITY(cur_gen, dep_gen, cur_name, dep_name) device_validate_causality_chain((cur_gen), (dep_gen), (cur_name), (dep_name))
#define DEVICE_VALIDATE_DATA_ORIGIN(ptr, from_loader) device_validate_data_flow_origin((ptr), (from_loader), #ptr)
#define DEVICE_VALIDATE_EPIGENETIC(val, gmin, gmax) device_validate_epigenetic_bounds((val), (gmin), (gmax), #val)
#define DEVICE_VALIDATE_TENSOR_CORE(ptr, m, n, k) device_validate_tensor_core_alignment((ptr), (m), (n), (k), #ptr)
#define DEVICE_VALIDATE_FLOW_LENIA(conc, vx, vy) device_validate_flow_lenia_state((conc), (vx), (vy), "flow_lenia_state")

#endif

#endif
