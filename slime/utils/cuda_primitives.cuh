#ifndef CUDA_PRIMITIVES_CUH
#define CUDA_PRIMITIVES_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/atomic>
#include <curand_kernel.h>
#include "../config/config.cu"
#include "../core/organism.cu"
#include "../debug/param_validator.cu"

#ifdef __CUDA_ARCH__
#define VALIDATE_WARP_UNIFORM(val, name) device_validate_warp_uniform(val, name)
#define VALIDATE_COALESCED(ptr, stride, name) device_validate_coalesced_access(ptr, stride, name)
#define VALIDATE_PROBABILITY(val, name) device_validate_probability(val, name)
#define VALIDATE_NORMALIZED(val, name) device_validate_normalized(val, name)
#define VALIDATE_POSITIVE_DEFINITE(val, name) device_validate_positive_definite(val, name)
#define VALIDATE_TENSOR_LAYOUT(ptr, rows, cols, name) device_validate_tensor_layout(ptr, rows, cols, name)
#define VALIDATE_GRADIENT_MAGNITUDE(grad, max_norm, name) device_validate_gradient_magnitude(grad, max_norm, name)
#define VALIDATE_MEMORY_ALIGNMENT(ptr, alignment, name) device_validate_memory_alignment(ptr, alignment, name)
#define VALIDATE_SHARED_MEMORY_BOUNDS(offset, max_size, name) device_validate_shared_memory_bounds(offset, max_size, name)
#define VALIDATE_GRID_COORDINATES(x, y, grid_size, name) device_validate_grid_coordinates(x, y, grid_size, name)
#define VALIDATE_BEHAVIORAL_DIMENSION(idx, name) device_validate_behavioral_dimension(idx, name)
#define VALIDATE_ARCHIVE_INDEX(idx, archive_size, name) device_validate_archive_index(idx, archive_size, name)
#define VALIDATE_GENOME_SLOT(slot, name) device_validate_genome_slot(slot, name)
#define VALIDATE_HARDWARE_COUNTER(val, min_plausible, max_plausible, name) device_validate_hardware_counter(val, min_plausible, max_plausible, name)
#define VALIDATE_FITNESS_COMPONENTS(fitness, coherence, rank, name) device_validate_fitness_components(fitness, coherence, rank, name)
#define VALIDATE_WARP_DIVERGENCE_ACCEPTABLE(ballot_mask, max_divergent_lanes, name) device_validate_warp_divergence_acceptable(ballot_mask, max_divergent_lanes, name)
#define VALIDATE_CAUSALITY_CHAIN(cur_gen, dep_gen, cur_name, dep_name) device_validate_causality_chain(cur_gen, dep_gen, cur_name, dep_name)
#define VALIDATE_DATA_FLOW_ORIGIN(ptr, from_loader, name) device_validate_data_flow_origin(ptr, from_loader, name)
#define VALIDATE_EPIGENETIC_BOUNDS(val, gmin, gmax, name) device_validate_epigenetic_bounds(val, gmin, gmax, name)
#define VALIDATE_TENSOR_CORE_ALIGNMENT(ptr, m, n, k, name) device_validate_tensor_core_alignment(ptr, m, n, k, name)
#define VALIDATE_FLOW_LENIA_STATE(conc, vx, vy, name) device_validate_flow_lenia_state(conc, vx, vy, name)
#define VALIDATE_PTR(ptr, name) device_validate_ptr(ptr, name)
#define VALIDATE_RANGE(val, min_val, max_val, name) device_validate_range(val, min_val, max_val, name)
#define VALIDATE_FINITE(val, name) device_validate_finite(val, name)
#define VALIDATE_GRID_BOUNDS(idx, size, name) device_validate_grid_bounds(idx, size, name)
#define VALIDATE_CAPACITY(count, cap, count_name, cap_name) device_validate_capacity(count, cap, count_name, cap_name)
#else
#define VALIDATE_WARP_UNIFORM(val, name) ((void)0)
#define VALIDATE_COALESCED(ptr, stride, name) ((void)0)
#define VALIDATE_PROBABILITY(val, name) ((void)0)
#define VALIDATE_NORMALIZED(val, name) ((void)0)
#define VALIDATE_POSITIVE_DEFINITE(val, name) ((void)0)
#define VALIDATE_TENSOR_LAYOUT(ptr, rows, cols, name) ((void)0)
#define VALIDATE_GRADIENT_MAGNITUDE(grad, max_norm, name) ((void)0)
#define VALIDATE_MEMORY_ALIGNMENT(ptr, alignment, name) ((void)0)
#define VALIDATE_SHARED_MEMORY_BOUNDS(offset, max_size, name) ((void)0)
#define VALIDATE_GRID_COORDINATES(x, y, grid_size, name) ((void)0)
#define VALIDATE_BEHAVIORAL_DIMENSION(idx, name) ((void)0)
#define VALIDATE_ARCHIVE_INDEX(idx, archive_size, name) ((void)0)
#define VALIDATE_GENOME_SLOT(slot, name) ((void)0)
#define VALIDATE_HARDWARE_COUNTER(val, min_plausible, max_plausible, name) ((void)0)
#define VALIDATE_FITNESS_COMPONENTS(fitness, coherence, rank, name) ((void)0)
#define VALIDATE_WARP_DIVERGENCE_ACCEPTABLE(ballot_mask, max_divergent_lanes, name) ((void)0)
#define VALIDATE_CAUSALITY_CHAIN(cur_gen, dep_gen, cur_name, dep_name) ((void)0)
#define VALIDATE_DATA_FLOW_ORIGIN(ptr, from_loader, name) ((void)0)
#define VALIDATE_EPIGENETIC_BOUNDS(val, gmin, gmax, name) ((void)0)
#define VALIDATE_TENSOR_CORE_ALIGNMENT(ptr, m, n, k, name) ((void)0)
#define VALIDATE_FLOW_LENIA_STATE(conc, vx, vy, name) ((void)0)
#define VALIDATE_PTR(ptr, name) ((void)0)
#define VALIDATE_RANGE(val, min_val, max_val, name) ((void)0)
#define VALIDATE_FINITE(val, name) ((void)0)
#define VALIDATE_GRID_BOUNDS(idx, size, name) ((void)0)
#define VALIDATE_CAPACITY(count, cap, count_name, cap_name) ((void)0)
#endif


__host__ __device__ __forceinline__ float safe_epsilon(float reference_scale) {
    return fmaxf(MACHINE_EPS * fabsf(reference_scale), FLOAT_MIN_NORMAL);
}

__device__ __forceinline__ bool is_meaningful(float value, float reference_scale) {
    return fabsf(value) > safe_epsilon(reference_scale);
}

__device__ __forceinline__ bool approx_equal(float a, float b) {
    float scale = fmaxf(fabsf(a), fabsf(b));
    return fabsf(a - b) <= safe_epsilon(scale);
}

__device__ __forceinline__ float safe_div(float numerator, float denominator) {
    return numerator / (denominator + safe_epsilon(denominator));
}

__device__ __forceinline__ float safe_log(float x) {
    return logf(fmaxf(x, safe_epsilon(1.0f)));
}

__device__ __forceinline__ float safe_sqrt_denom(float x) {
    return sqrtf(x) + safe_epsilon(x);
}

__device__ __forceinline__ float activation_relu(float x) {
    return fmaxf(0.0f, x);
}

__device__ __forceinline__ float activation_gelu(float x) {
    return GELU_SCALE * x * (GELU_OFFSET + tanhf(GELU_SQRT_2_OVER_PI * (x + GELU_CUBIC_COEFFICIENT * x * x * x)));
}

__device__ __forceinline__ float activation_sigmoid(float x) {
    return SIGMOID_SCALE / (SIGMOID_SCALE + expf(-x));
}

__device__ __forceinline__ float activation_gelu_backward(float x, float dL_dy) {
    float x2 = x * x, x3 = x2 * x;
    float inner = GELU_SQRT_2_OVER_PI * (x + GELU_CUBIC_COEFFICIENT * x3);
    float tanh_inner = tanhf(inner);
    float sech2 = GELU_OFFSET - tanh_inner * tanh_inner;
    float d_inner = GELU_SQRT_2_OVER_PI * (GELU_OFFSET + GELU_BACKWARD_X2_COEFF * GELU_CUBIC_COEFFICIENT * x2);
    return dL_dy * GELU_SCALE * ((GELU_OFFSET + tanh_inner) + x * sech2 * d_inner);
}

__device__ __forceinline__ float genome_slot_to_unit(const float* genome, int slot) {
    return (genome[slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
}

__device__ __forceinline__ float sample_neighborhood(
    const float* field, int idx, int grid_size, int radius = 1
) {
    if (!field) {
        return nanf("");
    }
    if (grid_size <= 0) {
        return nanf("");
    }
    int cx = idx % grid_size, cy = idx / grid_size;
    float sum = 0.0f;
    int n = 0;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int x = cx + dx, y = cy + dy;
            if (x >= 0 && x < grid_size && y >= 0 && y < grid_size) {
                sum += field[y * grid_size + x];
                n++;
            }
        }
    }
    if (n == 0) {
        return nanf("");
    }
    return sum / n;
}

__device__ __forceinline__ float validated_curand_normal(curandState* state, const char* caller, int idx) {
    float val = curand_normal(state);
    if (isnan(val) || isinf(val)) {
    }
    return val;
}

__device__ __forceinline__ float validated_curand_uniform(curandState* state, const char* caller, int idx) {
    float val = curand_uniform(state);
    if (isnan(val) || isinf(val) || val < 0.0f || val > 1.0f) {
    }
    return val;
}

__device__ __forceinline__ float ldg_float(const float* ptr) {
    if (ptr == nullptr) {
    }
    #if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
    #else
    return *ptr;
    #endif
}

__device__ __forceinline__ float4 ldg_float4(const float4* ptr) {
    if (ptr == nullptr) {
    }
    #if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
    #else
    return *ptr;
    #endif
}
#include <cooperative_groups.h>
#if __CUDA_ARCH__ >= 800
#include <cuda_pipeline.h>
#endif

namespace cg = cooperative_groups;






__device__ __forceinline__ int clamp(int x, int min, int max) {
    return x < min ? min : (x > max ? max : x);
}

__device__ __forceinline__ float clamp(float x, float min, float max) {
    return fminf(fmaxf(x, min), max);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    unsigned mask = __activemask();
    #pragma unroll
    for (int offset = WMMA_TILE_DIM; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    unsigned mask = __activemask();
    #pragma unroll
    for (int offset = WMMA_TILE_DIM; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(mask, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_min(float val) {
    unsigned mask = __activemask();
    #pragma unroll
    for (int offset = WMMA_TILE_DIM; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(mask, val, offset));
    }
    return val;
}

__device__ __forceinline__ int warp_vote_all(int predicate) {
    unsigned mask = __activemask();
    return __all_sync(mask, predicate);
}

__device__ __forceinline__ int warp_vote_any(int predicate) {
    unsigned mask = __activemask();
    return __any_sync(mask, predicate);
}

__device__ __forceinline__ float warp_scan_sum(float val) {
    unsigned mask = __activemask();
    int lane = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        float n = __shfl_up_sync(mask, val, offset);
        if (lane >= offset) val += n;
    }

    return val;
}

__device__ __forceinline__ int warp_scan_int(int val) {
    unsigned mask = __activemask();
    int lane = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        int n = __shfl_up_sync(mask, val, offset);
        if (lane >= offset) val += n;
    }

    return val;
}

__device__ __forceinline__ int warp_reduce_int(int val) {
    unsigned mask = __activemask();
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__device__ __forceinline__ unsigned int jenkins_hash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << JENKINS_MIX_SHIFT_A);
    a = (a ^ 0xc761c23c) ^ (a >> JENKINS_MIX_SHIFT_B);
    a = (a + 0x165667b1) + (a << JENKINS_MIX_SHIFT_C);
    a = (a + 0xd3a2646c) ^ (a << FLOW_KERNEL_SIZE);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> WMMA_TILE_DIM);
    return a;
}

__device__ unsigned long long content_hash(float* data, int size) {
    unsigned long long hash = 0;
    for (int i = 0; i < size; i++) {
        unsigned int bits = __float_as_uint(data[i]);
        hash += jenkins_hash(bits + i);
        hash = (hash << JENKINS_SHIFT_2) | (hash >> JENKINS_ROTATE);
    }
    return hash;
}





__device__ void convert_weights_to_fp16_device(Organism* organism) {
    float* weights_fp32 = organism->weights_fp32;
    half* weights_fp16 = organism->weights_fp16;
    int size = organism->weights_size;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights_fp16[idx] = __float2half(weights_fp32[idx]);
    }
}

__device__ void convert_weights_to_fp32_device(Organism* organism) {
    half* weights_fp16 = organism->weights_fp16;
    float* weights_fp32 = organism->weights_fp32;
    int size = organism->weights_size;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights_fp32[idx] = __half2float(weights_fp16[idx]);
    }
}

__device__ void convert_fp32_to_fp16_strided_device(Organism* organism) {
    const float* src = organism->strided_src_fp32;
    half* dst = organism->strided_dst_fp16;
    int batch_size = organism->strided_batch_size;
    int slice_size = organism->strided_slice_size;
    int src_stride = organism->strided_src_stride;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * slice_size;
    if (idx < total) {
        int batch_id = idx / slice_size;
        int local_idx = idx % slice_size;
        int src_idx = batch_id * src_stride + local_idx;

        dst[idx] = __float2half(src[src_idx]);
    }
}

__device__ void memcpy_to_strided_device(Organism* organism) {
    const float* src = organism->strided_src_fp32;
    float* dst = organism->strided_dst_fp32;
    int batch_size = organism->strided_batch_size;
    int slice_size = organism->strided_slice_size;
    int dst_stride = organism->strided_dst_stride;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * slice_size;
    if (idx < total) {
        int batch_id = idx / slice_size;
        int local_idx = idx % slice_size;
        int dst_idx = batch_id * dst_stride + local_idx;

        dst[dst_idx] = src[idx];
    }
}

template<typename T, int ALIGN = 16>
struct VectorizedLoad {
    __device__ static void load_float4(
        T* __restrict__ dest,
        const T* __restrict__ src,
        int count
    ) {
        int vec_count = count / 4;
        float4* dest4 = reinterpret_cast<float4*>(dest);
        const float4* src4 = reinterpret_cast<const float4*>(src);

        for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
            dest4[i] = src4[i];
        }

        int remainder_start = vec_count * 4;
        for (int i = remainder_start + threadIdx.x; i < count; i += blockDim.x) {
            dest[i] = src[i];
        }
    }

    __device__ static void load_float2(
        T* __restrict__ dest,
        const T* __restrict__ src,
        int count
    ) {
        int vec_count = count / 2;
        float2* dest2 = reinterpret_cast<float2*>(dest);
        const float2* src2 = reinterpret_cast<const float2*>(src);

        for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
            dest2[i] = src2[i];
        }

        int remainder_start = vec_count * 2;
        for (int i = remainder_start + threadIdx.x; i < count; i += blockDim.x) {
            dest[i] = src[i];
        }
    }
};

template<int TILE_DIM, int HALO, int BANK_OFFSET>
struct TiledSection2D {
    static constexpr int PADDED = TILE_DIM + 2 * HALO + BANK_OFFSET;

    template<typename T>
    __device__ static void load_with_halo(
        T (&tile)[PADDED][PADDED],
        const float* __restrict__ global,
        int resolution
    ) {
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int x = blockIdx.x * TILE_DIM + tx;
        int y = blockIdx.y * TILE_DIM + ty;

        
        #if __CUDA_ARCH__ >= 800
        if (x < resolution && y < resolution) {
            AsyncCopy<TILE_DIM>::memcpy_async_tile(&tile[ty + HALO][tx + HALO], &global[y * resolution + x], 1);
        }
        AsyncCopy<TILE_DIM>::commit_group();
        #else
        if (x < resolution && y < resolution) {
            tile[ty + HALO][tx + HALO] = global[y * resolution + x];
        }
        #endif

        
        if (tx < HALO && x >= HALO) {
            tile[ty + HALO][tx] = global[y * resolution + (x - HALO)];
        }
        if (tx >= TILE_DIM - HALO && x < resolution - HALO) {
            tile[ty + HALO][tx + 2 * HALO] = global[y * resolution + (x + HALO)];
        }
        if (ty < HALO && y >= HALO) {
            tile[ty][tx + HALO] = global[(y - HALO) * resolution + x];
        }
        if (ty >= TILE_DIM - HALO && y < resolution - HALO) {
            tile[ty + 2 * HALO][tx + HALO] = global[(y + HALO) * resolution + x];
        }

        #if __CUDA_ARCH__ >= 800
        AsyncCopy<TILE_DIM>::wait_group();
        #endif
    }

    template<typename T>
    __device__ static void store_from_tile(
        float* __restrict__ global,
        const T (&tile)[PADDED][PADDED],
        int resolution
    ) {
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int x = blockIdx.x * TILE_DIM + tx;
        int y = blockIdx.y * TILE_DIM + ty;

        if (x < resolution && y < resolution) {
            global[y * resolution + x] = tile[ty + HALO][tx + HALO];
        }
    }

    template<typename T>
    __device__ static T& at(T (&tile)[PADDED][PADDED], int tx, int ty, int dx, int dy) {
        return tile[ty + HALO + dy][tx + HALO + dx];
    }
};

template<int TILE_DIM, int HALO, int BANK_OFFSET>
struct TiledSection3D {
    static constexpr int PADDED = TILE_DIM + 2 * HALO + BANK_OFFSET;

    template<typename T>
    __device__ static void restrict_from_global(
        T (&section)[PADDED][PADDED][PADDED],
        const float* __restrict__ global,
        int resolution
    ) {
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int tz = threadIdx.z;
        int x = blockIdx.x * TILE_DIM + tx;
        int y = blockIdx.y * TILE_DIM + ty;
        int z = blockIdx.z * TILE_DIM + tz;

        if (x < resolution && y < resolution && z < resolution) {
            int idx = z * resolution * resolution + y * resolution + x;
            section[tz][ty][tx] = global[idx];
        }

        if (tx == TILE_DIM - 1 && x + 1 < resolution) {
            int idx = z * resolution * resolution + y * resolution + (x + 1);
            section[tz][ty][tx + 1] = global[idx];
        }
        if (ty == TILE_DIM - 1 && y + 1 < resolution) {
            int idx = z * resolution * resolution + (y + 1) * resolution + x;
            section[tz][ty + 1][tx] = global[idx];
        }
        if (tz == TILE_DIM - 1 && z + 1 < resolution) {
            int idx = (z + 1) * resolution * resolution + y * resolution + x;
            section[tz + 1][ty][tx] = global[idx];
        }
    }

    template<typename T, typename U>
    __device__ static void restrict_multiple(
        T (&section1)[PADDED][PADDED][PADDED],
        U (&section2)[PADDED][PADDED][PADDED],
        const float* __restrict__ global1,
        const unsigned char* __restrict__ global2,
        int resolution
    ) {
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int tz = threadIdx.z;
        int x = blockIdx.x * TILE_DIM + tx;
        int y = blockIdx.y * TILE_DIM + ty;
        int z = blockIdx.z * TILE_DIM + tz;
        int res = resolution;

        if (x < res && y < res && z < res) {
            int idx = z * res * res + y * res + x;
            section1[tz][ty][tx] = global1[idx];
            section2[tz][ty][tx] = global2[idx];
        }

        if (tx == TILE_DIM - 1 && x + 1 < res) {
            int idx = z * res * res + y * res + (x + 1);
            section1[tz][ty][tx + 1] = global1[idx];
        }
        if (ty == TILE_DIM - 1 && y + 1 < res) {
            int idx = z * res * res + (y + 1) * res + x;
            section1[tz][ty + 1][tx] = global1[idx];
        }
        if (tz == TILE_DIM - 1 && z + 1 < res) {
            int idx = (z + 1) * res * res + y * res + x;
            section1[tz + 1][ty][tx] = global1[idx];
        }
    }
};

template<int BLOCK_SIZE>
struct LinearSection {
    template<typename T>
    __device__ static float3 as_position(const float* __restrict__ data, int idx, int count) {
        if (idx < count) {
            return make_float3(data[idx * 3], data[idx * 3 + 1], data[idx * 3 + 2]);
        }
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    __device__ static float3 as_position_offset(const float* __restrict__ data, int offset, int idx) {
        return make_float3(data[offset + idx * 3], data[offset + idx * 3 + 1], data[offset + idx * 3 + 2]);
    }

    template<typename T>
    __device__ static void restrict_positions(
        T (&section)[BLOCK_SIZE],
        const float* __restrict__ global,
        int count
    ) {
        int tid = threadIdx.x;
        int idx = blockIdx.x * BLOCK_SIZE + tid;

        if (idx < count && tid < BLOCK_SIZE) {
            section[tid] = as_position<T>(global, idx, count);
        }
    }

    template<typename T>
    __device__ static void restrict_flat(
        T* section,
        const float* __restrict__ global,
        int count,
        int stride
    ) {
        int total = count * stride;
        if (total >= BLOCK_SIZE * 4) {
            VectorizedLoad<float>::load_float4(section, global, total);
        } else {
            int tid = threadIdx.x;
            int idx = blockIdx.x * BLOCK_SIZE + tid;

            if (idx < count) {
                for (int i = 0; i < stride; i++) {
                    section[tid * stride + i] = global[idx * stride + i];
                }
            }
        }
    }

    __device__ static void extend_position(
        float* __restrict__ global,
        int idx,
        int count,
        float3 pos
    ) {
        if (idx < count) {
            global[idx * 3] = pos.x;
            global[idx * 3 + 1] = pos.y;
            global[idx * 3 + 2] = pos.z;
        }
    }
};

struct TileBoundary {
    template<typename TileA, typename TileB>
    __device__ static bool verify_continuity(
        const TileA& t1,
        const TileB& t2,
        int overlap_size,
        float tolerance = 1e-6f
    ) {
        for (int i = threadIdx.x; i < overlap_size; i += blockDim.x) {
            if (fabsf(t1[i] - t2[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }
};

template<int TILE_SIZE = 32>
struct WarpReduce {
    __device__ static float sum(float val) {
        auto tile = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
        #pragma unroll
        for (int offset = TILE_SIZE / 2; offset > 0; offset >>= 1) {
            val += tile.shfl_down(val, offset);
        }
        return val;
    }

    __device__ static float max(float val) {
        auto tile = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
        #pragma unroll
        for (int offset = TILE_SIZE / 2; offset > 0; offset >>= 1) {
            val = fmaxf(val, tile.shfl_down(val, offset));
        }
        return val;
    }

    __device__ static float min(float val) {
        auto tile = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
        #pragma unroll
        for (int offset = TILE_SIZE / 2; offset > 0; offset >>= 1) {
            val = fminf(val, tile.shfl_down(val, offset));
        }
        return val;
    }

    __device__ static unsigned ballot(int predicate) {
        auto tile = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
        return tile.ballot(predicate);
    }

    __device__ static int any(int predicate) {
        auto tile = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
        return tile.any(predicate);
    }

    __device__ static int all(int predicate) {
        auto tile = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
        return tile.all(predicate);
    }

    __device__ static int thread_rank() {
        auto tile = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
        return tile.thread_rank();
    }
};

template<int BLOCK_SIZE, int WARP_SIZE = 32>
struct BlockReduce {
    __device__ static float sum(float val) {
        __shared__ float shared[BLOCK_SIZE / WARP_SIZE];
        int lane = threadIdx.x % WARP_SIZE;
        int wid = threadIdx.x / WARP_SIZE;

        val = WarpReduce<WARP_SIZE>::sum(val);

        if (lane == 0) shared[wid] = val;
        cg::this_grid().sync();

        val = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? shared[lane] : 0.0f;
        if (wid == 0) val = WarpReduce<WARP_SIZE>::sum(val);

        return val;
    }
};

template<int BLOCK_SIZE, int WARP_SIZE = 32>
struct BlockScan {
    __device__ static int exclusive_sum(int val, int& total) {
        __shared__ int warp_sums[BLOCK_SIZE / WARP_SIZE];
        int lane = threadIdx.x % WARP_SIZE;
        int wid = threadIdx.x / WARP_SIZE;

        int inclusive = warp_scan_int(val);

        if (lane == WARP_SIZE - 1) warp_sums[wid] = inclusive;
        cg::this_grid().sync();

        if (wid == 0) {
            int warp_val = (lane < BLOCK_SIZE / WARP_SIZE) ? warp_sums[lane] : 0;
            int warp_inclusive = warp_scan_int(warp_val);
            warp_sums[lane] = warp_inclusive;
        }
        cg::this_grid().sync();

        int warp_prefix = (wid > 0) ? warp_sums[wid - 1] : 0;
        int exclusive = inclusive - val + warp_prefix;

        total = warp_sums[BLOCK_SIZE / WARP_SIZE - 1];

        return exclusive;
    }

    __device__ static int compact_index(int predicate, int& total_kept) {
        int exclusive = exclusive_sum(predicate, total_kept);
        return predicate ? exclusive : -1;
    }
};

struct GridStride {
    __device__ static int thread_id() {
        return blockIdx.x * blockDim.x + threadIdx.x;
    }

    __device__ static int stride() {
        return blockDim.x * gridDim.x;
    }
};

struct Atomics {
    __device__ static int claim_slot(int* counter) {
        return atomicAdd(counter, 1);
    }

    __device__ static void add_float(float* address, float val) {
        atomicAdd(address, val);
    }

    __device__ static size_t add_size(size_t* address, size_t val) {
        return atomicAdd((unsigned long long*)address, (unsigned long long)val);
    }

    __device__ static size_t load_size(const size_t* address) {
        return *((volatile unsigned long long*)address);
    }

    __device__ static float cas_float(float* address, float compare, float val) {
        return atomicCAS((int*)address, __float_as_int(compare), __float_as_int(val));
    }

    __device__ static void increment_int(cuda::atomic<int, cuda::thread_scope_system>& counter) {
        atomicAdd((int*)&counter, 1);
    }

    __device__ static void decrement_int(cuda::atomic<int, cuda::thread_scope_system>& counter) {
        atomicSub((int*)&counter, 1);
    }

    __device__ static int load_int(const cuda::atomic<int, cuda::thread_scope_system>& counter) {
        return *((volatile int*)&counter);
    }

    __device__ static void store_int(cuda::atomic<int, cuda::thread_scope_system>& counter, int value) {
        *((volatile int*)&counter) = value;
    }
};

template<int TILE_SIZE>
struct AsyncCopy {
    template<typename T>
    __device__ static void memcpy_async_tile(
        T* __restrict__ dest,
        const T* __restrict__ src,
        int count
    ) {
        #if __CUDA_ARCH__ >= 800
        __pipeline_memcpy_async(dest, src, count * sizeof(T));
        #else
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            dest[i] = src[i];
        }
        #endif
    }

    __device__ static void commit_group() {
        #if __CUDA_ARCH__ >= 800
        __pipeline_commit();
        #else
        cg::this_grid().sync();
        #endif
    }

    __device__ static void wait_group() {
        #if __CUDA_ARCH__ >= 800
        __pipeline_wait_prior(0);
        #else
        cg::this_grid().sync();
        #endif
    }
};

struct Interpolation {
    __device__ static float linear(float a, float b, float t) {
        return __fmaf_rn(b - a, t, a);
    }

    __device__ static float3 linear(float3 a, float3 b, float t) {
        return make_float3(
            __fmaf_rn(b.x - a.x, t, a.x),
            __fmaf_rn(b.y - a.y, t, a.y),
            __fmaf_rn(b.z - a.z, t, a.z)
        );
    }

    __device__ static float bilinear(float tl, float tr, float bl, float br, float tx, float ty) {
        float top = linear(tl, tr, tx);
        float bot = linear(bl, br, tx);
        return linear(top, bot, ty);
    }

    __device__ static float bilinear_grad_x(float tl, float tr, float bl, float br, float ty) {
        return linear(tr - tl, br - bl, ty);
    }

    __device__ static float bilinear_grad_y(float tl, float tr, float bl, float br, float tx) {
        return linear(bl - tl, br - tr, tx);
    }

    __device__ static float3 bilinear_with_grad(float tl, float tr, float bl, float br, float tx, float ty) {
        return make_float3(
            bilinear(tl, tr, bl, br, tx, ty),
            bilinear_grad_x(tl, tr, bl, br, ty),
            bilinear_grad_y(tl, tr, bl, br, tx)
        );
    }

    __device__ static float4 bilinear_weights(float tx, float ty) {
        float omtx = 1.0f - tx;
        float omty = 1.0f - ty;
        return make_float4(omtx * omty, tx * omty, omtx * ty, tx * ty);
    }

    __device__ static void bilinear_weight_grads(float tx, float ty, float4* dw_dtx, float4* dw_dty) {
        float omtx = 1.0f - tx;
        float omty = 1.0f - ty;
        *dw_dtx = make_float4(-omty, omty, -ty, ty);
        *dw_dty = make_float4(-omtx, -tx, omtx, tx);
    }

    __device__ static float trilinear(
        float v000, float v100, float v010, float v110,
        float v001, float v101, float v011, float v111,
        float tx, float ty, float tz
    ) {
        float v00 = linear(v000, v100, tx);
        float v10 = linear(v010, v110, tx);
        float v01 = linear(v001, v101, tx);
        float v11 = linear(v011, v111, tx);
        float v0 = linear(v00, v10, ty);
        float v1 = linear(v01, v11, ty);
        return linear(v0, v1, tz);
    }
};

struct Stencils {
    
    __device__ static void load_3x3(
        float (&stencil)[3][3],
        const float* __restrict__ global,
        int x, int y,
        int grid_size,
        int stride
    ) {
        #pragma unroll
        for (int dy = -1; dy <= 1; dy++) {
            #pragma unroll
            for (int dx = -1; dx <= 1; dx++) {
                int nx = max(0, min(grid_size - 1, x + dx));
                int ny = max(0, min(grid_size - 1, y + dy));
                stencil[dy+1][dx+1] = ldg_float(&global[ny * grid_size * stride + nx * stride]);
            }
        }
    }

    
    __device__ static float laplacian_at(
        const float* __restrict__ global,
        int x, int y,
        int grid_size,
        int stride
    ) {
        float stencil[3][3];
        load_3x3(stencil, global, x, y, grid_size, stride);
        return laplacian_2d(stencil);
    }

    
    __device__ static float gradient_x_at(
        const float* __restrict__ global,
        int x, int y,
        int grid_size,
        int stride
    ) {
        float stencil[3][3];
        load_3x3(stencil, global, x, y, grid_size, stride);
        return gradient_x(stencil);
    }

    
    __device__ static float gradient_y_at(
        const float* __restrict__ global,
        int x, int y,
        int grid_size,
        int stride
    ) {
        float stencil[3][3];
        load_3x3(stencil, global, x, y, grid_size, stride);
        return gradient_y(stencil);
    }

    
    __device__ static void gradients_at(
        float& grad_x,
        float& grad_y,
        const float* __restrict__ global,
        int x, int y,
        int grid_size,
        int stride
    ) {
        float stencil[3][3];
        load_3x3(stencil, global, x, y, grid_size, stride);
        grad_x = gradient_x(stencil);
        grad_y = gradient_y(stencil);
    }

    
    __device__ static void all_operators(
        float& grad_x,
        float& grad_y,
        float& lap,
        float& center,
        const float* __restrict__ global,
        int x, int y,
        int grid_size,
        int stride
    ) {
        float stencil[3][3];
        load_3x3(stencil, global, x, y, grid_size, stride);
        grad_x = gradient_x(stencil);
        grad_y = gradient_y(stencil);
        lap = laplacian_2d(stencil);
        center = stencil[1][1];
    }

    
    template<int N>
    __device__ static float laplacian_2d(const float (&vals)[N][N]) {
        static_assert(N >= 3 && N % 2 == 1, "N must be odd and >= 3");
        constexpr int c = N / 2;
        return vals[c-1][c] + vals[c+1][c] + vals[c][c-1] + vals[c][c+1] - 4.0f * vals[c][c];
    }

    
    template<int N>
    __device__ static float gradient_x(const float (&vals)[N][N]) {
        static_assert(N >= 3 && N % 2 == 1, "N must be odd and >= 3");
        constexpr int c = N / 2;
        return (vals[c][c+1] - vals[c][c-1]) * 0.5f;
    }

    template<int N>
    __device__ static float gradient_y(const float (&vals)[N][N]) {
        static_assert(N >= 3 && N % 2 == 1, "N must be odd and >= 3");
        constexpr int c = N / 2;
        return (vals[c+1][c] - vals[c-1][c]) * 0.5f;
    }

    
    __device__ static float divergence_2d(float2 field, float2 fieldE, float2 fieldN) {
        return (fieldE.x - field.x) + (fieldN.y - field.y);
    }
};

struct FastMath {
    __device__ __forceinline__ static float distance_sq(float3 a, float3 b) {
        float dx = a.x - b.x;
        float dy = a.y - b.y;
        float dz = a.z - b.z;
        float result;
        asm volatile (
            "mul.f32 %0, %1, %1;\n\t"
            "fma.rn.f32 %0, %2, %2, %0;\n\t"
            "fma.rn.f32 %0, %3, %3, %0;\n\t"
            : "=f"(result)
            : "f"(dx), "f"(dy), "f"(dz)
        );
        return result;
    }

    __device__ __forceinline__ static float3 sub(float3 a, float3 b) {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    }

    __device__ __forceinline__ static float dot(float3 a, float3 b) {
        float result;
        asm volatile (
            "mul.f32 %0, %1, %4;\n\t"
            "fma.rn.f32 %0, %2, %5, %0;\n\t"
            "fma.rn.f32 %0, %3, %6, %0;\n\t"
            : "=f"(result)
            : "f"(a.x), "f"(a.y), "f"(a.z), "f"(b.x), "f"(b.y), "f"(b.z)
        );
        return result;
    }
};

struct Occupancy {
};

#include <mma.h>

#if __CUDA_ARCH__ >= 700

template<int M, int N, int K>
struct TensorCoreMatmul {
    __device__ static void multiply_accumulate(
        half* C,
        const half* A,
        const half* B,
        int lda,
        int ldb,
        int ldc
    ) {
        using namespace nvcuda::wmma;
        fragment<matrix_a, M, N, K, half, row_major> a_frag;
        fragment<matrix_b, M, N, K, half, row_major> b_frag;
        fragment<accumulator, M, N, K, half> c_frag;

        load_matrix_sync(a_frag, A, lda);
        load_matrix_sync(b_frag, B, ldb);
        fill_fragment(c_frag, __float2half(0.0f));

        mma_sync(c_frag, a_frag, b_frag, c_frag);

        store_matrix_sync(C, c_frag, ldc, mem_row_major);
    }

    __device__ static void multiply_tiled(
        half* C,
        const half* A,
        const half* B,
        int m, int n, int k,
        int lda, int ldb, int ldc
    ) {
        using namespace nvcuda::wmma;
        int warp_m = (blockIdx.y * blockDim.y + threadIdx.y) * M;
        int warp_n = (blockIdx.x * blockDim.x + threadIdx.x) * N;

        if (warp_m < m && warp_n < n) {
            fragment<accumulator, M, N, K, half> acc_frag;
            fill_fragment(acc_frag, __float2half(0.0f));

            for (int i = 0; i < k; i += K) {
                fragment<matrix_a, M, N, K, half, row_major> a_frag;
                fragment<matrix_b, M, N, K, half, row_major> b_frag;

                load_matrix_sync(a_frag, A + warp_m * lda + i, lda);
                load_matrix_sync(b_frag, B + i * ldb + warp_n, ldb);

                mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }

            store_matrix_sync(C + warp_m * ldc + warp_n, acc_frag, ldc, mem_row_major);
        }
    }
};
#endif

struct DIRESAOps {
    __device__ static void encode_latent_vectorized(
        const float* features,
        float* latent,
        int latent_dim
    ) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        int vec_count = latent_dim / 4;

        if (tid < vec_count) {
            float4 feat4 = ldg_float4((const float4*)&features[tid * 4]);
            ((float4*)latent)[tid] = feat4;
        }

        int remainder_start = vec_count * 4;
        if (tid == 0 && remainder_start < latent_dim) {
            for (int i = remainder_start; i < latent_dim; i++) {
                latent[i] = ldg_float(&features[i]);
            }
        }
    }

    __device__ static void decode_latent_warp(
        const float* latent,
        float* output,
        int latent_dim
    ) {
        auto warp = cg::tiled_partition<32>(cg::this_thread_block());
        int lane = warp.thread_rank();

        for (int i = lane * 4; i < latent_dim; i += 128) {
            if (i + 3 < latent_dim) {
                float4 lat4 = ldg_float4((const float4*)&latent[i]);
                ((float4*)output)[i / 4] = lat4;
            }
        }

        warp.sync();  
    }

    __device__ static float compute_latent_distance_sq(
        const float* latent1,
        const float* latent2,
        int latent_dim
    ) {
        int lane = threadIdx.x % WARP_SIZE;
        float local_sum = 0.0f;

        for (int i = lane * 4; i < latent_dim; i += WARP_SIZE * 4) {
            float4 l1 = ldg_float4((const float4*)&latent1[i]);
            float4 l2 = ldg_float4((const float4*)&latent2[i]);

            float dx = l1.x - l2.x;
            float dy = l1.y - l2.y;
            float dz = l1.z - l2.z;
            float dw = l1.w - l2.w;

            local_sum = __fmaf_rn(dx, dx, local_sum);
            local_sum = __fmaf_rn(dy, dy, local_sum);
            local_sum = __fmaf_rn(dz, dz, local_sum);
            local_sum = __fmaf_rn(dw, dw, local_sum);
        }

        return warp_reduce_sum(local_sum);
    }

    __device__ static void batch_convert_fp32_to_fp16(
        const float* input,
        half* output,
        int count
    ) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;

        if (tid * 4 < count) {
            float4 in4 = ldg_float4((const float4*)&input[tid * 4]);
            output[tid * 4 + 0] = __float2half(in4.x);
            output[tid * 4 + 1] = __float2half(in4.y);
            output[tid * 4 + 2] = __float2half(in4.z);
            output[tid * 4 + 3] = __float2half(in4.w);
        }
    }
};

struct SafeSize {
    static constexpr size_t vision_2d(int samples, int rows, int cols) {
        return static_cast<size_t>(samples) * rows * cols;
    }

    static constexpr size_t vision_3d(int samples, int rows, int cols, int channels) {
        return static_cast<size_t>(samples) * rows * cols * channels;
    }

    static constexpr size_t audio_spectral(int samples, int time, int mels, int channels) {
        return static_cast<size_t>(samples) * time * mels * channels * sizeof(float);
    }

    static constexpr size_t timeseries_multi(int samples, int timesteps, int features) {
        return static_cast<size_t>(samples) * timesteps * features * sizeof(float);
    }

    static constexpr size_t timeseries_single(int samples, int timesteps) {
        return static_cast<size_t>(samples) * timesteps * sizeof(float);
    }
};

__device__ void batched_tensor_core_gemm_device(Organism* organism) {
    const half* A = organism->gemm_A;
    const half* B = organism->gemm_B;
    float* C = organism->gemm_C;
    int M = organism->gemm_M;
    int N = organism->gemm_N;
    int K = organism->gemm_K;
    int A_head_stride = organism->gemm_A_head_stride;
    int B_head_stride = organism->gemm_B_head_stride;
    int C_head_stride = organism->gemm_C_head_stride;

    int head_id = blockIdx.z;
    const int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int warpN = blockIdx.y;

    const int tile_row = warpM * WMMA_TILE_DIM;
    const int tile_col = warpN * WMMA_TILE_DIM;

    if (tile_row < M && tile_col < N) {
        const half* A_head = A + head_id * A_head_stride;
        const half* B_head = B + head_id * B_head_stride;
        float* C_head = C + head_id * C_head_stride;

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::col_major> b_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, float> c_frag;

        nvcuda::wmma::fill_fragment(c_frag, 0.0f);

        for (int k_tile = 0; k_tile < K; k_tile += WMMA_TILE_DIM) {
            if (k_tile + WMMA_TILE_DIM <= K) {
                nvcuda::wmma::load_matrix_sync(a_frag, A_head + tile_row * K + k_tile, K);
                nvcuda::wmma::load_matrix_sync(b_frag, B_head + k_tile * N + tile_col, N);
                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        }
        nvcuda::wmma::store_matrix_sync(C_head + tile_row * N + tile_col, c_frag, N, nvcuda::wmma::mem_row_major);
    }
}

__device__ void batched_tensor_core_gemm_transA_device(Organism* organism) {
    const half* A = organism->gemm_A;
    const half* B = organism->gemm_B;
    float* C = organism->gemm_C;
    int M = organism->gemm_M;
    int N = organism->gemm_N;
    int K = organism->gemm_K;
    int A_head_stride = organism->gemm_A_head_stride;
    int B_head_stride = organism->gemm_B_head_stride;
    int C_head_stride = organism->gemm_C_head_stride;

    int head_id = blockIdx.z;
    const int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int warpN = blockIdx.y;

    const int tile_row = warpM * WMMA_TILE_DIM;
    const int tile_col = warpN * WMMA_TILE_DIM;

    if (tile_row < M && tile_col < N) {
        const half* A_head = A + head_id * A_head_stride;
        const half* B_head = B + head_id * B_head_stride;
        float* C_head = C + head_id * C_head_stride;

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::col_major> a_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, half, nvcuda::wmma::row_major> b_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_TILE_DIM, WMMA_TILE_DIM, WMMA_TILE_DIM, float> c_frag;

        nvcuda::wmma::fill_fragment(c_frag, 0.0f);

        for (int k_tile = 0; k_tile < K; k_tile += WMMA_TILE_DIM) {
            if (k_tile + WMMA_TILE_DIM <= K) {
                nvcuda::wmma::load_matrix_sync(a_frag, A_head + k_tile * M + tile_row, M);
                nvcuda::wmma::load_matrix_sync(b_frag, B_head + k_tile * N + tile_col, N);
                nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        }
        nvcuda::wmma::store_matrix_sync(C_head + tile_row * N + tile_col, c_frag, N, nvcuda::wmma::mem_row_major);
    }
}

__device__ void batched_transpose_fp16_device(Organism* organism) {
    const half* A = organism->transpose_A;
    half* B = organism->transpose_B;
    int M = organism->transpose_M;
    int N = organism->transpose_N;
    int A_head_stride = organism->transpose_A_head_stride;
    int B_head_stride = organism->transpose_B_head_stride;

    int head_id = blockIdx.z;
    __shared__ half tile[WMMA_TILE_DIM][WMMA_TILE_DIM + 1];

    int bx = blockIdx.x * WMMA_TILE_DIM, by = blockIdx.y * WMMA_TILE_DIM;
    int x = bx + threadIdx.x, y = by + threadIdx.y;

    const half* A_head = A + head_id * A_head_stride;
    half* B_head = B + head_id * B_head_stride;

    if (y < M && x < N) tile[threadIdx.y][threadIdx.x] = A_head[y * N + x];
    cg::this_grid().sync();

    int out_x = by + threadIdx.x, out_y = bx + threadIdx.y;
    if (out_y < N && out_x < M) B_head[out_y * M + out_x] = tile[threadIdx.x][threadIdx.y];
}

__device__ void batched_convert_fp32_to_fp16_strided_device(Organism* organism) {
    const float* src = organism->batched_strided_src_fp32;
    half* dst = organism->batched_strided_dst_fp16;
    Architecture arch = Architecture::maxBounds();
    int num_heads = arch.num_heads;
    int batch_size = organism->strided_batch_size;
    int slice_size = organism->strided_slice_size;
    int src_head_stride = organism->batched_src_head_stride;
    int src_batch_stride = organism->batched_src_batch_stride;
    int dst_head_stride = organism->batched_dst_head_stride;
    int batch_offset = organism->strided_batch_offset;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * batch_size * slice_size;
    if (idx < total) {
        int head_id = idx / (batch_size * slice_size);
        int remainder = idx % (batch_size * slice_size);
        int batch_id = remainder / slice_size;
        int local_idx = remainder % slice_size;

        int src_idx = head_id * src_head_stride + (batch_offset + batch_id) * src_batch_stride + local_idx;
        int dst_idx = head_id * dst_head_stride + batch_id * slice_size + local_idx;

        dst[dst_idx] = __float2half(src[src_idx]);
    }
}

__device__ void batched_memcpy_to_strided_device(Organism* organism) {
    const float* src = organism->batched_strided_src_fp32;
    float* dst = organism->batched_strided_dst_fp32;
    Architecture arch = Architecture::maxBounds();
    int num_heads = arch.num_heads;
    int batch_size = organism->strided_batch_size;
    int slice_size = organism->strided_slice_size;
    int src_head_stride = organism->batched_src_head_stride;
    int dst_head_stride = organism->batched_dst_head_stride;
    int dst_batch_stride = organism->batched_dst_batch_stride;
    int batch_offset = organism->strided_batch_offset;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * batch_size * slice_size;
    if (idx < total) {
        int head_id = idx / (batch_size * slice_size);
        int remainder = idx % (batch_size * slice_size);
        int batch_id = remainder / slice_size;
        int local_idx = remainder % slice_size;

        int src_idx = head_id * src_head_stride + batch_id * slice_size + local_idx;
        int dst_idx = head_id * dst_head_stride + (batch_offset + batch_id) * dst_batch_stride + local_idx;

        dst[dst_idx] = src[src_idx];
    }
}

__device__ void batched_accumulate_weight_grads_device(Organism* organism) {
    const float* dW = organism->weight_grad_src;
    float* grad_buffer = organism->grad_buffer;
    const int* head_offsets = organism->head_offsets;
    int weight_size = organism->weight_size;
    Architecture arch = Architecture::maxBounds();
    int num_heads = arch.num_heads;
    int dW_head_stride = organism->dW_head_stride;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * weight_size;
    if (idx < total) {
        int head_id = idx / weight_size;
        int local_idx = idx % weight_size;

        int src_idx = head_id * dW_head_stride + local_idx;
        int dst_idx = head_offsets[head_id] + local_idx;

        grad_buffer[dst_idx] = dW[src_idx];
    }
}

__device__ void batched_gelu_backward_device(Organism* organism) {
    const float* dL_dI = organism->backward_dL_dI;
    const float* pre_gelu = organism->backward_pre_gelu;
    float* dL_dpregelu = organism->backward_dL_dpregelu;
    Architecture arch = Architecture::maxBounds();
    int num_heads = arch.num_heads;
    int elements_per_head = organism->backward_elements_per_head;
    int src_head_stride = organism->backward_src_head_stride;
    int dst_head_stride = organism->backward_dst_head_stride;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * elements_per_head;
    if (idx < total) {
        int head_id = idx / elements_per_head;
        int local_idx = idx % elements_per_head;

        int src_idx = head_id * src_head_stride + local_idx;
        int dst_idx = head_id * dst_head_stride + local_idx;

        dL_dpregelu[dst_idx] = activation_gelu_backward(pre_gelu[src_idx], dL_dI[src_idx]);
    }
}

__device__ void batched_relu_backward_device(Organism* organism) {
    const float* dL_dP = organism->backward_dL_dP;
    const float* P = organism->backward_P;
    float* dL_dprerelu = organism->backward_dL_dprerelu;
    Architecture arch = Architecture::maxBounds();
    int num_heads = arch.num_heads;
    int elements_per_head = organism->backward_elements_per_head;
    int src_head_stride = organism->backward_src_head_stride;
    int dst_head_stride = organism->backward_dst_head_stride;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * elements_per_head;
    if (idx < total) {
        int head_id = idx / elements_per_head;
        int local_idx = idx % elements_per_head;

        int src_idx = head_id * src_head_stride + local_idx;
        int dst_idx = head_id * dst_head_stride + local_idx;

        dL_dprerelu[dst_idx] = dL_dP[src_idx] * ((P[src_idx] > 0.0f) ? 1.0f : 0.0f);
    }
}

__device__ void batched_im2col_device(Organism* organism) {
    const float* input = organism->im2col_input;
    float* col = organism->im2col_col;
    Architecture arch = Architecture::maxBounds();
    int num_heads = arch.num_heads;
    int batch_size = organism->im2col_batch_size;
    int grid_size = arch.grid_size;
    int channels = arch.channels;
    int input_head_stride = organism->im2col_input_head_stride;
    int col_head_stride = organism->im2col_col_head_stride;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_cells = grid_size * grid_size;
    int total = num_heads * batch_size * num_cells;
    if (idx < total) {
        int head_id = idx / (batch_size * num_cells);
        int remainder = idx % (batch_size * num_cells);
        int batch_id = remainder / num_cells;
        int cell_idx = remainder % num_cells;
        int cell_y = cell_idx / grid_size;
        int cell_x = cell_idx % grid_size;

        int col_width = 9 * channels;
        int col_row = batch_id * num_cells + cell_idx;

        const float* input_head = input + head_id * input_head_stride;
        float* col_head = col + head_id * col_head_stride;

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int ny = max(0, min(grid_size - 1, cell_y + dy));
                int nx = max(0, min(grid_size - 1, cell_x + dx));
                int patch_idx = (dy + 1) * 3 + (dx + 1);

                int input_base = batch_id * grid_size * grid_size * channels +
                                ny * grid_size * channels + nx * channels;

                for (int c = 0; c < channels; c++) {
                    col_head[col_row * col_width + patch_idx * channels + c] =
                        input_head[input_base + c];
                }
            }
        }
    }
}

__device__ void batched_col2im_device(Organism* organism) {
    const float* col = organism->col2im_col;
    float* output_grad = organism->col2im_output_grad;
    Architecture arch = Architecture::maxBounds();
    int num_heads = arch.num_heads;
    int batch_size = organism->col2im_batch_size;
    int grid_size = arch.grid_size;
    int channels = arch.channels;
    int col_head_stride = organism->col2im_col_head_stride;
    int output_head_stride = organism->col2im_output_head_stride;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_cells = grid_size * grid_size;
    int total = num_heads * batch_size * num_cells;
    if (idx < total) {
        int head_id = idx / (batch_size * num_cells);
        int remainder = idx % (batch_size * num_cells);
        int batch_id = remainder / num_cells;
        int cell_idx = remainder % num_cells;
        int cell_y = cell_idx / grid_size;
        int cell_x = cell_idx % grid_size;

        int col_width = 9 * channels;

        const float* col_head = col + head_id * col_head_stride;
        float* output_head = output_grad + head_id * output_head_stride;

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int out_y = cell_y - dy;
                int out_x = cell_x - dx;
                if (out_y >= 0 && out_y < grid_size && out_x >= 0 && out_x < grid_size) {
                    int out_cell = batch_id * num_cells + out_y * grid_size + out_x;
                    int patch_idx = (dy + 1) * 3 + (dx + 1);

                    int output_base = batch_id * grid_size * grid_size * channels +
                                     cell_y * grid_size * channels + cell_x * channels;

                    for (int c = 0; c < channels; c++) {
                        atomicAdd(&output_head[output_base + c],
                                 col_head[out_cell * col_width + patch_idx * channels + c]);
                    }
                }
            }
        }
    }
}

struct CooperativeSync {
    __device__ static void sync_warp() {
        auto warp = cg::tiled_partition<32>(cg::this_thread_block());
        warp.sync();
    }

    __device__ static void sync_block() {
        cg::this_grid().sync();
    }

    __device__ static void sync_grid() {
        cg::this_grid().sync();
    }

    __device__ static void sync_warp_if(bool condition) {
        if (condition) {
            auto warp = cg::tiled_partition<32>(cg::this_thread_block());
            warp.sync();
        }
    }
};

#endif
