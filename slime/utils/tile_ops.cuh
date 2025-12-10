#ifndef TILE_OPS_CUH
#define TILE_OPS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/atomic>

struct PoolEntry;

__device__ __forceinline__ float ldg_float(const float* ptr) {
    #if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
    #else
    return *ptr;
    #endif
}

__device__ __forceinline__ float4 ldg_float4(const float4* ptr) {
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

__device__ __forceinline__ int warp_vote_ballot(int predicate) {
    unsigned mask = __activemask();
    return __ballot_sync(mask, predicate);
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





__global__ void convert_weights_to_fp16(
    float* __restrict__ weights_fp32,
    half* __restrict__ weights_fp16,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights_fp16[idx] = __float2half(weights_fp32[idx]);
    }
}

__global__ void convert_weights_to_fp32(
    half* __restrict__ weights_fp16,
    float* __restrict__ weights_fp32,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights_fp32[idx] = __half2float(weights_fp16[idx]);
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

template<int TILE_DIM, int HALO, int BANK_OFFSET = 0>
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
    __device__ static T& at(T (&tile)[PADDED][PADDED], int tx, int ty, int dx = 0, int dy = 0) {
        return tile[ty + HALO + dy][tx + HALO + dx];
    }
};

template<int TILE_DIM, int HALO, int BANK_OFFSET = 0>
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
        __syncthreads();

        val = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? shared[lane] : 0.0f;
        if (wid == 0) val = WarpReduce<WARP_SIZE>::sum(val);

        return val;
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
        __syncthreads();
        #endif
    }

    __device__ static void wait_group() {
        #if __CUDA_ARCH__ >= 800
        __pipeline_wait_prior(0);
        #else
        __syncthreads();
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

    __device__ static float bilinear(float v00, float v10, float v01, float v11, float tx, float ty) {
        float v0 = linear(v00, v10, tx);
        float v1 = linear(v01, v11, tx);
        return linear(v0, v1, ty);
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
        int stride = 1
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
        int grid_size
    ) {
        float stencil[3][3];
        load_3x3(stencil, global, x, y, grid_size);
        return laplacian_2d(stencil);
    }

    
    __device__ static float gradient_x_at(
        const float* __restrict__ global,
        int x, int y,
        int grid_size
    ) {
        float stencil[3][3];
        load_3x3(stencil, global, x, y, grid_size);
        return gradient_x(stencil);
    }

    
    __device__ static float gradient_y_at(
        const float* __restrict__ global,
        int x, int y,
        int grid_size
    ) {
        float stencil[3][3];
        load_3x3(stencil, global, x, y, grid_size);
        return gradient_y(stencil);
    }

    
    __device__ static void gradients_at(
        float& grad_x,
        float& grad_y,
        const float* __restrict__ global,
        int x, int y,
        int grid_size
    ) {
        float stencil[3][3];
        load_3x3(stencil, global, x, y, grid_size);
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
        int grid_size
    ) {
        float stencil[3][3];
        load_3x3(stencil, global, x, y, grid_size);
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

// WMMA tensor core operations for sm_70+ (Volta/Turing/Ampere/Hopper)
#if __CUDA_ARCH__ >= 700

template<int M = 16, int N = 16, int K = 16>
struct TensorCoreMatmul {
    // FP16 matrix multiply: C[M×N] = A[M×K] × B[K×N]
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

    // Batched multiply for large matrices
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

        if (warp_m >= m || warp_n >= n) return;

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
};
#endif

// DIRESA-specific optimizations using tile_ops primitives
struct DIRESAOps {
    // Vectorized latent space encode (128 elements = 32 float4 loads)
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

        // Handle remainder
        int remainder_start = vec_count * 4;
        if (tid == 0 && remainder_start < latent_dim) {
            for (int i = remainder_start; i < latent_dim; i++) {
                latent[i] = ldg_float(&features[i]);
            }
        }
    }

    // Warp-synchronous latent decode (128D latent = 4 per thread × 32 threads)
    __device__ static void decode_latent_warp(
        const float* latent,
        float* output,
        int latent_dim
    ) {
        auto warp = cg::tiled_partition<32>(cg::this_thread_block());
        int lane = warp.thread_rank();

        // Each thread handles 4 latent elements
        for (int i = lane * 4; i < latent_dim; i += 128) {
            if (i + 3 < latent_dim) {
                float4 lat4 = ldg_float4((const float4*)&latent[i]);
                ((float4*)output)[i / 4] = lat4;
            }
        }

        warp.sync();  // Warp-level sync (5 cycles vs 1000+ for cudaDeviceSynchronize)
    }

    // Warp-reduce for distance computation
    __device__ static float compute_latent_distance_sq(
        const float* latent1,
        const float* latent2,
        int latent_dim
    ) {
        int lane = threadIdx.x % WARP_SIZE;
        float local_sum = 0.0f;

        // Each thread computes 4 squared differences
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

    // Convert genome batch to FP16 for tensor cores
    __device__ static void batch_convert_fp32_to_fp16(
        const float* input,
        half* output,
        int count
    ) {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;

        // Vectorized conversion: 4 floats at a time
        if (tid * 4 < count) {
            float4 in4 = ldg_float4((const float4*)&input[tid * 4]);
            output[tid * 4 + 0] = __float2half(in4.x);
            output[tid * 4 + 1] = __float2half(in4.y);
            output[tid * 4 + 2] = __float2half(in4.z);
            output[tid * 4 + 3] = __float2half(in4.w);
        }
    }
};

// Safe size computation (prevents integer overflow in dataset size calculations)
// Used in DATASET_REGISTRY_INIT to compute train_size_bytes and test_size_bytes
struct SafeSize {
    // 2D grayscale: samples × rows × cols (MNIST, Fashion-MNIST, ChestX-ray)
    static constexpr size_t vision_2d(int samples, int rows, int cols) {
        return static_cast<size_t>(samples) * rows * cols;
    }

    // 3D color: samples × rows × cols × channels (CIFAR-10, PathMNIST, RetinaMNIST)
    static constexpr size_t vision_3d(int samples, int rows, int cols, int channels) {
        return static_cast<size_t>(samples) * rows * cols * channels;
    }

    // Audio spectral: samples × time × mels × channels × sizeof(float) (ESC-50, Speech-Commands, UrbanSound8K)
    static constexpr size_t audio_spectral(int samples, int time, int mels, int channels) {
        return static_cast<size_t>(samples) * time * mels * channels * sizeof(float);
    }

    // Timeseries multi-feature: samples × timesteps × features × sizeof(float) (UCI-HAR, OPPORTUNITY)
    static constexpr size_t timeseries_multi(int samples, int timesteps, int features) {
        return static_cast<size_t>(samples) * timesteps * features * sizeof(float);
    }

    // Timeseries single: samples × timesteps × sizeof(float) (MIT-BIH-ECG)
    static constexpr size_t timeseries_single(int samples, int timesteps) {
        return static_cast<size_t>(samples) * timesteps * sizeof(float);
    }
};

// Cooperative group sync wrapper (replaces cudaDeviceSynchronize)
struct CooperativeSync {
    // Warp-level sync (32 threads, ~5 cycles)
    __device__ static void sync_warp() {
        auto warp = cg::tiled_partition<32>(cg::this_thread_block());
        warp.sync();
    }

    // Block-level sync (slower but still better than device sync)
    __device__ static void sync_block() {
        __syncthreads();
    }

    // Grid-level sync (avoid if possible, use streams instead)
    __device__ static void sync_grid() {
        cg::this_grid().sync();
    }

    // Conditional warp sync
    __device__ static void sync_warp_if(bool condition) {
        if (condition) {
            auto warp = cg::tiled_partition<32>(cg::this_thread_block());
            warp.sync();
        }
    }
};

#endif
