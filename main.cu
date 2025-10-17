#include <cuda_runtime.h>
#include <cuda.h>
#include <cudadevrt.h>
#include <cooperative_groups.h>
#include <cuda/std/atomic>
#include <nvcuda/wmma.hpp>
#include <stdio.h>
#include <stdlib.h>

namespace cg = cooperative_groups;
namespace wmma = nvcuda::wmma;

constexpr int WARP_SIZE = 32;
constexpr int SM_COUNT = 28;
constexpr int WARPS_PER_SM = 48;
constexpr int TENSOR_CORES_PER_SM = 4;
constexpr int L1_CACHE_SIZE = 128 * 1024;
constexpr int CACHE_LINE = 128;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

struct __align__(128) SlimeState {
    float* coherence;
    float* effective_rank;
    float* ca_grid;
    float* mass_buffer;
    cuda::std::atomic<int>* active_count;
};

__device__ __forceinline__ float warp_reduce_sum(float val) {
    unsigned mask = __activemask();
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    unsigned mask = __activemask();
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(mask, val, offset));
    }
    return val;
}

__global__ void tensor_core_ca_kernel(float* ca_grid, float* kernels, int width) {
    __shared__ half tile_a[WMMA_M][WMMA_K];
    __shared__ half tile_b[WMMA_K][WMMA_N];
    __shared__ float tile_c[WMMA_M][WMMA_N];

    int warp_m = blockIdx.y;
    int warp_n = blockIdx.x;
    int lane = threadIdx.x;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    #pragma unroll
    for (int i = 0; i < WMMA_M; i++) {
        if (lane < WMMA_K) {
            int row = warp_m * WMMA_M + i;
            int col = lane;
            tile_a[i][lane] = __float2half(ca_grid[row * width + col]);
        }
    }

    #pragma unroll
    for (int i = 0; i < WMMA_K; i++) {
        if (lane < WMMA_N) {
            tile_b[i][lane] = __float2half(kernels[i * WMMA_N + lane]);
        }
    }
    __syncthreads();

    wmma::load_matrix_sync(a_frag, (half*)tile_a, WMMA_K);
    wmma::load_matrix_sync(b_frag, (half*)tile_b, WMMA_N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync((float*)tile_c, c_frag, WMMA_N, wmma::mem_row_major);
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < WMMA_M; i++) {
        if (lane < WMMA_N) {
            int row = warp_m * WMMA_M + i;
            int col = warp_n * WMMA_N + lane;
            ca_grid[row * width + col] = tile_c[i][lane];
        }
    }
}

__global__ void flow_lenia_kernel(float* grid, float* params, int width) {
    extern __shared__ float tile[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * blockDim.x + tx;
    int gy = blockIdx.y * blockDim.y + ty;
    int tile_width = blockDim.x + 2;

    if (gx < width && gy < width) {
        tile[(ty + 1) * tile_width + (tx + 1)] = grid[gy * width + gx];
    }

    if (tx == 0 && gx > 0) {
        tile[(ty + 1) * tile_width] = grid[gy * width + (gx - 1)];
    }
    if (tx == blockDim.x - 1 && gx < width - 1) {
        tile[(ty + 1) * tile_width + (tx + 2)] = grid[gy * width + (gx + 1)];
    }
    if (ty == 0 && gy > 0) {
        tile[tx + 1] = grid[(gy - 1) * width + gx];
    }
    if (ty == blockDim.y - 1 && gy < width - 1) {
        tile[(ty + 2) * tile_width + (tx + 1)] = grid[(gy + 1) * width + gx];
    }
    __syncthreads();

    float sum = 0.0f;
    #pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
        #pragma unroll
        for (int dx = -1; dx <= 1; dx++) {
            sum += tile[(ty + 1 + dy) * tile_width + (tx + 1 + dx)] * params[(dy + 1) * 3 + (dx + 1)];
        }
    }

    float mu = params[9];
    float sigma = params[10];
    float growth = expf(-(sum - mu) * (sum - mu) / (2.0f * sigma * sigma));

    if (gx < width && gy < width) {
        grid[gy * width + gx] = tile[(ty + 1) * tile_width + (tx + 1)] * growth;
    }
}

__global__ void compute_fitness(SlimeState* state, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int pid = tid; pid < n; pid += stride) {
        float local_sum = 0.0f;
        int offset = pid * 65536;

        for (int i = threadIdx.x % 32; i < 65536; i += 32) {
            float val = state->ca_grid[offset + i];
            local_sum += val * val;
        }

        float warp_sum = warp_reduce_sum(local_sum);

        if (threadIdx.x % 32 == 0) {
            state->effective_rank[pid] = sqrtf(warp_sum);
            state->coherence[pid] = warp_sum / 65536.0f;
        }
    }
}

__global__ void archive_update(float* archive, float* genomes, float* fitness, int* centroids,
                               int n_centroids, int n_elites, int genome_size, int n_pods) {
    int cid = blockIdx.x;
    if (cid >= n_centroids) return;

    __shared__ float elite_fitness[10];
    __shared__ int best_idx;

    if (threadIdx.x < n_elites) {
        elite_fitness[threadIdx.x] = archive[(cid * n_elites + threadIdx.x) * genome_size];
    }
    __syncthreads();

    for (int pid = 0; pid < n_pods; pid++) {
        if (centroids[pid] == cid) {
            float pod_fit = fitness[pid];

            if (threadIdx.x == 0) {
                best_idx = -1;
                for (int i = 0; i < n_elites; i++) {
                    if (pod_fit > elite_fitness[i]) {
                        best_idx = i;
                        break;
                    }
                }
            }
            __syncthreads();

            if (best_idx >= 0) {
                int src = pid * genome_size;
                int dst = (cid * n_elites + best_idx) * genome_size;
                for (int i = threadIdx.x; i < genome_size; i += blockDim.x) {
                    archive[dst + i] = genomes[src + i];
                }
            }
        }
    }
}

__global__ void spawn_child_kernels(SlimeState* state, int depth) {
    if (depth > 5) return;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int active = state->active_count->load();
        int blocks = min((active + 255) / 256, SM_COUNT * 16);

        if (blocks > 0) {
            spawn_child_kernels<<<blocks, 256>>>(state, depth + 1);
        }

        dim3 ca_grid(16, 16);
        dim3 ca_block(16, 16);
        for (int p = 0; p < active; p++) {
            flow_lenia_kernel<<<ca_grid, ca_block, 18*18*sizeof(float)>>>(
                state->ca_grid + p * 65536, state->mass_buffer, 256
            );
        }

        compute_fitness<<<SM_COUNT * 2, 256>>>(state, active);

        archive_update<<<100, 128>>>(
            state->ca_grid, state->ca_grid, state->coherence,
            (int*)state->mass_buffer, 100, 10, 65536, active
        );
    }
}

__global__ void init_state(SlimeState* state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        size_t align = 128;
        cudaMalloc(&state->coherence, ((256 * sizeof(float) + align - 1) / align) * align);
        cudaMalloc(&state->effective_rank, ((256 * sizeof(float) + align - 1) / align) * align);
        cudaMalloc(&state->ca_grid, ((256 * 65536 * sizeof(float) + align - 1) / align) * align);
        cudaMalloc(&state->mass_buffer, ((11 * sizeof(float) + align - 1) / align) * align);
        cudaMalloc(&state->active_count, sizeof(cuda::std::atomic<int>));

        new (state->active_count) cuda::std::atomic<int>(16);

        float params[11] = {
            0.1f, 0.1f, 0.1f,
            0.1f, 0.2f, 0.1f,
            0.1f, 0.1f, 0.1f,
            3.0f, 1.0f
        };
        cudaMemcpy(state->mass_buffer, params, 11 * sizeof(float), cudaMemcpyHostToDevice);
    }
}

int main(int argc, char** argv) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (prop.major < 8) {
        fprintf(stderr, "sm_80+ required\n");
        return 1;
    }

    printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("SMs: %d, Tensor Cores: %d/SM\n", prop.multiProcessorCount, 4);

    SlimeState* d_state;
    cudaMalloc(&d_state, sizeof(SlimeState));
    init_state<<<1, 1>>>(d_state);
    cudaDeviceSynchronize();

    int steps = 1000;
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--steps=", 8) == 0) {
            steps = atoi(argv[i] + 8);
        }
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int step = 0; step < steps; step++) {
        spawn_child_kernels<<<1, 32>>>(d_state, 0);
        if (step % 100 == 0) {
            cudaDeviceSynchronize();
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Time: %.3f ms (%.2f steps/sec)\n", ms, steps * 1000.0f / ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDeviceReset();
    return 0;
}