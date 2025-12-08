
#ifndef TRITON_IMPL_CU
#define TRITON_IMPL_CU
#include "../config/config.cu"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda::wmma;

template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void tiled_matmul_kernel(
    half* __restrict__ A,
    half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {

    wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> c_frag;

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = blockIdx.y * blockDim.y + threadIdx.y;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += TILE_K) {
        int aRow = warpM * TILE_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * TILE_N;

        if (aRow < M && aCol < K && bRow < K && bCol < N) {

            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    int cRow = warpM * TILE_M;
    int cCol = warpN * TILE_N;

    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

__global__ void tiled_conv2d_kernel(
    float* __restrict__ input,
    float* __restrict__ kernel,
    float* __restrict__ output,
    int H, int W, int C,
    int kernel_size,
    int tile_size
) {
    extern __shared__ float shared[];

    float* tile = shared;
    float* kernel_shared = &shared[tile_size * tile_size];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int x = bx * tile_size + tx;
    int y = by * tile_size + ty;

    if (tx < kernel_size && ty < kernel_size) {
        kernel_shared[ty * kernel_size + tx] = kernel[ty * kernel_size + tx];
    }

    int halo = kernel_size / 2;
    int tile_with_halo = tile_size + 2 * halo;

    for (int dy = ty; dy < tile_with_halo; dy += blockDim.y) {
        for (int dx = tx; dx < tile_with_halo; dx += blockDim.x) {
            int gx = bx * tile_size + dx - halo;
            int gy = by * tile_size + dy - halo;

            gx = (gx + W) % W;
            gy = (gy + H) % H;

            tile[dy * tile_with_halo + dx] = input[gy * W + gx];
        }
    }

    __syncthreads();

    if (x < W && y < H) {
        float sum = 0.0f;

        for (int ky = 0; ky < kernel_size; ky++) {
            for (int kx = 0; kx < kernel_size; kx++) {
                int tile_y = ty + ky;
                int tile_x = tx + kx;

                sum += tile[tile_y * tile_with_halo + tile_x] *
                       kernel_shared[ky * kernel_size + kx];
            }
        }

        output[y * W + x] = sum;
    }
}

template<int TILE_SIZE>
__global__ void tiled_reduction_kernel(
    float* __restrict__ input,
    float* __restrict__ output,
    int N
) {
    __shared__ float shared[TILE_SIZE + BANK_PAD];

    int tid = threadIdx.x;
    int gid = blockIdx.x * TILE_SIZE + tid;

    float sum = 0.0f;

    for (int i = gid; i < N; i += gridDim.x * TILE_SIZE) {
        sum += input[i];
    }

    shared[tid] = sum;
    __syncthreads();

    for (int stride = TILE_SIZE / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

template<int TILE_DIM, int BLOCK_ROWS>
__global__ void tiled_transpose_kernel(
    float* __restrict__ input,
    float* __restrict__ output,
    int width, int height
) {
    __shared__ float tile[TILE_DIM][TILE_DIM + BANK_PAD];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * width + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            output[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

#endif
