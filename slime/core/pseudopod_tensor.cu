
#ifndef PSEUDOPOD_TENSOR_CU
#define PSEUDOPOD_TENSOR_CU

#include "../config/config.cu"
#include "organism.cu"
#include "../utils/cuda_primitives.cuh"
#include "../utils/genome_params.cuh"
#include "pseudopod.cu"
#include "../metrics/hardware_geometry.cu"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>

namespace cg = cooperative_groups;
namespace wmma = nvcuda::wmma;

__device__ void tensor_core_conv3x3_device(Organism* organism) {
    half* input = organism->tensor_A;
    half* kernel = organism->tensor_B;
    float* output = organism->tensor_C;
    Architecture arch = Architecture::maxBounds();
    int grid_size = arch.grid_size;
    int channels = arch.channels;

    const int warpId = (threadIdx.x + blockIdx.x * blockDim.x) / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;

    const int tile_x = (blockIdx.x * WMMA_TILE_DIM) % grid_size;
    const int tile_y = (blockIdx.y * WMMA_TILE_DIM) % grid_size;
    bool valid_tile = (tile_x < grid_size && tile_y < grid_size);

    float mass_before = 0.0f;
    float mass_after = 0.0f;

    using namespace nvcuda::wmma;

    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    fill_fragment(c_frag, 0.0f);

    __shared__ half tile_shared[WMMA_TILE_DIM][WMMA_TILE_DIM + BANK_PAD];
    __shared__ half kernel_shared[WMMA_TILE_DIM][WMMA_TILE_DIM + BANK_PAD];

    if (valid_tile && threadIdx.x < BLOCK_SIZE) {
        int ti = threadIdx.x / WMMA_TILE_DIM;
        int tj = threadIdx.x % WMMA_TILE_DIM;
        int y = tile_y + ti;
        int x = tile_x + tj;
        if (y < grid_size && x < grid_size && tj < channels) {
            tile_shared[ti][tj] = input[y * grid_size * channels + x * channels + tj];
            mass_before += __half2float(tile_shared[ti][tj]);
        } else {
            tile_shared[ti][tj] = __float2half(0.0f);
        }
    }

    mass_before = WarpReduce<WARP_SIZE>::sum(mass_before);

    cg::this_grid().sync();

    if (valid_tile) {
        load_matrix_sync(a_frag, (half*)tile_shared, WMMA_TILE_DIM);
    }

    for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
            if (valid_tile && threadIdx.x < BLOCK_SIZE) {
                int ki = threadIdx.x / WMMA_TILE_DIM;
                int kj = threadIdx.x % WMMA_TILE_DIM;
                if (ki < channels && kj < channels) {
                    kernel_shared[ki][kj] = kernel[ky * 3 * channels * channels + kx * channels * channels + ki * channels + kj];
                } else {
                    kernel_shared[ki][kj] = __float2half(0.0f);
                }
            }
            cg::this_grid().sync();

            if (valid_tile) {
                load_matrix_sync(b_frag, (half*)kernel_shared, WMMA_TILE_DIM);
                mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        }
    }

    __shared__ float result_shared[WMMA_TILE_DIM][WMMA_TILE_DIM + BANK_PAD];
    if (valid_tile) {
        store_matrix_sync(result_shared[0], c_frag, 16, mem_row_major);
    }
    cg::this_grid().sync();

    if (valid_tile && threadIdx.x < BLOCK_SIZE) {
        int ri = threadIdx.x / WMMA_TILE_DIM;
        int rj = threadIdx.x % WMMA_TILE_DIM;
        int y = tile_y + ri;
        int x = tile_x + rj;
        if (y < grid_size && x < grid_size && rj < channels) {
            int idx = y * grid_size * channels + x * channels + rj;
            output[idx] = result_shared[ri][rj];
            mass_after += result_shared[ri][rj];
        }
    }

    mass_after = WarpReduce<WARP_SIZE>::sum(mass_after);

    int mass_conserved = approx_equal(mass_after, mass_before);
    unsigned int ballot = WarpReduce<WARP_SIZE>::ballot(mass_conserved);

    __shared__ unsigned int warp_ballots[256 / WARP_SIZE];
    if (laneId == 0) {
        warp_ballots[warpId] = ballot;
    }
    cg::this_grid().sync();

    int all_converged = WarpReduce<WARP_SIZE>::all(
        (threadIdx.x < (blockDim.x / WARP_SIZE)) ?
        (warp_ballots[threadIdx.x] == 0xFFFFFFFF) : 1
    );

    if (valid_tile && all_converged && laneId == 0 && is_meaningful(mass_after, mass_before)) {
        float scale = mass_before / mass_after;

        auto tile = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());
        scale = tile.shfl(scale, 0);

        if (laneId < channels) {
            int idx = tile_y * grid_size * channels + tile_x * channels + laneId;
            output[idx] *= scale;
        }
    }
}


#endif
