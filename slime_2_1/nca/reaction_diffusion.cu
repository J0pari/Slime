// Sheet A-202: Reaction-Diffusion Field
//
// Unchanged from 2.0 in spec. Operates on chemical channels 0-5 of the CA
// grid. Reaction coefficients (channels 0-5 pairwise interactions) and
// diffusion rates come from the genome (A-301 bits 34-281).

#ifndef SLIME_2_1_NCA_REACTION_DIFFUSION_CU
#define SLIME_2_1_NCA_REACTION_DIFFUSION_CU

#include "../config/constants.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace slime::nca::rd {

// Per-organism reaction/diffusion coefficients decoded from the genome.
// Reaction coefficients: 6x6 matrix of pairwise rates (200 bits ~ quantized).
// Diffusion rates: one per chemical channel (48 bits ~ quantized).
struct Coefficients {
    float reaction[6 * 6];
    float diffusion[6];
};

// One reaction-diffusion step on chemical channels 0..5 of the grid.
// Coupling into channels 6..15 is read-only (the CA dynamics push activity
// back into the chemical field, but the RD step itself only writes chems).
//
// dt is fixed at 0.1; diffusion uses a 5-point Laplacian on the toroidal
// grid. Reaction terms are linear pairwise (A_i' += k_{ij} * A_j); the
// 6x6 coefficient matrix decodes from the genome.
__device__ inline void rd_step(__half* grid,
                               __half* scratch,
                               const Coefficients& coeffs) {
    constexpr int CHEM_N = 6;
    constexpr float DT = 0.1f;
    for (int by = 0; by < GRID_SIZE; by += blockDim.y) {
        for (int bx = 0; bx < GRID_SIZE; bx += blockDim.x) {
            int y = by + threadIdx.y;
            int x = bx + threadIdx.x;
            if (y >= GRID_SIZE || x >= GRID_SIZE) continue;
            auto idx = [](int yy, int xx, int c) {
                return (yy * GRID_SIZE + xx) * CA_CHANNELS + c;
            };
            int yp = (y + 1) % GRID_SIZE;
            int ym = (y - 1 + GRID_SIZE) % GRID_SIZE;
            int xp = (x + 1) % GRID_SIZE;
            int xm = (x - 1 + GRID_SIZE) % GRID_SIZE;
            float here[CHEM_N];
            #pragma unroll
            for (int c = 0; c < CHEM_N; ++c) {
                here[c] = __half2float(grid[idx(y, x, c)]);
            }
            #pragma unroll
            for (int c = 0; c < CHEM_N; ++c) {
                float lap = __half2float(grid[idx(yp, x, c)])
                          + __half2float(grid[idx(ym, x, c)])
                          + __half2float(grid[idx(y, xp, c)])
                          + __half2float(grid[idx(y, xm, c)])
                          - 4.f * here[c];
                float react = 0.f;
                #pragma unroll
                for (int j = 0; j < CHEM_N; ++j) {
                    react += coeffs.reaction[c * CHEM_N + j] * here[j];
                }
                float updated = here[c] + DT * (coeffs.diffusion[c] * lap + react);
                if (updated >  65504.f) updated =  65504.f;
                if (updated < -65504.f) updated = -65504.f;
                scratch[idx(y, x, c)] = __float2half(updated);
            }
        }
    }
    __syncthreads();
}

// Decode quantized coefficient bits from the genome into floats.
// Reaction coeffs: 200 bits across 36 entries (6x6) -> ~5.5 bits each.
// We quantize to 5 bits per entry centred on zero (-1.0 .. +1.0 in 32 steps),
// with the spare 20 bits ignored for now (reserved for sign or scale).
// Diffusion: 48 bits / 6 entries = 8 bits each in [0, 1].
__device__ inline void decode_coefficients(const uint32_t* genome_bits,
                                           Coefficients* out) {
    auto read_bits = [&](int start, int n) -> uint32_t {
        // Up to 8 contiguous bits across word boundaries.
        uint32_t lo_word = genome_bits[start / 32];
        uint32_t hi_word = genome_bits[(start + n - 1) / 32];
        int shift = start % 32;
        uint64_t combined = static_cast<uint64_t>(lo_word)
                          | (static_cast<uint64_t>(hi_word) << 32);
        uint32_t mask = (n == 32) ? 0xFFFFFFFFu : ((1u << n) - 1u);
        return static_cast<uint32_t>(combined >> shift) & mask;
    };
    // Reaction: 5 bits per entry, 36 entries = 180 bits (uses 180 of 200).
    for (int i = 0; i < 6 * 6; ++i) {
        uint32_t q = read_bits(GENOME_BIT_REACTION_LO + i * 5, 5);
        float v = (static_cast<float>(q) - 15.5f) / 15.5f;   // [-1, 1)
        out->reaction[i] = v;
    }
    // Diffusion: 8 bits per channel.
    for (int c = 0; c < 6; ++c) {
        uint32_t q = read_bits(GENOME_BIT_DIFFUSION_LO + c * 8, 8);
        out->diffusion[c] = static_cast<float>(q) / 255.0f;
    }
}

}  // namespace slime::nca::rd

#endif  // SLIME_2_1_NCA_REACTION_DIFFUSION_CU
