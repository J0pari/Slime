// Sheet A-202: Reaction-Diffusion Field
//
// Operates on chemical channels 0-5 of the CA grid. Reaction coefficients
// (pairwise interactions among the six chemical channels) and per-channel
// diffusion rates are decoded from the genome (A-301 bits 34-281).
//
// Integration: ca_step writes all 16 channels each step, so the CA is the
// source that drives the chemical channels 0-5 (they are part of the cell
// state). forward_kernel then calls rd_step, which ADDS spatial diffusion +
// decay to those channels on top of the cellwise update, when given a non-null
// per-organism Coefficients array. Passing null skips the diffusion/decay step
// but the chemicals still evolve cellwise via the CA. The decoded Coefficients
// are produced by decode_coefficients at decode time and passed to
// launch_forward. UNVERIFIED: this file has not been compiled with nvcc.

#ifndef COEVO_NCA_REACTION_DIFFUSION_CU
#define COEVO_NCA_REACTION_DIFFUSION_CU

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
// Reads and writes only the chemical channels: the reaction term couples the
// six chemical channels among themselves (A-202), and diffusion is per-channel.
// The CA reads the chemical field as context (sample_neighborhood reads all 16
// channels), but RD does not read the CA channels 6..15.
//
// Additive: called AFTER ca_step has written the cellwise update into `scratch`
// (next). rd_step reads the old chemical field from `grid` (curr) for the
// Laplacian, reaction, and decay, and ADDS the reaction-diffusion contribution
// on top of the cellwise value already in `scratch`. Reading curr (not next)
// for the stencil is the standard explicit operator split and is race-free
// (each thread writes only its own cell's scratch).
//
// dt = 0.1, decay = 0.05. Diffusion uses a 5-point Laplacian on the toroidal
// grid; dt*diffusion <= 0.1 < 0.25 (CFL) keeps diffusion stable and the decay
// keeps a continuously-sourced field bounded. The linear reaction term can
// still drive a channel toward saturation under adversarial genome
// coefficients; the FP16 clamp bounds it. Reaction is sum_j k_{ij} * A_j over
// the 6x6 genome-decoded coefficient matrix.
__device__ inline void rd_step(__half* grid,
                               __half* scratch,
                               const Coefficients& coeffs) {
    constexpr int CHEM_N = 6;
    constexpr float DT = 0.1f;
    constexpr float DECAY = 0.05f;
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
            float here[CHEM_N];   // old chemical field (curr)
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
                float base = __half2float(scratch[idx(y, x, c)]);  // cellwise update
                float updated = base
                    + DT * (coeffs.diffusion[c] * lap + react - DECAY * here[c]);
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

#endif  // COEVO_NCA_REACTION_DIFFUSION_CU
