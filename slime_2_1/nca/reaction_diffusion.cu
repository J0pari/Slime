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
__device__ void rd_step(__half* grid,
                        __half* scratch,
                        const Coefficients& coeffs);

// Decode quantized coefficient bits from the genome into floats.
__device__ void decode_coefficients(const uint32_t* genome_bits,
                                    Coefficients* out);

}  // namespace slime::nca::rd

#endif  // SLIME_2_1_NCA_REACTION_DIFFUSION_CU
