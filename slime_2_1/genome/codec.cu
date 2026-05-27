// Sheet A-301: Genome & Delta-Weight Codec (Role-Tagged)
//
// 1024-bit genome:
//   bits 0..1     role tag           (00 classifier, 01 predictor, 10/11 reserved)
//   bits 2..33    weight init seed   (32 bits)
//   bits 34..233  reaction coeffs    (200 bits quantized)
//   bits 234..281 diffusion rates    (48 bits quantized)
//   bits 282..1023 low-rank delta init prior
//
// Role mutation rate is 1e-4 per role-bit per spawn (vs 1e-2 baseline).
// Delta encoding, MAX_DELTA_FLOATS, and per-organism memory layout carry
// forward from 2.0.

#ifndef SLIME_2_1_GENOME_CODEC_CU
#define SLIME_2_1_GENOME_CODEC_CU

#include "../config/constants.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace slime::genome {

constexpr int GENOME_WORDS = GENOME_BITS / 32;  // 32 uint32_t words

struct Genome {
    uint32_t bits[GENOME_WORDS];
};

__host__ __device__ inline Role read_role(const Genome& g) {
    return static_cast<Role>(g.bits[0] & 0x3u);
}

__host__ __device__ inline void write_role(Genome& g, Role role) {
    uint32_t r = static_cast<uint32_t>(role) & 0x3u;
    g.bits[0] = (g.bits[0] & ~0x3u) | r;
}

__host__ __device__ inline uint32_t read_seed(const Genome& g) {
    // Bits 2..33 span words[0] (bits 2..31) and words[1] (bits 0..1).
    uint32_t lo = (g.bits[0] >> 2) & 0x3FFFFFFFu;          // 30 bits
    uint32_t hi = (g.bits[1] & 0x3u) << 30;                // 2 bits
    return lo | hi;
}

// Mutation. Each spawn uses the replica-assigned mutation rate from S-004 for
// non-role bits, and the fixed 1e-4 for role bits. Cross-role lineage drift
// is permitted as exploration.
__device__ void mutate(Genome* g,
                       float non_role_rate,
                       float role_rate,
                       uint32_t* rng_state);

// Single-point crossover, role-blind.
__device__ void crossover(const Genome& a,
                          const Genome& b,
                          Genome* out,
                          uint32_t* rng_state);

// Delta-weight codec (sparse weight updates layered on the base init).
// Layout unchanged from 2.0. Each organism holds up to MAX_DELTA_FLOATS
// (index, value) pairs encoding a low-rank perturbation initialised from
// bits 282..1023 of the genome.
constexpr int MAX_DELTA_FLOATS = 4096;  // from 2.0

struct DeltaWeights {
    uint32_t indices[MAX_DELTA_FLOATS];
    float    values[MAX_DELTA_FLOATS];
    int      count;
};

__device__ void apply_delta(const DeltaWeights& delta,
                            float* W_perc,
                            float* W_inter,
                            float* W_flow,
                            float* W_bmap);

// Initialise delta indices/values from the genome prior bits (282..1023).
__device__ void init_delta_from_prior(const Genome& g,
                                      DeltaWeights* delta,
                                      uint32_t* rng_state);

}  // namespace slime::genome

#endif  // SLIME_2_1_GENOME_CODEC_CU
