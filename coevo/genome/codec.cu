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
// Delta encoding stores sparse (index, value) weight perturbations layered on
// the base initialization, up to MAX_DELTA_FLOATS pairs per organism.

#ifndef COEVO_GENOME_CODEC_CU
#define COEVO_GENOME_CODEC_CU

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

// xorshift32 PRNG. Cheap, register-resident, suitable for per-thread state.
// The all-zero state is a fixed point of xorshift, so a zero seed (a valid
// genome seed value) would otherwise emit a constant stream — coerce it to a
// nonzero constant on entry.
__host__ __device__ inline uint32_t xorshift32(uint32_t* s) {
    uint32_t x = *s;
    if (x == 0u) x = 0x9E3779B9u;   // golden-ratio nonzero fallback
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

__host__ __device__ inline float rand_uniform(uint32_t* s) {
    return static_cast<float>(xorshift32(s)) * (1.0f / 4294967296.0f);
}

// Mutation. Each spawn uses the replica-assigned mutation rate from S-004 for
// non-role bits, and the fixed 1e-4 for role bits. Cross-role lineage drift
// is permitted as exploration. Bit-level Bernoulli flips per the two rates;
// role bits (0, 1) use role_rate, all others use non_role_rate.
__host__ __device__ inline void mutate(Genome* g,
                                       float non_role_rate,
                                       float role_rate,
                                       uint32_t* rng_state) {
    // Role bits live in word 0, positions 0 and 1.
    for (int b = 0; b < 2; ++b) {
        if (rand_uniform(rng_state) < role_rate) {
            g->bits[0] ^= (1u << b);
        }
    }
    // Remaining 1022 bits: word 0 bits 2..31, then words 1..31.
    for (int b = 2; b < 32; ++b) {
        if (rand_uniform(rng_state) < non_role_rate) {
            g->bits[0] ^= (1u << b);
        }
    }
    for (int w = 1; w < GENOME_WORDS; ++w) {
        for (int b = 0; b < 32; ++b) {
            if (rand_uniform(rng_state) < non_role_rate) {
                g->bits[w] ^= (1u << b);
            }
        }
    }
}

// Single-point crossover, role-blind. Cut point chosen uniformly in
// [1, GENOME_BITS - 1] so each parent contributes a nonempty span.
__host__ __device__ inline void crossover(const Genome& a,
                                          const Genome& b,
                                          Genome* out,
                                          uint32_t* rng_state) {
    uint32_t cut = 1u + (xorshift32(rng_state) % (GENOME_BITS - 1));
    uint32_t cut_word = cut >> 5;
    uint32_t cut_bit  = cut & 31u;

    for (int w = 0; w < GENOME_WORDS; ++w) {
        if (w < static_cast<int>(cut_word)) {
            out->bits[w] = a.bits[w];
        } else if (w > static_cast<int>(cut_word)) {
            out->bits[w] = b.bits[w];
        } else {
            uint32_t mask_a = (cut_bit == 0) ? 0u : ((1u << cut_bit) - 1u);
            out->bits[w] = (a.bits[w] & mask_a) | (b.bits[w] & ~mask_a);
        }
    }
}

// Delta-weight codec (sparse weight updates layered on the base init).
// Each organism holds up to MAX_DELTA_FLOATS (index, value) pairs encoding a
// low-rank perturbation initialised from bits 282..1023 of the genome.
constexpr int MAX_DELTA_FLOATS = 4096;

struct DeltaWeights {
    uint32_t indices[MAX_DELTA_FLOATS];
    float    values[MAX_DELTA_FLOATS];
    int      count;
};

// DECLARED ONLY — blueprint-in-place.
// apply_delta: scatter the sparse perturbation onto the learned weight buffers.
// The four learned weight banks (W_perc, W_inter, W_flow, W_bmap) are addressed
// as one concatenated index space; each delta.indices[k] selects a global
// weight slot and delta.values[k] is added to the base-initialized weight there.
// O(count) scatter, run at decode time before the forward pass.
__device__ void apply_delta(const DeltaWeights& delta,
                            float* W_perc,
                            float* W_inter,
                            float* W_flow,
                            float* W_bmap);

// DECLARED ONLY — blueprint-in-place.
// init_delta_from_prior: seed a fresh organism's delta from genome bits
// 282..1023. Interpret the prior bits as a low-rank generator: draw `count`
// (index, value) pairs where indices are spread across the concatenated weight
// space and values are small Gaussian-ish perturbations whose scale/rank are
// set by the prior. Determinism comes from rng_state seeded with the genome
// seed (read_seed), so the same genome always yields the same initial delta.
__device__ void init_delta_from_prior(const Genome& g,
                                      DeltaWeights* delta,
                                      uint32_t* rng_state);

}  // namespace slime::genome

#endif  // COEVO_GENOME_CODEC_CU
