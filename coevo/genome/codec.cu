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

__host__ inline Role read_role(const Genome& g) {
    return static_cast<Role>(g.bits[0] & 0x3u);
}

__host__ inline void write_role(Genome& g, Role role) {
    uint32_t r = static_cast<uint32_t>(role) & 0x3u;
    g.bits[0] = (g.bits[0] & ~0x3u) | r;
}

__host__ inline uint32_t read_seed(const Genome& g) {
    // Bits 2..33 span words[0] (bits 2..31) and words[1] (bits 0..1).
    uint32_t lo = (g.bits[0] >> 2) & 0x3FFFFFFFu;          // 30 bits
    uint32_t hi = (g.bits[1] & 0x3u) << 30;                // 2 bits
    return lo | hi;
}


// Mutation. Each spawn uses the replica-assigned mutation rate from S-004 for
// non-role bits, and the fixed 1e-4 for role bits. Cross-role lineage drift
// is permitted as exploration. Bit-level Bernoulli flips per the two rates;
// role bits (0, 1) use role_rate, all others use non_role_rate.
__host__ inline void mutate(Genome* g,
                            float non_role_rate,
                            float role_rate,
                            Pcg32* rng) {
    // Role bits live in word 0, positions 0 and 1.
    for (int b = 0; b < 2; ++b) {
        if (pcg32_float(rng) < role_rate) {
            g->bits[0] ^= (1u << b);
        }
    }
    // Remaining 1022 bits: word 0 bits 2..31, then words 1..31.
    for (int b = 2; b < 32; ++b) {
        if (pcg32_float(rng) < non_role_rate) {
            g->bits[0] ^= (1u << b);
        }
    }
    for (int w = 1; w < GENOME_WORDS; ++w) {
        for (int b = 0; b < 32; ++b) {
            if (pcg32_float(rng) < non_role_rate) {
                g->bits[w] ^= (1u << b);
            }
        }
    }
}

// Single-point crossover, role-blind. Cut point chosen uniformly in
// [1, GENOME_BITS - 1] so each parent contributes a nonempty span.
__host__ inline void crossover(const Genome& a,
                               const Genome& b,
                               Genome* out,
                               Pcg32* rng) {
    uint32_t cut = 1u + (pcg32_random(rng) % (GENOME_BITS - 1));
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

// Total concatenated weight space: W_perc + W_inter + W_flow + W_bmap.
// apply_delta and init_delta_from_prior address this as a flat index.
constexpr int TOTAL_WEIGHT_SLOTS = W_PERC_SIZE
    + (N_PERC_FILTERS * CA_CHANNELS * 32)   // PERC_DIM * HIDDEN_DIM
    + (32 * CA_CHANNELS)                      // HIDDEN_DIM * CA_CHANNELS
    + (CA_CHANNELS * BMAP_DIM);               // CA_CHANNELS * BMAP_DIM

// Offsets into the concatenated weight space.
constexpr int DELTA_OFF_PERC  = 0;
constexpr int DELTA_OFF_INTER = W_PERC_SIZE;
constexpr int DELTA_OFF_FLOW  = DELTA_OFF_INTER + (N_PERC_FILTERS * CA_CHANNELS * 32);
constexpr int DELTA_OFF_BMAP  = DELTA_OFF_FLOW + (32 * CA_CHANNELS);

// Scatter the sparse perturbation onto the learned weight buffers. The four
// weight banks are addressed as one concatenated index space. O(count) scatter,
// run at decode time before the forward pass.
__host__ __device__ inline void apply_delta(const DeltaWeights& delta,
                                   float* W_perc,
                                   float* W_inter,
                                   float* W_flow,
                                   float* W_bmap) {
    for (int k = 0; k < delta.count; ++k) {
        uint32_t idx = delta.indices[k] % TOTAL_WEIGHT_SLOTS;
        float val = delta.values[k];
        if (idx < static_cast<uint32_t>(DELTA_OFF_INTER)) {
            W_perc[idx - DELTA_OFF_PERC] += val;
        } else if (idx < static_cast<uint32_t>(DELTA_OFF_FLOW)) {
            W_inter[idx - DELTA_OFF_INTER] += val;
        } else if (idx < static_cast<uint32_t>(DELTA_OFF_BMAP)) {
            W_flow[idx - DELTA_OFF_FLOW] += val;
        } else {
            W_bmap[idx - DELTA_OFF_BMAP] += val;
        }
    }
}

// Seed a fresh organism's delta from genome bits 282..1023. The prior bits
// (742 bits) are interpreted as a low-rank generator:
//   - First 10 bits: count (clamped to [0, MAX_DELTA_FLOATS])
//   - Remaining bits seed index/value pairs via a local PCG32.
// Indices are spread across the concatenated weight space; values are small
// perturbations whose magnitude is controlled by the prior scale.
// Determinism: local Pcg32 is seeded from read_seed(g), so the same genome
// always yields the same initial delta.
__host__ inline void init_delta_from_prior(const Genome& g,
                                           DeltaWeights* delta) {
    // Extract a count from the first 10 bits of the delta prior region.
    // bits 282..291 = 10 bits -> up to 1024 entries.
    auto read_bits = [&](int start, int n) -> uint32_t {
        uint32_t lo_word = g.bits[start / 32];
        uint32_t hi_word = g.bits[(start + n - 1) / 32];
        int shift = start % 32;
        uint64_t combined = static_cast<uint64_t>(lo_word)
                          | (static_cast<uint64_t>(hi_word) << 32);
        uint32_t mask = (n == 32) ? 0xFFFFFFFFu : ((1u << n) - 1u);
        return static_cast<uint32_t>(combined >> shift) & mask;
    };
    uint32_t raw_count = read_bits(GENOME_BIT_DELTA_PRIOR_LO, 10);
    int count = static_cast<int>(raw_count);
    if (count > MAX_DELTA_FLOATS) count = MAX_DELTA_FLOATS;
    delta->count = count;

    // Seed a local PCG32 from the genome's weight init seed.
    Pcg32 local_rng;
    pcg32_seed(&local_rng, static_cast<uint64_t>(read_seed(g)), 0x5A17E001u);

    // Generate (index, value) pairs. Index is uniform in [0, TOTAL_WEIGHT_SLOTS).
    // Value is a small perturbation scaled by 0.01 (so initial perturbations
    // are ~1% of base weight scale). Uses uniform approximation to N(0,1):
    // (u - 0.5) * sqrt(12).
    constexpr float PRIOR_SCALE = 0.01f;
    for (int k = 0; k < count; ++k) {
        delta->indices[k] = pcg32_random(&local_rng) % TOTAL_WEIGHT_SLOTS;
        float u = pcg32_float(&local_rng);
        float approx_normal = (u - 0.5f) * 3.4641016f;  // sqrt(12)
        delta->values[k] = PRIOR_SCALE * approx_normal;
    }
}

}  // namespace slime::genome

#endif  // COEVO_GENOME_CODEC_CU
