// Reaction-diffusion coefficient schema and genome decode (A-202, A-301).
//
// Shared by the RD kernels and the host witnesses so the encoding cannot
// drift. Reaction: 5 bits per entry, sign-magnitude (bit 4 sign, bits 0-3
// magnitude); zero bits decode to zero. Diffusion: 8 bits per channel in
// [0, 1]; zero bits are zero.

#ifndef COEVO_NCA_RD_CODEC_CUH
#define COEVO_NCA_RD_CODEC_CUH

#include "../config/constants.cuh"

namespace slime::nca::rd {

// Per-organism reaction/diffusion coefficients decoded from the genome.
struct Coefficients {
    float reaction[6 * 6];
    float diffusion[6];
};

__host__ __device__ inline void decode_coefficients(const uint32_t* genome_bits,
                                                    Coefficients* out) {
    auto read_bits = [&](int start, int n) -> uint32_t {
        // Up to 8 contiguous bits across word boundaries.
        uint32_t lo_word = genome_bits[start / 32];
        uint32_t hi_word = genome_bits[(start + n - 1) / 32];
        int shift = start % WORD_BITS;
        uint64_t combined = static_cast<uint64_t>(lo_word)
                          | (static_cast<uint64_t>(hi_word) << 32);
        uint32_t mask = (n == WORD_BITS) ? 0xFFFFFFFFu : ((1u << n) - 1u);
        return static_cast<uint32_t>(combined >> shift) & mask;
    };
    // Reaction: 5 bits per entry, 36 entries = 180 bits (uses 180 of 200).
    for (int i = 0; i < 6 * 6; ++i) {
        uint32_t q = read_bits(GENOME_BIT_REACTION_LO + i * 5, 5);
        float mag = static_cast<float>(q & 0xFu) / 15.0f;
        out->reaction[i] = (q & 0x10u) ? mag : -mag;
    }
    // Diffusion: 8 bits per channel.
    for (int c = 0; c < 6; ++c) {
        uint32_t q = read_bits(GENOME_BIT_DIFFUSION_LO + c * 8, 8);
        out->diffusion[c] = static_cast<float>(q) / 255.0f;
    }
}

}  // namespace slime::nca::rd

#endif  // COEVO_NCA_RD_CODEC_CUH
