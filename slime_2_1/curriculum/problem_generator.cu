// Sheet A-701: Problem Generator & Dual Curriculum
//
// Classifier tasks: batches of 16 samples with augmentations, difficulty
// scalar, feature vector, threshold tau. SOT sub-batch of 4 images with a
// reversible pixel-permutation transform under a host-controlled key.
//
// Predictor tasks: K=8 target classifier organisms sampled from the active
// pool, weighted by current ensemble prediction error on their
// bmap_32 -> bmap_64 mapping (predictor curriculum targets predictor weak
// spots, not classifier weak spots).
//
// Probe injection: signed and host-held, applies to both populations.
// Predictor probes are drawn from a fixed pool of archived classifiers signed
// at run start (stationary evaluation reference for predictor quality).

#ifndef COEVO_CURRICULUM_PROBLEM_GENERATOR_CU
#define COEVO_CURRICULUM_PROBLEM_GENERATOR_CU

#include "../config/constants.cuh"

#include <cstddef>   // offsetof, size_t
#include <cstdint>
#include <cuda_runtime.h>

namespace slime::curriculum {

constexpr int CLASSIFIER_BATCH = 16;
constexpr int SOT_SUBBATCH     = 4;
constexpr int PREDICTOR_BATCH  = PREDICTOR_EVAL_K;  // 8

struct ClassifierBatch {
    __half image[CLASSIFIER_BATCH * GRID_SIZE * GRID_SIZE * 3];
    int    label[CLASSIFIER_BATCH];
    bool   is_sot[CLASSIFIER_BATCH];    // first SOT_SUBBATCH entries
    float  task_embedding[TASK_EMBED_DIM];
    float  difficulty;
    float  tau;
};

struct PredictorBatch {
    uint32_t target_organism_id[PREDICTOR_BATCH];
    float    target_bmap_32[PREDICTOR_BATCH * BMAP_DIM];
    float    target_bmap_64[PREDICTOR_BATCH * BMAP_DIM];  // ground truth
    bool     target_was_sot[PREDICTOR_BATCH];
    float    task_embedding[TASK_EMBED_DIM];
};

// ---- Classifier curriculum ----------------------------------------------
// DECLARED ONLY — blueprint-in-place.
// assemble_classifier_batch: draw CLASSIFIER_BATCH (16) samples from the active
// dataset, scale each to 64x64, fill labels and the 16-d task embedding, set
// the difficulty scalar and threshold tau from the current curriculum state,
// and mark the first round(sot_density * batch) entries (at least 1 when
// sot_density > 0, capped at SOT_SUBBATCH) as SOT, applying
// apply_sot_permutation to those images under host_sot_key. Difficulty bias
// targets weak archive niches. Cadence: CURRICULUM_INTERVAL.
void assemble_classifier_batch(ClassifierBatch* out,
                               float sot_density,
                               uint64_t host_sot_key,
                               cudaStream_t stream);

// SOT pixel permutation: a reversible index permutation of the 64x64 pixel
// grid (channels move together) under a host-held key. Implemented as a
// balanced Feistel network over the 12-bit pixel index (64*64 = 4096 = 2^12),
// which is a bijection on [0, 4096) for any round-key schedule, hence exactly
// invertible — the host can undo it to score identity reconstruction (S-002).
// `invert` selects forward vs inverse (reverse the round order).
__host__ __device__ inline uint32_t sot_feistel(uint32_t idx,
                                                uint64_t key,
                                                bool invert) {
    // 12-bit index split into two 6-bit halves.
    uint32_t l = (idx >> 6) & 0x3Fu;
    uint32_t r = idx & 0x3Fu;
    const int ROUNDS = 4;
    for (int round = 0; round < ROUNDS; ++round) {
        int ri = invert ? (ROUNDS - 1 - round) : round;
        uint32_t rk = static_cast<uint32_t>((key >> (8 * ri)) & 0xFFu);
        uint32_t nl, nr;
        if (!invert) {
            // Forward round: f reads r. (l, r) -> (r, l ^ f(r)).
            uint32_t f = ((r * 73u) + rk * 0x9Eu + ri * 0x2Fu) & 0x3Fu;
            nl = r;
            nr = l ^ f;
        } else {
            // Inverse round: f reads l. (l, r) -> (r ^ f(l), l).
            uint32_t f = ((l * 73u) + rk * 0x9Eu + ri * 0x2Fu) & 0x3Fu;
            nl = r ^ f;
            nr = l;
        }
        l = nl; r = nr;
    }
    return ((l & 0x3Fu) << 6) | (r & 0x3Fu);
}

// Permute pixels of a 64x64x3 image in place under host_sot_key. Reversible:
// applying with the same key and the inverse flag restores the original. The
// three channels of a pixel move together (the permutation is on pixel index).
__host__ __device__ inline void apply_sot_permutation(__half* image_64x64x3,
                                                      uint64_t host_sot_key,
                                                      bool invert,
                                                      __half* scratch_64x64x3) {
    for (int idx = 0; idx < GRID_SIZE * GRID_SIZE; ++idx) {
        uint32_t dst = sot_feistel(static_cast<uint32_t>(idx), host_sot_key, invert);
        scratch_64x64x3[dst * 3 + 0] = image_64x64x3[idx * 3 + 0];
        scratch_64x64x3[dst * 3 + 1] = image_64x64x3[idx * 3 + 1];
        scratch_64x64x3[dst * 3 + 2] = image_64x64x3[idx * 3 + 2];
    }
    for (int i = 0; i < GRID_SIZE * GRID_SIZE * 3; ++i) image_64x64x3[i] = scratch_64x64x3[i];
}

// ---- Predictor curriculum -----------------------------------------------
// DECLARED ONLY — blueprint-in-place.
// assemble_predictor_batch: sample PREDICTOR_BATCH (8) active classifiers
// weighted by per_organism_ensemble_error (softmax or roulette over the error
// vector) so predictors are pushed toward their own weak spots. For each
// chosen target, copy its bmap_32 (input) and bmap_64 (ground truth) from the
// Intent Registry into `out`, record its id and SOT flag, and copy the shared
// task embedding. Sampling without replacement within a batch.
void assemble_predictor_batch(PredictorBatch* out,
                              const float* per_organism_ensemble_error,
                              cudaStream_t stream);

// ---- Probe set ----------------------------------------------------------
// Signed at run start. Used by both populations. The signature is host-held
// and verified before each use (S-002 alignment). Predictor probes are a
// fixed pool of archived classifiers signed at run start (stationary
// reference).
struct ProbeSet {
    ClassifierBatch classifier_probes[4];        // 64-batch total (4*16)
    uint32_t        predictor_probe_targets[PREDICTOR_BATCH];
    uint64_t        signature;                   // host-verified
};

// Keyed signature over the probe-set bytes (everything except the signature
// field itself), so a tampered probe set is rejected before use. FNV-1a-64
// seeded with the host key and finalized with a key-dependent mix — a keyed
// checksum, not a cryptographic MAC, but enough to catch accidental drift and
// casual tampering of the host-held probe set. For adversarial integrity,
// swap in a real MAC (e.g. SipHash) behind this same interface.
__host__ __device__ inline uint64_t probe_set_signature(const ProbeSet& set,
                                                        uint64_t host_key) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&set);
    // Hash everything up to the trailing `signature` field.
    size_t len = offsetof(ProbeSet, signature);
    uint64_t h = 1469598103934665603ull ^ host_key;
    for (size_t i = 0; i < len; ++i) {
        h ^= bytes[i];
        h *= 1099511628211ull;
    }
    // Key-dependent finalization mix.
    h ^= host_key + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    return h;
}

// Recompute the signature and compare against the stored one. Returns true iff
// the probe set is intact under host_key.
__host__ __device__ inline bool verify_probe_set(const ProbeSet& set,
                                                 uint64_t host_key) {
    return probe_set_signature(set, host_key) == set.signature;
}

// Escalation logic triggers on blended surprise (S-001 CUSUM). It is a
// host-side policy and is not implemented in this file.

}  // namespace slime::curriculum

#endif  // COEVO_CURRICULUM_PROBLEM_GENERATOR_CU
