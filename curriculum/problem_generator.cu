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
#include "../config/strong_ids.cuh"

#include <cstddef>   // offsetof, size_t
#include <cstdint>
#include <cuda_runtime.h>

namespace slime::curriculum {

constexpr int CLASSIFIER_BATCH = CLASSIFIER_BATCH_SIZE;
constexpr int SOT_SUBBATCH     = SOT_SUBBATCH_SIZE;
constexpr int PREDICTOR_BATCH  = PREDICTOR_EVAL_K;  // 8
// Upper bound on SOT reference roll-outs needed when every pool organism
// assigned to an SOT sample gets its own reference forward (per-organism
// effective weights): at most (pool / batch) organisms per SOT image.
constexpr int SOT_MAX_REFS = SOT_SUBBATCH * (POOL_SIZE / CLASSIFIER_BATCH);  // 16

struct ClassifierBatch {
    __half image[CLASSIFIER_BATCH * GRID_SIZE * GRID_SIZE * 3];
    int    label[CLASSIFIER_BATCH];
    bool   is_sot[CLASSIFIER_BATCH];    // first SOT_SUBBATCH entries
    float  task_embedding[TASK_EMBED_DIM];
    float  difficulty;
    float  tau;
};

struct PredictorBatch {
    // Pool-slot identity of each target (-1 for the stationary probe slots,
    // which need not correspond to a live pool organism); the lineage id is
    // carried separately for provenance and archive linkage.
    int      target_pool_slot[PREDICTOR_BATCH];
    LineageId target_lineage_id[PREDICTOR_BATCH];
    float    target_bmap_32[PREDICTOR_BATCH * BMAP_DIM];
    float    target_bmap_64[PREDICTOR_BATCH * BMAP_DIM];  // ground truth
    bool     target_was_sot[PREDICTOR_BATCH];
    float    task_embedding[TASK_EMBED_DIM];
};

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
    const int ROUNDS = FEISTEL_ROUNDS;
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

// Assemble a classifier batch. Draws CLASSIFIER_BATCH (16) samples, fills
// labels and task embedding, marks SOT entries and applies pixel permutation.
// Images are procedurally generated patterns (deterministic per label) that
// give classifiers distinct spatial structure to differentiate. The rng_state
// is advanced so successive calls produce different batches.
inline void assemble_classifier_batch(ClassifierBatch* out,
                                      float sot_density,
                                      uint64_t host_sot_key,
                                      Pcg32* rng) {
    // Task embedding: deterministic from PCG32.
    for (int d = 0; d < TASK_EMBED_DIM; ++d) {
        out->task_embedding[d] = pcg32_float(rng) - 0.5f;
    }
    out->difficulty = 1.0f;
    out->tau = 0.5f;

    // Number of SOT entries.
    int n_sot = static_cast<int>(sot_density * CLASSIFIER_BATCH + 0.5f);
    if (sot_density > 0.f && n_sot < 1) n_sot = 1;
    if (n_sot > SOT_SUBBATCH) n_sot = SOT_SUBBATCH;

    // Scratch buffer for SOT permutation (24KB on stack).
    __half sot_scratch[GRID_SIZE * GRID_SIZE * 3];

    for (int s = 0; s < CLASSIFIER_BATCH; ++s) {
        // Label: cycle through 0..NUM_CLASSES-1.
        out->label[s] = static_cast<int>(pcg32_random(rng) % NUM_CLASSES);
        int lab = out->label[s];

        // Generate a distinct spatial pattern per label: sinusoidal bars whose
        // frequency and phase depend on the label. Each class has a visually
        // distinct structure the NCA can learn to decode.
        __half* img = &out->image[s * GRID_SIZE * GRID_SIZE * 3];
        float freq = 1.0f + static_cast<float>(lab);
        float phase = static_cast<float>(lab) * 0.3f;
        for (int y = 0; y < GRID_SIZE; ++y) {
            for (int xp = 0; xp < GRID_SIZE; ++xp) {
                int idx = (y * GRID_SIZE + xp) * 3;
                float vy = sinf(freq * 6.2831853f * static_cast<float>(y) / GRID_SIZE + phase);
                float vx = cosf(freq * 6.2831853f * static_cast<float>(xp) / GRID_SIZE + phase);
                img[idx + 0] = __float2half(0.5f + 0.4f * vy);
                img[idx + 1] = __float2half(0.5f + 0.4f * vx);
                img[idx + 2] = __float2half(0.5f + 0.2f * vy * vx);
            }
        }

        // Mark and permute SOT entries.
        out->is_sot[s] = (s < n_sot);
        if (out->is_sot[s]) {
            apply_sot_permutation(img, host_sot_key, false, sot_scratch);
        }
    }
}

// Predictor batch assembly is implemented at Stage 6 when role machinery is
// wired. The PredictorBatch struct above is retained as it is referenced by
// the integration layer's type declarations.

// ---- Probe set ----------------------------------------------------------
// Signed at run start. Used by both populations. The signature is host-held
// and verified before each use (S-002 alignment). Predictor probes are a
// fixed pool of archived classifiers signed at run start (stationary
// reference).
struct ProbeSet {
    ClassifierBatch classifier_probes[4];        // 64-batch total (4*16)
    LineageId       predictor_probe_targets[PREDICTOR_BATCH];

    // Predictor probe references (A-601/A-701): a fixed pool of archived
    // classifiers signed at bootstrap. Their bmap_32 and ground-truth
    // bmap_64 are frozen at signing time, so predictor quality is measured
    // against a stationary reference. Signed=false until bootstrap fires.
    // These fields precede `signature` so the keyed checksum covers them.
    bool  predictor_probes_signed;
    float predictor_probe_bmap32[PREDICTOR_BATCH * BMAP_DIM];
    float predictor_probe_bmap64[PREDICTOR_BATCH * BMAP_DIM];

    // Held-out probe tuples for the reference regressor (A-601): signed
    // classifier (bmap_64, task_embedding, fitness) tuples snapshotted from
    // the replay buffer at bootstrap, never trained on afterwards. This is
    // the reference's ground-truth held-out signal.
    bool  probe_tuples_signed;
    float probe_bmap[PROBE_BATCH * BMAP_DIM];
    float probe_task_emb[PROBE_BATCH * TASK_EMBED_DIM];
    float probe_fitness[PROBE_BATCH];

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

// Initialize the probe set: generate 4 classifier probe batches (64 total
// samples) and sign with host_sot_key. Per A-601/Q-001: signed at run start,
// host-held, applies to both populations. The predictor probe references are
// signed later, at the bootstrap crossing (see sign_predictor_probes).
inline void init_probe_set(ProbeSet* ps, uint64_t host_sot_key, Pcg32* rng) {
    for (int b = 0; b < 4; ++b) {
        assemble_classifier_batch(&ps->classifier_probes[b],
                                  0.f,  // no SOT in probes
                                  host_sot_key, rng);
    }
    for (int i = 0; i < PREDICTOR_BATCH; ++i) {
        ps->predictor_probe_targets[i] = LineageId();
    }
    ps->predictor_probes_signed = false;
    std::memset(ps->predictor_probe_bmap32, 0, sizeof(ps->predictor_probe_bmap32));
    std::memset(ps->predictor_probe_bmap64, 0, sizeof(ps->predictor_probe_bmap64));
    ps->probe_tuples_signed = false;
    std::memset(ps->probe_bmap, 0, sizeof(ps->probe_bmap));
    std::memset(ps->probe_task_emb, 0, sizeof(ps->probe_task_emb));
    std::memset(ps->probe_fitness, 0, sizeof(ps->probe_fitness));
    ps->signature = probe_set_signature(*ps, host_sot_key);
}

// Sign the held-out probe tuples at bootstrap: snapshot PROBE_BATCH real
// (bmap_64, task_embedding, fitness) tuples from the replay buffer and
// re-sign the probe set. After this the tuples never change, so the
// reference's held-out error is a stationary signal.
inline void sign_probe_tuples(ProbeSet* ps,
                              const float* bmap_rows,      // [PROBE_BATCH][BMAP_DIM]
                              const float* task_rows,      // [PROBE_BATCH][TASK_EMBED_DIM]
                              const float* fitness_rows,   // [PROBE_BATCH]
                              uint64_t host_sot_key) {
    std::memcpy(ps->probe_bmap, bmap_rows,
                PROBE_BATCH * BMAP_DIM * sizeof(float));
    std::memcpy(ps->probe_task_emb, task_rows,
                PROBE_BATCH * TASK_EMBED_DIM * sizeof(float));
    std::memcpy(ps->probe_fitness, fitness_rows,
                PROBE_BATCH * sizeof(float));
    ps->probe_tuples_signed = true;
    ps->signature = probe_set_signature(*ps, host_sot_key);
}

// Sign the predictor probe references at the bootstrap crossing: freeze the
// bmap_32 / bmap_64 of the chosen archived classifiers (their trajectories
// come from the Intent Registry) and re-sign the probe set. After this the
// references never change: predictor quality is measured against a
// stationary evaluation pool.
inline void sign_predictor_probes(ProbeSet* ps,
                                           const LineageId* target_ids,
                                  const float* bmap32_rows,   // [K][BMAP_DIM]
                                  const float* bmap64_rows,   // [K][BMAP_DIM]
                                  uint64_t host_sot_key) {
    for (int i = 0; i < PREDICTOR_BATCH; ++i) {
        ps->predictor_probe_targets[i] = target_ids[i];
        std::memcpy(&ps->predictor_probe_bmap32[i * BMAP_DIM],
                    &bmap32_rows[i * BMAP_DIM], BMAP_DIM * sizeof(float));
        std::memcpy(&ps->predictor_probe_bmap64[i * BMAP_DIM],
                    &bmap64_rows[i * BMAP_DIM], BMAP_DIM * sizeof(float));
    }
    ps->predictor_probes_signed = true;
    ps->signature = probe_set_signature(*ps, host_sot_key);
}

// Assemble a predictor batch (A-701). Slots 0..PREDICTOR_PROBE_SLOTS-1 carry
// the signed stationary probe references; the remaining slots carry pool
// targets sampled weighted by the ensemble prediction error tracked per
// organism (error_ema), so the predictor curriculum targets weak spots.
// task_embedding is the current classifier batch embedding.
constexpr int PREDICTOR_PROBE_SLOTS = PREDICTOR_PROBE_SLOT_COUNT;
constexpr int PREDICTOR_POOL_SLOTS = PREDICTOR_POOL_SLOT_COUNT;

// A predictor is scored against one target per generation; the slot rotates
// with the generation so that over PREDICTOR_BATCH generations it covers
// every target, and its fitness uses the per-predictor loss EMA.
__host__ __device__ inline int predictor_target_slot(int org, int gen) {
    return (org + gen) % PREDICTOR_BATCH;
}

inline void assemble_predictor_batch(PredictorBatch* out,
                                     const ProbeSet& probes,
                                     const LineageId* pool_lineage_ids,
                                     const float* error_ema,     // [POOL_SIZE]
                                     const float* bmap32_rows,   // [POOL_SIZE][BMAP_DIM]
                                     const float* bmap64_rows,   // [POOL_SIZE][BMAP_DIM]
                                     const float* task_embedding,
                                     const Role* pool_roles,     // [POOL_SIZE]
                                     const bool* pool_was_sot,   // [POOL_SIZE]
                                     Pcg32* rng) {
    std::memset(out, 0, sizeof(*out));

    // Stationary probe slots.
    int slot = 0;
    if (probes.predictor_probes_signed) {
        for (; slot < PREDICTOR_PROBE_SLOTS && slot < PREDICTOR_BATCH; ++slot) {
            out->target_pool_slot[slot] = -1;
            out->target_lineage_id[slot] =
                LineageId(probes.predictor_probe_targets[slot]);
            std::memcpy(&out->target_bmap_32[slot * BMAP_DIM],
                        &probes.predictor_probe_bmap32[slot * BMAP_DIM],
                        BMAP_DIM * sizeof(float));
            std::memcpy(&out->target_bmap_64[slot * BMAP_DIM],
                        &probes.predictor_probe_bmap64[slot * BMAP_DIM],
                        BMAP_DIM * sizeof(float));
            out->target_was_sot[slot] = false;
        }
    }

    // Pool slots weighted by the error EMA (roulette selection with a small
    // floor so every organism stays reachable). Only classifier organisms
    // are eligible targets: a predictor models classifier behavior.
    for (; slot < PREDICTOR_BATCH; ++slot) {
        float total = 0.f;
        for (int i = 0; i < POOL_SIZE; ++i) {
            if (canonical_role(pool_roles[i]) != Role::Classifier) continue;
            total += error_ema[i] + PREDICTOR_CURRICULUM_ERROR_FLOOR;
        }
        int chosen = -1;
        if (total > 0.f) {
            float r = pcg32_float(rng) * total;
            float acc = 0.f;
            for (int i = 0; i < POOL_SIZE; ++i) {
                if (canonical_role(pool_roles[i]) != Role::Classifier) continue;
                acc += error_ema[i] + PREDICTOR_CURRICULUM_ERROR_FLOOR;
                if (r <= acc) { chosen = i; break; }
            }
        }
        if (chosen < 0) {
            int count = 0;
            for (int i = 0; i < POOL_SIZE; ++i) {
                if (canonical_role(pool_roles[i]) != Role::Classifier) continue;
                count++;
                if (pcg32_random(rng) % static_cast<uint32_t>(count) == 0u) {
                    chosen = i;
                }
            }
        }
        if (chosen < 0) continue;  // no classifier targets available
        out->target_pool_slot[slot] = chosen;
        out->target_lineage_id[slot] = pool_lineage_ids[chosen];
        std::memcpy(&out->target_bmap_32[slot * BMAP_DIM],
                    &bmap32_rows[chosen * BMAP_DIM], BMAP_DIM * sizeof(float));
        std::memcpy(&out->target_bmap_64[slot * BMAP_DIM],
                    &bmap64_rows[chosen * BMAP_DIM], BMAP_DIM * sizeof(float));
        out->target_was_sot[slot] = pool_was_sot[chosen];
    }

    for (int d = 0; d < TASK_EMBED_DIM; ++d) {
        out->task_embedding[d] = task_embedding[d];
    }
}

// Escalation logic triggers on blended surprise (S-001 CUSUM). It is a
// host-side policy and is not implemented in this file.

}  // namespace slime::curriculum

#endif  // COEVO_CURRICULUM_PROBLEM_GENERATOR_CU


