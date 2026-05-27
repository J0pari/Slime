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

#ifndef SLIME_2_1_CURRICULUM_PROBLEM_GENERATOR_CU
#define SLIME_2_1_CURRICULUM_PROBLEM_GENERATOR_CU

#include "../config/constants.cuh"

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
// Sample-difficulty bias targets classifiers toward weak archive niches.
// Launched once per CURRICULUM_INTERVAL (= 50) generations.
void assemble_classifier_batch(ClassifierBatch* out,
                               float sot_density,
                               uint64_t host_sot_key,
                               cudaStream_t stream);

// SOT pixel permutation: reversible under host_sot_key. Applied to the first
// SOT_SUBBATCH images. Marker bit set on the corresponding is_sot entries.
__device__ void apply_sot_permutation(__half* image_64x64x3,
                                      uint64_t host_sot_key);

// ---- Predictor curriculum -----------------------------------------------
// Weight active classifiers by recent ensemble prediction error on their
// bmap_32 -> bmap_64 mapping. Sample 8 by that weight.
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

bool verify_probe_set(const ProbeSet& set, uint64_t host_key);

// Escalation logic: triggers on blended surprise (S-001 CUSUM). Implemented
// host-side per 2.0 conventions; not duplicated here.

}  // namespace slime::curriculum

#endif  // SLIME_2_1_CURRICULUM_PROBLEM_GENERATOR_CU
