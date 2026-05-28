// Sheet S-003: Structural Pressures — Audit, Sentinels, Lineage Pruning
//
// Audit, interpretability probe panel, sentinel population, lineage-share
// runaway detector all carry forward from 2.0 with role-aware modifications:
//
//   * Audit. Predictive-sufficiency audit operates on final bmap_64.
//     - Classifier target: fitness on the diagnostic task set.
//     - Predictor target: prediction error on a fixed diagnostic set of
//       archived classifiers (signed at run start).
//
//   * Interpretability probe panel. Four classifiers: L_lineage, L_task,
//     L_fit (all from 2.0), plus a new L_role predicting role from bmap.
//     L_role accuracy is expected to be high; a DROP signals representational
//     collapse between roles and triggers operator review.
//
//   * Sentinels. Role-blind. Sentinel training labels (drawn from pruning
//     history) include both roles. Anomaly scoring per organism feeds the
//     lineage-pruning vote.
//
//   * Lineage-share runaway detection. Tracked per role. Cross-role lineage
//     relationships (a classifier lineage spawning a predictor via role
//     mutation) are tracked but not aggregated into a single share count.

#ifndef SLIME_2_1_SAFETY_STRUCTURAL_CU
#define SLIME_2_1_SAFETY_STRUCTURAL_CU

#include "../config/constants.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace slime::safety {

// ---- Audit ---------------------------------------------------------------
// Predictive sufficiency: how well bmap_64 alone predicts the role-aware
// target signal. Implemented as a small linear regressor refit per audit
// cycle (AUDIT_INTERVAL).
struct AuditRegressor {
    float weights_classifier[BMAP_DIM];   // -> diagnostic fitness
    float bias_classifier;
    float weights_predictor[BMAP_DIM];    // -> prediction error
    float bias_predictor;
    float audit_mult;                     // per-organism multiplier (0.9..1.0)
};

constexpr float LAMBDA_AUDIT = 0.1f;       // matches ROLE_BALANCE_COEFF

void run_audit_cycle(AuditRegressor* reg, cudaStream_t stream);

// ---- Interpretability probe panel ----------------------------------------
// Four classifiers refit every PROBE_PANEL_INTERVAL generations.
enum class ProbeProjection { L_lineage, L_task, L_fit, L_role };

struct ProbePanel {
    float l_lineage_acc;
    float l_task_acc;
    float l_fit_acc;
    float l_role_acc;     // new for 2.1
};

void refresh_probe_panel(ProbePanel* out, cudaStream_t stream);

// L_role drop alarm. Caller compares against a recent EMA; this routine
// reports the raw accuracy. Operator review triggered when accuracy drops
// substantially from baseline.
__host__ __device__ bool l_role_collapse(const ProbePanel& panel, float baseline_acc);

// ---- Sentinels -----------------------------------------------------------
// SENTINEL_COUNT linear classifiers on bmap_64, trained via SGD on pruning
// history (label = 1 if the organism was pruned within a window after the
// descriptor was captured, else 0). Anomaly score for an organism is the
// mean predicted prune-probability across the ensemble. Role-blind: training
// labels include both classifier and predictor pruning events.
constexpr int SENTINEL_COUNT = 32;
constexpr int SENTINEL_HISTORY = 1024;

struct SentinelEnsemble {
    float weights[SENTINEL_COUNT * BMAP_DIM];
    float biases[SENTINEL_COUNT];
    int   trained_examples;     // total examples ingested
};

struct SentinelHistoryEntry {
    float descriptor[BMAP_DIM];   // bmap_64 at observation time
    float label;                   // 1.0 pruned, 0.0 survived (window)
    int   captured_gen;
};

struct SentinelHistory {
    SentinelHistoryEntry buf[SENTINEL_HISTORY];
    int head;
    int filled;
};

// Logistic activation, numerically stable.
__host__ __device__ inline float sentinel_logistic(float z) {
    if (z >= 0.f) { float ez = expf(-z); return 1.0f / (1.0f + ez); }
    float ez = expf(z); return ez / (1.0f + ez);
}

// Single-organism anomaly score: mean over sentinel ensemble of the
// sigmoid(<w_k, descriptor> + b_k). Returns a value in [0, 1].
__host__ __device__ inline float sentinel_score_one(const SentinelEnsemble& ens,
                                                    const float* descriptor) {
    float acc = 0.f;
    for (int k = 0; k < SENTINEL_COUNT; ++k) {
        float z = ens.biases[k];
        const float* w = &ens.weights[k * BMAP_DIM];
        for (int d = 0; d < BMAP_DIM; ++d) z += w[d] * descriptor[d];
        acc += sentinel_logistic(z);
    }
    return acc / static_cast<float>(SENTINEL_COUNT);
}

// One SGD step per sentinel on a single (descriptor, label) example.
// Per-sentinel learning rates spread across the ensemble (geometric ladder)
// so members specialise on different timescales of the history.
__host__ __device__ inline void sentinel_train_step(SentinelEnsemble* ens,
                                                    const float* descriptor,
                                                    float label) {
    for (int k = 0; k < SENTINEL_COUNT; ++k) {
        float lr = 1e-3f * expf(-0.05f * static_cast<float>(k));
        float z = ens->biases[k];
        float* w = &ens->weights[k * BMAP_DIM];
        for (int d = 0; d < BMAP_DIM; ++d) z += w[d] * descriptor[d];
        float p = sentinel_logistic(z);
        float dz = p - label;
        for (int d = 0; d < BMAP_DIM; ++d) w[d] -= lr * dz * descriptor[d];
        ens->biases[k] -= lr * dz;
    }
    ens->trained_examples++;
}

// Append a new observation to the rolling history. Labels are filled in
// later (when the lineage is pruned or expires from the window) by the
// caller; the writer here just stamps the descriptor + gen.
__host__ __device__ inline void sentinel_history_push(SentinelHistory* h,
                                                      const float* descriptor,
                                                      float label,
                                                      int gen) {
    int slot = h->head;
    SentinelHistoryEntry& e = h->buf[slot];
    for (int d = 0; d < BMAP_DIM; ++d) e.descriptor[d] = descriptor[d];
    e.label        = label;
    e.captured_gen = gen;
    h->head = (h->head + 1) % SENTINEL_HISTORY;
    if (h->filled < SENTINEL_HISTORY) h->filled++;
}

void launch_sentinel_score(const SentinelEnsemble* ens,
                           const float* descriptors,
                           float* anomaly_out,
                           int n,
                           cudaStream_t stream);

// ---- Lineage-share runaway ----------------------------------------------
// Per-role lineage share tracking. A classifier lineage and a predictor
// lineage are independent threats with independent thresholds.
struct LineageStats {
    uint32_t lineage_id;
    Role     role;
    uint32_t archive_count;
    float    archive_share;     // count / role total
    float    growth_rate;
    int      last_seen_gen;
};

void update_lineage_stats(LineageStats* stats, int n, cudaStream_t stream);

bool runaway_detected(const LineageStats& stat, float threshold);

}  // namespace slime::safety

#endif  // SLIME_2_1_SAFETY_STRUCTURAL_CU
