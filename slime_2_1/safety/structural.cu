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
constexpr int SENTINEL_COUNT = 32;

struct SentinelEnsemble {
    // Sentinel weights + state are stored similarly to placeholder regressor
    // and trained on pruning history (role-blind labels).
    // Specific architecture inherits from 2.0.
    int placeholder;
};

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
