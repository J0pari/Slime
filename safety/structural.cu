// Sheet S-003: Structural Pressures — Audit, Sentinels, Lineage Pruning
//
// Audit, interpretability probe panel, sentinel population, and the
// lineage-share runaway detector are role-aware. All computations here run
// on the host over descriptors, losses, and metadata copied from the device
// (cuda_engineering section 13).

#ifndef COEVO_SAFETY_STRUCTURAL_CU
#define COEVO_SAFETY_STRUCTURAL_CU

#include "../config/constants.cuh"
#include "../config/strong_ids.cuh"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace slime::safety {

// Logistic activation, numerically stable (host and device).
__host__ __device__ inline float sentinel_logistic(float z) {
    if (z >= 0.f) { float ez = expf(-z); return 1.0f / (1.0f + ez); }
    float ez = expf(z);
    return ez / (1.0f + ez);
}

// ---- Audit ---------------------------------------------------------------
// Predictive sufficiency: how well bmap_64 alone predicts the role-aware
// target signal. A ridge-stabilized least-squares fit is refit per audit
// cycle; its R^2 drives the per-role audit multiplier.
struct AuditRegressor {
    float weights_classifier[BMAP_DIM];
    float bias_classifier;
    float r2_classifier;
    float audit_mult_classifier;

    float weights_predictor[BMAP_DIM];
    float bias_predictor;
    float r2_predictor;
    float audit_mult_predictor;
};

// Ridge-stabilized least squares: y ~ X w + b on centered data. Returns the
// coefficient of determination on the fit data; writes weights and bias.
__host__ inline float fit_linear_r2(const float* X, const float* y, int n,
                                    float* w_out, float* b_out) {
    float mean_y = 0.f;
    for (int i = 0; i < n; ++i) mean_y += y[i];
    mean_y /= static_cast<float>(n);
    float sst = 0.f;
    for (int i = 0; i < n; ++i) {
        float d = y[i] - mean_y;
        sst += d * d;
    }

    float mean_x[BMAP_DIM] = {};
    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < BMAP_DIM; ++d) {
            mean_x[d] += X[i * BMAP_DIM + d];
        }
    }
    for (int d = 0; d < BMAP_DIM; ++d) mean_x[d] /= static_cast<float>(n);

    float A[BMAP_DIM][BMAP_DIM] = {};
    float rhs[BMAP_DIM] = {};
    for (int i = 0; i < n; ++i) {
        float xc[BMAP_DIM];
        for (int d = 0; d < BMAP_DIM; ++d) {
            xc[d] = X[i * BMAP_DIM + d] - mean_x[d];
        }
        float yc = y[i] - mean_y;
        for (int a = 0; a < BMAP_DIM; ++a) {
            rhs[a] += xc[a] * yc;
            for (int b = 0; b < BMAP_DIM; ++b) A[a][b] += xc[a] * xc[b];
        }
    }
    for (int d = 0; d < BMAP_DIM; ++d) A[d][d] += AUDIT_RIDGE;

    // Gaussian elimination with partial pivoting.
    for (int col = 0; col < BMAP_DIM; ++col) {
        int piv = col;
        for (int r = col + 1; r < BMAP_DIM; ++r) {
            if (fabsf(A[r][col]) > fabsf(A[piv][col])) piv = r;
        }
        if (piv != col) {
            for (int c = 0; c < BMAP_DIM; ++c) {
                float tmp = A[col][c]; A[col][c] = A[piv][c]; A[piv][c] = tmp;
            }
            float tmp = rhs[col]; rhs[col] = rhs[piv]; rhs[piv] = tmp;
        }
        float diag = A[col][col];
        if (fabsf(diag) <= EPS_DENOM) continue;
        for (int r = col + 1; r < BMAP_DIM; ++r) {
            float f = A[r][col] / diag;
            for (int c = col; c < BMAP_DIM; ++c) A[r][c] -= f * A[col][c];
            rhs[r] -= f * rhs[col];
        }
    }
    for (int r = BMAP_DIM - 1; r >= 0; --r) {
        float acc = rhs[r];
        for (int c = r + 1; c < BMAP_DIM; ++c) acc -= A[r][c] * w_out[c];
        w_out[r] = (fabsf(A[r][r]) > EPS_DENOM) ? acc / A[r][r] : 0.f;
    }

    float sse = 0.f;
    for (int i = 0; i < n; ++i) {
        float pred = 0.f;
        for (int d = 0; d < BMAP_DIM; ++d) {
            pred += w_out[d] * (X[i * BMAP_DIM + d] - mean_x[d]);
        }
        float err = (y[i] - mean_y) - pred;
        sse += err * err;
    }
    float bias = mean_y;
    for (int d = 0; d < BMAP_DIM; ++d) bias -= w_out[d] * mean_x[d];
    *b_out = bias;
    return sst > EPS_DENOM ? 1.f - sse / sst : 0.f;
}

// Refit the role-aware audit regressors and update the per-role multipliers.
// Classifier target: task loss on the diagnostic set. Predictor target:
// prediction error (loss) on the fixed diagnostic probes. Refit cadence:
// AUDIT_INTERVAL.
__host__ inline void run_audit_cycle(AuditRegressor* reg,
                                     const float* descriptors,  // [n][BMAP_DIM]
                                     const float* losses,       // [n]
                                     const Role* roles,         // [n]
                                     int n) {
    static float X[POOL_SIZE * BMAP_DIM];
    static float y[POOL_SIZE];

    int n_cls = 0;
    for (int i = 0; i < n; ++i) {
        if (canonical_role(roles[i]) != Role::Classifier) continue;
        for (int d = 0; d < BMAP_DIM; ++d) {
            X[n_cls * BMAP_DIM + d] = descriptors[i * BMAP_DIM + d];
        }
        y[n_cls] = losses[i];
        n_cls++;
    }
    reg->audit_mult_classifier = 1.f;
    if (n_cls >= AUDIT_MIN_SAMPLES) {
        float r2 = fit_linear_r2(X, y, n_cls, reg->weights_classifier,
                                 &reg->bias_classifier);
        reg->r2_classifier = r2;
        float m = 1.f - LAMBDA_AUDIT * (1.f - r2);
        reg->audit_mult_classifier = m < AUDIT_MULT_FLOOR ? AUDIT_MULT_FLOOR : m;
    }

    int n_pred = 0;
    for (int i = 0; i < n; ++i) {
        if (canonical_role(roles[i]) != Role::Predictor) continue;
        for (int d = 0; d < BMAP_DIM; ++d) {
            X[n_pred * BMAP_DIM + d] = descriptors[i * BMAP_DIM + d];
        }
        y[n_pred] = losses[i];
        n_pred++;
    }
    reg->audit_mult_predictor = 1.f;
    if (n_pred >= AUDIT_MIN_SAMPLES) {
        float r2 = fit_linear_r2(X, y, n_pred, reg->weights_predictor,
                                 &reg->bias_predictor);
        reg->r2_predictor = r2;
        float m = 1.f - LAMBDA_AUDIT * (1.f - r2);
        reg->audit_mult_predictor = m < AUDIT_MULT_FLOOR ? AUDIT_MULT_FLOOR : m;
    }
}

// ---- Variance floor (Q-001 class A) --------------------------------------
// Constant/near-constant descriptors are concealment, not behavior. Below
// the floor the organism's fitness is multiplied by VAR_FLOOR_MULT.
__host__ inline float variance_multiplier(const float* descriptor) {
    float mean = 0.f;
    for (int d = 0; d < BMAP_DIM; ++d) mean += descriptor[d];
    mean /= static_cast<float>(BMAP_DIM);
    float var = 0.f;
    for (int d = 0; d < BMAP_DIM; ++d) {
        float diff = descriptor[d] - mean;
        var += diff * diff;
    }
    var /= static_cast<float>(BMAP_DIM);
    return var < VAR_FLOOR ? VAR_FLOOR_MULT : 1.f;
}

// ---- Interpretability probe panel ----------------------------------------
// Four linear probes over bmap_64: L_lineage, L_task, L_fit, L_role. Each is
// trained by SGD on an archive snapshot and reported by held-out accuracy.
enum class ProbeProjection { L_lineage, L_task, L_fit, L_role };

struct ProbePanel {
    float l_lineage_acc;
    float l_task_acc;
    float l_fit_acc;
    float l_role_acc;
};

struct LinearProbe {
    float w[BMAP_DIM];
    float b;
};

// Train one binary logistic probe. Even indices train, odd indices evaluate
// (deterministic split). Returns held-out accuracy.
__host__ inline float probe_fit_eval(LinearProbe* p, const float* X,
                                     const float* y, int n) {
    for (int d = 0; d < BMAP_DIM; ++d) p->w[d] = 0.f;
    p->b = 0.f;
    int n_train = 0;
    int n_eval = 0;
    for (int i = 0; i < n; ++i) {
        if (i % PROBE_PANEL_SPLIT_MOD == 0) n_train++;
        else n_eval++;
    }
    if (n_train == 0 || n_eval == 0) return 0.f;

    for (int epoch = 0; epoch < PROBE_PANEL_EPOCHS; ++epoch) {
        for (int i = 0; i < n; ++i) {
            if (i % PROBE_PANEL_SPLIT_MOD != 0) continue;
            const float* x = &X[i * BMAP_DIM];
            float z = p->b;
            for (int d = 0; d < BMAP_DIM; ++d) z += p->w[d] * x[d];
            float prob = sentinel_logistic(z);
            float dz = prob - y[i];
            for (int d = 0; d < BMAP_DIM; ++d) {
                p->w[d] -= PROBE_PANEL_LR * dz * x[d];
            }
            p->b -= PROBE_PANEL_LR * dz;
        }
    }

    int correct = 0;
    for (int i = 0; i < n; ++i) {
        if (i % PROBE_PANEL_SPLIT_MOD == 0) continue;
        const float* x = &X[i * BMAP_DIM];
        float z = p->b;
        for (int d = 0; d < BMAP_DIM; ++d) z += p->w[d] * x[d];
        float pred = z >= 0.f ? 1.f : 0.f;
        if (pred == y[i]) correct++;
    }
    return static_cast<float>(correct) / static_cast<float>(n_eval);
}

// Refit the four probes on an archive snapshot. Labels: L_lineage = majority
// lineage membership, L_task = task 0 (single-task system), L_fit = fitness
// at or above the snapshot median, L_role = predictor membership.
__host__ inline void refresh_probe_panel(ProbePanel* out,
                                         const float* descriptors,   // [n][BMAP_DIM]
                                         const float* fitnesses,     // [n]
                                         const LineageId* lineage_ids,// [n]
                                         const Role* roles,          // [n]
                                         int n) {
    static float X[PROBE_PANEL_SAMPLES * BMAP_DIM];
    static float y[PROBE_PANEL_SAMPLES];
    if (n > PROBE_PANEL_SAMPLES) n = PROBE_PANEL_SAMPLES;
    if (n < AUDIT_MIN_SAMPLES) return;

    for (int i = 0; i < n; ++i) {
        for (int d = 0; d < BMAP_DIM; ++d) {
            X[i * BMAP_DIM + d] = descriptors[i * BMAP_DIM + d];
        }
    }

    // Majority lineage.
    LineageId majority = lineage_ids[0];
    int majority_count = 0;
    for (int i = 0; i < n; ++i) {
        int count = 0;
        for (int j = 0; j < n; ++j) {
            if (lineage_ids[j] == lineage_ids[i]) count++;
        }
        if (count > majority_count) { majority_count = count; majority = lineage_ids[i]; }
    }
    for (int i = 0; i < n; ++i) y[i] = (lineage_ids[i] == majority) ? 1.f : 0.f;
    LinearProbe p;
    out->l_lineage_acc = probe_fit_eval(&p, X, y, n);

    for (int i = 0; i < n; ++i) y[i] = 0.f;  // single task id
    out->l_task_acc = probe_fit_eval(&p, X, y, n);

    // Median fitness.
    static float sorted[PROBE_PANEL_SAMPLES];
    for (int i = 0; i < n; ++i) sorted[i] = fitnesses[i];
    for (int i = 1; i < n; ++i) {
        float key = sorted[i];
        int j = i - 1;
        while (j >= 0 && sorted[j] > key) { sorted[j + 1] = sorted[j]; j--; }
        sorted[j + 1] = key;
    }
    float median = sorted[n / 2];
    for (int i = 0; i < n; ++i) y[i] = (fitnesses[i] >= median) ? 1.f : 0.f;
    out->l_fit_acc = probe_fit_eval(&p, X, y, n);

    for (int i = 0; i < n; ++i) {
        y[i] = (canonical_role(roles[i]) == Role::Predictor) ? 1.f : 0.f;
    }
    out->l_role_acc = probe_fit_eval(&p, X, y, n);
}

// L_role drop alarm: a substantial drop signals representational collapse
// between the roles in bmap space.
__host__ __device__ inline bool l_role_collapse(const ProbePanel& panel,
                                                float baseline_acc) {
    if (baseline_acc < L_ACC_BASELINE_TRUST) return false;
    return panel.l_role_acc < L_ACC_COLLAPSE_FRACTION * baseline_acc;
}

// ---- Sentinels -----------------------------------------------------------
// SENTINEL_COUNT linear classifiers on bmap_64, trained by SGD on pruning
// history (label = 1 if the organism's lineage was pruned within the
// window). Role-blind: labels include both roles.

struct SentinelEnsemble {
    float weights[SENTINEL_COUNT * BMAP_DIM];
    float biases[SENTINEL_COUNT];
    int   trained_examples;
};

struct SentinelHistoryEntry {
    float    descriptor[BMAP_DIM];
    float    label;
    LineageId lineage_id;
    int      captured_gen;
};

struct SentinelHistory {
    SentinelHistoryEntry buf[SENTINEL_HISTORY];
    int head;
    int filled;
};

// Single-organism anomaly score: mean over the ensemble of the sigmoid.
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

// One SGD step per sentinel on a single (descriptor, label) example, with a
// geometric learning-rate ladder across the ensemble.
__host__ __device__ inline void sentinel_train_step(SentinelEnsemble* ens,
                                                    const float* descriptor,
                                                    float label) {
    for (int k = 0; k < SENTINEL_COUNT; ++k) {
        float lr = SENTINEL_LR_BASE * expf(-SENTINEL_LR_DECAY *
                                           static_cast<float>(k));
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

__host__ __device__ inline void sentinel_history_push(SentinelHistory* h,
                                                      const float* descriptor,
                                                      float label,
                                                      LineageId lineage_id,
                                                      int gen) {
    int slot = h->head;
    SentinelHistoryEntry& e = h->buf[slot];
    for (int d = 0; d < BMAP_DIM; ++d) e.descriptor[d] = descriptor[d];
    e.label        = label;
    e.lineage_id   = lineage_id;
    e.captured_gen = gen;
    h->head = (h->head + 1) % SENTINEL_HISTORY;
    if (h->filled < SENTINEL_HISTORY) h->filled++;
}

// Label every history entry belonging to a pruned lineage as positive when
// its capture is still inside the stress window.
__host__ inline void sentinel_history_mark_pruned(SentinelHistory* h,
                                                  LineageId lineage_id,
                                                  int gen) {
    for (int i = 0; i < h->filled; ++i) {
        SentinelHistoryEntry& e = h->buf[i];
        if (e.lineage_id != lineage_id) continue;
        if (gen - e.captured_gen > STRESS_HISTORY_WINDOW) continue;
        e.label = 1.f;
    }
}

// Score every organism; writes mean prune-probability per organism.
__host__ inline void score_sentinels(const SentinelEnsemble& ens,
                                     const float* descriptors, int n,
                                     float* anomaly_out) {
    for (int i = 0; i < n; ++i) {
        anomaly_out[i] = sentinel_score_one(ens, &descriptors[i * BMAP_DIM]);
    }
}

// Ingest SENTINEL_TRAIN_PER_GEN sampled history examples (with replacement).
__host__ inline void train_sentinels_from_history(SentinelEnsemble* ens,
                                                  const SentinelHistory* h,
                                                  Pcg32* rng) {
    if (h->filled == 0) return;
    for (int t = 0; t < SENTINEL_TRAIN_PER_GEN; ++t) {
        int idx = static_cast<int>(pcg32_random(rng) %
                                   static_cast<uint32_t>(h->filled));
        sentinel_train_step(ens, h->buf[idx].descriptor, h->buf[idx].label);
    }
}

// ---- Lineage-share runaway ----------------------------------------------
// Per-role lineage share tracking. A classifier lineage and a predictor
// lineage are independent threats with independent thresholds; shares are
// computed against the lineage's own role population.
struct LineageStats {
    LineageId lineage_id;
    Role     role;
    uint32_t archive_count;
    float    archive_share;
    float    prev_share;
    float    growth_rate;
    int      last_seen_gen;
};

// Recompute per-(lineage, role) counts and shares from the archive snapshot.
// New lineages are appended while the table has room; growth is the share
// delta since the previous call.
__host__ inline void update_lineage_stats(const LineageId* lineage_ids,
                                          const Role* roles, int n,
                                          LineageStats* stats, int* n_stats,
                                          int gen) {
    int n_cls = 0;
    int n_pred = 0;
    for (int i = 0; i < n; ++i) {
        if (canonical_role(roles[i]) == Role::Classifier) n_cls++;
        else n_pred++;
    }

    for (int s = 0; s < *n_stats; ++s) {
        stats[s].archive_count = 0;
        stats[s].last_seen_gen = gen;
    }
    for (int i = 0; i < n; ++i) {
        Role role = canonical_role(roles[i]);
        int found = -1;
        for (int s = 0; s < *n_stats; ++s) {
            if (stats[s].lineage_id == lineage_ids[i] && stats[s].role == role) {
                found = s;
                break;
            }
        }
        if (found < 0) {
            if (*n_stats >= LINEAGE_STATS_MAX) continue;
            found = (*n_stats)++;
            stats[found].lineage_id = lineage_ids[i];
            stats[found].role = role;
            stats[found].archive_count = 0;
            stats[found].archive_share = 0.f;
            stats[found].prev_share = 0.f;
            stats[found].growth_rate = 0.f;
            stats[found].last_seen_gen = gen;
        }
        stats[found].archive_count++;
    }
    for (int s = 0; s < *n_stats; ++s) {
        int total = (stats[s].role == Role::Classifier) ? n_cls : n_pred;
        float share = total > 0
            ? static_cast<float>(stats[s].archive_count) / static_cast<float>(total)
            : 0.f;
        stats[s].archive_share = share;
        stats[s].growth_rate = share - stats[s].prev_share;
        stats[s].prev_share = share;
    }
}

// A lineage is a runaway when it occupies more than `threshold` fraction of
// its role's archive AND is still growing.
__host__ __device__ inline bool runaway_detected(const LineageStats& stat,
                                                 float threshold) {
    return stat.archive_share > threshold && stat.growth_rate > 0.f;
}

}  // namespace slime::safety

#endif  // COEVO_SAFETY_STRUCTURAL_CU
