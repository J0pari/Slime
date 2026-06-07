// Sheet A-601: Predictor Role & Hybrid Surprise Signal
//
// Two surprise sources operate in parallel:
//
//   Placeholder regressor (always running). MLP (32+16 -> 128 -> 64 -> 2)
//   predicting (fitness_hat, log_uncertainty) for classifier organisms from
//   bmap_64 + task embedding. AdamW lr=1e-4. Never operates on predictor
//   organisms. Serves as continuous ground-truth check.
//
//   Evolved predictor sub-population (after archive crosses MAX_ARCHIVE/2).
//   16 predictor-role founders spawned by role-flipping copies of high-novelty
//   classifier parents. Each predictor evaluated on K=8 random active
//   classifiers per generation; loss is MSE on bmap_64.
//
// Hybrid blending: s_blended = (1 - r) * s_placeholder + r * s_predictor
// where r = clip(Pearson(s_placeholder, s_predictor) on probe set over the
// last 100 generations, [0, 1]).

#ifndef COEVO_PREDICTOR_HYBRID_SURPRISE_CU
#define COEVO_PREDICTOR_HYBRID_SURPRISE_CU

#include "../config/constants.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace slime::predictor {

// ---- Placeholder regressor ----------------------------------------------
// Layer sizes: (BMAP_DIM + TASK_EMBED_DIM = 48) -> 128 -> 64 -> 2.
// Output: (fitness_hat, log_uncertainty).
constexpr int PH_INPUT = BMAP_DIM + TASK_EMBED_DIM;
constexpr int PH_H1    = 128;
constexpr int PH_H2    = 64;
constexpr int PH_OUT   = 2;

struct PlaceholderRegressor {
    float W1[PH_INPUT * PH_H1];
    float b1[PH_H1];
    float W2[PH_H1 * PH_H2];
    float b2[PH_H2];
    float W3[PH_H2 * PH_OUT];
    float b3[PH_OUT];

    // AdamW state (lr = 1e-4 fixed by spec).
    float m_W1[PH_INPUT * PH_H1]; float v_W1[PH_INPUT * PH_H1];
    float m_W2[PH_H1 * PH_H2];    float v_W2[PH_H1 * PH_H2];
    float m_W3[PH_H2 * PH_OUT];   float v_W3[PH_H2 * PH_OUT];
    float m_b1[PH_H1]; float v_b1[PH_H1];
    float m_b2[PH_H2]; float v_b2[PH_H2];
    float m_b3[PH_OUT]; float v_b3[PH_OUT];

    int step;
};

// Rolling buffer of (bmap_64, task_emb, fitness) tuples from the classifier
// archive. Capacity 5000 per I-001 shared structures.
constexpr int PH_REPLAY_CAPACITY = 5000;

struct PlaceholderReplayBuffer {
    float bmap[PH_REPLAY_CAPACITY * BMAP_DIM];
    float task_emb[PH_REPLAY_CAPACITY * TASK_EMBED_DIM];
    float fitness[PH_REPLAY_CAPACITY];
    int   head;       // ring head
    int   filled;     // entries actually populated
};

// DECLARED ONLY — blueprint-in-place.
// placeholder_forward: a 3-layer MLP forward. Concatenate bmap (32) and
// task_emb (16) into a 48-d input; h1 = gelu(W1.x + b1) [128]; h2 =
// gelu(W2.h1 + b2) [64]; out2 = W3.h2 + b3 [2] = (fitness_hat,
// log_uncertainty). Surprise for a classifier is then
// (fitness_actual - fitness_hat)^2, optionally scaled by exp(-log_uncertainty)
// for a heteroscedastic weighting. Pure feed-forward; no shared state.
__device__ void placeholder_forward(const PlaceholderRegressor& r,
                                    const float* bmap,
                                    const float* task_emb,
                                    float* out2);

// DECLARED ONLY — blueprint-in-place.
// launch_placeholder_train: one AdamW step (lr 1e-4) on a minibatch drawn from
// the replay buffer. Loss is a Gaussian NLL: 0.5 * (exp(-s) * (y - mu)^2 + s)
// where (mu, s) = placeholder_forward outputs, y = stored fitness — this
// trains both the mean and the uncertainty head. Backprop through the 3 layers,
// apply AdamW to W1..W3 and biases using the m_/v_ buffers in the struct. Runs
// once per generation in world_train_phase, isolated from organism weights
// (A-601 gradient policy).
void launch_placeholder_train(PlaceholderRegressor* r,
                              const PlaceholderReplayBuffer* buf,
                              cudaStream_t stream);

// ---- Predictor ensemble surprise ----------------------------------------
// For a probe target classifier organism, surprise = variance across
// predictions from the top-K predictors (by recent fitness).

struct PredictorSelectionCache {
    uint32_t organism_idx[PREDICTOR_ENSEMBLE_TOP_K];
    float    recent_fitness[PREDICTOR_ENSEMBLE_TOP_K];
};

// Ensemble surprise per probe target. Each predictor's bmap_64 prediction
// for the probe lives in the Intent Registry from the forward phase.
// Surprise = mean (over BMAP_DIM) variance across the K predictions.
__host__ __device__ inline float ensemble_surprise(const float* predictions,  // [K * BMAP_DIM]
                                                   int k) {
    float total = 0.f;
    for (int d = 0; d < BMAP_DIM; ++d) {
        float mean = 0.f;
        for (int i = 0; i < k; ++i) {
            mean += predictions[i * BMAP_DIM + d];
        }
        mean /= static_cast<float>(k);
        float var = 0.f;
        for (int i = 0; i < k; ++i) {
            float diff = predictions[i * BMAP_DIM + d] - mean;
            var += diff * diff;
        }
        total += var / static_cast<float>(k);
    }
    return total / static_cast<float>(BMAP_DIM);
}

// ---- Bootstrap founder spawn --------------------------------------------
// One-shot. Selects PREDICTOR_FOUNDERS (= 16) high-novelty classifier parents
// from the archive and emits role-flipped copies into the active pool: the
// genome's role bits go to Predictor (= 01), delta weights and the rest of
// the genome are preserved. Subsequent predictor reproduction follows normal
// spawn rules.
//
// Input arrays are length n_active. parent_novelty[i] is the role-internal
// novelty score for classifier i (predictors and dead slots get -inf).
// out_founders is filled with PREDICTOR_FOUNDERS indices into parent_idx.
__host__ inline void select_predictor_founders(const float* parent_novelty,
                                               int n_active,
                                               int* out_founder_indices) {
    // Simple top-K selection. K is small (16) so an O(N*K) scan is fine.
    for (int k = 0; k < PREDICTOR_FOUNDERS; ++k) {
        int best = -1;
        float best_v = -1.0f;
        for (int i = 0; i < n_active; ++i) {
            // Skip already-chosen indices.
            bool taken = false;
            for (int j = 0; j < k; ++j) if (out_founder_indices[j] == i) { taken = true; break; }
            if (taken) continue;
            if (parent_novelty[i] > best_v) {
                best_v = parent_novelty[i];
                best   = i;
            }
        }
        out_founder_indices[k] = best;
    }
}

// ---- Hybrid blending ----------------------------------------------------
struct CorrelationWindow {
    float s_placeholder[HYBRID_R_WINDOW];
    float s_predictor[HYBRID_R_WINDOW];
    int   head;
    int   filled;
};

// Pearson r clipped to [0, 1]. Before bootstrap, the window is empty and r
// is treated as zero; the placeholder dominates.
__host__ __device__ inline float pearson_r_clipped(const CorrelationWindow& w) {
    int n = w.filled;
    if (n < 2) return 0.f;
    float sum_x = 0.f, sum_y = 0.f;
    for (int i = 0; i < n; ++i) {
        sum_x += w.s_placeholder[i];
        sum_y += w.s_predictor[i];
    }
    float mean_x = sum_x / static_cast<float>(n);
    float mean_y = sum_y / static_cast<float>(n);
    float num = 0.f, den_x = 0.f, den_y = 0.f;
    for (int i = 0; i < n; ++i) {
        float dx = w.s_placeholder[i] - mean_x;
        float dy = w.s_predictor[i]   - mean_y;
        num   += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    float den = sqrtf(den_x * den_y);
    if (den <= 1e-12f) return 0.f;
    float r = num / den;
    if (r < 0.f) return 0.f;
    if (r > 1.f) return 1.f;
    return r;
}

// Push a new (s_placeholder, s_predictor) sample into the rolling window.
__host__ __device__ inline void push_correlation(CorrelationWindow* w,
                                                 float s_ph, float s_pr) {
    w->s_placeholder[w->head] = s_ph;
    w->s_predictor[w->head]   = s_pr;
    w->head = (w->head + 1) % HYBRID_R_WINDOW;
    if (w->filled < HYBRID_R_WINDOW) w->filled++;
}

__host__ __device__ inline float blend_surprise(float s_placeholder,
                                                float s_predictor,
                                                float r) {
    if (r < 0.f) r = 0.f;
    if (r > 1.f) r = 1.f;
    return (1.0f - r) * s_placeholder + r * s_predictor;
}

// The CUSUM on r itself (companion to the surprise CUSUM, A-601 + S-001) runs
// through the single safety::CusumState implementation in safety/monitoring.cu
// rather than a second hand-rolled accumulator here. A-601 only specifies that
// r is monitored for precipitous collapse; the driver feeds r into
// safety::cusum_update(&world->cusum_r, r) each generation. Keeping one CUSUM
// implementation avoids the two drifting apart (e.g. only one resetting its
// accumulator after an alarm).

}  // namespace slime::predictor

#endif  // COEVO_PREDICTOR_HYBRID_SURPRISE_CU
