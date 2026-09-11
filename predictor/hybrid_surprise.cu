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

// GELU approximation (Hendrycks-Gimpel) for host-side placeholder MLP.
__host__ inline float ph_gelu(float x) {
    const float k = 0.7978845608f;
    return 0.5f * x * (1.f + tanhf(k * (x + 0.044715f * x * x * x)));
}

__host__ inline float ph_gelu_derivative(float x) {
    const float k = 0.7978845608f;
    float x3 = x * x * x;
    float inner = k * (x + 0.044715f * x3);
    float t = tanhf(inner);
    float sech2 = 1.f - t * t;
    float d_inner = k * (1.f + 3.f * 0.044715f * x * x);
    return 0.5f * (1.f + t) + 0.5f * x * sech2 * d_inner;
}

// Initialize placeholder regressor weights via Kaiming He (fan_in-based).
__host__ inline void init_placeholder_regressor(PlaceholderRegressor* r,
                                                Pcg32* rng) {
    auto box_muller = [](Pcg32* rng) -> float {
        float u1 = pcg32_float(rng);
        float u2 = pcg32_float(rng);
        if (u1 < 1e-30f) u1 = 1e-30f;
        return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
    };

    // W1: fan_in = PH_INPUT (48), GELU -> sqrt(2/48)
    float s1 = sqrtf(2.0f / static_cast<float>(PH_INPUT));
    for (int i = 0; i < PH_INPUT * PH_H1; ++i) r->W1[i] = box_muller(rng) * s1;
    for (int i = 0; i < PH_H1; ++i) r->b1[i] = 0.f;

    // W2: fan_in = PH_H1 (128), GELU -> sqrt(2/128)
    float s2 = sqrtf(2.0f / static_cast<float>(PH_H1));
    for (int i = 0; i < PH_H1 * PH_H2; ++i) r->W2[i] = box_muller(rng) * s2;
    for (int i = 0; i < PH_H2; ++i) r->b2[i] = 0.f;

    // W3: fan_in = PH_H2 (64), linear -> sqrt(1/64)
    float s3 = sqrtf(1.0f / static_cast<float>(PH_H2));
    for (int i = 0; i < PH_H2 * PH_OUT; ++i) r->W3[i] = box_muller(rng) * s3;
    for (int i = 0; i < PH_OUT; ++i) r->b3[i] = 0.f;

    // Zero AdamW state.
    for (int i = 0; i < PH_INPUT * PH_H1; ++i) { r->m_W1[i] = 0.f; r->v_W1[i] = 0.f; }
    for (int i = 0; i < PH_H1 * PH_H2; ++i) { r->m_W2[i] = 0.f; r->v_W2[i] = 0.f; }
    for (int i = 0; i < PH_H2 * PH_OUT; ++i) { r->m_W3[i] = 0.f; r->v_W3[i] = 0.f; }
    for (int i = 0; i < PH_H1; ++i) { r->m_b1[i] = 0.f; r->v_b1[i] = 0.f; }
    for (int i = 0; i < PH_H2; ++i) { r->m_b2[i] = 0.f; r->v_b2[i] = 0.f; }
    for (int i = 0; i < PH_OUT; ++i) { r->m_b3[i] = 0.f; r->v_b3[i] = 0.f; }
    r->step = 0;
}

// Host-side placeholder forward pass. Produces (fitness_hat, log_uncertainty).
// Per A-601: input = concat(bmap_64[32], task_emb[16]) = 48-d.
// h1 = gelu(W1*x + b1) [128]; h2 = gelu(W2*h1 + b2) [64];
// out = W3*h2 + b3 [2].
__host__ inline void placeholder_forward(const PlaceholderRegressor& r,
                                         const float* bmap,
                                         const float* task_emb,
                                         float* h1_out,    // [PH_H1] scratch
                                         float* h2_out,    // [PH_H2] scratch
                                         float* out2) {
    // Concatenate input.
    float input[PH_INPUT];
    for (int i = 0; i < BMAP_DIM; ++i) input[i] = bmap[i];
    for (int i = 0; i < TASK_EMBED_DIM; ++i) input[BMAP_DIM + i] = task_emb[i];

    // Layer 1: h1 = gelu(W1 * input + b1)
    for (int h = 0; h < PH_H1; ++h) {
        float acc = r.b1[h];
        for (int j = 0; j < PH_INPUT; ++j) {
            acc += r.W1[j * PH_H1 + h] * input[j];
        }
        h1_out[h] = ph_gelu(acc);
    }

    // Layer 2: h2 = gelu(W2 * h1 + b2)
    for (int h = 0; h < PH_H2; ++h) {
        float acc = r.b2[h];
        for (int j = 0; j < PH_H1; ++j) {
            acc += r.W2[j * PH_H2 + h] * h1_out[j];
        }
        h2_out[h] = ph_gelu(acc);
    }

    // Layer 3: out = W3 * h2 + b3
    for (int o = 0; o < PH_OUT; ++o) {
        float acc = r.b3[o];
        for (int j = 0; j < PH_H2; ++j) {
            acc += r.W3[j * PH_OUT + o] * h2_out[j];
        }
        out2[o] = acc;
    }
}

// Compute placeholder surprise for a single classifier organism.
// Surprise = (fitness_actual - fitness_hat)^2, heteroscedastic weighting
// by exp(-log_uncertainty) per A-601.
__host__ inline float placeholder_surprise(const PlaceholderRegressor& r,
                                           const float* bmap,
                                           const float* task_emb,
                                           float fitness_actual) {
    float h1[PH_H1], h2[PH_H2], out[PH_OUT];
    placeholder_forward(r, bmap, task_emb, h1, h2, out);
    float mu = out[0];
    float log_unc = out[1];
    float diff = fitness_actual - mu;
    return diff * diff * expf(-log_unc);
}

// Push a (bmap_64, task_emb, fitness) tuple into the replay buffer.
__host__ inline void replay_buffer_push(PlaceholderReplayBuffer* buf,
                                        const float* bmap,
                                        const float* task_emb,
                                        float fitness) {
    int idx = buf->head;
    for (int d = 0; d < BMAP_DIM; ++d)
        buf->bmap[idx * BMAP_DIM + d] = bmap[d];
    for (int d = 0; d < TASK_EMBED_DIM; ++d)
        buf->task_emb[idx * TASK_EMBED_DIM + d] = task_emb[d];
    buf->fitness[idx] = fitness;
    buf->head = (buf->head + 1) % PH_REPLAY_CAPACITY;
    if (buf->filled < PH_REPLAY_CAPACITY) buf->filled++;
}

// AdamW update for a single parameter array.
// Per A-601: lr = 1e-4 fixed by spec.
constexpr float PH_LR = 1e-4f;
constexpr float PH_BETA1 = 0.9f;
constexpr float PH_BETA2 = 0.999f;
constexpr float PH_EPS = 1e-8f;
constexpr float PH_WD = 0.01f;

__host__ inline void adamw_update(float* param, float* grad,
                                  float* m, float* v,
                                  int size, int step) {
    float bc1 = 1.0f - powf(PH_BETA1, static_cast<float>(step));
    float bc2 = 1.0f - powf(PH_BETA2, static_cast<float>(step));
    for (int i = 0; i < size; ++i) {
        m[i] = PH_BETA1 * m[i] + (1.0f - PH_BETA1) * grad[i];
        v[i] = PH_BETA2 * v[i] + (1.0f - PH_BETA2) * grad[i] * grad[i];
        float m_hat = m[i] / bc1;
        float v_hat = v[i] / bc2;
        param[i] -= PH_LR * (m_hat / (sqrtf(v_hat) + PH_EPS) + PH_WD * param[i]);
    }
}

// Placeholder training: one AdamW step on a single sample from the replay buffer.
// Loss = Gaussian NLL: 0.5 * (exp(-s) * (y - mu)^2 + s)
// where (mu, s) = placeholder_forward outputs, y = fitness.
// Per A-601 gradient policy: isolated from organism weights.
constexpr int PH_TRAIN_MINIBATCH = 8;

__host__ inline void placeholder_train_step(PlaceholderRegressor* r,
                                            const PlaceholderReplayBuffer* buf,
                                            Pcg32* rng) {
    if (buf->filled < PH_TRAIN_MINIBATCH) return;
    r->step++;

    // Gradient accumulators (zeroed).
    float dW1[PH_INPUT * PH_H1] = {};
    float db1[PH_H1] = {};
    float dW2[PH_H1 * PH_H2] = {};
    float db2[PH_H2] = {};
    float dW3[PH_H2 * PH_OUT] = {};
    float db3[PH_OUT] = {};

    float total_loss = 0.f;

    for (int mb = 0; mb < PH_TRAIN_MINIBATCH; ++mb) {
        // Sample a random entry from the replay buffer.
        int idx = static_cast<int>(pcg32_random(rng) % static_cast<uint32_t>(buf->filled));
        const float* bmap = &buf->bmap[idx * BMAP_DIM];
        const float* temb = &buf->task_emb[idx * TASK_EMBED_DIM];
        float y = buf->fitness[idx];

        // Forward pass with intermediates.
        float input[PH_INPUT];
        for (int i = 0; i < BMAP_DIM; ++i) input[i] = bmap[i];
        for (int i = 0; i < TASK_EMBED_DIM; ++i) input[BMAP_DIM + i] = temb[i];

        float pre_h1[PH_H1], h1[PH_H1];
        for (int h = 0; h < PH_H1; ++h) {
            float acc = r->b1[h];
            for (int j = 0; j < PH_INPUT; ++j) acc += r->W1[j * PH_H1 + h] * input[j];
            pre_h1[h] = acc;
            h1[h] = ph_gelu(acc);
        }

        float pre_h2[PH_H2], h2[PH_H2];
        for (int h = 0; h < PH_H2; ++h) {
            float acc = r->b2[h];
            for (int j = 0; j < PH_H1; ++j) acc += r->W2[j * PH_H2 + h] * h1[j];
            pre_h2[h] = acc;
            h2[h] = ph_gelu(acc);
        }

        float out[PH_OUT];
        for (int o = 0; o < PH_OUT; ++o) {
            float acc = r->b3[o];
            for (int j = 0; j < PH_H2; ++j) acc += r->W3[j * PH_OUT + o] * h2[j];
            out[o] = acc;
        }

        float mu = out[0];
        float s = out[1];
        float diff = y - mu;
        float exp_neg_s = expf(-s);

        // Gaussian NLL: L = 0.5 * (exp(-s) * (y - mu)^2 + s)
        total_loss += 0.5f * (exp_neg_s * diff * diff + s);

        // dL/d_mu = -exp(-s) * (y - mu)
        // dL/d_s  = 0.5 * (-exp(-s) * (y - mu)^2 + 1)
        float d_mu = -exp_neg_s * diff;
        float d_s = 0.5f * (-exp_neg_s * diff * diff + 1.0f);
        float d_out[PH_OUT] = { d_mu, d_s };

        // Backprop layer 3: out = W3 * h2 + b3
        float d_h2[PH_H2] = {};
        for (int o = 0; o < PH_OUT; ++o) {
            db3[o] += d_out[o];
            for (int j = 0; j < PH_H2; ++j) {
                dW3[j * PH_OUT + o] += h2[j] * d_out[o];
                d_h2[j] += r->W3[j * PH_OUT + o] * d_out[o];
            }
        }

        // Backprop layer 2: h2 = gelu(W2 * h1 + b2)
        float d_pre_h2[PH_H2];
        for (int h = 0; h < PH_H2; ++h) {
            d_pre_h2[h] = d_h2[h] * ph_gelu_derivative(pre_h2[h]);
        }
        float d_h1[PH_H1] = {};
        for (int h = 0; h < PH_H2; ++h) {
            db2[h] += d_pre_h2[h];
            for (int j = 0; j < PH_H1; ++j) {
                dW2[j * PH_H2 + h] += h1[j] * d_pre_h2[h];
                d_h1[j] += r->W2[j * PH_H2 + h] * d_pre_h2[h];
            }
        }

        // Backprop layer 1: h1 = gelu(W1 * input + b1)
        float d_pre_h1[PH_H1];
        for (int h = 0; h < PH_H1; ++h) {
            d_pre_h1[h] = d_h1[h] * ph_gelu_derivative(pre_h1[h]);
        }
        for (int h = 0; h < PH_H1; ++h) {
            db1[h] += d_pre_h1[h];
            for (int j = 0; j < PH_INPUT; ++j) {
                dW1[j * PH_H1 + h] += input[j] * d_pre_h1[h];
            }
        }
    }

    // Average gradients over minibatch.
    float inv_mb = 1.0f / static_cast<float>(PH_TRAIN_MINIBATCH);
    for (int i = 0; i < PH_INPUT * PH_H1; ++i) dW1[i] *= inv_mb;
    for (int i = 0; i < PH_H1; ++i) db1[i] *= inv_mb;
    for (int i = 0; i < PH_H1 * PH_H2; ++i) dW2[i] *= inv_mb;
    for (int i = 0; i < PH_H2; ++i) db2[i] *= inv_mb;
    for (int i = 0; i < PH_H2 * PH_OUT; ++i) dW3[i] *= inv_mb;
    for (int i = 0; i < PH_OUT; ++i) db3[i] *= inv_mb;

    // AdamW updates (lr = 1e-4, per A-601).
    adamw_update(r->W1, dW1, r->m_W1, r->v_W1, PH_INPUT * PH_H1, r->step);
    adamw_update(r->b1, db1, r->m_b1, r->v_b1, PH_H1, r->step);
    adamw_update(r->W2, dW2, r->m_W2, r->v_W2, PH_H1 * PH_H2, r->step);
    adamw_update(r->b2, db2, r->m_b2, r->v_b2, PH_H2, r->step);
    adamw_update(r->W3, dW3, r->m_W3, r->v_W3, PH_H2 * PH_OUT, r->step);
    adamw_update(r->b3, db3, r->m_b3, r->v_b3, PH_OUT, r->step);
}

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
