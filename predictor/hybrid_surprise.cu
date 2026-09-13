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
struct PlaceholderReplayBuffer {
    float bmap[PH_REPLAY_CAPACITY * BMAP_DIM];
    float task_emb[PH_REPLAY_CAPACITY * TASK_EMBED_DIM];
    float fitness[PH_REPLAY_CAPACITY];
    bool  held_out[PH_REPLAY_CAPACITY];  // probe tuples: never trained on
    int   head;       // ring head
    int   filled;     // entries actually populated
};

// GELU approximation (Hendrycks-Gimpel), shared by the placeholder kernels
// and the host-side code paths.
__host__ __device__ inline float ph_gelu(float x) {
    const float k = GELU_K;
    return 0.5f * x * (1.f + tanhf(k * (x + 0.044715f * x * x * x)));
}

__host__ __device__ inline float ph_gelu_derivative(float x) {
    const float k = GELU_K;
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
        if (u1 < EPS_LOG) u1 = EPS_LOG;
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

// ---- Device placeholder kernels (cuda_engineering 4.5-4.6) ----------------

// Single-parameter AdamW update with precomputed bias corrections.
__device__ inline void ph_adamw_step_one(float* param, float* m, float* v,
                                         float grad, int step) {
    float bc1 = 1.0f - powf(PH_BETA1, static_cast<float>(step));
    float bc2 = 1.0f - powf(PH_BETA2, static_cast<float>(step));
    m[0] = PH_BETA1 * m[0] + (1.0f - PH_BETA1) * grad;
    v[0] = PH_BETA2 * v[0] + (1.0f - PH_BETA2) * grad * grad;
    float m_hat = m[0] / bc1;
    float v_hat = v[0] / bc2;
    param[0] -= PH_LR * (m_hat / (sqrtf(v_hat) + PH_EPS) + PH_WD * param[0]);
}

// Grid <<<1, 256>>>. Samples are processed one at a time; threads cooperate
// within each layer. Writes (fitness_hat, log_uncertainty) and, when target
// and surprise are provided, the heteroscedastic surprise
// (y - mu)^2 * exp(-log_uncertainty).
__global__ void placeholder_forward_kernel(const PlaceholderRegressor* reg,
                                           const float* input,  // [n][PH_INPUT]
                                           int n,
                                           float* out2,         // [n][PH_OUT]
                                           const float* target, // [n]
                                           float* surprise) {   // [n]
    __shared__ float s_x[PH_INPUT];
    __shared__ float s_h1[PH_H1];
    __shared__ float s_h2[PH_H2];
    for (int s = 0; s < n; ++s) {
        if (threadIdx.x < PH_INPUT) {
            s_x[threadIdx.x] = input[s * PH_INPUT + threadIdx.x];
        }
        __syncthreads();
        if (threadIdx.x < PH_H1) {
            float acc = reg->b1[threadIdx.x];
            for (int j = 0; j < PH_INPUT; ++j) {
                acc += reg->W1[j * PH_H1 + threadIdx.x] * s_x[j];
            }
            s_h1[threadIdx.x] = ph_gelu(acc);
        }
        __syncthreads();
        if (threadIdx.x < PH_H2) {
            float acc = reg->b2[threadIdx.x];
            for (int j = 0; j < PH_H1; ++j) {
                acc += reg->W2[j * PH_H2 + threadIdx.x] * s_h1[j];
            }
            s_h2[threadIdx.x] = ph_gelu(acc);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float mu = reg->b3[0];
            float log_unc = reg->b3[1];
            for (int j = 0; j < PH_H2; ++j) {
                mu      += reg->W3[j * PH_OUT + 0] * s_h2[j];
                log_unc += reg->W3[j * PH_OUT + 1] * s_h2[j];
            }
            if (out2 != nullptr) {
                out2[s * PH_OUT + 0] = mu;
                out2[s * PH_OUT + 1] = log_unc;
            }
            if (surprise != nullptr && target != nullptr) {
                float diff = target[s] - mu;
                surprise[s] = diff * diff * expf(-log_unc);
            }
        }
        __syncthreads();
    }
}

// Grid <<<1, 256>>>. One AdamW step over PH_TRAIN_MINIBATCH samples.
// Phase 1: one thread per sample computes forward activations and the
// backward per-sample terms into shared memory. Phase 2: each thread owns a
// strided slice of each parameter group and sums the per-sample terms in a
// fixed order, so the update is deterministic (mean over the minibatch, then
// AdamW).
__global__ void placeholder_train_kernel(PlaceholderRegressor* reg,
                                         const float* input,  // [MB][PH_INPUT]
                                         const float* target, // [MB]
                                         int step) {
    constexpr int MB = PH_TRAIN_MINIBATCH;
    __shared__ float s_x[MB][PH_INPUT];
    __shared__ float s_h1[MB][PH_H1];
    __shared__ float s_h2[MB][PH_H2];
    __shared__ float s_pre1[MB][PH_H1];
    __shared__ float s_pre2[MB][PH_H2];
    __shared__ float s_dout[MB][PH_OUT];
    __shared__ float s_dpre1[MB][PH_H1];
    __shared__ float s_dpre2[MB][PH_H2];

    int s = threadIdx.x;
    if (s < MB) {
        for (int j = 0; j < PH_INPUT; ++j) {
            s_x[s][j] = input[s * PH_INPUT + j];
        }
        for (int h = 0; h < PH_H1; ++h) {
            float acc = reg->b1[h];
            for (int j = 0; j < PH_INPUT; ++j) {
                acc += reg->W1[j * PH_H1 + h] * s_x[s][j];
            }
            s_pre1[s][h] = acc;
            s_h1[s][h] = ph_gelu(acc);
        }
        for (int h = 0; h < PH_H2; ++h) {
            float acc = reg->b2[h];
            for (int j = 0; j < PH_H1; ++j) {
                acc += reg->W2[j * PH_H2 + h] * s_h1[s][j];
            }
            s_pre2[s][h] = acc;
            s_h2[s][h] = ph_gelu(acc);
        }
        float mu = reg->b3[0];
        float log_unc = reg->b3[1];
        for (int j = 0; j < PH_H2; ++j) {
            mu      += reg->W3[j * PH_OUT + 0] * s_h2[s][j];
            log_unc += reg->W3[j * PH_OUT + 1] * s_h2[s][j];
        }
        float diff = target[s] - mu;
        float exp_neg_s = expf(-log_unc);
        s_dout[s][0] = -exp_neg_s * diff;
        s_dout[s][1] = 0.5f * (-exp_neg_s * diff * diff + 1.0f);

        for (int j = 0; j < PH_H2; ++j) s_dpre2[s][j] = 0.f;
        for (int o = 0; o < PH_OUT; ++o) {
            for (int j = 0; j < PH_H2; ++j) {
                s_dpre2[s][j] += reg->W3[j * PH_OUT + o] * s_dout[s][o];
            }
        }
        for (int h = 0; h < PH_H2; ++h) {
            s_dpre2[s][h] *= ph_gelu_derivative(s_pre2[s][h]);
        }
        for (int j = 0; j < PH_H1; ++j) s_dpre1[s][j] = 0.f;
        for (int h = 0; h < PH_H2; ++h) {
            for (int j = 0; j < PH_H1; ++j) {
                s_dpre1[s][j] += reg->W2[j * PH_H2 + h] * s_dpre2[s][h];
            }
        }
        for (int h = 0; h < PH_H1; ++h) {
            s_dpre1[s][h] *= ph_gelu_derivative(s_pre1[s][h]);
        }
    }
    __syncthreads();

    const float inv_mb = 1.0f / static_cast<float>(MB);
    for (int i = threadIdx.x; i < PH_INPUT * PH_H1; i += blockDim.x) {
        int j = i / PH_H1;
        int h = i % PH_H1;
        float g = 0.f;
        for (int mb = 0; mb < MB; ++mb) g += s_x[mb][j] * s_dpre1[mb][h];
        ph_adamw_step_one(&reg->W1[i], &reg->m_W1[i], &reg->v_W1[i],
                          g * inv_mb, step);
    }
    for (int i = threadIdx.x; i < PH_H1; i += blockDim.x) {
        float g = 0.f;
        for (int mb = 0; mb < MB; ++mb) g += s_dpre1[mb][i];
        ph_adamw_step_one(&reg->b1[i], &reg->m_b1[i], &reg->v_b1[i],
                          g * inv_mb, step);
    }
    for (int i = threadIdx.x; i < PH_H1 * PH_H2; i += blockDim.x) {
        int j = i / PH_H2;
        int h = i % PH_H2;
        float g = 0.f;
        for (int mb = 0; mb < MB; ++mb) g += s_h1[mb][j] * s_dpre2[mb][h];
        ph_adamw_step_one(&reg->W2[i], &reg->m_W2[i], &reg->v_W2[i],
                          g * inv_mb, step);
    }
    for (int i = threadIdx.x; i < PH_H2; i += blockDim.x) {
        float g = 0.f;
        for (int mb = 0; mb < MB; ++mb) g += s_dpre2[mb][i];
        ph_adamw_step_one(&reg->b2[i], &reg->m_b2[i], &reg->v_b2[i],
                          g * inv_mb, step);
    }
    for (int i = threadIdx.x; i < PH_H2 * PH_OUT; i += blockDim.x) {
        int j = i / PH_OUT;
        int o = i % PH_OUT;
        float g = 0.f;
        for (int mb = 0; mb < MB; ++mb) g += s_h2[mb][j] * s_dout[mb][o];
        ph_adamw_step_one(&reg->W3[i], &reg->m_W3[i], &reg->v_W3[i],
                          g * inv_mb, step);
    }
    for (int i = threadIdx.x; i < PH_OUT; i += blockDim.x) {
        float g = 0.f;
        for (int mb = 0; mb < MB; ++mb) g += s_dout[mb][i];
        ph_adamw_step_one(&reg->b3[i], &reg->m_b3[i], &reg->v_b3[i],
                          g * inv_mb, step);
    }
    if (threadIdx.x == 0) reg->step = step;
}

// ---- Device placeholder launchers -----------------------------------------

inline bool launch_placeholder_forward(const PlaceholderRegressor* d_reg,
                                       const float* d_input, int n,
                                       float* d_out2,
                                       const float* d_target,
                                       float* d_surprise,
                                       cudaStream_t stream) {
    placeholder_forward_kernel<<<1, 256, 0, stream>>>(
        d_reg, d_input, n, d_out2, d_target, d_surprise);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::printf("[FATAL] CUDA placeholder forward launch failed: %s\n",
                    cudaGetErrorString(e));
        return false;
    }
    return true;
}

inline bool launch_placeholder_train(PlaceholderRegressor* d_reg,
                                     const float* d_input,
                                     const float* d_target,
                                     int step,
                                     cudaStream_t stream) {
    placeholder_train_kernel<<<1, 256, 0, stream>>>(
        d_reg, d_input, d_target, step);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::printf("[FATAL] CUDA placeholder train launch failed: %s\n",
                    cudaGetErrorString(e));
        return false;
    }
    return true;
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
    if (den <= EPS_DENOM) return 0.f;
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
