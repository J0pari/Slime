// Global context broadcast and its adjoint (A-203, I5).
//
// The forward computes the 16-channel spatial mean of the pre-broadcast
// state, maps it through W_ctx to CH_AUX_LAST-CH_AUX_FIRST+1 context values,
// and overwrites the aux channels of every cell:
//
//   summary[c]      = mean_cells state_pre[cell, c]
//   ctx[k]          = sum_c W_ctx[c*K + k] * summary[c]
//   state_post[.,k] = ctx[k]                    (aux channels, overwritten)
//   state_post[.,c] = state_pre[.,c]            (all other channels)
//
// The adjoint consumes d_state_post and produces dW_ctx and d_state_pre.
// Direct-path gradients of the overwritten aux channels are zero, but the
// summary depends on every pre-broadcast channel, so the mean adjoint
// re-adds d_summary[c]/n_cells to ALL channels including the aux ones.
//
// This file is the single definition of the formulas: the device kernels and
// the host finite-difference witness both use it, so they cannot drift.

#ifndef COEVO_NCA_CONTEXT_ADJOINT_CUH
#define COEVO_NCA_CONTEXT_ADJOINT_CUH

#include "../config/constants.cuh"

namespace slime::nca {

constexpr int CTX_K = CH_AUX_LAST - CH_AUX_FIRST + 1;

// ctx[k] = sum_c W_ctx[c*CTX_K + k] * summary[c].
__host__ __device__ inline void context_map(const float* summary,
                                            const float* W_ctx,
                                            float* ctx) {
    for (int k = 0; k < CTX_K; ++k) {
        float acc = 0.f;
        for (int c = 0; c < CA_CHANNELS; ++c) {
            acc += W_ctx[c * CTX_K + k] * summary[c];
        }
        ctx[k] = acc;
    }
}

// Forward broadcast on a host/device state array [n_cells * CA_CHANNELS].
__host__ __device__ inline void context_broadcast(const float* state_pre,
                                                  int n_cells,
                                                  const float* W_ctx,
                                                  float* state_post) {
    float summary[CA_CHANNELS];
    for (int c = 0; c < CA_CHANNELS; ++c) {
        float acc = 0.f;
        for (int cell = 0; cell < n_cells; ++cell) {
            acc += state_pre[cell * CA_CHANNELS + c];
        }
        summary[c] = acc / static_cast<float>(n_cells);
    }
    float ctx[CTX_K];
    context_map(summary, W_ctx, ctx);
    for (int cell = 0; cell < n_cells; ++cell) {
        for (int c = 0; c < CA_CHANNELS; ++c) {
            state_post[cell * CA_CHANNELS + c] =
                state_pre[cell * CA_CHANNELS + c];
        }
        for (int k = 0; k < CTX_K; ++k) {
            state_post[cell * CA_CHANNELS + CH_AUX_FIRST + k] = ctx[k];
        }
    }
}

// Adjoint: d_state_post -> dW_ctx, d_state_pre.
__host__ __device__ inline void context_backward(const float* state_pre,
                                                 int n_cells,
                                                 const float* W_ctx,
                                                 const float* d_state_post,
                                                 float* dW_ctx,
                                                 float* d_state_pre) {
    float summary[CA_CHANNELS];
    for (int c = 0; c < CA_CHANNELS; ++c) {
        float acc = 0.f;
        for (int cell = 0; cell < n_cells; ++cell) {
            acc += state_pre[cell * CA_CHANNELS + c];
        }
        summary[c] = acc / static_cast<float>(n_cells);
    }

    // d_ctx[k] = sum_cells d_state_post[cell, aux+k].
    float d_ctx[CTX_K];
    for (int k = 0; k < CTX_K; ++k) d_ctx[k] = 0.f;
    for (int cell = 0; cell < n_cells; ++cell) {
        for (int k = 0; k < CTX_K; ++k) {
            d_ctx[k] += d_state_post[cell * CA_CHANNELS + CH_AUX_FIRST + k];
        }
    }

    // dW_ctx[c*CTX_K + k] = summary[c] * d_ctx[k].
    for (int c = 0; c < CA_CHANNELS; ++c) {
        for (int k = 0; k < CTX_K; ++k) {
            dW_ctx[c * CTX_K + k] += summary[c] * d_ctx[k];
        }
    }

    // d_summary[c] = sum_k W_ctx[c*CTX_K + k] * d_ctx[k]; the mean's adjoint
    // adds d_summary[c]/n_cells to every cell and channel.
    for (int c = 0; c < CA_CHANNELS; ++c) {
        float acc = 0.f;
        for (int k = 0; k < CTX_K; ++k) {
            acc += W_ctx[c * CTX_K + k] * d_ctx[k];
        }
        float mean_grad = acc / static_cast<float>(n_cells);
        for (int cell = 0; cell < n_cells; ++cell) {
            float direct = 0.f;
            bool is_aux = (c >= CH_AUX_FIRST && c <= CH_AUX_LAST);
            if (!is_aux) {
                direct = d_state_post[cell * CA_CHANNELS + c];
            }
            d_state_pre[cell * CA_CHANNELS + c] = direct + mean_grad;
        }
    }
}

}  // namespace slime::nca

#endif  // COEVO_NCA_CONTEXT_ADJOINT_CUH
