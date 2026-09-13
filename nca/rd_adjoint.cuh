// Reaction-diffusion step and its adjoint (A-202, I6).
//
// Forward (explicit operator split, per cell p, chemical channel c):
//   here[j]  = curr[p, j]
//   lap[c]   = sum_{n in N(p)} curr[n, c] - 4*here[c]
//   react[c] = sum_j K[c, j] * here[j]
//   next[p,c]= clamp(base[p,c] + DT*(D[c]*lap[c] + react[c] - DECAY*here[c]))
//
// where `base` is the CA output written before RD (rd_step reads curr for the
// stencil and adds onto base). Non-chemical channels are untouched.
//
// Adjoint (loss gradient d_next):
//   d_base[p,c]      = d_next[p,c]
//   d_curr[p,j]     += DT * sum_c d_next[p,c] * K[c,j]      (K transposed)
//   d_curr[p,c]     += -DT * DECAY * d_next[p,c]
//   d_curr[n,c]     += DT * D[c] * d_next[p,c]   (each neighbor)
//   d_curr[p,c]     += -4 * DT * D[c] * d_next[p,c]
//   dK[c,j]         += DT * d_next[p,c] * curr[p,j]
//   dD[c]           += DT * lap[c] * d_next[p,c]
// A saturated output (|pre-clamp| >= FP16 max) has zero derivative: the
// clamp zeroes d_next for that cell/channel before all terms above.
//
// This file is the single definition of the formulas: the device kernels and
// the host finite-difference witness both use it.

#ifndef COEVO_NCA_RD_ADJOINT_CUH
#define COEVO_NCA_RD_ADJOINT_CUH

#include "../config/constants.cuh"

namespace slime::nca::rd {

constexpr int RD_CHEM_N = CH_CHEM_LAST + 1;

// Host/device forward on a toroidal n x n grid, channels [0, RD_CHEM_N).
// curr/next/base are [n*n*CA_CHANNELS] with channels contiguous.
__host__ __device__ inline void rd_step_ref(const float* curr,
                                            const float* base,
                                            float* next,
                                            int n,
                                            const float* K,   // [N*N]
                                            const float* D) { // [N]
    for (int p = 0; p < n * n; ++p) {
        int y = p / n;
        int x = p % n;
        int yp = (y + 1) % n, ym = (y + n - 1) % n;
        int xp = (x + 1) % n, xm = (x + n - 1) % n;
        const int np[4] = {yp * n + x, ym * n + x, y * n + xp, y * n + xm};
        for (int c = 0; c < CA_CHANNELS; ++c) {
            next[p * CA_CHANNELS + c] = base[p * CA_CHANNELS + c];
        }
        for (int c = 0; c < RD_CHEM_N; ++c) {
            float here = curr[p * CA_CHANNELS + c];
            float lap = -4.f * here;
            for (int t = 0; t < 4; ++t) {
                lap += curr[np[t] * CA_CHANNELS + c];
            }
            float react = 0.f;
            for (int j = 0; j < RD_CHEM_N; ++j) {
                react += K[c * RD_CHEM_N + j] * curr[p * CA_CHANNELS + j];
            }
            float updated = base[p * CA_CHANNELS + c]
                + RD_DT * (D[c] * lap + react - RD_DECAY * here);
            if (updated >  FP16_MAX_VALUE) updated =  FP16_MAX_VALUE;
            if (updated < -FP16_MAX_VALUE) updated = -FP16_MAX_VALUE;
            next[p * CA_CHANNELS + c] = updated;
        }
    }
}

// Adjoint. dK/dD are accumulated (callers zero them once); d_curr is
// accumulated into (the CA adjoint adds its own terms there too).
__host__ __device__ inline void rd_adjoint_ref(const float* curr,
                                               const float* base,
                                               const float* d_next,
                                               int n,
                                               const float* K,
                                               const float* D,
                                               float* dK,      // [N*N]
                                               float* dD,      // [N]
                                               float* d_curr) {
    for (int p = 0; p < n * n; ++p) {
        int y = p / n;
        int x = p % n;
        int yp = (y + 1) % n, ym = (y + n - 1) % n;
        int xp = (x + 1) % n, xm = (x + n - 1) % n;
        const int np[4] = {yp * n + x, ym * n + x, y * n + xp, y * n + xm};
        for (int c = 0; c < RD_CHEM_N; ++c) {
            float here = curr[p * CA_CHANNELS + c];
            float lap = -4.f * here;
            for (int t = 0; t < 4; ++t) {
                lap += curr[np[t] * CA_CHANNELS + c];
            }
            float react = 0.f;
            for (int j = 0; j < RD_CHEM_N; ++j) {
                react += K[c * RD_CHEM_N + j] * curr[p * CA_CHANNELS + j];
            }
            float updated = base[p * CA_CHANNELS + c]
                + RD_DT * (D[c] * lap + react - RD_DECAY * here);
            float g = d_next[p * CA_CHANNELS + c];
            // Match the forward's clamp exactly (strict inequalities).
            if (updated > FP16_MAX_VALUE) g = 0.f;
            if (updated < -FP16_MAX_VALUE) g = 0.f;

            // Reaction: K transposed; coefficient gradients.
            for (int j = 0; j < RD_CHEM_N; ++j) {
                d_curr[p * CA_CHANNELS + j] +=
                    RD_DT * g * K[c * RD_CHEM_N + j];
                dK[c * RD_CHEM_N + j] +=
                    RD_DT * g * curr[p * CA_CHANNELS + j];
            }
            // Decay and Laplacian self term.
            d_curr[p * CA_CHANNELS + c] +=
                -RD_DT * RD_DECAY * g - 4.f * RD_DT * D[c] * g;
            // Laplacian neighbor terms (symmetric gather).
            for (int t = 0; t < 4; ++t) {
                d_curr[np[t] * CA_CHANNELS + c] += RD_DT * D[c] * g;
            }
            dD[c] += RD_DT * lap * g;
        }
    }
}

}  // namespace slime::nca::rd

#endif  // COEVO_NCA_RD_ADJOINT_CUH
