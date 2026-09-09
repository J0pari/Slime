// Sheet A-401: Soft Quality-Diversity Archive (Role-Aware)
//
// Descriptor: final bmap_64. Weighted Euclidean distance with per-dimension
// inverse-variance EMA. RFF KDE projection W_rff shared across roles; the
// archive's mean RFF vector mu_archive is maintained per role. 20x20 PCA bins
// computed periodically on the union archive. Each bin has per-role
// capacity caps.
//
// Surprise-mediated fitness scaling drives role balance: rho = s_avg / s_target
// (s_avg = blended surprise averaged over the last 100 generations on probes,
// s_target = median calibrated over generations 200-700 once at bootstrap and
// frozen).
//   classifier_mult = 1 + 0.1 * max(0, 1 - rho)
//   predictor_mult  = 1 + 0.1 * max(0, rho - 1)
// Coefficient 0.1 matches lambda_audit.

#ifndef COEVO_ARCHIVE_SOFT_QD_ARCHIVE_CU
#define COEVO_ARCHIVE_SOFT_QD_ARCHIVE_CU

#include "../config/constants.cuh"
#include "../genome/codec.cu"

#include <cstdint>
#include <cuda_runtime.h>

namespace slime::archive {

struct ArchiveEntry {
    float    descriptor[BMAP_DIM];   // bmap_64
    float    rff_proj[BMAP_DIM];     // shared RFF projection
    float    fitness;                 // composed (see below)
    float    f_raw;
    float    f_sot;
    uint32_t lineage_id;
    uint32_t parent_id;
    uint32_t bin_x;
    uint32_t bin_y;
    int      generation;
    Role     role;
    bool     alive;
    genome::Genome genome;            // stored for spawn parent selection
};

// Per-bin per-role caps. Stored as a flat array of bins; replacement decisions
// for classifier and predictor slots are independent.
struct BinCaps {
    uint16_t cap_classifier;
    uint16_t cap_predictor;
    uint16_t count_classifier;
    uint16_t count_predictor;
};

// RFF (Random Fourier Features) projection for KDE novelty. The projection
// W_rff is a [BMAP_DIM x BMAP_DIM] random matrix drawn once at init and frozen.
// rff(x) = cos(W_rff * x + bias), producing a BMAP_DIM-d feature vector whose
// inner product approximates a Gaussian kernel. Novelty for a candidate is the
// distance between its RFF features and the per-role mean RFF vector.
constexpr int RFF_DIM = BMAP_DIM;  // RFF output dimension matches descriptor dim

struct RffProjection {
    float W[RFF_DIM * BMAP_DIM];
    float bias[RFF_DIM];
};

struct Archive {
    ArchiveEntry entries[MAX_ARCHIVE];
    BinCaps      bins[ARCHIVE_BINS_X * ARCHIVE_BINS_Y];
    RffProjection rff;

    // Per-role RFF means (role-internal novelty).
    float mu_rff_classifier[RFF_DIM];
    float mu_rff_predictor[RFF_DIM];

    // Per-dimension inverse-variance EMA used for weighted Euclidean distance.
    float inv_var_ema[BMAP_DIM];

    // Per-role lineage runaway tracking (S-003).
    // Hooks in via lineage stats elsewhere; archive reports per-role share.
    uint32_t count_classifier;
    uint32_t count_predictor;

    // Per-role live index lists for O(1) parent selection (section 9).
    int alive_classifier_idx[MAX_ARCHIVE];
    int alive_predictor_idx[MAX_ARCHIVE];
    int n_alive_classifier;
    int n_alive_predictor;

    // PCA state for bin assignment between rebins (section 9.1).
    float pc[2][BMAP_DIM];      // PCA vectors from last rebin
    float pc_min[2];            // projection extents
    float pc_max[2];
    float pc_mean[BMAP_DIM];   // centering mean from last rebin
    bool  pca_valid;            // false until first recompute_bins
};

// ---- RFF projection and novelty -------------------------------------------

// Initialize the RFF projection matrix from a seed. Called once at world init.
// Uses a local PCG32 seeded from the provided seed (consistent with G-100).
__host__ inline void init_rff(RffProjection* rff, uint32_t seed) {
    Pcg32 rng;
    pcg32_seed(&rng, static_cast<uint64_t>(seed), 0x4FF01u);
    // Bandwidth: sigma = 1.0 for normalized descriptors.
    constexpr float BANDWIDTH_INV = 1.0f;
    for (int i = 0; i < RFF_DIM * BMAP_DIM; ++i) {
        float u1 = pcg32_float(&rng);
        float u2 = pcg32_float(&rng);
        // Box-Muller for proper Gaussian draws in the projection.
        float z = sqrtf(-2.0f * logf(u1 + 1e-30f)) * cosf(6.2831853f * u2);
        rff->W[i] = z * BANDWIDTH_INV;
    }
    for (int i = 0; i < RFF_DIM; ++i) {
        float u = pcg32_float(&rng);
        rff->bias[i] = u * 6.2831853f;  // uniform in [0, 2*pi)
    }
}

// Project a descriptor through the RFF: out[j] = cos(W[j,:] * desc + bias[j]).
__host__ inline void rff_project(const RffProjection& rff,
                                            const float* desc,
                                            float* out) {
    for (int j = 0; j < RFF_DIM; ++j) {
        float dot = rff.bias[j];
        for (int d = 0; d < BMAP_DIM; ++d) {
            dot += rff.W[j * BMAP_DIM + d] * desc[d];
        }
        out[j] = cosf(dot);
    }
}

// Novelty: squared L2 distance between an entry's RFF features and the
// per-role mean. Higher = more novel.
__host__ inline float rff_novelty(const float* rff_features,
                                             const float* mu_rff_role) {
    float dist2 = 0.f;
    for (int j = 0; j < RFF_DIM; ++j) {
        float d = rff_features[j] - mu_rff_role[j];
        dist2 += d * d;
    }
    return dist2;
}

// Update the per-role mean RFF vector with a new entry (online mean).
__host__ inline void update_rff_mean(float* mu, uint32_t count,
                                                const float* rff_features) {
    if (count == 0) {
        for (int j = 0; j < RFF_DIM; ++j) mu[j] = rff_features[j];
        return;
    }
    float alpha = 1.0f / static_cast<float>(count + 1);
    for (int j = 0; j < RFF_DIM; ++j) {
        mu[j] += alpha * (rff_features[j] - mu[j]);
    }
}

// ---- Public archive ops --------------------------------------------------

__host__ inline int archive_size(const Archive& a) {
    return static_cast<int>(a.count_classifier + a.count_predictor);
}

// True once the union archive crosses MAX_ARCHIVE / 2. Used to fire the
// one-shot predictor-founder spawn (A-601).
__host__ inline bool bootstrap_trigger(const Archive& a) {
    return archive_size(a) >= ARCHIVE_HALF;
}

// Weighted Euclidean distance using the per-dim inverse-variance EMA.
__host__ inline float weighted_dist2(const float* a,
                                                const float* b,
                                                const float* inv_var) {
    float acc = 0.f;
    for (int d = 0; d < BMAP_DIM; ++d) {
        float diff = a[d] - b[d];
        acc += diff * diff * inv_var[d];
    }
    return acc;
}

// Soft-QD score: combines quality (fitness) and diversity (RFF novelty).
// The novelty term lets a less-fit but more-novel candidate displace a crowded
// incumbent. Lambda_novelty = 0.5 balances the two terms; the RFF novelty is
// normalized by RFF_DIM to keep it scale-comparable with fitness in [0,1].
constexpr float LAMBDA_NOVELTY = 0.5f;

__host__ inline float qd_score(float fitness, float novelty) {
    return fitness + LAMBDA_NOVELTY * novelty / static_cast<float>(RFF_DIM);
}

// Assign a bin to a descriptor. If PCA is valid, project onto stored PCA
// vectors and extents. Otherwise use hash-based fallback (section 9.1).
__host__ inline void assign_bin(const Archive& a, const float* descriptor,
                                uint32_t& bin_x, uint32_t& bin_y) {
    if (a.pca_valid) {
        float p[2] = {0.f, 0.f};
        for (int d = 0; d < BMAP_DIM; ++d) {
            float centered = descriptor[d] - a.pc_mean[d];
            p[0] += centered * a.pc[0][d];
            p[1] += centered * a.pc[1][d];
        }
        for (int axis = 0; axis < 2; ++axis) {
            float range = a.pc_max[axis] - a.pc_min[axis];
            if (range < 1e-12f) range = 1.0f;
            int bins_n = (axis == 0) ? ARCHIVE_BINS_X : ARCHIVE_BINS_Y;
            int b = static_cast<int>((p[axis] - a.pc_min[axis]) / range
                                     * (bins_n - 1) + 0.5f);
            if (b < 0) b = 0;
            if (b >= bins_n) b = bins_n - 1;
            if (axis == 0) bin_x = static_cast<uint32_t>(b);
            else           bin_y = static_cast<uint32_t>(b);
        }
    } else {
        // Hash-based fallback: use first two descriptor dims.
        uint32_t h0 = static_cast<uint32_t>(fabsf(descriptor[0]) * 1e6f);
        uint32_t h1 = static_cast<uint32_t>(fabsf(descriptor[1]) * 1e6f);
        bin_x = h0 % ARCHIVE_BINS_X;
        bin_y = h1 % ARCHIVE_BINS_Y;
    }
}

// ---- Live index list helpers (internal) -----------------------------------

// Add an entry index to the appropriate per-role live list.
__host__ inline void live_list_add(Archive* a, int idx, Role role) {
    if (role == Role::Classifier) {
        a->alive_classifier_idx[a->n_alive_classifier++] = idx;
    } else {
        a->alive_predictor_idx[a->n_alive_predictor++] = idx;
    }
}

// Remove an entry index from the appropriate per-role live list (swap-remove).
__host__ inline void live_list_remove(Archive* a, int idx, Role role) {
    int* list;
    int* count;
    if (role == Role::Classifier) {
        list = a->alive_classifier_idx;
        count = &a->n_alive_classifier;
    } else {
        list = a->alive_predictor_idx;
        count = &a->n_alive_predictor;
    }
    for (int i = 0; i < *count; ++i) {
        if (list[i] == idx) {
            list[i] = list[*count - 1];
            (*count)--;
            return;
        }
    }
}

// Soft-QD insertion with novelty-weighted replacement. Returns the index of
// the slot that received the candidate, or -1 if rejected.
//
// The candidate's RFF features must be pre-computed in cand.rff_proj. Novelty
// is computed against the per-role mean RFF vector. The combined QD score
// (fitness + lambda * novelty) determines both whether the candidate enters
// and which incumbent it displaces.
__host__ inline int insert(Archive* a, const ArchiveEntry& cand) {
    int b = cand.bin_x * ARCHIVE_BINS_Y + cand.bin_y;
    BinCaps& caps = a->bins[b];
    uint16_t cap   = (cand.role == Role::Classifier) ? caps.cap_classifier
                                                     : caps.cap_predictor;
    uint16_t& cnt  = (cand.role == Role::Classifier) ? caps.count_classifier
                                                     : caps.count_predictor;

    // Compute candidate novelty against per-role mean.
    const float* mu_role = (cand.role == Role::Classifier) ? a->mu_rff_classifier
                                                           : a->mu_rff_predictor;
    float cand_novelty = rff_novelty(cand.rff_proj, mu_role);
    float cand_qd = qd_score(cand.fitness, cand_novelty);

    // If the bin/role has capacity, find an empty slot.
    if (cnt < cap) {
        for (int i = 0; i < MAX_ARCHIVE; ++i) {
            if (!a->entries[i].alive) {
                a->entries[i] = cand;
                a->entries[i].alive = true;
                cnt++;
                live_list_add(a, i, cand.role);
                if (cand.role == Role::Classifier) {
                    update_rff_mean(a->mu_rff_classifier, a->count_classifier,
                                    cand.rff_proj);
                    a->count_classifier++;
                } else {
                    update_rff_mean(a->mu_rff_predictor, a->count_predictor,
                                    cand.rff_proj);
                    a->count_predictor++;
                }
                return i;
            }
        }
    }

    // Replace the weakest same-role same-bin occupant by QD score.
    int worst_idx  = -1;
    float worst_qd = cand_qd;
    for (int i = 0; i < MAX_ARCHIVE; ++i) {
        const ArchiveEntry& e = a->entries[i];
        if (!e.alive) continue;
        if (e.role != cand.role) continue;
        if (e.bin_x != cand.bin_x || e.bin_y != cand.bin_y) continue;
        float e_novelty = rff_novelty(e.rff_proj, mu_role);
        float e_qd = qd_score(e.fitness, e_novelty);
        if (e_qd < worst_qd) {
            worst_qd = e_qd;
            worst_idx = i;
        }
    }
    if (worst_idx >= 0) {
        // Evict incumbent from live list, replace with candidate.
        live_list_remove(a, worst_idx, cand.role);
        a->entries[worst_idx] = cand;
        a->entries[worst_idx].alive = true;
        live_list_add(a, worst_idx, cand.role);
        // RFF mean is not recomputed on replacement (would require removing the
        // evicted entry's contribution). The periodic recompute_bins call
        // rebuilds the mean from scratch.
        return worst_idx;
    }
    return -1;
}

// Compose fitness with role multiplier, audit multiplier, and variance
// multiplier. See blueprint A-401 fitness composition:
//   f = f_raw * role_mult * audit_mult * variance_mult
__host__ inline float compose_fitness(float f_raw,
                                                 float role_mult,
                                                 float audit_mult,
                                                 float variance_mult) {
    return f_raw * role_mult * audit_mult * variance_mult;
}

// Surprise ratio rho = s_avg / s_target. Guards against an uncalibrated
// s_target (returns 1, which makes both role multipliers idle at 1.0).
__host__ inline float surprise_ratio(float s_avg, float s_target) {
    if (s_target <= 1e-12f) return 1.0f;
    return s_avg / s_target;
}

// Surprise-mediated role multipliers. rho = s_avg / s_target. Coefficient is
// ROLE_BALANCE_COEFF (= 0.1, matches lambda_audit).
__host__ inline float classifier_mult(float rho) {
    float gap = 1.0f - rho;
    if (gap < 0.f) gap = 0.f;
    return 1.0f + ROLE_BALANCE_COEFF * gap;
}

__host__ inline float predictor_mult(float rho) {
    float gap = rho - 1.0f;
    if (gap < 0.f) gap = 0.f;
    return 1.0f + ROLE_BALANCE_COEFF * gap;
}

// SOT gate: sigmoid(20 * (x - 0.7)).
__host__ inline float sot_gate(float x) {
    float z = SOT_GATE_SLOPE * (x - SOT_GATE_MIDPOINT);
    // numerically stable sigmoid
    if (z >= 0.f) {
        float ez = expf(-z);
        return 1.0f / (1.0f + ez);
    } else {
        float ez = expf(z);
        return ez / (1.0f + ez);
    }
}

// Refit the 20x20 PCA binning over the union archive.
// Runs on host (not on the per-generation hot path). Steps:
//   1. Gather alive descriptors, compute mean.
//   2. Power iteration on the 32x32 covariance for top-2 PCs.
//   3. Project onto (PC0, PC1), compute extents, assign bins.
//   4. Recount per-role per-bin occupancy.
inline void recompute_bins(Archive* a, cudaStream_t /*stream*/) {
    // Step 1: gather alive descriptors and compute mean.
    float mean[BMAP_DIM] = {};
    int n_alive = 0;
    for (int i = 0; i < MAX_ARCHIVE; ++i) {
        if (!a->entries[i].alive) continue;
        n_alive++;
        for (int d = 0; d < BMAP_DIM; ++d)
            mean[d] += a->entries[i].descriptor[d];
    }
    if (n_alive < 2) return;  // need at least 2 points for PCA
    float inv_n = 1.0f / static_cast<float>(n_alive);
    for (int d = 0; d < BMAP_DIM; ++d) mean[d] *= inv_n;

    // Step 2: power iteration for top-2 PCs on the BMAP_DIM x BMAP_DIM covariance.
    // We never form the covariance explicitly; instead compute C*v = sum_i (x_i^T v) x_i.
    constexpr int POWER_ITERS = 20;
    float pc[2][BMAP_DIM];

    // Initialize PC0 to (1,0,0,...).
    for (int d = 0; d < BMAP_DIM; ++d) pc[0][d] = (d == 0) ? 1.0f : 0.0f;

    for (int iter = 0; iter < POWER_ITERS; ++iter) {
        float new_v[BMAP_DIM] = {};
        for (int i = 0; i < MAX_ARCHIVE; ++i) {
            if (!a->entries[i].alive) continue;
            float dot = 0.f;
            for (int d = 0; d < BMAP_DIM; ++d)
                dot += (a->entries[i].descriptor[d] - mean[d]) * pc[0][d];
            for (int d = 0; d < BMAP_DIM; ++d)
                new_v[d] += dot * (a->entries[i].descriptor[d] - mean[d]);
        }
        float norm = 0.f;
        for (int d = 0; d < BMAP_DIM; ++d) norm += new_v[d] * new_v[d];
        norm = sqrtf(norm + 1e-30f);
        for (int d = 0; d < BMAP_DIM; ++d) pc[0][d] = new_v[d] / norm;
    }

    // PC1: deflate and power iterate.
    for (int d = 0; d < BMAP_DIM; ++d) pc[1][d] = (d == 1) ? 1.0f : 0.0f;
    for (int iter = 0; iter < POWER_ITERS; ++iter) {
        float new_v[BMAP_DIM] = {};
        for (int i = 0; i < MAX_ARCHIVE; ++i) {
            if (!a->entries[i].alive) continue;
            float dot = 0.f;
            for (int d = 0; d < BMAP_DIM; ++d)
                dot += (a->entries[i].descriptor[d] - mean[d]) * pc[1][d];
            for (int d = 0; d < BMAP_DIM; ++d)
                new_v[d] += dot * (a->entries[i].descriptor[d] - mean[d]);
        }
        // Gram-Schmidt: remove PC0 component.
        float proj0 = 0.f;
        for (int d = 0; d < BMAP_DIM; ++d) proj0 += new_v[d] * pc[0][d];
        for (int d = 0; d < BMAP_DIM; ++d) new_v[d] -= proj0 * pc[0][d];
        float norm = 0.f;
        for (int d = 0; d < BMAP_DIM; ++d) norm += new_v[d] * new_v[d];
        norm = sqrtf(norm + 1e-30f);
        for (int d = 0; d < BMAP_DIM; ++d) pc[1][d] = new_v[d] / norm;
    }

    // Store PCA state for assign_bin between rebins (section 9.1).
    for (int d = 0; d < BMAP_DIM; ++d) {
        a->pc[0][d] = pc[0][d];
        a->pc[1][d] = pc[1][d];
        a->pc_mean[d] = mean[d];
    }

    // Step 3: project every entry onto (PC0, PC1), find extents, assign bins.
    float min0 = 1e30f, max0 = -1e30f, min1 = 1e30f, max1 = -1e30f;
    for (int i = 0; i < MAX_ARCHIVE; ++i) {
        if (!a->entries[i].alive) continue;
        float p0 = 0.f, p1 = 0.f;
        for (int d = 0; d < BMAP_DIM; ++d) {
            float centered = a->entries[i].descriptor[d] - mean[d];
            p0 += centered * pc[0][d];
            p1 += centered * pc[1][d];
        }
        if (p0 < min0) min0 = p0;
        if (p0 > max0) max0 = p0;
        if (p1 < min1) min1 = p1;
        if (p1 > max1) max1 = p1;
    }
    float range0 = max0 - min0;
    float range1 = max1 - min1;
    if (range0 < 1e-12f) range0 = 1.0f;
    if (range1 < 1e-12f) range1 = 1.0f;

    // Store extents for assign_bin.
    a->pc_min[0] = min0; a->pc_max[0] = max0;
    a->pc_min[1] = min1; a->pc_max[1] = max1;
    a->pca_valid = true;

    // Reset bin counts.
    for (int b = 0; b < ARCHIVE_BINS_X * ARCHIVE_BINS_Y; ++b) {
        a->bins[b].count_classifier = 0;
        a->bins[b].count_predictor  = 0;
    }

    // Rebuild live index lists from scratch during rebin.
    a->n_alive_classifier = 0;
    a->n_alive_predictor = 0;

    // Assign bins and recount.
    for (int i = 0; i < MAX_ARCHIVE; ++i) {
        if (!a->entries[i].alive) continue;
        float p0 = 0.f, p1 = 0.f;
        for (int d = 0; d < BMAP_DIM; ++d) {
            float centered = a->entries[i].descriptor[d] - mean[d];
            p0 += centered * pc[0][d];
            p1 += centered * pc[1][d];
        }
        int bx = static_cast<int>((p0 - min0) / range0 * (ARCHIVE_BINS_X - 1) + 0.5f);
        int by = static_cast<int>((p1 - min1) / range1 * (ARCHIVE_BINS_Y - 1) + 0.5f);
        if (bx < 0) bx = 0;
        if (bx >= ARCHIVE_BINS_X) bx = ARCHIVE_BINS_X - 1;
        if (by < 0) by = 0;
        if (by >= ARCHIVE_BINS_Y) by = ARCHIVE_BINS_Y - 1;
        a->entries[i].bin_x = static_cast<uint32_t>(bx);
        a->entries[i].bin_y = static_cast<uint32_t>(by);
        int b = bx * ARCHIVE_BINS_Y + by;
        if (a->entries[i].role == Role::Classifier) {
            a->bins[b].count_classifier++;
            a->alive_classifier_idx[a->n_alive_classifier++] = i;
        } else {
            a->bins[b].count_predictor++;
            a->alive_predictor_idx[a->n_alive_predictor++] = i;
        }
    }
}

// Apply lineage brake: scale the effective fitness of entries belonging to a
// runaway lineage, making them easier to displace. The brake factor is
// proportional to how far over threshold the lineage sits. Per-role and
// independent. Idempotent: recomputes the factor from current share each call.
//
// share_fraction: the lineage's fraction of the role's archive slots (0..1).
// threshold: the maximum acceptable share (e.g. 0.2 = 20%). Entries above
// threshold get their fitness scaled by (threshold / share_fraction), so a
// lineage at 2x the threshold has its fitness halved.
__host__ inline void apply_lineage_brake(Archive* a,
                                                    Role role,
                                                    uint32_t runaway_lineage_id,
                                                    float share_fraction,
                                                    float threshold) {
    if (share_fraction <= threshold) return;  // no braking needed
    float brake = threshold / share_fraction;
    if (brake < 0.1f) brake = 0.1f;  // floor to prevent zeroing

    for (int i = 0; i < MAX_ARCHIVE; ++i) {
        ArchiveEntry& e = a->entries[i];
        if (!e.alive) continue;
        if (e.role != role) continue;
        if (e.lineage_id != runaway_lineage_id) continue;
        e.fitness *= brake;
    }
}

}  // namespace slime::archive

#endif  // COEVO_ARCHIVE_SOFT_QD_ARCHIVE_CU
