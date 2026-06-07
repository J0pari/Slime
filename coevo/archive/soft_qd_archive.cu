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
};

// Per-bin per-role caps. Stored as a flat array of bins; replacement decisions
// for classifier and predictor slots are independent.
struct BinCaps {
    uint16_t cap_classifier;
    uint16_t cap_predictor;
    uint16_t count_classifier;
    uint16_t count_predictor;
};

struct Archive {
    ArchiveEntry entries[MAX_ARCHIVE];
    BinCaps      bins[ARCHIVE_BINS_X * ARCHIVE_BINS_Y];

    // Per-role RFF means (role-internal novelty).
    float mu_rff_classifier[BMAP_DIM];
    float mu_rff_predictor[BMAP_DIM];

    // Per-dimension inverse-variance EMA used for weighted Euclidean distance.
    float inv_var_ema[BMAP_DIM];

    // Per-role lineage runaway tracking (S-003).
    // Hooks in via lineage stats elsewhere; archive reports per-role share.
    uint32_t count_classifier;
    uint32_t count_predictor;
};

// ---- Public archive ops --------------------------------------------------

__host__ __device__ inline int archive_size(const Archive& a) {
    return static_cast<int>(a.count_classifier + a.count_predictor);
}

// True once the union archive crosses MAX_ARCHIVE / 2. Used to fire the
// one-shot predictor-founder spawn (A-601).
__host__ __device__ inline bool bootstrap_trigger(const Archive& a) {
    return archive_size(a) >= ARCHIVE_HALF;
}

// Weighted Euclidean distance using the per-dim inverse-variance EMA.
__host__ __device__ inline float weighted_dist2(const float* a,
                                                const float* b,
                                                const float* inv_var) {
    float acc = 0.f;
    for (int d = 0; d < BMAP_DIM; ++d) {
        float diff = a[d] - b[d];
        acc += diff * diff * inv_var[d];
    }
    return acc;
}

// Soft-QD insertion. Returns the index of the slot that received the
// candidate, or -1 if rejected.
//
// Implemented so far: per-role bin capacity, then a fitness-based replacement
// of the weakest same-role occupant of the candidate's PCA bin. This is the
// quality half of the soft-QD rule.
//
// NOT YET IMPLEMENTED (A-401): the diversity half. The full rule weights the replacement
// decision by role-internal novelty (RFF KDE against mu_classifier /
// mu_predictor), so a less-fit but more-novel candidate can still displace a
// crowded incumbent. weighted_dist2() and the per-role mu_rff vectors are
// staged for that term; until it lands, insertion is purely quality-ranked
// within a bin and the archive under-rewards novelty.
__host__ __device__ inline int insert(Archive* a, const ArchiveEntry& cand) {
    int b = cand.bin_x * ARCHIVE_BINS_Y + cand.bin_y;
    BinCaps& caps = a->bins[b];
    uint16_t cap   = (cand.role == Role::Classifier) ? caps.cap_classifier
                                                     : caps.cap_predictor;
    uint16_t& cnt  = (cand.role == Role::Classifier) ? caps.count_classifier
                                                     : caps.count_predictor;

    // If the bin/role has capacity, find an empty slot.
    if (cnt < cap) {
        for (int i = 0; i < MAX_ARCHIVE; ++i) {
            if (!a->entries[i].alive) {
                a->entries[i] = cand;
                a->entries[i].alive = true;
                cnt++;
                if (cand.role == Role::Classifier) a->count_classifier++;
                else                                a->count_predictor++;
                return i;
            }
        }
    }
    // Otherwise replace the weakest same-role same-bin occupant if the
    // candidate's fitness exceeds it.
    int worst_idx  = -1;
    float worst_f  = cand.fitness;
    for (int i = 0; i < MAX_ARCHIVE; ++i) {
        const ArchiveEntry& e = a->entries[i];
        if (!e.alive) continue;
        if (e.role != cand.role) continue;
        if (e.bin_x != cand.bin_x || e.bin_y != cand.bin_y) continue;
        if (e.fitness < worst_f) {
            worst_f = e.fitness;
            worst_idx = i;
        }
    }
    if (worst_idx >= 0) {
        a->entries[worst_idx] = cand;
        a->entries[worst_idx].alive = true;
        return worst_idx;
    }
    return -1;
}

// Compose fitness with role multiplier, audit multiplier, and variance
// multiplier. See blueprint A-401 fitness composition:
//   f = f_raw * role_mult * audit_mult * variance_mult
__host__ __device__ inline float compose_fitness(float f_raw,
                                                 float role_mult,
                                                 float audit_mult,
                                                 float variance_mult) {
    return f_raw * role_mult * audit_mult * variance_mult;
}

// Surprise ratio rho = s_avg / s_target. Guards against an uncalibrated
// s_target (returns 1, which makes both role multipliers idle at 1.0).
__host__ __device__ inline float surprise_ratio(float s_avg, float s_target) {
    if (s_target <= 1e-12f) return 1.0f;
    return s_avg / s_target;
}

// Surprise-mediated role multipliers. rho = s_avg / s_target. Coefficient is
// ROLE_BALANCE_COEFF (= 0.1, matches lambda_audit).
__host__ __device__ inline float classifier_mult(float rho) {
    float gap = 1.0f - rho;
    if (gap < 0.f) gap = 0.f;
    return 1.0f + ROLE_BALANCE_COEFF * gap;
}

__host__ __device__ inline float predictor_mult(float rho) {
    float gap = rho - 1.0f;
    if (gap < 0.f) gap = 0.f;
    return 1.0f + ROLE_BALANCE_COEFF * gap;
}

// SOT gate: sigmoid(20 * (x - 0.7)).
__host__ __device__ inline float sot_gate(float x) {
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

// DECLARED ONLY — blueprint-in-place.
// recompute_bins: refit the 20x20 PCA binning over the union archive.
//   1. Gather all alive descriptors (bmap_64) into a [N x BMAP_DIM] matrix.
//   2. Mean-center; compute the top-2 principal components (power iteration on
//      the 32x32 covariance is enough for 2 PCs and avoids a full eig).
//   3. Project every descriptor onto (PC0, PC1); take min/max per axis to fix
//      the bin grid extents; assign each entry bin_x, bin_y in [0,20).
//   4. Recount BinCaps.count_classifier / count_predictor from the new
//      assignments. Caps themselves are policy and set elsewhere.
//   Cadence: periodic (every few hundred generations); cost is O(N*BMAP_DIM)
//   per power-iteration step. Run on the host or a single block; it is not on
//   the per-generation hot path.
void recompute_bins(Archive* a, cudaStream_t stream);

// DECLARED ONLY — blueprint-in-place.
// apply_lineage_brake: when lineage `runaway_lineage_id` of `role` exceeds its
// per-role archive-share threshold (S-003), raise that role's replacement bar
// inside every bin the lineage occupies, so its members are displaced sooner
// and new members insert harder. Concretely: scale the effective fitness used
// in insert()'s weakest-occupant comparison by a brake factor < 1 for entries
// whose lineage_id matches, proportional to how far over threshold the lineage
// sits. Per-role and independent: a classifier brake never touches predictor
// replacement decisions. Must be idempotent across repeated calls in a
// generation (recompute the factor from current share, do not accumulate).
__device__ void apply_lineage_brake(Archive* a,
                                    Role role,
                                    uint32_t runaway_lineage_id);

}  // namespace slime::archive

#endif  // COEVO_ARCHIVE_SOFT_QD_ARCHIVE_CU
