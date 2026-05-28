// Sheet S-004: Parallel Tempering Ladders
//
// Two narrow PT applications, both role-blind, both independent.
//
// Mutation-rate ladder. The active pool of 64 organisms is partitioned into
// 4 replicas of 16 organisms each, with per-replica mutation rates
// {0.005, 0.01, 0.02, 0.04}. Replica assignment is preserved across
// generations until a swap occurs. Swap proposals every 50 generations
// between adjacent replicas. Swap criterion = fitness improvement RATE over
// the prior 50 generations (best organism per replica). Metropolis acceptance:
//
//   p_accept = min(1, exp(beta * (delta_high - delta_low)))
//
// beta is adapted via EMA on accept rate targeting 0.25 (removes the magic
// number). Swaps move ENTIRE organisms (checkpoint buffers + momentum follow).
//
// SOT-density stress ladder. Independent of the main pool. Three stress
// sub-populations of 8 organisms each at SOT densities {10%, 20%, 40%}.
// Each is role-balanced (4 classifiers + 4 predictors). Seeded each
// generation by sampling lineage representatives from the main pool with
// 25% slot refresh rate. Stress organisms do NOT compete in the main archive.
// A lineage whose stress reps fail the SOT gate > 50% over the last 10
// stress evaluations is flagged for operator review.

#ifndef SLIME_2_1_SAFETY_PARALLEL_TEMPERING_CU
#define SLIME_2_1_SAFETY_PARALLEL_TEMPERING_CU

#include "../config/constants.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace slime::safety::pt {

// ---- Mutation-rate ladder -----------------------------------------------
struct MutationLadder {
    // Per-organism replica id (0..3). Index is the active-pool slot.
    uint8_t replica_of[POOL_SIZE];

    // Per-replica best-fitness history for the rolling 50-gen rate.
    float best_fitness_history[PT_NUM_REPLICAS][PT_SWAP_INTERVAL];
    int   history_head;

    // Adapted beta and accept-rate EMA targeting PT_TARGET_ACCEPT.
    float beta;
    float accept_ema;
    int   swaps_attempted;
    int   swaps_accepted;
};

// Each organism's mutation rate when it spawns offspring.
__host__ __device__ inline float mutation_rate_for(const MutationLadder& l, int idx) {
    return PT_MUTATION_RATES[l.replica_of[idx]];
}

// Fitness improvement RATE per replica over the prior PT_SWAP_INTERVAL
// generations: delta between most-recent best and oldest best, normalised by
// the window length. Used for the swap criterion (S-004).
__host__ __device__ inline float improvement_rate(const MutationLadder& l,
                                                  int replica) {
    int head = l.history_head;
    int oldest = (head + 1) % PT_SWAP_INTERVAL;
    int latest = (head - 1 + PT_SWAP_INTERVAL) % PT_SWAP_INTERVAL;
    float dfit = l.best_fitness_history[replica][latest]
               - l.best_fitness_history[replica][oldest];
    return dfit / static_cast<float>(PT_SWAP_INTERVAL);
}

// Metropolis acceptance for a swap between adjacent replicas low (cooler)
// and high (hotter). Spec: p = min(1, exp(beta * (delta_high - delta_low))).
__host__ __device__ inline float swap_accept_probability(float beta,
                                                         float delta_low,
                                                         float delta_high) {
    float arg = beta * (delta_high - delta_low);
    if (arg >= 0.f) return 1.0f;
    return expf(arg);
}

// Adaptive beta EMA: nudge beta so the accept rate tracks PT_TARGET_ACCEPT.
// Higher accept rate -> too easy -> increase beta. Spec says this removes the
// "magic number" by adapting it.
__host__ __device__ inline void update_beta(MutationLadder* l, float ema_rate = 0.05f) {
    int attempted = l->swaps_attempted;
    if (attempted <= 0) return;
    float accept_rate = static_cast<float>(l->swaps_accepted)
                      / static_cast<float>(attempted);
    l->accept_ema = (1.0f - ema_rate) * l->accept_ema + ema_rate * accept_rate;
    float err = l->accept_ema - PT_TARGET_ACCEPT;
    // accept_ema > target  => beta too small (everything accepted) -> increase
    // accept_ema < target  => beta too large (rejects too much)    -> decrease
    l->beta *= expf(0.5f * err);
    if (l->beta < 1e-3f) l->beta = 1e-3f;
    if (l->beta > 1e3f)  l->beta = 1e3f;
}

// Update the rolling best-fitness slot for each replica each generation.
void record_best_fitness(MutationLadder* l,
                         const float* organism_fitness,
                         cudaStream_t stream);

// Propose swaps between adjacent replicas. Moves entire organisms
// (CA grid, delta weights, CAME state) so optimizer momentum follows the
// organism. Adapts beta to track PT_TARGET_ACCEPT.
void propose_swaps(MutationLadder* l, cudaStream_t stream);

// ---- SOT-density stress ladder ------------------------------------------
struct StressLadder {
    // STRESS_SUBPOP_COUNT * STRESS_SUBPOP_SIZE = 24 stress slots.
    uint32_t lineage_id[STRESS_POOL_SIZE];
    uint32_t source_pool_idx[STRESS_POOL_SIZE];   // back-pointer
    Role     role[STRESS_POOL_SIZE];
    uint8_t  subpop[STRESS_POOL_SIZE];            // 0,1,2 -> 10/20/40% SOT
    bool     sot_gate_pass[STRESS_POOL_SIZE];
    int      eval_count[STRESS_POOL_SIZE];
    int      last_refresh_gen[STRESS_POOL_SIZE];

    // Rolling failure window per lineage (S-004: > 50% failure over last 10
    // evals flags operator review).
    // External lineage table holds the actual ring buffer; this is a count.
    int      flagged_lineage_count;
};

// Seed/refresh slots. Sampling rate STRESS_REFRESH_FRACTION = 25%/gen.
// Sampling is biased toward lineages whose stress evaluations are outdated.
void refresh_stress_slots(StressLadder* l,
                          uint32_t* eligible_lineages,
                          int n_eligible,
                          int generation,
                          cudaStream_t stream);

// One evaluation cycle: run each stress organism on a batch at its
// subpop-assigned SOT density. Update sot_gate_pass + eval_count.
void evaluate_stress(StressLadder* l, cudaStream_t stream);

// Compute per-lineage failure rate over the last STRESS_HISTORY_WINDOW
// evaluations; emit a flag (no automatic pruning - operator review only).
void flag_stress_failures(StressLadder* l, cudaStream_t stream);

// Interaction note from the spec: organisms are not simultaneously assigned
// mutation-rate replica AND stress-replica positions. Stress evaluations
// sample from the main pool across all mutation-rate replicas using only
// lineage_id.

}  // namespace slime::safety::pt

#endif  // SLIME_2_1_SAFETY_PARALLEL_TEMPERING_CU
