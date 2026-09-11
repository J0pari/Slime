// Sheet S-004: Parallel Tempering Ladder — pure host logic
//
// The mutation-rate ladder state and its ring-buffer / Metropolis math live
// here so the production loop (safety/parallel_tempering.cu) and the host unit
// test suite execute the same functions. Everything in this header is pure
// host logic with no CUDA runtime dependency beyond the constants header.

#ifndef COEVO_SAFETY_PT_LADDER_CUH
#define COEVO_SAFETY_PT_LADDER_CUH

#include "../config/constants.cuh"

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
#ifdef __CUDA_ARCH__
__device__ inline float mutation_rate_for(const MutationLadder& l, int idx) {
    return d_PT_MUTATION_RATES[l.replica_of[idx]];
}
#else
__host__ inline float mutation_rate_for(const MutationLadder& l, int idx) {
    return PT_MUTATION_RATES[l.replica_of[idx]];
}
#endif

// Fitness improvement RATE per replica over the prior PT_SWAP_INTERVAL
// generations: delta between most-recent best and oldest best, normalised by
// the window length. Used for the swap criterion (S-004).
//
// Ring convention: record_best_fitness writes at history_head and then
// advances it, so history_head points at the next write slot, which is also
// the oldest entry once the ring is full. latest = last written entry
// (head - 1 mod interval); oldest = next-to-be-overwritten entry (head).
__host__ __device__ inline float improvement_rate(const MutationLadder& l,
                                                  int replica) {
    int head = l.history_head;
    int oldest = head % PT_SWAP_INTERVAL;
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
// Called once per swap round after propose_swaps tallies that round's
// attempts/accepts. Resets per-round counters.
__host__ __device__ inline void update_beta(MutationLadder* l, float ema_rate = 0.2f) {
    int attempted = l->swaps_attempted;
    if (attempted <= 0) return;
    float round_rate = static_cast<float>(l->swaps_accepted)
                     / static_cast<float>(attempted);
    l->accept_ema = (1.0f - ema_rate) * l->accept_ema + ema_rate * round_rate;
    float err = l->accept_ema - PT_TARGET_ACCEPT;
    l->beta *= expf(0.5f * err);
    if (l->beta < 1e-3f) l->beta = 1e-3f;
    if (l->beta > 1e3f)  l->beta = 1e3f;
    l->swaps_attempted = 0;
    l->swaps_accepted  = 0;
}

// For each replica, scan its member slots, take the max fitness, and write it
// into best_fitness_history[replica][history_head]; then advance history_head.
// Called once per generation.
inline void record_best_fitness(MutationLadder* l,
                                const float* organism_fitness) {
    for (int r = 0; r < PT_NUM_REPLICAS; ++r) {
        float best = -1e30f;
        for (int i = 0; i < POOL_SIZE; ++i) {
            if (l->replica_of[i] == r && organism_fitness[i] > best) {
                best = organism_fitness[i];
            }
        }
        l->best_fitness_history[r][l->history_head] = best;
    }
    l->history_head = (l->history_head + 1) % PT_SWAP_INTERVAL;
}

}  // namespace slime::safety::pt

#endif  // COEVO_SAFETY_PT_LADDER_CUH
