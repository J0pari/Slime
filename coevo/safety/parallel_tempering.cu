// Sheet S-004: Parallel Tempering Ladders
//
// Per cuda_engineering.md section 13 and blueprint S-004.
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
// beta is adapted via EMA on accept rate targeting 0.25.
//
// On acceptance of a swap between replicas R_lo and R_hi, ALL organisms in
// R_lo exchange pool slots with ALL organisms in R_hi. This is a full data
// swap: OrganismState (device), CheckpointBuffer (device), GradBuffers
// (device), and OrganismTable rows (host). The pool slot's replica_tag stays
// fixed to the slot — it identifies the temperature, not the organism.
//
// SOT-density stress ladder. Independent of the main pool. Three stress
// sub-populations of 8 organisms each at SOT densities {10%, 20%, 40%}.

#ifndef COEVO_SAFETY_PARALLEL_TEMPERING_CU
#define COEVO_SAFETY_PARALLEL_TEMPERING_CU

#include "../config/constants.cuh"
#include "../nca/engine.cu"
#include "../autodiff/warp_tape.cu"
#include "../genome/codec.cu"

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

namespace slime::safety::pt {

using nca::OrganismState;
using nca::ForwardInputs;
using autodiff::CheckpointBuffer;
using autodiff::GradBuffers;

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

// ---- SwapContext -----------------------------------------------------------
// Holds all pointers needed for full organism data swap. Passed to
// propose_swaps instead of a World* to avoid circular header dependency.
// The integration layer constructs this from the World struct.
struct SwapContext {
    // Device pointers for organism data swap.
    OrganismState*    d_organisms;
    CheckpointBuffer* d_checkpoints;
    GradBuffers*      d_grads;
    // Temp buffers for device-side swap (pre-allocated in World).
    OrganismState*    d_swap_org;
    CheckpointBuffer* d_swap_ckpt;
    GradBuffers*      d_swap_grad;
    // Host-side organism table arrays for row swaps.
    genome::Genome*       genomes;
    genome::DeltaWeights* deltas;
    uint32_t*             lineage_id;
    uint32_t*             parent_id;
    int*                  spawn_gen;
    float*                fitness;
    float*                f_raw;
    float*                f_sot;
    Role*                 role;
    // Stream for device memcpy.
    cudaStream_t stream;
};

// Swap device data for two pool-slot organisms through temp buffer.
// A ↔ B via: temp = A; A = B; B = temp.
static inline void swap_device_organism(SwapContext& ctx, int slot_a, int slot_b) {
    // OrganismState swap.
    cudaMemcpyAsync(ctx.d_swap_org, &ctx.d_organisms[slot_a],
                    sizeof(OrganismState), cudaMemcpyDeviceToDevice, ctx.stream);
    cudaMemcpyAsync(&ctx.d_organisms[slot_a], &ctx.d_organisms[slot_b],
                    sizeof(OrganismState), cudaMemcpyDeviceToDevice, ctx.stream);
    cudaMemcpyAsync(&ctx.d_organisms[slot_b], ctx.d_swap_org,
                    sizeof(OrganismState), cudaMemcpyDeviceToDevice, ctx.stream);

    // CheckpointBuffer swap.
    cudaMemcpyAsync(ctx.d_swap_ckpt, &ctx.d_checkpoints[slot_a],
                    sizeof(CheckpointBuffer), cudaMemcpyDeviceToDevice, ctx.stream);
    cudaMemcpyAsync(&ctx.d_checkpoints[slot_a], &ctx.d_checkpoints[slot_b],
                    sizeof(CheckpointBuffer), cudaMemcpyDeviceToDevice, ctx.stream);
    cudaMemcpyAsync(&ctx.d_checkpoints[slot_b], ctx.d_swap_ckpt,
                    sizeof(CheckpointBuffer), cudaMemcpyDeviceToDevice, ctx.stream);

    // GradBuffers swap.
    cudaMemcpyAsync(ctx.d_swap_grad, &ctx.d_grads[slot_a],
                    sizeof(GradBuffers), cudaMemcpyDeviceToDevice, ctx.stream);
    cudaMemcpyAsync(&ctx.d_grads[slot_a], &ctx.d_grads[slot_b],
                    sizeof(GradBuffers), cudaMemcpyDeviceToDevice, ctx.stream);
    cudaMemcpyAsync(&ctx.d_grads[slot_b], ctx.d_swap_grad,
                    sizeof(GradBuffers), cudaMemcpyDeviceToDevice, ctx.stream);
}

// Swap host-side OrganismTable row data between two pool slots.
static inline void swap_host_organism(SwapContext& ctx, int slot_a, int slot_b) {
    // Genome.
    genome::Genome tmp_genome = ctx.genomes[slot_a];
    ctx.genomes[slot_a] = ctx.genomes[slot_b];
    ctx.genomes[slot_b] = tmp_genome;

    // DeltaWeights.
    genome::DeltaWeights tmp_delta = ctx.deltas[slot_a];
    ctx.deltas[slot_a] = ctx.deltas[slot_b];
    ctx.deltas[slot_b] = tmp_delta;

    // Scalars.
    {
        uint32_t t;
        t = ctx.lineage_id[slot_a]; ctx.lineage_id[slot_a] = ctx.lineage_id[slot_b]; ctx.lineage_id[slot_b] = t;
        t = ctx.parent_id[slot_a];  ctx.parent_id[slot_a]  = ctx.parent_id[slot_b];  ctx.parent_id[slot_b]  = t;
    }
    {
        int t = ctx.spawn_gen[slot_a]; ctx.spawn_gen[slot_a] = ctx.spawn_gen[slot_b]; ctx.spawn_gen[slot_b] = t;
    }
    {
        float t;
        t = ctx.fitness[slot_a]; ctx.fitness[slot_a] = ctx.fitness[slot_b]; ctx.fitness[slot_b] = t;
        t = ctx.f_raw[slot_a];   ctx.f_raw[slot_a]   = ctx.f_raw[slot_b];   ctx.f_raw[slot_b]   = t;
        t = ctx.f_sot[slot_a];   ctx.f_sot[slot_a]   = ctx.f_sot[slot_b];   ctx.f_sot[slot_b]   = t;
    }
    {
        Role t = ctx.role[slot_a]; ctx.role[slot_a] = ctx.role[slot_b]; ctx.role[slot_b] = t;
    }
}

// Every PT_SWAP_INTERVAL generations, for each adjacent replica pair: compute
// improvement_rate, draw Metropolis accept, swap ALL organisms between the two
// replicas on accept. Per section 13: full organism data swap. The pool slot's
// replica_tag stays fixed — it identifies the temperature, not the organism.
//
// Swap timing: before backward in the generation loop, so swapped organisms
// contribute gradients in their new replica context.
inline void propose_swaps(MutationLadder* l,
                          const float* organism_fitness,
                          Pcg32* rng,
                          SwapContext& ctx) {
    for (int pair = 0; pair < PT_NUM_REPLICAS - 1; ++pair) {
        int lo = pair;
        int hi = pair + 1;
        float rate_lo = improvement_rate(*l, lo);
        float rate_hi = improvement_rate(*l, hi);
        float p = swap_accept_probability(l->beta, rate_lo, rate_hi);
        l->swaps_attempted++;

        float u = pcg32_float(rng);

        if (u < p) {
            l->swaps_accepted++;

            // Collect all organism indices in each replica.
            int lo_slots[PT_REPLICA_SIZE];
            int hi_slots[PT_REPLICA_SIZE];
            int n_lo = 0, n_hi = 0;
            for (int i = 0; i < POOL_SIZE; ++i) {
                if (l->replica_of[i] == lo && n_lo < PT_REPLICA_SIZE) {
                    lo_slots[n_lo++] = i;
                }
                if (l->replica_of[i] == hi && n_hi < PT_REPLICA_SIZE) {
                    hi_slots[n_hi++] = i;
                }
            }

            // Swap paired organisms (by offset within replica).
            int n_pairs = (n_lo < n_hi) ? n_lo : n_hi;
            for (int k = 0; k < n_pairs; ++k) {
                int slot_a = lo_slots[k];
                int slot_b = hi_slots[k];

                // Device data swap (OrganismState + Checkpoint + Grads).
                swap_device_organism(ctx, slot_a, slot_b);

                // Host data swap (genome, delta, metadata).
                swap_host_organism(ctx, slot_a, slot_b);
            }

            // Sync device copies before proceeding to next pair.
            cudaStreamSynchronize(ctx.stream);
        }
    }
    update_beta(l);
}

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

    int      flagged_lineage_count;
};

// DECLARED ONLY — blueprint-in-place.
void refresh_stress_slots(StressLadder* l,
                          uint32_t* eligible_lineages,
                          int n_eligible,
                          int generation,
                          cudaStream_t stream);

// DECLARED ONLY — blueprint-in-place.
void evaluate_stress(StressLadder* l, cudaStream_t stream);

// DECLARED ONLY — blueprint-in-place.
void flag_stress_failures(StressLadder* l, cudaStream_t stream);

}  // namespace slime::safety::pt

#endif  // COEVO_SAFETY_PARALLEL_TEMPERING_CU
