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
#include "../config/strong_ids.cuh"
#include "../nca/engine.cu"
#include "../autodiff/warp_tape.cu"
#include "../genome/codec.cu"
#include "pt_ladder.cuh"
#include "stress_ladder.cuh"

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

namespace slime::safety::pt {

using nca::OrganismState;
using nca::ForwardInputs;
using autodiff::CheckpointBuffer;
using autodiff::GradBuffers;

// MutationLadder, improvement_rate, swap_accept_probability, update_beta, and
// record_best_fitness live in pt_ladder.cuh (pure host logic shared with the
// host unit tests). This file adds the transactional swap machinery.

// ---- SwapContext -----------------------------------------------------------
// Holds all pointers needed for full organism data swap. Passed to
// propose_swaps instead of a World* to avoid circular header dependency.
// The integration layer constructs this from the World struct.
struct SwapContext {
    // Device pointers for organism data swap.
    OrganismState*    d_organisms;
    CheckpointBuffer* d_checkpoints;
    GradBuffers*      d_grads;
    // Per-organism effective weight banks (W_shared + genome delta). These
    // must move with the organism: backward reconstructs each checkpoint with
    // the organism's own bank, so a swap that leaves the bank behind would
    // differentiate trajectory B with phenotype A.
    float*            d_eff_weights;   // [pool_size * TOTAL_WEIGHTS]
    // Temp buffers for device-side swap (pre-allocated in World).
    OrganismState*    d_swap_org;
    CheckpointBuffer* d_swap_ckpt;
    GradBuffers*      d_swap_grad;
    float*            d_swap_wbank;    // [TOTAL_WEIGHTS]
    // Host-side organism table arrays for row swaps.
    genome::Genome*       genomes;
    genome::DeltaWeights* deltas;
    LineageId*            lineage_id;
    ArchiveSlot*          parent_id;
    int*                  spawn_gen;
    float*                fitness;
    float*                f_raw;
    float*                f_sot;
    Role*                 role;
    // Evaluation correlation state. The seed gradient rows and batch sample
    // assignment belong to the logical organism: they must move with it or
    // backward will differentiate one trajectory using another organism's
    // objective (A-501 transaction invariant).
    float*                seed_grad;        // [pool_size * BMAP_DIM] host rows
    int*                  batch_sample_idx; // [pool_size] host
    float*                predictor_error_ema; // [pool_size] host
    float*                predictor_loss_ema;  // [pool_size] host
    // Stream for device memcpy.
    cudaStream_t stream;
};

// Swap device data for two pool-slot organisms through temp buffer.
// A ↔ B via: temp = A; A = B; B = temp. Returns false (with a report) if any
// enqueue fails; the caller aborts the run.
static inline bool swap_device_organism(SwapContext& ctx, int slot_a, int slot_b) {
    cudaError_t cuda_err = cudaSuccess;
    auto copy = [&](void* dst, const void* src, size_t bytes) {
        if (cuda_err != cudaSuccess) return;
        cuda_err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, ctx.stream);
    };

    // OrganismState swap.
    copy(ctx.d_swap_org, &ctx.d_organisms[slot_a], sizeof(OrganismState));
    copy(&ctx.d_organisms[slot_a], &ctx.d_organisms[slot_b], sizeof(OrganismState));
    copy(&ctx.d_organisms[slot_b], ctx.d_swap_org, sizeof(OrganismState));

    // CheckpointBuffer swap.
    copy(ctx.d_swap_ckpt, &ctx.d_checkpoints[slot_a], sizeof(CheckpointBuffer));
    copy(&ctx.d_checkpoints[slot_a], &ctx.d_checkpoints[slot_b], sizeof(CheckpointBuffer));
    copy(&ctx.d_checkpoints[slot_b], ctx.d_swap_ckpt, sizeof(CheckpointBuffer));

    // GradBuffers swap.
    copy(ctx.d_swap_grad, &ctx.d_grads[slot_a], sizeof(GradBuffers));
    copy(&ctx.d_grads[slot_a], &ctx.d_grads[slot_b], sizeof(GradBuffers));
    copy(&ctx.d_grads[slot_b], ctx.d_swap_grad, sizeof(GradBuffers));

    // Effective-weight bank swap: backward must re-forward each trajectory
    // with the phenotype that produced it.
    copy(ctx.d_swap_wbank,
         &ctx.d_eff_weights[slot_a * autodiff::TOTAL_WEIGHTS],
         autodiff::TOTAL_WEIGHTS * sizeof(float));
    copy(&ctx.d_eff_weights[slot_a * autodiff::TOTAL_WEIGHTS],
         &ctx.d_eff_weights[slot_b * autodiff::TOTAL_WEIGHTS],
         autodiff::TOTAL_WEIGHTS * sizeof(float));
    copy(&ctx.d_eff_weights[slot_b * autodiff::TOTAL_WEIGHTS],
         ctx.d_swap_wbank,
         autodiff::TOTAL_WEIGHTS * sizeof(float));

    if (cuda_err != cudaSuccess) {
        std::printf("[FATAL] CUDA PT swap failed: %s\n", cudaGetErrorString(cuda_err));
        return false;
    }
    return true;
}

// Swap host-side OrganismTable row data between two pool slots, including the
// evaluation correlation state (seed gradients, batch assignment) so every
// field belonging to a logical organism moves together.
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
        LineageId t_id;
        t_id = ctx.lineage_id[slot_a]; ctx.lineage_id[slot_a] = ctx.lineage_id[slot_b]; ctx.lineage_id[slot_b] = t_id;
        ArchiveSlot t_slot;
        t_slot = ctx.parent_id[slot_a];  ctx.parent_id[slot_a]  = ctx.parent_id[slot_b];  ctx.parent_id[slot_b]  = t_slot;
    }
    {
        int t = ctx.spawn_gen[slot_a]; ctx.spawn_gen[slot_a] = ctx.spawn_gen[slot_b]; ctx.spawn_gen[slot_b] = t;
        t = ctx.batch_sample_idx[slot_a]; ctx.batch_sample_idx[slot_a] = ctx.batch_sample_idx[slot_b]; ctx.batch_sample_idx[slot_b] = t;
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

    // Seed-gradient rows (BMAP_DIM floats each). These are the loss gradients
    // produced by scoring the pre-swap rollouts; the logical organism must
    // carry its own objective through the swap.
    {
        float tmp_sg[BMAP_DIM];
        float* sa = &ctx.seed_grad[slot_a * BMAP_DIM];
        float* sb = &ctx.seed_grad[slot_b * BMAP_DIM];
        std::memcpy(tmp_sg, sa, sizeof(tmp_sg));
        std::memcpy(sa, sb, sizeof(tmp_sg));
        std::memcpy(sb, tmp_sg, sizeof(tmp_sg));
    }

    // Predictor curriculum error estimate: it describes the organism as a
    // target, so it moves with the organism.
    {
        float t = ctx.predictor_error_ema[slot_a];
        ctx.predictor_error_ema[slot_a] = ctx.predictor_error_ema[slot_b];
        ctx.predictor_error_ema[slot_b] = t;
    }
    {
        float t = ctx.predictor_loss_ema[slot_a];
        ctx.predictor_loss_ema[slot_a] = ctx.predictor_loss_ema[slot_b];
        ctx.predictor_loss_ema[slot_b] = t;
    }
}

// Every PT_SWAP_INTERVAL generations, for each adjacent replica pair: compute
// improvement_rate, draw Metropolis accept, swap ALL organisms between the two
// replicas on accept. Per section 13: full organism data swap. The pool slot's
// replica_tag stays fixed — it identifies the temperature, not the organism.
//
// Swap timing: before backward in the generation loop, so swapped organisms
// contribute gradients in their new replica context. Returns false (run
// invalidated) if any device operation fails.
inline bool propose_swaps(MutationLadder* l,
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

                // Device data swap (OrganismState + Checkpoint + Grads +
                // effective weights).
                if (!swap_device_organism(ctx, slot_a, slot_b)) return false;

                // Host data swap (genome, delta, metadata).
                swap_host_organism(ctx, slot_a, slot_b);
            }

            // Sync device copies before proceeding to next pair.
            cudaError_t e = cudaStreamSynchronize(ctx.stream);
            if (e != cudaSuccess) {
                std::printf("[FATAL] CUDA PT sync failed: %s\n", cudaGetErrorString(e));
                return false;
            }
        }
    }
    update_beta(l);
    return true;
}

}  // namespace slime::safety::pt

#endif  // COEVO_SAFETY_PARALLEL_TEMPERING_CU




