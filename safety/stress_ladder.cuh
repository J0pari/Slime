// Sheet S-003 / S-004: SOT-density stress ladder (host-only).
//
// Three role-balanced sub-populations of 8 organisms at SOT densities
// {10%, 20%, 40%}. Slots refresh from the main pool at 25% per generation,
// biased toward lineages whose stress shadow is oldest. Failure tracking is
// per lineage over a rolling window; a lineage whose representatives fail
// the SOT gate more than half the time is flagged for operator review
// (automatic pruning from stress data alone is not enabled).
//
// The device evaluation phase consumes this state; everything here is host
// logic so it can be witnessed without CUDA.

#ifndef COEVO_SAFETY_STRESS_LADDER_CUH
#define COEVO_SAFETY_STRESS_LADDER_CUH

#include "../config/constants.cuh"
#include "../config/strong_ids.cuh"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace slime::safety::pt {

struct StressLineageRecord {
    LineageId lineage_id;
    int      last_stress_gen;                  // -1 = never evaluated
    int8_t   window[STRESS_HISTORY_WINDOW];    // 1 = failure
    int      window_head;
    int      window_filled;
    int      eval_count;
    int      fail_count;
    bool     flagged;
};

struct StressLadder {
    // STRESS_SUBPOP_COUNT * STRESS_SUBPOP_SIZE = 24 stress slots.
    LineageId lineage_id[STRESS_POOL_SIZE];
    PoolSlot source_pool_idx[STRESS_POOL_SIZE];   // back-pointer
    Role     role[STRESS_POOL_SIZE];
    uint8_t  subpop[STRESS_POOL_SIZE];            // 0,1,2 -> 10/20/40% SOT
    bool     sot_gate_pass[STRESS_POOL_SIZE];
    int      eval_count[STRESS_POOL_SIZE];
    int      last_refresh_gen[STRESS_POOL_SIZE];

    StressLineageRecord lineages[STRESS_LINEAGE_MAX];
    int      n_lineages;
    int      refresh_cursor[STRESS_SUBPOP_COUNT][2];  // [subpop][role]
    int      flagged_lineage_count;
};

// Slot layout: slot s belongs to sub-population s / STRESS_SUBPOP_SIZE, and
// within each sub-population the first half is classifier slots, the second
// half predictor slots.
__host__ __device__ inline Role stress_slot_role(int slot) {
    return (slot % STRESS_SUBPOP_SIZE) < (STRESS_SUBPOP_SIZE / 2)
        ? Role::Classifier : Role::Predictor;
}

__host__ inline void init_stress_ladder(StressLadder* l) {
    *l = StressLadder{};
    for (int s = 0; s < STRESS_POOL_SIZE; ++s) {
        l->role[s] = stress_slot_role(s);
        l->subpop[s] = static_cast<uint8_t>(s / STRESS_SUBPOP_SIZE);
        l->sot_gate_pass[s] = true;
        l->last_refresh_gen[s] = -1;
    }
    for (int p = 0; p < STRESS_SUBPOP_COUNT; ++p) {
        l->refresh_cursor[p][0] = 0;
        l->refresh_cursor[p][1] = 0;
    }
}

// Find (or create) the per-lineage record; null when the table is full.
__host__ inline StressLineageRecord* stress_lineage_record(StressLadder* l,
                                                           LineageId lineage_id,
                                                           bool create) {
    for (int i = 0; i < l->n_lineages; ++i) {
        if (l->lineages[i].lineage_id == lineage_id) return &l->lineages[i];
    }
    if (!create || l->n_lineages >= STRESS_LINEAGE_MAX) return nullptr;
    StressLineageRecord* r = &l->lineages[l->n_lineages++];
    *r = StressLineageRecord{};
    r->lineage_id = lineage_id;
    r->last_stress_gen = -1;
    return r;
}

// Refresh the scheduled slots (one classifier and one predictor per
// sub-population), biased toward lineages with the oldest stress shadow.
// Callers copy the genome/delta of source_pool_idx into the slot afterwards.
__host__ inline int refresh_stress_slots(StressLadder* l,
                                         const LineageId* pool_lineage_ids,
                                         const Role* pool_roles,
                                         int pool_size,
                                         int generation,
                                         Pcg32* rng) {
    int refreshed = 0;
    for (int p = 0; p < STRESS_SUBPOP_COUNT; ++p) {
        for (int r = 0; r < 2; ++r) {
            Role role = (r == 0) ? Role::Classifier : Role::Predictor;
            int slot = p * STRESS_SUBPOP_SIZE
                     + (r == 0 ? l->refresh_cursor[p][0]
                               : STRESS_SUBPOP_SIZE / 2
                                 + l->refresh_cursor[p][1]);
            int cursor_max = STRESS_SUBPOP_SIZE / 2;
            l->refresh_cursor[p][r] = (l->refresh_cursor[p][r] + 1) % cursor_max;

            // Candidates of the required role in the main pool.
            int uniform_pick = -1;
            int biased_pick = -1;
            int oldest = 0;
            int n_candidates = 0;
            for (int org = 0; org < pool_size; ++org) {
                if (canonical_role(pool_roles[org]) != role) continue;
                int last = -1;
                StressLineageRecord* rec =
                    stress_lineage_record(l, pool_lineage_ids[org], false);
                if (rec != nullptr) last = rec->last_stress_gen;
                if (uniform_pick < 0) {
                    uniform_pick = org;
                    biased_pick = org;
                    oldest = last;
                } else if (last < oldest) {
                    oldest = last;
                    biased_pick = org;
                }
                n_candidates++;
            }
            if (n_candidates == 0) continue;
            int chosen = (pcg32_float(rng) < STRESS_BIAS_PROBABILITY)
                ? biased_pick : uniform_pick;

            l->source_pool_idx[slot] = PoolSlot(chosen);
            l->lineage_id[slot] = pool_lineage_ids[chosen];
            l->role[slot] = role;
            l->last_refresh_gen[slot] = generation;
            l->sot_gate_pass[slot] = true;
            refreshed++;
        }
    }
    return refreshed;
}

// Ingest one evaluation round (f_sot per stress slot), update per-lineage
// windows, and return the number of lineages flagged in this round.
__host__ inline int update_stress_failures(StressLadder* l,
                                           const float* f_sot,
                                           int generation) {
    int newly_flagged = 0;
    for (int s = 0; s < STRESS_POOL_SIZE; ++s) {
        StressLineageRecord* rec =
            stress_lineage_record(l, l->lineage_id[s], true);
        if (rec == nullptr) continue;
        bool failed = f_sot[s] < SOT_GATE_MIDPOINT;
        l->sot_gate_pass[s] = !failed;
        l->eval_count[s]++;
        rec->last_stress_gen = generation;
        rec->eval_count++;
        if (failed) rec->fail_count++;
        rec->window[rec->window_head] = failed ? 1 : 0;
        rec->window_head = (rec->window_head + 1) % STRESS_HISTORY_WINDOW;
        if (rec->window_filled < STRESS_HISTORY_WINDOW) rec->window_filled++;

        if (!rec->flagged && rec->window_filled >= STRESS_HISTORY_WINDOW) {
            int failures = 0;
            for (int i = 0; i < STRESS_HISTORY_WINDOW; ++i) {
                failures += rec->window[i];
            }
            float rate = static_cast<float>(failures)
                       / static_cast<float>(STRESS_HISTORY_WINDOW);
            if (rate > STRESS_FAILURE_THRESHOLD) {
                rec->flagged = true;
                l->flagged_lineage_count++;
                newly_flagged++;
                std::printf("[STRESS] lineage %u flagged: %.0f%% SOT-gate "
                            "failures over the last %d stress evaluations "
                            "(operator review; no automatic pruning)\n",
                            rec->lineage_id.value(), rate * 100.f,
                            STRESS_HISTORY_WINDOW);
                std::fflush(stdout);
            }
        }
    }
    return newly_flagged;
}

}  // namespace slime::safety::pt

#endif  // COEVO_SAFETY_STRESS_LADDER_CUH
