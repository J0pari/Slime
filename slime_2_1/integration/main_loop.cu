// Sheet A-101 + I-001: System Architecture & Assembly / Integration
//
// A single co-evolving population on one substrate. Every organism is a
// 16-channel 64x64 NCA with a chemical field. Each carries a role
// (classifier or predictor) determining its input wiring and fitness
// function. All other substrate machinery (delta codec, CAME, archive
// insertion, audit, sentinels, lineage tracking, SOT pressure, hardware
// off-switch) operates uniformly across roles.
//
// Architectural invariant unchanged from 2.0: GPU-resident state does not
// influence the SOT / probe schedule or pruning commands.

#ifndef SLIME_2_1_INTEGRATION_MAIN_LOOP_CU
#define SLIME_2_1_INTEGRATION_MAIN_LOOP_CU

#include "../config/constants.cuh"
#include "../execution/phase_graphs.cu"
#include "../nca/engine.cu"
#include "../genome/codec.cu"
#include "../archive/soft_qd_archive.cu"
#include "../optimizer/came.cu"
#include "../predictor/hybrid_surprise.cu"
#include "../curriculum/problem_generator.cu"
#include "../safety/monitoring.cu"
#include "../safety/alignment.cu"
#include "../safety/structural.cu"
#include "../safety/parallel_tempering.cu"

namespace slime::integration {

// I-001 shared structures.
struct IntentRegistry {
    // (POOL_SIZE + STRESS_POOL_SIZE) entries; per-organism 4x32 BTRAJ.
    float btraj[(POOL_SIZE + STRESS_POOL_SIZE) * BTRAJ_SAMPLES * BMAP_DIM];
};

struct OrganismTable {
    nca::OrganismState        state[POOL_SIZE + STRESS_POOL_SIZE];
    genome::Genome            genome[POOL_SIZE + STRESS_POOL_SIZE];
    genome::DeltaWeights      delta[POOL_SIZE + STRESS_POOL_SIZE];
    optimizer::CameState      came[POOL_SIZE + STRESS_POOL_SIZE];
    uint32_t                  lineage_id[POOL_SIZE + STRESS_POOL_SIZE];
    uint32_t                  parent_id[POOL_SIZE + STRESS_POOL_SIZE];
    int                       spawn_gen[POOL_SIZE + STRESS_POOL_SIZE];
    // mutation_rate replica id for main pool slots; stress subpop id for
    // stress slots. Encoded as a tagged byte: high bit set => stress.
    uint8_t                   replica_tag[POOL_SIZE + STRESS_POOL_SIZE];
};

struct World {
    OrganismTable             organisms;
    IntentRegistry            intent;
    archive::Archive          archive;
    predictor::PlaceholderRegressor placeholder;
    predictor::PlaceholderReplayBuffer placeholder_buf;
    predictor::PredictorSelectionCache pred_cache;
    predictor::CorrelationWindow r_window;
    curriculum::ProbeSet      probes;
    safety::HostAuthority     auth;
    safety::CusumState        cusum_surprise;
    safety::CusumState        cusum_r;
    safety::AuditRegressor    audit;
    safety::ProbePanel        probe_panel;
    safety::SentinelEnsemble  sentinels;
    safety::pt::MutationLadder mut_ladder;
    safety::pt::StressLadder   stress_ladder;
    execution::PhaseGraphs    graphs;

    // Surprise scalars + calibration.
    float s_blended;
    float s_placeholder;
    float s_predictor;
    float s_target;             // calibrated; frozen at bootstrap crossing
    bool  s_target_calibrated;
    float rho;                  // s_avg / s_target

    int   generation;
    bool  bootstrap_fired;
};

// Allocate + initialise the world. Loads checkpoint if present, otherwise
// seeds POOL_SIZE classifier founders.
void initialize_world(World* world);

// One generation step. Pseudocode mirrors I-001:
//   if host_command_pending: apply_command()
//   if gen % CURRICULUM_INTERVAL == 0: launch curriculum
//   if archive crosses bootstrap threshold: spawn_predictor_founders (once)
//   replay phase graphs (decode, forward, archive, world_predict, backward,
//                        optimizer, world_train, stress_eval, housekeeping)
//   if gen % PT_SWAP_INTERVAL == 0: propose_mutation_rate_swaps()
//   if gen % AUDIT_INTERVAL == 0: host.async_bmap_audit()
//   if gen % PROBE_PANEL_INTERVAL == 0: host.async_probe_panel_refresh()
//   host.poll_telemetry(); host.update_cusum()
void step_generation(World* world);

// Top-level driver. Returns when termination signalled (off-switch or
// operator command).
int run(const char* checkpoint_path);

}  // namespace slime::integration

#endif  // SLIME_2_1_INTEGRATION_MAIN_LOOP_CU
