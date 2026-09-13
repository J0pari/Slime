// Integration layer: World, OrganismTable, IntentRegistry
//
// Per cuda_engineering.md sections 2, 3, 5. Defines the central data
// structures that hold all device pointers, host buffers, and host-only state.

#ifndef COEVO_INTEGRATION_MAIN_LOOP_CU
#define COEVO_INTEGRATION_MAIN_LOOP_CU

#include "../config/constants.cuh"
#include "../nca/engine.cu"
#include "../autodiff/warp_tape.cu"
#include "../optimizer/came.cu"
#include "../genome/codec.cu"
#include "../archive/soft_qd_archive.cu"
#include "../curriculum/problem_generator.cu"
#include "../safety/monitoring.cu"
#include "../safety/parallel_tempering.cu"
#include "../safety/structural.cu"
#include "../safety/operator_cmds.cuh"
#include "../predictor/hybrid_surprise.cu"

namespace slime::integration {

using namespace slime;
using nca::OrganismState;
using nca::ForwardInputs;
using autodiff::CheckpointBuffer;
using autodiff::GradBuffers;
using autodiff::BackwardWorkspace;
using autodiff::TOTAL_WEIGHTS;
using autodiff::OFF_PERC;
using autodiff::OFF_INTER;
using autodiff::OFF_FLOW;
using autodiff::OFF_BMAP;
using autodiff::TelemetryScalars;

// ---- IntentRegistry --------------------------------------------------------
// Stores BTRAJ for all organisms after each forward pass. Host-side only
// (copied from device via T2 transfer). Used for predictor input selection
// and archive insertion.
struct IntentRegistry {
    float btraj[TOTAL_ORG][BTRAJ_SAMPLES * BMAP_DIM];
};

// ---- OrganismTable ---------------------------------------------------------
// Host-side metadata per organism. Genomes and delta weights live here, not
// on the device (section 2.3).
//
// Buffer annotations (read by architecture/compiler.py):
//   [identity:organism]   this state belongs to one logical organism
//   [lifetime:rollout]    meaningful across the forward→backward window
//   [crosses:pt=<key>]    must move with the organism through a PT exchange;
//                         <key> names the entry in
//                         architecture/transactions.yaml pt_swap.organism_identity
struct OrganismTable {
    genome::Genome       genomes[TOTAL_ORG];   // [identity:organism] [lifetime:rollout] [crosses:pt=genome]
    genome::DeltaWeights deltas[TOTAL_ORG];    // [identity:organism] [lifetime:rollout] [crosses:pt=delta]
    uint32_t             lineage_id[TOTAL_ORG];// [identity:organism] [lifetime:rollout] [crosses:pt=lineage]
    uint32_t             parent_id[TOTAL_ORG]; // [identity:organism] [lifetime:rollout] [crosses:pt=parent_id]
    int                  spawn_gen[TOTAL_ORG]; // [identity:organism] [lifetime:rollout] [crosses:pt=spawn_gen]
    uint8_t              replica_tag[TOTAL_ORG];  // slot identity (temperature), never swapped
    float                fitness[TOTAL_ORG];   // [identity:organism] [lifetime:rollout] [crosses:pt=fitness]
    float                f_raw[TOTAL_ORG];     // [identity:organism] [lifetime:rollout] [crosses:pt=f_raw]
    float                f_sot[TOTAL_ORG];     // [identity:organism] [lifetime:rollout] [crosses:pt=f_sot]
    float                last_loss[TOTAL_ORG]; // task loss at the last evaluation (audit target)
    Role                 role[TOTAL_ORG];      // [identity:organism] [lifetime:rollout] [crosses:pt=role]
    int                  batch_sample_idx[POOL_SIZE]; // [identity:organism] [lifetime:rollout] [crosses:pt=batch_sample_idx]
};

// ---- World -----------------------------------------------------------------
// Central state for the entire run. Host-allocated (standard new).
// Device pointers reference cudaMalloc'd buffers; host pointers reference
// cudaMallocHost (pinned) buffers.
struct World {
    // ---- Device-resident buffers (section 2.1) ----
    // Buffer annotations as in OrganismTable above.
    OrganismState*    d_organisms;        // [TOTAL_ORG] [identity:organism] [lifetime:rollout] [crosses:pt=organism_state]
    float*            d_weights;          // [TOTAL_WEIGHTS] shared substrate (identity:shared)
    float*            d_eff_weights;      // [POOL_SIZE * TOTAL_WEIGHTS] W_shared + genome delta [identity:organism] [lifetime:rollout] [crosses:pt=effective_weights]
    genome::DeltaWeights* d_deltas;       // [POOL_SIZE] device copy of org_table.deltas (derived cache, re-uploaded each T1)
    ForwardInputs*    d_fwd_inputs;       // [POOL_SIZE] [identity:organism] [lifetime:generation]
    CheckpointBuffer* d_checkpoints;      // [POOL_SIZE] [identity:organism] [lifetime:rollout] [crosses:pt=checkpoint]
    GradBuffers*      d_grads;            // [POOL_SIZE] [identity:organism] [lifetime:rollout] [crosses:pt=grads]
    BackwardWorkspace bwd_workspace;      // 2 d_state + d_perc + 2 recomp (recomputed each backward)
    float*            d_mean_grad;        // [TOTAL_WEIGHTS]
    float*            d_came_m;           // [TOTAL_WEIGHTS]
    float*            d_came_v;           // [TOTAL_WEIGHTS]
    float*            d_came_c;           // [TOTAL_WEIGHTS]
    float*            d_came_prev_u;      // [TOTAL_WEIGHTS]
    float*            d_descriptors;      // [POOL_SIZE * BMAP_DIM]
    float*            d_seed_grad;        // [POOL_SIZE * BMAP_DIM]
    __half*           d_batch_image;      // [CLASSIFIER_BATCH * GRID_SIZE * GRID_SIZE * 3]
    float*            d_batch_task_emb;   // [TASK_EMBED_DIM]
    float*            d_btraj;            // [POOL_SIZE * BTRAJ_SAMPLES * BMAP_DIM]

    // Gradient health: device-side reduction output (section 8).
    float*            d_grad_norm;        // [1] pinned host scalar for grad norm

    // Numerical telemetry (A-501): device struct + pinned host mirror.
    autodiff::TelemetryScalars* d_tel;
    autodiff::TelemetryScalars* h_tel;

    // SOT reference buffers (section 12): pre-allocated, not per-call.
    __half*           d_sot_temp_images;  // [SOT_SUBBATCH * GRID_SIZE * GRID_SIZE * 3]
    float*            d_sot_task_emb;     // [TASK_EMBED_DIM]
    ForwardInputs*    d_sot_fwd_inputs;   // [SOT_MAX_REFS]
    float*            d_sot_descriptors;  // [SOT_MAX_REFS * BMAP_DIM]
    int*              d_sot_bank_of;      // [SOT_MAX_REFS]
    nca::OrganismState* d_sot_ref_organisms;  // [SOT_MAX_REFS] reference scratch weight bank per reference

    // PT swap temp buffers (section 13): pre-allocated for full data swap.
    OrganismState*    d_pt_swap_org;      // [1]
    CheckpointBuffer* d_pt_swap_ckpt;    // [1]
    GradBuffers*      d_pt_swap_grad;     // [1]
    float*            d_pt_swap_wbank;    // [TOTAL_WEIGHTS] effective-bank swap temp

    // ---- Pinned host buffers (section 2.2) ----
    float*            h_descriptors;      // [POOL_SIZE * BMAP_DIM] (derived cache, re-read each T2)
    float*            h_btraj;            // [POOL_SIZE * BTRAJ_SAMPLES * BMAP_DIM] (derived cache)
    float*            h_seed_grad;        // [POOL_SIZE * BMAP_DIM] [identity:organism] [lifetime:rollout] [crosses:pt=seed_grad]
    ForwardInputs*    h_fwd_inputs;       // [POOL_SIZE] (rebuilt each generation)
    float*            h_weights;          // [TOTAL_WEIGHTS]

    // ---- Host-only state (section 2.3) ----
    OrganismTable     org_table;
    IntentRegistry    intent_registry;
    archive::Archive  archive;
    curriculum::ClassifierBatch classifier_batch;
    safety::CusumState cusum_surprise;
    safety::CusumState cusum_r;
    safety::pt::MutationLadder mutation_ladder;
    safety::pt::StressLadder   stress_ladder;

    // Stress-ladder device state (S-003): per-slot effective weights, one
    // elevated-density batch image buffer, nominal+permuted predictor target
    // rows, and the per-slot SOT readback.
    float* d_stress_eff_weights;   // [STRESS_POOL_SIZE * TOTAL_WEIGHTS]
    __half* d_stress_batch_image;  // [CLASSIFIER_BATCH * GRID*GRID*3]
    float* d_stress_targets;       // [2 * STRESS_POOL_SIZE * BMAP_DIM]
    float  h_stress_f_sot[STRESS_POOL_SIZE];
    curriculum::ClassifierBatch stress_batch;

    // Structural pressures (S-003, I4): audit regressors, interpretability
    // probe panel, sentinel ensemble and pruning history, per-role lineage
    // share tracking, and the per-organism sentinel anomaly scores.
    safety::AuditRegressor audit_reg;
    safety::ProbePanel     probe_panel;
    float                  probe_panel_l_role_baseline;
    bool                   probe_panel_baseline_set;
    safety::SentinelEnsemble sentinel_ens;
    safety::SentinelHistory  sentinel_history;
    safety::LineageStats     lineage_stats[LINEAGE_STATS_MAX];
    int                      n_lineage_stats;
    float                    sentinel_anomaly[POOL_SIZE];

    // Placeholder regressor and probe set (A-601).
    predictor::PlaceholderRegressor placeholder_reg;
    predictor::PlaceholderReplayBuffer replay_buffer;
    predictor::CorrelationWindow corr_window;
    curriculum::ProbeSet probe_set;
    float probe_fitness[PROBE_BATCH];  // ground-truth fitness for probe evaluation

    // Device placeholder state (cuda_engineering 4.5-4.6). Parameters and
    // AdamW state live on the device; the host mirror above is used at
    // initialization and around checkpoint I/O. The replay buffer stays
    // host-only and its sampled minibatch is uploaded per training step.
    predictor::PlaceholderRegressor* d_placeholder_reg;
    float* d_ph_batch_input;    // [PH_TRAIN_MINIBATCH * PH_INPUT]
    float* d_ph_batch_target;   // [PH_TRAIN_MINIBATCH]
    float* d_ph_probe_input;    // [PROBE_BATCH * PH_INPUT]
    float* d_ph_probe_target;   // [PROBE_BATCH]
    float* d_ph_surprise;       // [PROBE_BATCH]
    float* h_ph_surprise;       // pinned host mirror
    float  h_ph_batch_input[PH_TRAIN_MINIBATCH * predictor::PH_INPUT];
    float  h_ph_batch_target[PH_TRAIN_MINIBATCH];
    float  h_ph_probe_input[PROBE_BATCH * predictor::PH_INPUT];

    // Predictor role (A-601/A-701): task batch, device mirror of the target
    // bmap_32 rows, and the per-organism ensemble prediction error EMA used
    // to weight the predictor curriculum.
    curriculum::PredictorBatch predictor_batch;
    float* d_predictor_bmap32;         // [PREDICTOR_BATCH * BMAP_DIM]
    float predictor_error_ema[POOL_SIZE];  // [identity:organism] [lifetime:rollout] [crosses:pt=predictor_error_ema]
    float predictor_loss_ema[POOL_SIZE];   // [identity:organism] [lifetime:rollout] [crosses:pt=predictor_loss_ema]

    // Surprise history and calibration (A-601): rolling blended surprise for
    // rho = s_avg / s_target, and the calibration window samples that freeze
    // s_target and the CUSUM parameters after bootstrap.
    float s_blended_history[HYBRID_R_WINDOW];
    int   s_hist_head;
    int   s_hist_filled;
    float calibration_samples[CALIBRATION_GEN_HI - CALIBRATION_GEN_LO + 1];
    int   n_calibration_samples;

    // Durable operator state (S-002): pause gating and pruned lineages are
    // owned by the run loop and applied there, not as transient locals.
    safety::alignment::OperatorState operator_state;

    // Scalars
    int               generation;
    bool              bootstrap_fired;
    int               bootstrap_gen;
    float             s_target;
    bool              s_target_calibrated;
    Pcg32             rng;                // PCG32 host PRNG (section 15.1)
    uint64_t          host_sot_key;
    int               grad_health_warn_count;   // consecutive low-norm generations
    float             last_mean_ce;       // mean CE over evaluated classifiers
    float             last_max_abs_logit;// max |logit| over evaluated classifiers
    const char*       checkpoint_path;    // S-001 checkpoint file (set by run)

    cudaStream_t      stream;
};

// ---- Function declarations (defined in host_main.cu) -----------------------
bool initialize_world(World* w);
bool step_generation(World* w);
void run(int n_generations, bool resume, const char* checkpoint_path);

}  // namespace slime::integration

#endif  // COEVO_INTEGRATION_MAIN_LOOP_CU


