// Wave 2: Integration Layer — World Struct, OrganismTable, IntentRegistry
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
struct OrganismTable {
    genome::Genome       genomes[TOTAL_ORG];
    genome::DeltaWeights deltas[TOTAL_ORG];
    uint32_t             lineage_id[TOTAL_ORG];
    uint32_t             parent_id[TOTAL_ORG];
    int                  spawn_gen[TOTAL_ORG];
    uint8_t              replica_tag[TOTAL_ORG];  // mirrors MutationLadder::replica_of
    float                fitness[TOTAL_ORG];       // composed fitness per organism
    float                f_raw[TOTAL_ORG];
    float                f_sot[TOTAL_ORG];
    Role                 role[TOTAL_ORG];          // cached from genome
    int                  batch_sample_idx[POOL_SIZE]; // which batch sample this org got
};

// ---- World -----------------------------------------------------------------
// Central state for the entire run. Host-allocated (standard new).
// Device pointers reference cudaMalloc'd buffers; host pointers reference
// cudaMallocHost (pinned) buffers.
struct World {
    // ---- Device-resident buffers (section 2.1) ----
    OrganismState*    d_organisms;        // [TOTAL_ORG]
    float*            d_weights;          // [TOTAL_WEIGHTS]
    ForwardInputs*    d_fwd_inputs;       // [POOL_SIZE]
    CheckpointBuffer* d_checkpoints;      // [POOL_SIZE]
    GradBuffers*      d_grads;            // [POOL_SIZE]
    BackwardWorkspace bwd_workspace;      // 2 d_state + d_perc + 2 recomp
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

    // SOT reference buffers (section 12): pre-allocated, not per-call.
    __half*           d_sot_temp_images;  // [SOT_SUBBATCH * GRID_SIZE * GRID_SIZE * 3]
    float*            d_sot_task_emb;     // [TASK_EMBED_DIM]
    ForwardInputs*    d_sot_fwd_inputs;   // [SOT_SUBBATCH]
    float*            d_sot_descriptors;  // [SOT_SUBBATCH * BMAP_DIM]

    // PT swap temp buffers (section 13): pre-allocated for full data swap.
    OrganismState*    d_pt_swap_org;      // [1]
    CheckpointBuffer* d_pt_swap_ckpt;    // [1]
    GradBuffers*      d_pt_swap_grad;     // [1]

    // ---- Pinned host buffers (section 2.2) ----
    float*            h_descriptors;      // [POOL_SIZE * BMAP_DIM]
    float*            h_btraj;            // [POOL_SIZE * BTRAJ_SAMPLES * BMAP_DIM]
    float*            h_seed_grad;        // [POOL_SIZE * BMAP_DIM]
    ForwardInputs*    h_fwd_inputs;       // [POOL_SIZE]
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

    // Scalars
    int               generation;
    bool              bootstrap_fired;
    int               bootstrap_gen;
    float             s_target;
    bool              s_target_calibrated;
    Pcg32             rng;                // PCG32 host PRNG (section 15.1)
    uint64_t          host_sot_key;
    int               grad_health_warn_count;   // consecutive low-norm generations

    cudaStream_t      stream;
};

// ---- Function declarations (defined in host_main.cu) -----------------------
void initialize_world(World* w);
void step_generation(World* w);
void run(int n_generations);

}  // namespace slime::integration

#endif  // COEVO_INTEGRATION_MAIN_LOOP_CU
