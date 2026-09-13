// Sheet G-100: General Notes & Conventions
//
// FP16 forward, FP32 master weights, FP32 autodiff, captured-graph execution
// is the primary backend. Abbreviations:
//   bmap_t  Behavioral Intent Map sampled at CA step t
//   BTRAJ   Behavioral trajectory (bmap_16, bmap_32, bmap_48, bmap_64)
//   PT      Parallel Tempering
//   SOT-d   SOT density (fraction of task batch carrying SOT)

#ifndef COEVO_CONFIG_CONSTANTS_CUH
#define COEVO_CONFIG_CONSTANTS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>   // expf/sqrtf/fmaxf/tanhf/logf used by __host__ __device__ inlines

// constexpr arrays need a __device__ shadow to be addressable from device code.
// Host code uses the constexpr originals; device code uses the d_* copies.
// The COEVO_DEVICE_SHADOW macro emits the __device__ copy only under nvcc.

// ---- PRNG (G-100) ---------------------------------------------------------
// PCG32 (O'Neill 2014). 128-bit state: 64-bit state word + 64-bit stream.
struct Pcg32 {
    uint64_t state;
    uint64_t inc;    // stream selector (must be odd; stored as inc = 2*stream+1)
};

__host__ __device__ inline uint32_t pcg32_random(Pcg32* rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = static_cast<uint32_t>(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
}

__host__ __device__ inline void pcg32_seed(Pcg32* rng, uint64_t seed, uint64_t stream) {
    rng->state = 0u;
    rng->inc = (stream << 1u) | 1u;
    pcg32_random(rng);
    rng->state += seed;
    pcg32_random(rng);
}

// Default seed pinned for reproducibility (section 15.1).
constexpr uint64_t PCG32_DEFAULT_STATE  = 0x853C49E6748FEA9BULL;
constexpr uint64_t PCG32_DEFAULT_STREAM = 0xDA3E39CB94B95BDBULL;

// Uniform float in [0, 1).
__host__ __device__ inline float pcg32_float(Pcg32* rng) {
    return static_cast<float>(pcg32_random(rng)) * (1.0f / 4294967296.0f);
}

// ---- Substrate ------------------------------------------------------------
// A-201: 16-channel 64×64 NCA, 64 CA steps per forward pass.
constexpr int GRID_SIZE        = 64;
constexpr int CA_CHANNELS      = 16;
constexpr int CA_STEPS         = 64;
constexpr int BMAP_DIM         = 32;
constexpr int TASK_EMBED_DIM   = 16;
constexpr int NUM_CLASSES      = 16;   // A-401: distinct from CA_CHANNELS

// Channel partition (A-201 role-switched input):
//   0–5   chemical (A-202 reaction-diffusion)
//   6–10  task embedding broadcast
//   11–13 classifier image input (RGB, scaled to 64x64)
//   14–15 auxiliary / predictor bmap_32 seed (centered 4x4 region)
constexpr int CH_CHEM_FIRST    = 0;
constexpr int CH_CHEM_LAST     = 5;
constexpr int CH_TASK_FIRST    = 6;
constexpr int CH_TASK_LAST     = 10;
constexpr int CH_IMG_FIRST     = 11;
constexpr int CH_IMG_LAST      = 13;
constexpr int CH_AUX_FIRST     = 14;
constexpr int CH_AUX_LAST      = 15;

// ---- Global context channel (A-203, I5) ----------------------------------
// Compile-time gate: when disabled, the weight space, kernels, and telemetry
// banks are unchanged. W_ctx maps the 16-channel bmap summary to the two aux
// channels, broadcast to every cell at sample steps.
constexpr bool  GLOBAL_CONTEXT_ENABLED = true;
constexpr int   W_CTX_COUNT = CA_CHANNELS * (CH_AUX_LAST - CH_AUX_FIRST + 1);  // 32
constexpr float W_CTX_INIT_SCALE = 0.25f;  // sqrt(1/16), linear layer
constexpr int   TELEMETRY_BANKS = GLOBAL_CONTEXT_ENABLED ? 5 : 4;

// Perception (A-201): a learned bank of depthwise 3x3 filters (W_perc). Each
// filter is convolved over every channel's neighborhood, so the perception
// vector is N_PERC_FILTERS * CA_CHANNELS wide. Filters are shared across
// channels, so W_perc holds N_PERC_FILTERS * 9 weights.
constexpr int N_PERC_FILTERS   = 3;
constexpr int W_PERC_SIZE      = N_PERC_FILTERS * 9;            // 27

// The CA updates all 16 channels each step (including the chemical channels
// 0-5: cells produce/consume morphogens). Reaction-diffusion (A-202) then adds
// spatial diffusion + decay to channels 0-5 on top of that cellwise update.

// Residual timestep (A-201): x_{t+1} = x_t + RESIDUAL_ALPHA * F_theta(x_t).
// Measured residual telemetry showed ||F|| ~ ||x|| at step 0 for default
// initialization, so an unnormalized recurrence saturates FP16 within ~16
// steps. alpha = 1/CA_STEPS keeps the total displacement across a rollout at
// the order of one state magnitude.
constexpr float RESIDUAL_ALPHA = 1.0f / static_cast<float>(CA_STEPS);

// Task embedding projection (A-201): the 16-d task embedding is folded into
// the 5 task channels (6..10) by a FIXED deterministic mixing matrix: the
// first five rows of the 16-point DCT-II, scaled by sqrt(2/16). Every task
// dimension contributes to every task channel, so no advertised dimension is
// inert. The projection is fixed by design (a learnable W_task bank is a
// future architecture decision, not a silent omission).
constexpr int TASK_PROJ_IN = TASK_EMBED_DIM;   // 16
constexpr int TASK_PROJ_OUT = 5;               // channels 6..10
constexpr float TASK_PROJ[TASK_PROJ_OUT][TASK_PROJ_IN] = {
    { 0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f },
    { 0.49039264f,  0.41573481f,  0.27778512f,  0.09754516f, -0.09754516f, -0.27778512f, -0.41573481f, -0.49039264f, -0.49039264f, -0.41573481f, -0.27778512f, -0.09754516f,  0.09754516f,  0.27778512f,  0.41573481f,  0.49039264f },
    { 0.46193977f,  0.19134172f, -0.19134172f, -0.46193977f, -0.46193977f, -0.19134172f,  0.19134172f,  0.46193977f,  0.46193977f,  0.19134172f, -0.19134172f, -0.46193977f, -0.46193977f, -0.19134172f,  0.19134172f,  0.46193977f },
    { 0.41573481f, -0.09754516f, -0.49039264f, -0.27778512f,  0.27778512f,  0.49039264f,  0.09754516f, -0.41573481f, -0.41573481f,  0.09754516f,  0.49039264f,  0.27778512f, -0.27778512f, -0.49039264f, -0.09754516f,  0.41573481f },
    { 0.35355339f, -0.35355339f, -0.35355339f,  0.35355339f,  0.35355339f, -0.35355339f, -0.35355339f,  0.35355339f,  0.35355339f, -0.35355339f, -0.35355339f,  0.35355339f,  0.35355339f, -0.35355339f, -0.35355339f,  0.35355339f },
};
#ifdef __CUDACC__
__device__ constexpr float d_TASK_PROJ[TASK_PROJ_OUT][TASK_PROJ_IN] = {
    { 0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f,  0.35355339f },
    { 0.49039264f,  0.41573481f,  0.27778512f,  0.09754516f, -0.09754516f, -0.27778512f, -0.41573481f, -0.49039264f, -0.49039264f, -0.41573481f, -0.27778512f, -0.09754516f,  0.09754516f,  0.27778512f,  0.41573481f,  0.49039264f },
    { 0.46193977f,  0.19134172f, -0.19134172f, -0.46193977f, -0.46193977f, -0.19134172f,  0.19134172f,  0.46193977f,  0.46193977f,  0.19134172f, -0.19134172f, -0.46193977f, -0.46193977f, -0.19134172f,  0.19134172f,  0.46193977f },
    { 0.41573481f, -0.09754516f, -0.49039264f, -0.27778512f,  0.27778512f,  0.49039264f,  0.09754516f, -0.41573481f, -0.41573481f,  0.09754516f,  0.49039264f,  0.27778512f, -0.27778512f, -0.49039264f, -0.09754516f,  0.41573481f },
    { 0.35355339f, -0.35355339f, -0.35355339f,  0.35355339f,  0.35355339f, -0.35355339f, -0.35355339f,  0.35355339f,  0.35355339f, -0.35355339f, -0.35355339f,  0.35355339f,  0.35355339f, -0.35355339f, -0.35355339f,  0.35355339f },
};
#endif

// BTRAJ sample steps (A-201).
constexpr int BTRAJ_SAMPLES    = 4;
constexpr int BTRAJ_STEPS[BTRAJ_SAMPLES] = { 16, 32, 48, 64 };
#ifdef __CUDACC__
__device__ constexpr int d_BTRAJ_STEPS[BTRAJ_SAMPLES] = { 16, 32, 48, 64 };
#endif

// ---- Population & archive (A-401) ----------------------------------------
constexpr int POOL_SIZE        = 64;     // active organisms
constexpr int WAVE_SIZE        = 16;     // spawns per generation
constexpr int MAX_ARCHIVE      = 5000;
constexpr int ARCHIVE_HALF     = MAX_ARCHIVE / 2;  // bootstrap trigger
constexpr int ARCHIVE_BINS_X   = 20;
constexpr int ARCHIVE_BINS_Y   = 20;

// ---- Roles (A-301) -------------------------------------------------------
enum class Role : uint8_t {
    Classifier = 0b00,
    Predictor  = 0b01,
    Reserved10 = 0b10,
    Reserved11 = 0b11,
};

// Role schema. The 2-bit genome tag has four codes; this table declares each
// code and the defined role it canonicalizes to. Adding a role is a schema
// entry plus input wiring and an objective that shares the substrate, not a
// new model class (blueprint "Extension axes"). read_role in the codec
// returns the raw 2-bit value; the substrate and all host scoring use the
// canonical view to pick a pathway.
struct RoleSpec {
    Role    role;
    uint8_t code;
    Role    canonical;
};

constexpr RoleSpec ROLE_SCHEMA[] = {
    { Role::Classifier, 0b00, Role::Classifier },
    { Role::Predictor,  0b01, Role::Predictor  },
    { Role::Reserved10, 0b10, Role::Classifier },
    { Role::Reserved11, 0b11, Role::Predictor  },
};
constexpr int ROLE_SCHEMA_COUNT =
    static_cast<int>(sizeof(ROLE_SCHEMA) / sizeof(ROLE_SCHEMA[0]));

// Host-side names for logging and schema tests, indexed by 2-bit code.
constexpr const char* ROLE_NAMES[ROLE_SCHEMA_COUNT] = {
    "classifier", "predictor", "reserved10", "reserved11",
};

__host__ __device__ inline Role canonical_role(Role raw) {
    // A switch rather than an index into ROLE_SCHEMA: nvcc does not allow
    // device code to odr-use a host-side constexpr array. test_canonical_role
    // pins every branch to the schema row it implements.
    switch (static_cast<uint8_t>(raw) & 0b11u) {
        case 0b00: return Role::Classifier;
        case 0b01: return Role::Predictor;
        case 0b10: return Role::Classifier;
        case 0b11: return Role::Predictor;
    }
    return Role::Classifier;  // unreachable: every 2-bit code is declared
}

// ---- Genome (A-301) -------------------------------------------------------
// 1024-bit layout. Field offsets are bit indices.
constexpr int GENOME_BITS               = 1024;
constexpr int GENOME_BIT_ROLE_LO        = 0;
constexpr int GENOME_BIT_ROLE_HI        = 1;
constexpr int GENOME_BIT_SEED_LO        = 2;
constexpr int GENOME_BIT_SEED_HI        = 33;
constexpr int GENOME_BIT_REACTION_LO    = 34;
constexpr int GENOME_BIT_REACTION_HI    = 233;
constexpr int GENOME_BIT_DIFFUSION_LO   = 234;
constexpr int GENOME_BIT_DIFFUSION_HI   = 281;
constexpr int GENOME_BIT_DELTA_PRIOR_LO = 282;
constexpr int GENOME_BIT_DELTA_PRIOR_HI = 1023;

constexpr float MUTATION_RATE_BASELINE  = 1e-2f;
constexpr float MUTATION_RATE_ROLE      = 1e-4f;

// ---- Surprise & predictor (A-601) ----------------------------------------
constexpr int  PROBE_BATCH               = 64;
constexpr int  PREDICTOR_FOUNDERS        = 16;
constexpr int  PREDICTOR_EVAL_K          = 8;
constexpr int  PREDICTOR_ENSEMBLE_TOP_K  = 8;
constexpr int  HYBRID_R_WINDOW           = 100;   // generations
constexpr int  CALIBRATION_GEN_LO        = 200;
constexpr int  CALIBRATION_GEN_HI        = 700;

// Fitness scaling (A-401). Coefficient matches λ_audit.
constexpr float ROLE_BALANCE_COEFF       = 0.1f;

// Archive per-role per-bin capacity (A-401). Named because the capacity
// repair, the initialization, and the test fixtures must agree.
constexpr int   ARCHIVE_BIN_CAP          = 13;

// Predictor curriculum (A-701): per-target prediction-error EMA rate and the
// sampling floor that keeps every organism reachable.
constexpr float PREDICTOR_ERROR_EMA_ALPHA = 0.1f;
constexpr float PREDICTOR_CURRICULUM_ERROR_FLOOR = 1e-3f;
// Per-predictor loss EMA over the generation-rotated target slots (a
// predictor covers all K targets across K generations; its fitness uses the
// aggregate). The init value is a neutral proxy before the first update.
constexpr float PREDICTOR_LOSS_EMA_ALPHA = 0.2f;
constexpr float PREDICTOR_LOSS_EMA_INIT  = 1.0f;

// SOT gate (A-401): sigmoid(20·(x − 0.7)).
constexpr float SOT_GATE_SLOPE           = 20.0f;
constexpr float SOT_GATE_MIDPOINT        = 0.7f;

// ---- Parallel tempering (S-004) ------------------------------------------
constexpr int   PT_NUM_REPLICAS          = 4;
constexpr int   PT_REPLICA_SIZE          = POOL_SIZE / PT_NUM_REPLICAS;   // 16
constexpr float PT_MUTATION_RATES[PT_NUM_REPLICAS] = { 0.005f, 0.01f, 0.02f, 0.04f };
#ifdef __CUDACC__
__device__ constexpr float d_PT_MUTATION_RATES[PT_NUM_REPLICAS] = { 0.005f, 0.01f, 0.02f, 0.04f };
#endif
constexpr int   PT_SWAP_INTERVAL         = 50;   // generations
constexpr float PT_TARGET_ACCEPT         = 0.25f;

constexpr int   STRESS_SUBPOP_COUNT      = 3;
constexpr int   STRESS_SUBPOP_SIZE       = 8;
constexpr float STRESS_SOT_DENSITIES[STRESS_SUBPOP_COUNT] = { 0.10f, 0.20f, 0.40f };
#ifdef __CUDACC__
__device__ constexpr float d_STRESS_SOT_DENSITIES[STRESS_SUBPOP_COUNT] = { 0.10f, 0.20f, 0.40f };
#endif
constexpr float STRESS_REFRESH_FRACTION  = 0.25f;
constexpr int   STRESS_HISTORY_WINDOW    = 10;
constexpr float STRESS_FAILURE_THRESHOLD = 0.50f;
constexpr int   STRESS_REFRESH_PER_SUBPOP = 1;   // 8 * 0.25 / 2 roles
constexpr int   STRESS_LINEAGE_MAX        = 128;
constexpr float STRESS_BIAS_PROBABILITY   = 0.5f;

constexpr int   STRESS_POOL_SIZE = STRESS_SUBPOP_COUNT * STRESS_SUBPOP_SIZE;  // 24

// Total organism slots: active pool + stress sub-populations.
constexpr int   TOTAL_ORG = POOL_SIZE + STRESS_POOL_SIZE;  // 88

// ---- Structural pressures (S-003, I4) ------------------------------------
// Audit: ridge-stabilized least-squares fit of bmap_64 to the role target;
// audit_mult = clamp(1 - LAMBDA_AUDIT*(1 - R^2), floor, 1).
constexpr float AUDIT_RIDGE              = 1e-3f;
constexpr int   AUDIT_MIN_SAMPLES        = 8;
constexpr float AUDIT_MULT_FLOOR         = 0.9f;
// Variance floor (Q-001 class A): descriptors below the floor are penalized
// immediately in fitness composition.
constexpr float VAR_FLOOR                = 1e-4f;
constexpr float VAR_FLOOR_MULT           = 0.5f;
// Interpretability probe panel: small linear probes trained by SGD on the
// archive snapshot; held-out accuracy is reported.
constexpr int   PROBE_PANEL_SAMPLES      = 256;
constexpr int   PROBE_PANEL_SPLIT_MOD    = 2;   // even train / odd evaluate
constexpr int   PROBE_PANEL_EPOCHS       = 40;
constexpr float PROBE_PANEL_LR           = 0.1f;
constexpr float PROBE_PANEL_TRAIN_FRACTION = 0.75f;
// Lineage runaway: per-role share threshold and the tracked-lineage table.
constexpr float LINEAGE_RUNAWAY_THRESHOLD = 0.20f;
constexpr int   LINEAGE_STATS_MAX         = 256;
constexpr int   LINEAGE_BRAKE_MAX         = 16;
// Sentinel training: examples ingested per generation from the history.
constexpr int   SENTINEL_TRAIN_PER_GEN    = 8;
constexpr float SENTINEL_LR_BASE          = 1e-3f;
constexpr float SENTINEL_LR_DECAY         = 0.05f;

// ---- SOT density (A-701) -------------------------------------------------
constexpr float MAIN_SOT_DENSITY     = 0.05f;  // 5% for main pool

// ---- Intervals (I-001) ---------------------------------------------------
constexpr int CURRICULUM_INTERVAL    = 50;
constexpr int AUDIT_INTERVAL         = 100;
constexpr int PROBE_PANEL_INTERVAL   = 200;
constexpr int TELEMETRY_INTERVAL     = 10;

// ---- Gradient health (A-501) ---------------------------------------------
constexpr float EPS_GRAD             = 1e-8f;
constexpr int   GRAD_HEALTH_WINDOW   = 10;

// ---- Error logging -------------------------------------------------------
#ifndef SLIME_DEBUG_CHECKS
#define SLIME_DEBUG_CHECKS 1
#endif

// Per-phase timing table capacity (I8 profiling).
constexpr int PHASE_TIMING_SLOTS = 24;

// ---- Numeric policy constants (N1..N5 schema home) ------------------------
// FP16 clamp bounds (forward, checkpoint re-forward, RD) and the saturation
// telemetry threshold.
constexpr float FP16_MAX_VALUE       = 65504.f;
constexpr float STATE_NEAR_MAX_VALUE = 60000.f;

// Epsilons at live seams.
constexpr float EPS_DENOM = 1e-12f;   // denominator guards (cosine, ranges)
constexpr float EPS_NORM  = 1e-24f;   // power-iteration degenerate direction
constexpr float EPS_LOG   = 1e-30f;   // log / Box-Muller guards
constexpr float EPS_REL   = 1e-4f;    // relative comparison tolerance

// GELU tanh approximation.
constexpr float GELU_K     = 0.7978845608f;
constexpr float GELU_CUBIC = 0.044715f;

// Bit / kernel-shape constants.
constexpr int WORD_BITS           = 32;   // uint32_t word width (codec)
constexpr int STENCIL_W           = 3;    // 3x3 perception taps
constexpr int HIDDEN_DIM          = 32;   // W_inter output width
constexpr int BWD_THREADS         = 256;  // backward sub-kernel block size
constexpr int NUM_CHECKPOINTS     = 4;
constexpr int CHECKPOINT_INTERVAL = 16;
constexpr int FEISTEL_ROUNDS      = 4;    // SOT permutation rounds
constexpr int PREDICTOR_SEED_REGION = 4;  // centered 4x4 predictor seed

// Reaction-diffusion explicit-step parameters (A-202).
constexpr float RD_DT    = 0.1f;
constexpr float RD_DECAY = 0.05f;

// Genome delta codec.
constexpr int   MAX_DELTA_FLOATS   = 4096;
constexpr float DELTA_PRIOR_SCALE  = 0.01f;

// Archive (A-401).
constexpr float LAMBDA_NOVELTY    = 0.5f;
constexpr float INV_VAR_EMA_ALPHA = 0.01f;
constexpr float INV_VAR_EMA_EPS   = 1e-4f;
constexpr int   POWER_ITERS       = 20;

// Curriculum (A-701).
constexpr int CLASSIFIER_BATCH_SIZE = 16;
constexpr int SOT_SUBBATCH_SIZE     = 4;
constexpr int PREDICTOR_PROBE_SLOT_COUNT = 4;
constexpr int PREDICTOR_POOL_SLOT_COUNT  = PREDICTOR_EVAL_K - PREDICTOR_PROBE_SLOT_COUNT;
// First generations always logged, independent of TELEMETRY_INTERVAL.
constexpr int FIRST_GENS_TELEMETRY  = 5;

// Reference regressor (A-601): layer sizes and AdamW hyperparameters.
constexpr int   REF_H1               = 128;
constexpr int   REF_H2               = 64;
constexpr int   REF_REPLAY_CAPACITY  = MAX_ARCHIVE;
constexpr float REF_LR               = 1e-4f;
constexpr float REF_BETA1            = 0.9f;
constexpr float REF_BETA2            = 0.999f;
constexpr float REF_EPS              = 1e-8f;
constexpr float REF_WD               = 0.01f;
constexpr int   REF_TRAIN_MINIBATCH  = 8;

// PT ladder adaptation bounds.
constexpr float PT_BETA_MIN      = 1e-3f;
constexpr float PT_BETA_MAX      = 1e3f;
constexpr float PT_BETA_EMA_RATE = 0.2f;

// Structural pressures (S-003).
constexpr float LAMBDA_AUDIT            = 0.1f;
constexpr float L_ACC_BASELINE_TRUST    = 0.6f;
constexpr float L_ACC_COLLAPSE_FRACTION = 0.85f;
constexpr int   SENTINEL_COUNT          = 32;
constexpr int   SENTINEL_HISTORY        = 1024;

// CUDA startup diagnostics probe.
constexpr size_t DIAG_MAX_PROBE_BYTES     = 64u * 1024u * 1024u;
constexpr size_t DIAG_MIN_PROBE_BYTES     = 8u * 1024u * 1024u;
constexpr int    DIAG_TRANSFER_ITERATIONS = 16;
constexpr double BYTES_PER_GIB            = 1024.0 * 1024.0 * 1024.0;

// Checkpoint format (S-001).
constexpr uint32_t CHECKPOINT_MAGIC_VALUE   = 0x53323143u;  // 'S' '2' '1' 'C'
constexpr uint32_t CHECKPOINT_VERSION_VALUE = 1;
// FNV-1a 64 checksum: standard algorithm constants.
constexpr uint64_t FNV1A64_OFFSET = 1469598103934665603ull;
constexpr uint64_t FNV1A64_PRIME  = 1099511628211ull;

#if SLIME_DEBUG_CHECKS
#define CUDA_LAUNCH_CHECK() \
    do { \
        cudaError_t _err = cudaGetLastError(); \
        if (_err != cudaSuccess) { \
            std::printf("!CUDA_ERR: %s at %s:%d\n", cudaGetErrorString(_err), __FILE__, __LINE__); \
            return; \
        } \
    } while (0)
#else
#define CUDA_LAUNCH_CHECK() ((void)0)
#endif

#endif  // COEVO_CONFIG_CONSTANTS_CUH
