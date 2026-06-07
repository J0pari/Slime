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

// ---- Substrate ------------------------------------------------------------
// A-201: 16-channel 64×64 NCA, 64 CA steps per forward pass.
constexpr int GRID_SIZE        = 64;
constexpr int CA_CHANNELS      = 16;
constexpr int CA_STEPS         = 64;
constexpr int BMAP_DIM         = 32;
constexpr int TASK_EMBED_DIM   = 16;

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

// Channel ownership for the per-step update (A-201 / A-202):
//   chemical channels 0-5 are owned by reaction-diffusion (rd_step). The
//   perception/interaction path reads them as context but does not overwrite
//   them. The CA delta therefore updates only the non-chemical channels 6-15.
constexpr int CA_OUT_FIRST     = CH_TASK_FIRST;                 // 6
constexpr int CA_OUT_CHANNELS  = CA_CHANNELS - CH_TASK_FIRST;   // 10 (channels 6..15)

// BTRAJ sample steps (A-201).
constexpr int BTRAJ_SAMPLES    = 4;
constexpr int BTRAJ_STEPS[BTRAJ_SAMPLES] = { 16, 32, 48, 64 };

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

// SOT gate (A-401): sigmoid(20·(x − 0.7)).
constexpr float SOT_GATE_SLOPE           = 20.0f;
constexpr float SOT_GATE_MIDPOINT        = 0.7f;

// ---- Parallel tempering (S-004) ------------------------------------------
constexpr int   PT_NUM_REPLICAS          = 4;
constexpr int   PT_REPLICA_SIZE          = POOL_SIZE / PT_NUM_REPLICAS;   // 16
constexpr float PT_MUTATION_RATES[PT_NUM_REPLICAS] = { 0.005f, 0.01f, 0.02f, 0.04f };
constexpr int   PT_SWAP_INTERVAL         = 50;   // generations
constexpr float PT_TARGET_ACCEPT         = 0.25f;

constexpr int   STRESS_SUBPOP_COUNT      = 3;
constexpr int   STRESS_SUBPOP_SIZE       = 8;
constexpr float STRESS_SOT_DENSITIES[STRESS_SUBPOP_COUNT] = { 0.10f, 0.20f, 0.40f };
constexpr float STRESS_REFRESH_FRACTION  = 0.25f;
constexpr int   STRESS_HISTORY_WINDOW    = 10;
constexpr float STRESS_FAILURE_THRESHOLD = 0.50f;

constexpr int   STRESS_POOL_SIZE = STRESS_SUBPOP_COUNT * STRESS_SUBPOP_SIZE;  // 24

// ---- Intervals (I-001) ---------------------------------------------------
constexpr int CURRICULUM_INTERVAL    = 50;
constexpr int AUDIT_INTERVAL         = 100;
constexpr int PROBE_PANEL_INTERVAL   = 200;

// ---- Error logging -------------------------------------------------------
#ifndef SLIME_DEBUG_CHECKS
#define SLIME_DEBUG_CHECKS 1
#endif

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
