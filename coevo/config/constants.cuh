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

// Perception (A-201): a learned bank of depthwise 3x3 filters (W_perc). Each
// filter is convolved over every channel's neighborhood, so the perception
// vector is N_PERC_FILTERS * CA_CHANNELS wide. Filters are shared across
// channels, so W_perc holds N_PERC_FILTERS * 9 weights.
constexpr int N_PERC_FILTERS   = 3;
constexpr int W_PERC_SIZE      = N_PERC_FILTERS * 9;            // 27

// The CA updates all 16 channels each step (including the chemical channels
// 0-5: cells produce/consume morphogens). Reaction-diffusion (A-202) then adds
// spatial diffusion + decay to channels 0-5 on top of that cellwise update.

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

constexpr int   STRESS_POOL_SIZE = STRESS_SUBPOP_COUNT * STRESS_SUBPOP_SIZE;  // 24

// Total organism slots: active pool + stress sub-populations.
constexpr int   TOTAL_ORG = POOL_SIZE + STRESS_POOL_SIZE;  // 88

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
