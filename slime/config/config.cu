#ifndef CONFIG_CU
#define CONFIG_CU

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>

// ============================================================================
// DEBUG CONFIGURATION
// Set to 0 for release builds to eliminate error-check branching overhead
// ============================================================================
#ifndef SLIME_DEBUG_CHECKS
#define SLIME_DEBUG_CHECKS 1
#endif

// Error checking macros - compile out in release builds
#if SLIME_DEBUG_CHECKS
#define CUDA_LAUNCH_CHECK() \
    do { \
        cudaError_t _err = cudaGetLastError(); \
        if (_err != cudaSuccess) { \
            printf("!CUDA_ERR: %s at %s:%d\n", cudaGetErrorString(_err), __FILE__, __LINE__); \
            return; \
        } \
    } while(0)

#define CUDA_LAUNCH_CHECK_VAL(retval) \
    do { \
        cudaError_t _err = cudaGetLastError(); \
        if (_err != cudaSuccess) { \
            printf("!CUDA_ERR: %s at %s:%d\n", cudaGetErrorString(_err), __FILE__, __LINE__); \
            return retval; \
        } \
    } while(0)

#define SLIME_DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
// Release mode: no-op macros
#define CUDA_LAUNCH_CHECK() ((void)0)
#define CUDA_LAUNCH_CHECK_VAL(retval) ((void)0)
#define SLIME_DEBUG_PRINT(...) ((void)0)
#endif







enum SpawnWorkspaceSlot {
    SPAWN_WS_TEMP_GENOME = 0,
    SPAWN_WS_TEMP_PARENT,
    SPAWN_WS_PARENT_GENOME,
    SPAWN_WS_CHILD_GENOME,
    SPAWN_WS_PARENT_PARENT_TEMP,
    SPAWN_WS_COUNT
};

enum ChemFieldSlot {
    CHEM_CONCENTRATION = 0,
    CHEM_GRADIENT_X,
    CHEM_GRADIENT_Y,
    CHEM_LAPLACIAN,
    CHEM_SOURCES,
    CHEM_DECAY_FACTORS,
    CHEM_FIELD_COUNT
};

enum RDFieldSlot {
    RD_RESOURCE_DENSITY = 0,
    RD_RESOURCE_NEXT,
    RD_FITNESS_LANDSCAPE,
    RD_RESOURCE_GRADIENT_X,
    RD_RESOURCE_GRADIENT_Y,
    RD_FIELD_COUNT
};

constexpr int WMMA_TILE_DIM = 16;
constexpr int WARP_SIZE = 32;
constexpr int BANK_PAD = 1;

// CUDA Dynamic Parallelism (CDP) configuration
constexpr int CDP_SYNC_DEPTH = 4;
constexpr int CDP_PENDING_LAUNCH_COUNT = 2048;
constexpr int CDP_STACK_SIZE = 16384;
constexpr size_t DEVICE_MALLOC_HEAP_MB = 512;

constexpr int MAX_KERNEL_CYCLES = 100000;
constexpr int KERNEL_CYCLES_MIN = 1000;
constexpr int KERNEL_CYCLES_MAX = 50000;        


constexpr int TILE_M = WMMA_TILE_DIM;
constexpr int TILE_N = WMMA_TILE_DIM;
constexpr int TILE_K = WMMA_TILE_DIM;
constexpr int TILE_SIZE = WMMA_TILE_DIM;
constexpr int TILE_DIM = 2 * WMMA_TILE_DIM;








constexpr int GENOME_SIZE = 1024;
constexpr int POOL_CAPACITY_MIN = 8;
constexpr int POOL_CAPACITY_MAX = 64;
constexpr int MAX_ARCHIVE_SIZE = 10000;
constexpr int PARENT_COUNT = 2;
constexpr int MAX_CELLS = GENOME_SIZE;
constexpr int MAX_MEMORY_SIZE = GENOME_SIZE;
constexpr int TAPE_CAPACITY = 10 * GENOME_SIZE;
constexpr int VALUE_CAPACITY = 50 * GENOME_SIZE;
constexpr int TRACE_CAPACITY = GENOME_SIZE;
constexpr int MAX_HISTORY_LENGTH = GENOME_SIZE;
constexpr int MAX_DELTAS_PER_ENTRY = 128;
constexpr int MAX_WEIGHT_DELTAS_PER_ELITE = 512;  // Weight deltas for Lamarckian inheritance
constexpr int MAX_TAPE_SIZE = TAPE_CAPACITY * POOL_CAPACITY_MAX;
constexpr int MAX_TAPE_VALUES = VALUE_CAPACITY * POOL_CAPACITY_MAX;
constexpr int MAX_JACOBI_SWEEPS = 100;   
constexpr int MAX_SPARSE_NEIGHBORS = 10;
constexpr int MAX_CA_KERNEL_SIZE = 3;
constexpr int MAX_FLOW_KERNEL_SIZE = 3;
constexpr int CA_KERNEL_SIZE = 3;
constexpr int CA_KERNEL_CELL_COUNT = CA_KERNEL_SIZE * CA_KERNEL_SIZE;
constexpr int CA_KERNEL_NEIGHBOR_COUNT = CA_KERNEL_CELL_COUNT - 1;
constexpr int FLOW_KERNEL_SIZE = 3;







// Dataset registry
constexpr int NUM_DATASETS = 12;

// Active datasets: 0=MNIST, 1=Fashion-MNIST, 2=CIFAR-10, 3=PathMNIST, 7=UCI-HAR
#define ACTIVE_DATASET_LIST 0, 1, 2, 3, 7
constexpr int NUM_ACTIVE_DATASETS = 5;

__device__ __constant__ int ACTIVE_DATASET_IDS[NUM_ACTIVE_DATASETS] = {ACTIVE_DATASET_LIST};
constexpr int HOST_ACTIVE_DATASET_IDS[NUM_ACTIVE_DATASETS] = {ACTIVE_DATASET_LIST};

// Genome-derived dataset dimension ranges (no hardcoded MNIST values)
constexpr int NUM_CLASSES_MIN = 2;
constexpr int NUM_CLASSES_MAX = 1000;
constexpr int SAMPLE_DIM_MIN = 8;
constexpr int SAMPLE_DIM_MAX = 256;

// Dataset dimensions
constexpr int MNIST_ROWS = 28;
constexpr int MNIST_COLS = 28;
constexpr int MNIST_CHANNELS = 1;
constexpr int MNIST_CLASSES = 10;
constexpr int MNIST_TRAIN_SAMPLES = 60000;
constexpr int MNIST_TEST_SAMPLES = 10000;

constexpr int FASHION_MNIST_ROWS = 28;
constexpr int FASHION_MNIST_COLS = 28;
constexpr int FASHION_MNIST_CHANNELS = 1;
constexpr int FASHION_MNIST_CLASSES = 10;
constexpr int FASHION_MNIST_TRAIN_SAMPLES = 60000;
constexpr int FASHION_MNIST_TEST_SAMPLES = 10000;

constexpr int CIFAR10_ROWS = 32;
constexpr int CIFAR10_COLS = 32;
constexpr int CIFAR10_CHANNELS = 3;
constexpr int CIFAR10_CLASSES = 10;
constexpr int CIFAR10_TRAIN_SAMPLES = 50000;
constexpr int CIFAR10_TEST_SAMPLES = 10000;

constexpr int PATHMNIST_ROWS = 28;
constexpr int PATHMNIST_COLS = 28;
constexpr int PATHMNIST_CHANNELS = 3;
constexpr int PATHMNIST_CLASSES = 9;
constexpr int PATHMNIST_TRAIN_SAMPLES = 89996;
constexpr int PATHMNIST_TEST_SAMPLES = 7180;

constexpr int RETINAMNIST_ROWS = 28;
constexpr int RETINAMNIST_COLS = 28;
constexpr int RETINAMNIST_CHANNELS = 3;
constexpr int RETINAMNIST_CLASSES = 5;
constexpr int RETINAMNIST_TRAIN_SAMPLES = 1080;
constexpr int RETINAMNIST_TEST_SAMPLES = 120;

constexpr int CHESTXRAY_ROWS = 1024;
constexpr int CHESTXRAY_COLS = 1024;
constexpr int CHESTXRAY_CHANNELS = 1;
constexpr int CHESTXRAY_CLASSES = 14;
constexpr int CHESTXRAY_TRAIN_SAMPLES = 86524;
constexpr int CHESTXRAY_TEST_SAMPLES = 25596;
constexpr int CHESTXRAY_PYRAMID_LEVELS = 10;
constexpr int CHESTXRAY_HILBERT_ORDER = 10;

constexpr int UCIHAR_TIMESTEPS = 561;
constexpr int UCIHAR_FEATURES = 1;
constexpr int UCIHAR_CHANNELS = 1;
constexpr int UCIHAR_CLASSES = 6;
constexpr int UCIHAR_TRAIN_SAMPLES = 7352;
constexpr int UCIHAR_TEST_SAMPLES = 2947;

constexpr int MITBIH_TIMESTEPS = 187;
constexpr int MITBIH_CHANNELS = 1;
constexpr int MITBIH_CLASSES = 5;
constexpr int MITBIH_TRAIN_SAMPLES = 87554;
constexpr int MITBIH_TEST_SAMPLES = 21892;

constexpr int OPPORTUNITY_TIMESTEPS = 24;
constexpr int OPPORTUNITY_FEATURES = 113;
constexpr int OPPORTUNITY_CHANNELS = 1;
constexpr int OPPORTUNITY_CLASSES = 18;
constexpr int OPPORTUNITY_TRAIN_SAMPLES = 557963;
constexpr int OPPORTUNITY_TEST_SAMPLES = 118750;

// Audio spectrogram parameters
constexpr int AUDIO_N_FFT_SMALL = 512;
constexpr int AUDIO_N_FFT_LARGE = 2048;
constexpr int AUDIO_HOP_SMALL = 160;
constexpr int AUDIO_HOP_MEDIUM = 512;
constexpr int AUDIO_HOP_LARGE = 1024;
constexpr int AUDIO_N_MELS = 40;
constexpr int AUDIO_SPEC_CHANNELS = 3;

constexpr int AUDIO_TIME_SHORT = 101;
constexpr int AUDIO_TIME_MEDIUM = 172;
constexpr int AUDIO_TIME_LONG = 216;

constexpr int ESC50_CLASSES = 50;
constexpr int ESC50_TRAIN_SAMPLES = 1600;
constexpr int ESC50_TEST_SAMPLES = 400;

constexpr int SPEECH_COMMANDS_CLASSES = 35;
constexpr int SPEECH_COMMANDS_TRAIN_SAMPLES = 84662;
constexpr int SPEECH_COMMANDS_TEST_SAMPLES = 11005;

constexpr int URBANSOUND8K_CLASSES = 10;
constexpr int URBANSOUND8K_TRAIN_SAMPLES = 7079;
constexpr int URBANSOUND8K_TEST_SAMPLES = 1654;

constexpr int BIT_DEPTH_8 = 8;
constexpr int BIT_DEPTH_16 = 16;







constexpr int SHA256_HASH_SIZE = 32;
constexpr int PATH_BUFFER_SIZE = 512;
constexpr int BYTES_PER_KB = 1024;
constexpr int BYTES_PER_MB = BYTES_PER_KB * BYTES_PER_KB;
constexpr int TELEMETRY_EVERY_GEN = 1;
constexpr int TELEMETRY_DETAILED = 10;
constexpr int TELEMETRY_COMPREHENSIVE = 100;
constexpr int CHECKPOINT_INTERVAL = TELEMETRY_DETAILED;







// Spatial dimensions (fixed by toroidal 2D space architecture)
constexpr int AGENT_SPATIAL_DIMS = 4;  // position[2] + velocity[2]

// Hardware feature dimensions for execution trace compression
constexpr int HARDWARE_FEATURES_DIM = 15;






constexpr int GRADIENT_HISTORY = 2 * WMMA_TILE_DIM;
constexpr int NUM_CHEMICAL_FIELD_ARRAYS = 6;
constexpr int GRADIENT_SAMPLE_SIZE = 100;
constexpr int VORONOI_EXPORT_LIMIT = 100;
constexpr float OCCUPANCY_VARIANCE_WEIGHT = 1.0f;  
constexpr float ARCHIVE_DENSITY_MARGIN = 0.1f;  

constexpr int BLOCK_SIZE = 8 * WARP_SIZE;  
constexpr int BLOCK_ROWS = WMMA_TILE_DIM / 2;
constexpr int BLOCK_M = 8 * WMMA_TILE_DIM;
constexpr int BLOCK_N = 8 * WMMA_TILE_DIM;
constexpr int BLOCK_K = WMMA_TILE_DIM / 2;
















constexpr float TAU = 6.283185307179586476925286766559f;
constexpr float OCTAVE_MULTIPLIER = 2.0f;
constexpr int BASE_FEATURES_COUNT = 4;







constexpr float GELU_SQRT_2_OVER_PI = 0.7978845608f;
constexpr float GELU_CUBIC_COEFFICIENT = 0.044715f;
constexpr float GELU_SCALE = 0.5f;
constexpr float GELU_OFFSET = 1.0f;

constexpr float GAUSSIAN_VARIANCE_DENOMINATOR = 2.0f;
constexpr float CENTERED_DIFFERENCE_SCALE = 0.5f;
constexpr float QUARTER_SCALE = 0.25f;
constexpr float RECONSTRUCTION_GRADIENT_SCALE = 2.0f;











constexpr int RNG_MASK_BITS = 24;
constexpr int XORSHIFT_STATE_BITS = 64;
constexpr float RNG_NORMALIZATION_SCALE = (float)(1u << RNG_MASK_BITS);
constexpr double XORSHIFT_NORMALIZATION_SCALE = 18446744073709551616.0;
constexpr float FRACTIONAL_OU_KERNEL_OFFSET = 1.5f;

constexpr unsigned int RNG_SEED_MULTIPLIER = 1337u;
constexpr unsigned int LCG_MULTIPLIER = 1664525u;
constexpr unsigned int LCG_INCREMENT = 1013904223u;

// Knuth's golden ratio multipliers for xorshift128+ initialization
constexpr uint64_t XORSHIFT_GOLDEN_RATIO_A = 0x9e3779b97f4a7c15ULL;
constexpr uint64_t XORSHIFT_GOLDEN_RATIO_B = 0xbf58476d1ce4e5b9ULL;

// Golden ratio constant for hash mixing (phi * 2^32)
constexpr uint32_t HASH_GOLDEN_RATIO_32 = 0x9e3779b9u;

// SplitMix64 and MurmurHash3 mixing constants
constexpr uint64_t HASH_MIX_CONSTANT_A = 0xff51afd7ed558ccdULL;
constexpr uint64_t HASH_MIX_CONSTANT_B = 0xc4ceb9fe1a85ec53ULL;

// Default seed for curandState initialization
constexpr unsigned long CURAND_DEFAULT_SEED = 0x12345678UL;

constexpr int XORSHIFT128_ROTL_A = 24;
constexpr int XORSHIFT128_ROTL_B = 37;
constexpr int XORSHIFT128_SHIFT_C = 16;

constexpr int JENKINS_SHIFT_1 = 3;
constexpr int JENKINS_SHIFT_2 = 2;
constexpr int JENKINS_ROTATE = 59;

constexpr int JENKINS_MIX_SHIFT_A = 12;
constexpr int JENKINS_MIX_SHIFT_B = 19;
constexpr int JENKINS_MIX_SHIFT_C = 5;
constexpr int JENKINS_MIX_SHIFT_D = 7;
constexpr int JENKINS_MIX_ROTATE = 57;

constexpr int JENKINS_FINAL_SHIFT_A = 6;
constexpr int JENKINS_FINAL_SHIFT_B = 11;
constexpr int HASH_FINALIZER_SHIFT = 33;

constexpr unsigned int DIRESA_INIT_SEED = 42;

// DIRESA (Distance-Regularized Siamese Twin Autoencoder) - All genome-derived
constexpr int NUM_TEMPERING_REPLICAS_MIN = 1;
constexpr int NUM_TEMPERING_REPLICAS_MAX = 8;
constexpr int DIRESA_HIDDEN1_MIN = 16;
constexpr int DIRESA_HIDDEN1_MAX = 128;
constexpr int DIRESA_HIDDEN2_MIN = 8;
constexpr int DIRESA_HIDDEN2_MAX = 64;
constexpr int DIRESA_BATCH_SIZE_MIN = 128;
constexpr int DIRESA_BATCH_SIZE_MAX = 1024;
constexpr float ANNEAL_STEP_MIN = 0.05f;
constexpr float ANNEAL_STEP_MAX = 0.5f;
constexpr float COV_TARGET_MIN = 1e-6f;
constexpr float COV_TARGET_MAX = 1e-4f;
constexpr float DIST_WEIGHT_MIN = 0.1f;
constexpr float DIST_WEIGHT_MAX = 10.0f;
constexpr float RECON_WEIGHT_MIN = 0.1f;
constexpr float RECON_WEIGHT_MAX = 10.0f;
constexpr float DIRESA_DISTANCE_EXPONENT_MIN = 0.3f;
constexpr float DIRESA_DISTANCE_EXPONENT_MAX = 2.0f;
constexpr float DIRESA_QUALITY_WEIGHT_MIN = 0.01f;
constexpr float DIRESA_QUALITY_WEIGHT_MAX = 1.0f;

// Critical exponents for power-law scaling
constexpr float VORONOI_CORRELATION_EXPONENT_MIN = 0.5f;
constexpr float VORONOI_CORRELATION_EXPONENT_MAX = 1.0f;
constexpr float VORONOI_BASE_RADIUS_MIN = 0.01f;
constexpr float VORONOI_BASE_RADIUS_MAX = 0.5f;

constexpr float RANK_RENYI_ORDER_MIN = 0.5f;
constexpr float RANK_RENYI_ORDER_MAX = 2.0f;

constexpr float CHEMOTAXIS_LEVY_ALPHA_MIN = 0.5f;
constexpr float CHEMOTAXIS_LEVY_ALPHA_MAX = 2.0f;
constexpr float CHEMOTAXIS_HURST_EXPONENT_MIN = 0.1f;
constexpr float CHEMOTAXIS_HURST_EXPONENT_MAX = 0.9f;

constexpr float FITNESS_RANK_EXPONENT_MIN = 0.3f;
constexpr float FITNESS_RANK_EXPONENT_MAX = 2.0f;
constexpr float FITNESS_COHERENCE_EXPONENT_MIN = 0.3f;
constexpr float FITNESS_COHERENCE_EXPONENT_MAX = 2.0f;
constexpr float FITNESS_COUPLING_EXPONENT_MIN = -0.5f;
constexpr float FITNESS_COUPLING_EXPONENT_MAX = 0.5f;

constexpr float FOURIER_SPECTRUM_EXPONENT_MIN = 0.0f;
constexpr float FOURIER_SPECTRUM_EXPONENT_MAX = 2.0f;
constexpr float FOURIER_BASE_FREQ_MIN = 0.1f;
constexpr float FOURIER_BASE_FREQ_MAX = 2.0f;
constexpr int FOURIER_NUM_OCTAVES_MIN = 2;
constexpr int FOURIER_NUM_OCTAVES_MAX = 6;

constexpr int COHERENCE_WINDOW_SIZE_MIN = 10;
constexpr int COHERENCE_WINDOW_SIZE_MAX = 200;

// Task-grounded fitness exponents
constexpr float FITNESS_TASK_EXPONENT_MIN = 0.3f;
constexpr float FITNESS_TASK_EXPONENT_MAX = 2.0f;
constexpr float FITNESS_GEN_EXPONENT_MIN = 0.3f;
constexpr float FITNESS_GEN_EXPONENT_MAX = 2.0f;
constexpr float FITNESS_EFFICIENCY_EXPONENT_MIN = 0.1f;
constexpr float FITNESS_EFFICIENCY_EXPONENT_MAX = 1.5f;

// Baldwin Effect sensitivity
constexpr float BALDWIN_SENSITIVITY_MIN = 0.01f;
constexpr float BALDWIN_SENSITIVITY_MAX = 1.0f;

// Manifold compression dimensions
constexpr int ATTRACTOR_DIM_MIN = 4;
constexpr int ATTRACTOR_DIM_MAX = 16;
constexpr int CA_STATE_DIM_MIN = 4;
constexpr int CA_STATE_DIM_MAX = 16;
constexpr int GENOME_LATENT_DIM_MIN = 32;
constexpr int GENOME_LATENT_DIM_MAX = 128;

// Three-axis behavioral dimensions
constexpr int BEHAVIORAL_DIM_HW_MIN = 4;
constexpr int BEHAVIORAL_DIM_HW_MAX = 16;
constexpr int BEHAVIORAL_DIM_TASK_MIN = 4;
constexpr int BEHAVIORAL_DIM_TASK_MAX = 32;
constexpr int BEHAVIORAL_DIM_GEN_MIN = 2;
constexpr int BEHAVIORAL_DIM_GEN_MAX = 16;

// Compile-time maximum for stack allocations (sum of three axes)
constexpr int BEHAVIORAL_DIM_MAX = BEHAVIORAL_DIM_HW_MAX + BEHAVIORAL_DIM_TASK_MAX + BEHAVIORAL_DIM_GEN_MAX;

// DIRESA weight pool strides (6-layer autoencoder: in→H1→H2→latent→H2→H1→out)
constexpr size_t DIRESA_HW_STRIDE = HARDWARE_FEATURES_DIM * DIRESA_HIDDEN1_MAX + DIRESA_HIDDEN1_MAX + DIRESA_HIDDEN1_MAX * DIRESA_HIDDEN2_MAX + DIRESA_HIDDEN2_MAX + DIRESA_HIDDEN2_MAX * BEHAVIORAL_DIM_HW_MAX + BEHAVIORAL_DIM_HW_MAX + BEHAVIORAL_DIM_HW_MAX * DIRESA_HIDDEN2_MAX + DIRESA_HIDDEN2_MAX + DIRESA_HIDDEN2_MAX * DIRESA_HIDDEN1_MAX + DIRESA_HIDDEN1_MAX + DIRESA_HIDDEN1_MAX * HARDWARE_FEATURES_DIM + HARDWARE_FEATURES_DIM;
constexpr size_t DIRESA_TASK_STRIDE = BEHAVIORAL_DIM_TASK_MAX * DIRESA_HIDDEN1_MAX + DIRESA_HIDDEN1_MAX + DIRESA_HIDDEN1_MAX * DIRESA_HIDDEN2_MAX + DIRESA_HIDDEN2_MAX + DIRESA_HIDDEN2_MAX * BEHAVIORAL_DIM_TASK_MAX + BEHAVIORAL_DIM_TASK_MAX + BEHAVIORAL_DIM_TASK_MAX * DIRESA_HIDDEN2_MAX + DIRESA_HIDDEN2_MAX + DIRESA_HIDDEN2_MAX * DIRESA_HIDDEN1_MAX + DIRESA_HIDDEN1_MAX + DIRESA_HIDDEN1_MAX * BEHAVIORAL_DIM_TASK_MAX + BEHAVIORAL_DIM_TASK_MAX;
constexpr size_t DIRESA_GEN_STRIDE = 1 * DIRESA_HIDDEN1_MAX + DIRESA_HIDDEN1_MAX + DIRESA_HIDDEN1_MAX * DIRESA_HIDDEN2_MAX + DIRESA_HIDDEN2_MAX + DIRESA_HIDDEN2_MAX * BEHAVIORAL_DIM_GEN_MAX + BEHAVIORAL_DIM_GEN_MAX + BEHAVIORAL_DIM_GEN_MAX * DIRESA_HIDDEN2_MAX + DIRESA_HIDDEN2_MAX + DIRESA_HIDDEN2_MAX * DIRESA_HIDDEN1_MAX + DIRESA_HIDDEN1_MAX + DIRESA_HIDDEN1_MAX * 1 + 1;
constexpr size_t DIRESA_GENOME_STRIDE = GENOME_SIZE * DIRESA_HIDDEN1_MAX + DIRESA_HIDDEN1_MAX + DIRESA_HIDDEN1_MAX * DIRESA_HIDDEN2_MAX + DIRESA_HIDDEN2_MAX + DIRESA_HIDDEN2_MAX * GENOME_LATENT_DIM_MAX + GENOME_LATENT_DIM_MAX + GENOME_LATENT_DIM_MAX * DIRESA_HIDDEN2_MAX + DIRESA_HIDDEN2_MAX + DIRESA_HIDDEN2_MAX * DIRESA_HIDDEN1_MAX + DIRESA_HIDDEN1_MAX + DIRESA_HIDDEN1_MAX * GENOME_SIZE + GENOME_SIZE;

// Behavioral embedding update frequency
constexpr int EMBEDDING_UPDATE_FREQ = 10;

// Adaptive curriculum thresholds
constexpr float CURRICULUM_ACCURACY_THRESHOLD_MIN = 0.5f;
constexpr float CURRICULUM_ACCURACY_THRESHOLD_MAX = 0.95f;
constexpr float CURRICULUM_DIVERSITY_THRESHOLD_MIN = 0.1f;
constexpr float CURRICULUM_DIVERSITY_THRESHOLD_MAX = 0.8f;
constexpr float CURRICULUM_MIN_GENERATIONS_MIN = 10.0f;
constexpr float CURRICULUM_MIN_GENERATIONS_MAX = 500.0f;





constexpr int FEATURE_WARP_DIVERGENCE_ENTROPY = 0;
constexpr int FEATURE_WARP_CONVERGENCE_RATE = 1;
constexpr int FEATURE_ACTIVE_THREAD_FRACTION = 2;
constexpr int FEATURE_MEMORY_COALESCING_EFFICIENCY = 3;
constexpr int FEATURE_CACHE_LINE_UTILIZATION = 4;
constexpr int FEATURE_MEMORY_DIVERGENCE_SPREAD = 5;
constexpr int FEATURE_BANK_CONFLICT_DENSITY = 6;
constexpr int FEATURE_TENSOR_CORE_USAGE = 7;
constexpr int FEATURE_TENSOR_MEMORY_BANDWIDTH = 8;
constexpr int FEATURE_INSTRUCTION_THROUGHPUT = 9;
constexpr int FEATURE_PIPELINE_STALL_FRACTION = 10;
constexpr int FEATURE_OCCUPANCY_VARIANCE = 11;
constexpr int FEATURE_ARITHMETIC_INTENSITY = 12;
constexpr int FEATURE_MEMORY_BANDWIDTH_SATURATION = 13;
constexpr int FEATURE_INTERACTION_TERM = 14;







constexpr float PERFECT_EFFICIENCY = 1.0f;
constexpr float PERFECT_COALESCING = 1.0f;
constexpr float PERFECT_CACHE_UTILIZATION = 1.0f;

constexpr float NORMALIZED_MIN = 0.0f;
constexpr float NORMALIZED_MAX = 1.0f;

constexpr float GENOME_VALUE_MIN = -1.0f;
constexpr float GENOME_VALUE_MAX = 1.0f;
constexpr float GENOME_RANGE_SCALE = 2.0f;
constexpr float GENOME_TO_UNIT_OFFSET = 1.0f;
constexpr float GENOME_TO_UNIT_SCALE = 0.5f;














constexpr float LIFECYCLE_COHERENCE_STRESSED_MIN = 0.1f;
constexpr float LIFECYCLE_COHERENCE_STRESSED_MAX = 0.5f;
constexpr float LIFECYCLE_COHERENCE_RECOVER_MIN = 0.5f;
constexpr float LIFECYCLE_COHERENCE_RECOVER_MAX = 0.9f;
constexpr float LIFECYCLE_STRESS_ACCUM_RATE_MIN = 0.01f;
constexpr float LIFECYCLE_STRESS_ACCUM_RATE_MAX = 0.2f;
constexpr float LIFECYCLE_STRESS_DECAY_RATE_MIN = 0.01f;
constexpr float LIFECYCLE_STRESS_DECAY_RATE_MAX = 0.1f;
constexpr float LIFECYCLE_STRESS_THRESHOLD_MIN = 0.3f;
constexpr float LIFECYCLE_STRESS_THRESHOLD_MAX = 0.8f;
constexpr float LIFECYCLE_FITNESS_MULTIPLIER_MIN = 0.5f;
constexpr float LIFECYCLE_FITNESS_MULTIPLIER_MAX = 2.0f;
constexpr float LIFECYCLE_GRADIENT_STAGNATION_MIN = 0.001f;
constexpr float LIFECYCLE_GRADIENT_STAGNATION_MAX = 0.1f;
constexpr float LIFECYCLE_DORMANT_STRESS_MULT_MIN = 0.1f;
constexpr float LIFECYCLE_DORMANT_STRESS_MULT_MAX = 0.5f;


constexpr float LIFECYCLE_CRISIS_FITNESS_MULT_MIN = 0.3f;
constexpr float LIFECYCLE_CRISIS_FITNESS_MULT_MAX = 0.8f;
constexpr float LIFECYCLE_CRISIS_COHERENCE_MIN = 0.2f;
constexpr float LIFECYCLE_CRISIS_COHERENCE_MAX = 0.7f;
constexpr float LIFECYCLE_ELITE_FITNESS_INHERIT_MIN = 0.5f;
constexpr float LIFECYCLE_FITNESS_INHERIT_CENTER_MIN = 0.3f;
constexpr float LIFECYCLE_FITNESS_INHERIT_CENTER_MAX = 0.8f;
constexpr float LIFECYCLE_FITNESS_INHERIT_STEEPNESS_MIN = 5.0f;
constexpr float LIFECYCLE_FITNESS_INHERIT_STEEPNESS_MAX = 20.0f;
constexpr float LIFECYCLE_ELITE_FITNESS_INHERIT_MAX = 1.0f;
constexpr float LIFECYCLE_ELITE_COHERENCE_RESET_MIN = 0.3f;
constexpr float LIFECYCLE_ELITE_COHERENCE_RESET_MAX = 0.8f;


constexpr float LIFECYCLE_FITNESS_THRESHOLD_CENTER_MIN = 0.3f;
constexpr float LIFECYCLE_FITNESS_THRESHOLD_CENTER_MAX = 0.8f;
constexpr float LIFECYCLE_FITNESS_THRESHOLD_STEEPNESS_MIN = 5.0f;
constexpr float LIFECYCLE_FITNESS_THRESHOLD_STEEPNESS_MAX = 20.0f;

constexpr float LIFECYCLE_BOOST_THRESHOLD_CENTER_MIN = 0.6f;
constexpr float LIFECYCLE_BOOST_THRESHOLD_CENTER_MAX = 1.2f;
constexpr float LIFECYCLE_BOOST_THRESHOLD_STEEPNESS_MIN = 5.0f;
constexpr float LIFECYCLE_BOOST_THRESHOLD_STEEPNESS_MAX = 20.0f;

constexpr float LIFECYCLE_CRISIS_THRESHOLD_CENTER_MIN = 0.2f;
constexpr float LIFECYCLE_CRISIS_THRESHOLD_CENTER_MAX = 0.6f;
constexpr float LIFECYCLE_CRISIS_THRESHOLD_STEEPNESS_MIN = 5.0f;
constexpr float LIFECYCLE_CRISIS_THRESHOLD_STEEPNESS_MAX = 20.0f;






constexpr int MIN_DENSITY_INIT = MAX_ARCHIVE_SIZE * 100 - 1;







constexpr float DIFFUSIVITY_BASE_MIN = 0.001f;
constexpr float DIFFUSIVITY_BASE_MAX = 0.50f;
constexpr float REACTION_ORDER_MIN = 1.0f;
constexpr float REACTION_ORDER_MAX = 5.0f;
constexpr float REACTION_RATE_MIN = -1.0f;
constexpr float REACTION_RATE_MAX = 1.0f;
constexpr float ADVECTION_BASE_MIN = 0.0f;
constexpr float ADVECTION_BASE_MAX = 0.20f;      
constexpr float FLOW_DT_BASE_MIN = 0.001f;       
constexpr float FLOW_DT_BASE_MAX = 0.10f;        


constexpr float CHEMICAL_DECAY_BASE_MIN = 0.90f;  
constexpr float CHEMICAL_DECAY_BASE_MAX = 0.999f; 


constexpr float CHEMOTAXIS_THETA_BASE_MIN = 0.05f;   
constexpr float CHEMOTAXIS_THETA_BASE_MAX = 0.30f;   
constexpr float CHEMOTAXIS_SIGMA_BASE_MIN = 0.05f;   
constexpr float CHEMOTAXIS_SIGMA_BASE_MAX = 0.50f;   
constexpr float SENSITIVITY_BASE_MIN = 0.001f;       
constexpr float SENSITIVITY_BASE_MAX = 0.10f;        
constexpr float EXPLORATION_BASE_MIN = 0.0f;         
constexpr float EXPLORATION_BASE_MAX = 0.5f;         


constexpr float SENSITIVITY_THRESHOLD_BASE_MIN = 0.001f;  
constexpr float SENSITIVITY_THRESHOLD_BASE_MAX = 0.05f;   
constexpr float EXPLORATION_GROWTH_BASE_MIN = 1.01f;      
constexpr float EXPLORATION_GROWTH_BASE_MAX = 1.20f;      
constexpr float SENSITIVITY_DECAY_BASE_MIN = 0.80f;       
constexpr float SENSITIVITY_DECAY_BASE_MAX = 0.99f;       
constexpr float EXPLORATION_DECAY_BASE_MIN = 0.80f;       
constexpr float EXPLORATION_DECAY_BASE_MAX = 0.99f;       
constexpr float SENSITIVITY_GROWTH_BASE_MIN = 1.01f;      
constexpr float SENSITIVITY_GROWTH_BASE_MAX = 1.15f;      


constexpr float MIN_SENSITIVITY_CLAMP_BASE_MIN = 0.01f;   
constexpr float MIN_SENSITIVITY_CLAMP_BASE_MAX = 0.30f;   
constexpr float MAX_SENSITIVITY_CLAMP_BASE_MIN = 1.0f;    
constexpr float MAX_SENSITIVITY_CLAMP_BASE_MAX = 5.0f;    
constexpr float MIN_EXPLORATION_CLAMP_BASE_MIN = 0.01f;   
constexpr float MIN_EXPLORATION_CLAMP_BASE_MAX = 0.30f;   
constexpr float MAX_EXPLORATION_CLAMP_BASE_MIN = 0.5f;    
constexpr float MAX_EXPLORATION_CLAMP_BASE_MAX = 2.0f;    


constexpr float ATTRACTOR_SIGMA_BASE_MIN = 0.01f;         
constexpr float ATTRACTOR_SIGMA_BASE_MAX = 0.20f;         
constexpr float BEHAVIORAL_FIELD_SIGMA_BASE_MIN = 0.02f;  
constexpr float BEHAVIORAL_FIELD_SIGMA_BASE_MAX = 0.30f;  
constexpr float AGENT_EMBEDDING_SCALE_BASE_MIN = 0.01f;   
constexpr float AGENT_EMBEDDING_SCALE_BASE_MAX = 0.50f;   


constexpr float GRADIENT_MIX_WEIGHT_BASE_MIN = 0.0f;      
constexpr float GRADIENT_MIX_WEIGHT_BASE_MAX = 1.0f;      
constexpr float MEMORY_DECAY_RATE_BASE_MIN = 0.5f;        
constexpr float MEMORY_DECAY_RATE_BASE_MAX = 5.0f;        


constexpr float INIT_EXPLORATION_BASE_MIN = 0.1f;         
constexpr float INIT_EXPLORATION_BASE_MAX = 1.0f;         
constexpr float INIT_SENSITIVITY_BASE_MIN = 0.3f;         
constexpr float INIT_SENSITIVITY_BASE_MAX = 3.0f;         


constexpr float AGENT_SOURCE_SIGMA_BASE_MIN = 0.5f;       
constexpr float AGENT_SOURCE_SIGMA_BASE_MAX = 5.0f;       
constexpr float AGENT_SOURCE_STRENGTH_BASE_MIN = 0.1f;    
constexpr float AGENT_SOURCE_STRENGTH_BASE_MAX = 3.0f;    


constexpr float MAX_AGENT_VELOCITY_BASE_MIN = 0.001f;      
constexpr float MAX_AGENT_VELOCITY_BASE_MAX = 0.1f;        


constexpr int NUM_HEADS_MIN = 1;
constexpr int NUM_HEADS_MAX = 8;

constexpr int WMMA_ALIGNMENT = 8;
constexpr int HEAD_DIM_TILES_MIN = 1;
constexpr int HEAD_DIM_TILES_MAX = 4;
constexpr int CHANNELS_OCTETS_MIN = 1;
constexpr int CHANNELS_OCTETS_MAX = 2;

constexpr int HEAD_DIM_MIN = HEAD_DIM_TILES_MIN * WMMA_TILE_DIM;
constexpr int HEAD_DIM_MAX = HEAD_DIM_TILES_MAX * WMMA_TILE_DIM;
constexpr int CHANNELS_MIN = CHANNELS_OCTETS_MIN * WMMA_ALIGNMENT;
constexpr int CHANNELS_MAX = CHANNELS_OCTETS_MAX * WMMA_ALIGNMENT;
constexpr int HIDDEN_DIM_MIN = HEAD_DIM_MIN;
constexpr int HIDDEN_DIM_MAX = NUM_HEADS_MAX * HEAD_DIM_MAX;
constexpr int GRID_SIZE_MIN = 64;
constexpr int GRID_SIZE_MAX = 64;
constexpr int MAX_HEAD_DIM = HEAD_DIM_MAX;
constexpr int MAX_CHANNELS = CHANNELS_MAX;

constexpr int FLOW_FIELD_DIMS = 2;
constexpr int AFFINITY_REDUCED_DIMS = 1;

constexpr int CA_FIELD_SIZE = GRID_SIZE_MAX * GRID_SIZE_MAX;
constexpr int CA_CONCENTRATION_SIZE = CA_FIELD_SIZE * CHANNELS_MAX;
constexpr int CA_OUTPUT_SIZE = CA_FIELD_SIZE * NUM_HEADS_MAX * HEAD_DIM_MAX;
constexpr int CA_AFFINITY_SIZE = CA_FIELD_SIZE * AFFINITY_REDUCED_DIMS;
constexpr int CA_FLOW_SIZE = CA_FIELD_SIZE * FLOW_FIELD_DIMS;
constexpr int CA_REINTEGRATION_SIZE = CA_FIELD_SIZE * CHANNELS_MAX;
constexpr int CA_STATE_STRIDE = CA_CONCENTRATION_SIZE + CA_OUTPUT_SIZE + CA_AFFINITY_SIZE + CA_FLOW_SIZE + CA_REINTEGRATION_SIZE;

// Per-entry CA weight sizes (FP16 half precision)
constexpr int CA_PERCEPTION_WEIGHT_SIZE = NUM_HEADS_MAX * CHANNELS_MAX * HEAD_DIM_MAX;
constexpr int CA_INTERACTION_WEIGHT_SIZE = NUM_HEADS_MAX * HEAD_DIM_MAX * HEAD_DIM_MAX;
constexpr int CA_VALUE_WEIGHT_SIZE = NUM_HEADS_MAX * HEAD_DIM_MAX * CHANNELS_MAX;
constexpr int CA_WEIGHTS_PER_ENTRY_STRIDE = CA_PERCEPTION_WEIGHT_SIZE + CA_INTERACTION_WEIGHT_SIZE + CA_VALUE_WEIGHT_SIZE;

// Batch size limits (needed before SAVED_ACTIVATION_SIZE)
constexpr int BATCH_SIZE_MIN = 8;
constexpr int BATCH_SIZE_MAX = 32;

// Per-entry saved activation sizes for gradient training
// Layout: batch_size * num_heads * (grid_size * grid_size) * head_dim
constexpr int SAVED_ACTIVATION_SIZE = BATCH_SIZE_MAX * NUM_HEADS_MAX * CA_FIELD_SIZE * HEAD_DIM_MAX;

// Per-entry autodiff tape buffer sizes
constexpr int TAPE_ENTRIES_PER_ENTRY = TAPE_CAPACITY;
constexpr int TAPE_VALUES_PER_ENTRY = VALUE_CAPACITY;

// Backward pass workspace chunking - sized for memory-bounded gradient computation
// col_width = 9 * channels for 3x3 kernel im2col
constexpr int COL_WIDTH_MAX = 9 * CHANNELS_MAX;
// Chunk size determines memory/parallelism tradeoff: larger = more parallel, more memory
constexpr int BACKWARD_CHUNK_SAMPLES = 8192;
// Per-sample workspace costs at max architecture
constexpr size_t BACKWARD_WS_FP16_A_SIZE = (size_t)NUM_HEADS_MAX * BACKWARD_CHUNK_SAMPLES * HIDDEN_DIM_MAX * sizeof(half);
constexpr size_t BACKWARD_WS_FP16_B_SIZE = (size_t)NUM_HEADS_MAX * BACKWARD_CHUNK_SAMPLES * HIDDEN_DIM_MAX * sizeof(half);
constexpr size_t BACKWARD_WS_DW_SIZE = (size_t)NUM_HEADS_MAX * HIDDEN_DIM_MAX * HIDDEN_DIM_MAX * sizeof(float);
constexpr size_t BACKWARD_WS_DI_SIZE = (size_t)NUM_HEADS_MAX * BACKWARD_CHUNK_SAMPLES * HIDDEN_DIM_MAX * sizeof(float);
constexpr size_t BACKWARD_WS_W_T_SIZE = (size_t)NUM_HEADS_MAX * HIDDEN_DIM_MAX * HIDDEN_DIM_MAX * sizeof(half);
constexpr size_t BACKWARD_WS_IM2COL_SIZE = (size_t)NUM_HEADS_MAX * BACKWARD_CHUNK_SAMPLES * COL_WIDTH_MAX * sizeof(float);
constexpr size_t BACKWARD_WS_DPREGELU_SIZE = (size_t)NUM_HEADS_MAX * BACKWARD_CHUNK_SAMPLES * HIDDEN_DIM_MAX * sizeof(float);

constexpr float LEARNING_RATE_MIN = 0.0001f;
constexpr float LEARNING_RATE_MAX = 0.01f;
constexpr float BATCH_SIZE_NORM_MIN = 0.0f;
constexpr float BATCH_SIZE_NORM_MAX = 1.0f;
constexpr float DECAY_RATE_MIN = 0.9f;
constexpr float DECAY_RATE_MAX = 0.999f;
constexpr float ADAM_BETA1_MIN = 0.85f;
constexpr float ADAM_BETA1_MAX = 0.95f;
constexpr float ADAM_BETA2_MIN = 0.99f;
constexpr float ADAM_BETA2_MAX = 0.9999f;
constexpr float GRADIENT_CLIP_MIN = 0.1f;
constexpr float GRADIENT_CLIP_MAX = 10.0f;
constexpr float SPAWN_RATE_MIN = 0.01f;
constexpr float SPAWN_RATE_MAX = 0.5f;
constexpr float DECAY_THRESHOLD_MIN = 0.05f;
constexpr float DECAY_THRESHOLD_MAX = 0.3f;
constexpr float PRUNING_THRESHOLD_MIN = 0.3f;
constexpr float PRUNING_THRESHOLD_MAX = 0.8f;
constexpr float CONSOLIDATION_THRESHOLD_MIN = 0.3f;
constexpr float CONSOLIDATION_THRESHOLD_MAX = 0.8f;
constexpr float SPAWN_PROBABILITY_MIN_MIN = 0.0f;
constexpr float SPAWN_PROBABILITY_MIN_MAX = 0.1f;
constexpr float HUNGER_THRESHOLD_MIN = 0.5f;
constexpr float HUNGER_THRESHOLD_MAX = 0.99f;
constexpr float FITNESS_CULLING_MULT_MIN = 0.5f;
constexpr float FITNESS_CULLING_MULT_MAX = 2.0f;
constexpr int ARCHIVE_PRUNING_INTERVAL_MIN = 50;
constexpr int ARCHIVE_PRUNING_INTERVAL_MAX = 500;


constexpr float RESOURCE_FLOW_DT_MIN = 0.001f;
constexpr float RESOURCE_FLOW_DT_MAX = 0.1f;
constexpr float CHEMICAL_DIFFUSION_DT_MIN = 0.001f;
constexpr float CHEMICAL_DIFFUSION_DT_MAX = 0.1f;
constexpr float CHEMOTAXIS_DT_MIN = 0.01f;
constexpr float CHEMOTAXIS_DT_MAX = 0.5f;
constexpr float FLOW_LENIA_DT_MIN = 0.01f;
constexpr float FLOW_LENIA_DT_MAX = 0.5f;
constexpr float VORONOI_INIT_DT_MIN = 0.001f;
constexpr float VORONOI_INIT_DT_MAX = 0.1f;
constexpr float DEFAULT_DECAY_RATE_MIN = 0.9f;
constexpr float DEFAULT_DECAY_RATE_MAX = 0.999f;
constexpr float WARP_CA_GROWTH_RATE_MIN = 0.01f;
constexpr float WARP_CA_GROWTH_RATE_MAX = 0.5f;


constexpr float RD_U_INIT_MIN = 0.5f;
constexpr float RD_U_INIT_MAX = 1.5f;
constexpr float RD_PERTURBATION_RADIUS_MIN = 0.05f;
constexpr float RD_PERTURBATION_RADIUS_MAX = 0.3f;
constexpr float RD_FEED_RATE_MIN = 0.01f;
constexpr float RD_FEED_RATE_MAX = 0.1f;
constexpr float RD_KILL_RATE_MIN = 0.03f;
constexpr float RD_KILL_RATE_MAX = 0.09f;
constexpr float RD_DIFFUSION_COEFF_MIN = 0.5f;
constexpr float RD_DIFFUSION_COEFF_MAX = 2.0f;
constexpr float FLOW_LENIA_S_MIN = 0.1f;
constexpr float FLOW_LENIA_S_MAX = 3.0f;
constexpr float FLOW_LENIA_BETA_A_MIN = 1.0f;
constexpr float FLOW_LENIA_BETA_A_MAX = 20.0f;
constexpr float FLOW_LENIA_N_MIN = 1.0f;
constexpr float FLOW_LENIA_N_MAX = 4.0f;
constexpr float FLOW_LENIA_ALPHA_MIN_MIN = 0.0f;
constexpr float FLOW_LENIA_ALPHA_MIN_MAX = 0.3f;
constexpr float FLOW_LENIA_ALPHA_MAX_MIN = 0.7f;
constexpr float FLOW_LENIA_ALPHA_MAX_MAX = 1.0f;
constexpr float FLOW_LENIA_SHARPNESS_MIN = 1.0f;
constexpr float FLOW_LENIA_SHARPNESS_MAX = 50.0f;
constexpr float RD_V_PERTURBATION_MIN = 0.1f;
constexpr float RD_V_PERTURBATION_MAX = 0.5f;
constexpr float RESOURCE_INIT_MIN = 0.5f;
constexpr float RESOURCE_INIT_MAX = 2.0f;
constexpr float RESOURCE_NOISE_MIN = 0.0f;
constexpr float RESOURCE_NOISE_MAX = 0.3f;

constexpr float ARCHIVE_ACCEPTANCE_NOVELTY_WEIGHT = 0.5f;
constexpr float ARCHIVE_ACCEPTANCE_QUALITY_WEIGHT = 0.5f;

constexpr int AUDIT_SAMPLE_COUNT = 8;

struct AuditBuffer {
    volatile int ready;
    volatile int consumed;
    int generation;
    int batch_size;
    int num_classes;
    int grid_size;
    int correct_count;
    float loss;
    float accuracy;

    // Arrays sized by CA_FIELD_SIZE - data is resampled to grid_size
    unsigned char sample_images[AUDIT_SAMPLE_COUNT * CA_FIELD_SIZE];
    int sample_labels[AUDIT_SAMPLE_COUNT];
    float sample_logits[AUDIT_SAMPLE_COUNT * NUM_CLASSES_MAX];
    int sample_predictions[AUDIT_SAMPLE_COUNT];
    float sample_confidences[AUDIT_SAMPLE_COUNT];

    float ca_snapshot[CA_FIELD_SIZE];

    float train_accuracy;
    float test_accuracy;
    float generalization_gap;

    // Pool metrics
    int pool_alive_count;
    int pool_capacity;

    // === MAP-ELITES EVOLUTIONARY ARC ===

    // Coverage dynamics
    int archive_occupied_cells;
    int frontier_cells_gained;
    int frontier_cells_lost;
    int sparse_cell_count;
    float niche_entropy;
    float novelty_gradient;

    // Quality dynamics
    float elite_fitness_best;
    float elite_fitness_mean;
    float elite_fitness_delta;
    float quality_floor;
    float quality_mean;
    float quality_range;

    // Density distribution
    float density_mean;
    float density_max;
    float density_variance;

    // 3-axis behavioral spread (hw, task, gen)
    float hw_axis_min, hw_axis_max, hw_axis_mean;
    float task_axis_min, task_axis_max, task_axis_mean;
    float gen_axis_min, gen_axis_max, gen_axis_mean;

    // Population flow
    int total_population;
    int births_this_gen;
    int deaths_this_gen;

    // === DIRESA (pre vs post encoding) ===
    float diresa_recon_loss_hw;
    float diresa_recon_loss_task;
    float diresa_recon_loss_gen;
    float diresa_recon_loss_total;
    float diresa_behavioral_drift;
    float diresa_latent_utilization;

    // === GENOME COMPLEXITY ===
    int genome_unique_hashes;
    float genome_hash_entropy;
    float genome_avg_deltas;

    // Per-class accuracy tracking
    float per_class_correct[NUM_CLASSES_MAX];
    float per_class_total[NUM_CLASSES_MAX];

    // === POOL ENTRY SNAPSHOTS ===
    // Per-entry data for forensic pool state export
    int pool_entry_alive[POOL_CAPACITY_MAX];
    float pool_entry_fitness[POOL_CAPACITY_MAX];
    float pool_entry_hunger[POOL_CAPACITY_MAX];
    int pool_entry_age[POOL_CAPACITY_MAX];
    int pool_entry_num_deltas[POOL_CAPACITY_MAX];
    uint64_t pool_entry_genome_hash[POOL_CAPACITY_MAX];
};

#endif
