#ifndef GENOME_OPS_CUH
#define GENOME_OPS_CUH

#include "../config/config.cu"
#include <cuda_runtime.h>

struct GPUElite;
struct PoolEntry;

__device__ constexpr uint64_t fnv1a_hash(const char* str) {
    uint64_t hash = 14695981039346656037ULL;
    while (*str) {
        hash ^= (uint64_t)(*str++);
        hash *= 1099511628211ULL;
    }
    return hash;
}



__device__ __forceinline__ int derive_param_slot(uint64_t genome_hash, const char* param_id) {
    uint64_t id_hash = fnv1a_hash(param_id);

    
    uint64_t combined = genome_hash ^ id_hash;
    combined ^= (combined >> 33);
    combined *= 0xff51afd7ed558ccdULL;
    combined ^= (combined >> 33);
    combined *= 0xc4ceb9fe1a85ec53ULL;
    combined ^= (combined >> 33);

    return (int)(combined % GENOME_SIZE);
}


struct InitContext {
    float metabolic;
    float stress;
    float morphogen;

    __device__ __forceinline__ void derive_from_genome(uint64_t genome_hash, const float* genome) {
        int ctx_metabolic_slot = derive_param_slot(genome_hash, "init_context_metabolic");
        int ctx_stress_slot = derive_param_slot(genome_hash, "init_context_stress");
        int ctx_morphogen_slot = derive_param_slot(genome_hash, "init_context_morphogen");
        metabolic = (genome[ctx_metabolic_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        stress = (genome[ctx_stress_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        morphogen = (genome[ctx_morphogen_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
    }
};

struct BehavioralDimensions {
    int hw_dim;
    int task_dim;
    int gen_dim;

    // Must be called with genome reconstructed via reconstruct_genome_from_archive()
    // Architectural enforcement: genomes only exist after decompression
    __device__ __forceinline__ void derive_from_genome(uint64_t genome_hash, const float* genome) {
        int hw_dim_slot = derive_param_slot(genome_hash, "behavioral_dim_hw");
        int task_dim_slot = derive_param_slot(genome_hash, "behavioral_dim_task");
        int gen_dim_slot = derive_param_slot(genome_hash, "behavioral_dim_gen");

        float hw_norm = (genome[hw_dim_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float task_norm = (genome[task_dim_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float gen_norm = (genome[gen_dim_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;

        hw_dim = BEHAVIORAL_DIM_HW_MIN + (int)(hw_norm * (BEHAVIORAL_DIM_HW_MAX - BEHAVIORAL_DIM_HW_MIN));
        task_dim = BEHAVIORAL_DIM_TASK_MIN + (int)(task_norm * (BEHAVIORAL_DIM_TASK_MAX - BEHAVIORAL_DIM_TASK_MIN));
        gen_dim = BEHAVIORAL_DIM_GEN_MIN + (int)(gen_norm * (BEHAVIORAL_DIM_GEN_MAX - BEHAVIORAL_DIM_GEN_MIN));
    }

    __device__ __forceinline__ int total() const {
        return hw_dim + task_dim + gen_dim;
    }
};

__device__ __forceinline__ float genome_to_param_impl(
    const float* genome,
    const float* epigenetic,
    int primary_slot,
    float context_metabolic,
    float context_stress,
    float context_morphogen,
    float context_complexity,
    float context_niche,
    float context_learning,
    float context_performance,
    float min_val,
    float max_val
) {
    #ifndef GENOME_SIZE
    #define GENOME_SIZE 1024
    #endif


    float base_val = genome[primary_slot];



    int epi_slot = (primary_slot * 0x9e3779b9 + 0x7f4a7c15) % GENOME_SIZE;
    float epigenetic_sensitivity = (genome[epi_slot] + 1.0f) * 0.5f;


    int epi_bounds_slot = (primary_slot * 0x3c9f82a5 + 0x1ce4e5b9) % GENOME_SIZE;
    float epi_min = genome[epi_bounds_slot] * 0.5f;
    float epi_max = 2.0f + genome[epi_bounds_slot];

    float epigenetic_factor = 1.0f + epigenetic_sensitivity * epigenetic[primary_slot];
    epigenetic_factor = fminf(fmaxf(epigenetic_factor, epi_min), epi_max);


    int enhancer_1_slot = (primary_slot * 0x51b3e1f4 + 0x3f2a9c71) % GENOME_SIZE;
    int enhancer_2_slot = (primary_slot * 0x7a2e914c + 0x5d8b4e3a) % GENOME_SIZE;
    int enhancer_3_slot = (primary_slot * 0x2f1a8c9d + 0x9b7e5f2c) % GENOME_SIZE;
    int enhancer_4_slot = (primary_slot * 0x4b7c2e91 + 0x6a5d3f1c) % GENOME_SIZE;  // Telemetry enhancers
    int enhancer_5_slot = (primary_slot * 0x9c3e7a2f + 0x8b4f1d6e) % GENOME_SIZE;
    int enhancer_6_slot = (primary_slot * 0x6f1b8d3c + 0x2e9a4c5b) % GENOME_SIZE;
    int enhancer_7_slot = (primary_slot * 0x3a7f5e2c + 0x1d6b9f4a) % GENOME_SIZE;

    float enhancer_1 = genome[enhancer_1_slot];
    float enhancer_2 = genome[enhancer_2_slot];
    float enhancer_3 = genome[enhancer_3_slot];
    float enhancer_4 = genome[enhancer_4_slot];
    float enhancer_5 = genome[enhancer_5_slot];
    float enhancer_6 = genome[enhancer_6_slot];
    float enhancer_7 = genome[enhancer_7_slot];


    int ctx_slot = (primary_slot * 0x8e5f3c2a + 0x4d9b7e1f) % GENOME_SIZE;
    float context_coupling = (genome[ctx_slot] + 1.0f) * 0.25f;

    float context_modulation =
        context_coupling * (
            enhancer_1 * context_metabolic +
            enhancer_2 * context_stress +
            enhancer_3 * context_morphogen +
            enhancer_4 * context_complexity +    // Self-awareness (genetic diversity perception)
            enhancer_5 * context_niche +         // Environmental awareness (archive topology)
            enhancer_6 * context_learning +      // Learning state awareness (DIRESA drift)
            enhancer_7 * context_performance     // Performance awareness (MNIST accuracy)
        );

    float expressed_val = base_val * epigenetic_factor * (1.0f + context_modulation);


    float normalized = tanhf(expressed_val);
    normalized = (normalized + 1.0f) * 0.5f;

    return min_val + (max_val - min_val) * normalized;
}

__device__ __forceinline__ float genome_to_param(
    const float* genome,
    const float* epigenetic,
    int primary_slot,
    float context_metabolic,
    float context_stress,
    float context_morphogen,
    float context_complexity,
    float context_niche,
    float context_learning,
    float context_performance,
    float min_val,
    float max_val
) {
    return genome_to_param_impl(genome, epigenetic, primary_slot,
                                context_metabolic, context_stress, context_morphogen,
                                context_complexity, context_niche, context_learning, context_performance,
                                min_val, max_val);
}

#endif
