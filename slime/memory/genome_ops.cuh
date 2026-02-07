#ifndef GENOME_OPS_CUH
#define GENOME_OPS_CUH

#include "../config/config.cu"
#include "../debug/param_validator.cu"
#include <cuda_runtime.h>

struct GPUElite;
struct MultiHeadCAState;

struct PoolEntry {
    int id;
    float fitness;
    float coherence;
    float task_accuracy;
    float train_accuracy;
    float test_accuracy;
    float task_loss;
    float classification_stability;
    float avg_confidence;
    int per_class_correct[NUM_CLASSES_MAX];
    int per_class_total[NUM_CLASSES_MAX];
    float generalization_gap;
    float hardware_efficiency;
    float gradient_magnitude;
    float effective_rank;
    float recon_loss_hw;
    float recon_loss_task;
    float recon_loss_gen;
    float recon_loss_total;
    float behavioral_drift_rate;
    float latent_utilization;
    float compression_ratio;
    float hardware_feature_correlation;
    float hunger;
    int age;
    bool alive;
    uint64_t genome_hash;
    int generation;
    float* gradients;
    uint64_t parent_hash;
    int parent_idx;
    uint16_t num_deltas;
    uint16_t max_deltas;
    uint16_t* delta_indices;
    float* delta_values;
    int num_heads;
    int channels;
    int hidden_dim;
    int head_dim;
    int grid_size;
    int num_tempering_replicas;
    int diresa_hidden1;
    int diresa_hidden2;
    int diresa_batch_size;
    float anneal_step;
    float cov_target;
    unsigned long long active_warps;
    unsigned long long divergent_branches;
    unsigned long long total_branches;
    unsigned long long global_loads;
    unsigned long long global_stores;
    unsigned long long l2_transactions;
    unsigned long long dram_transactions;
    unsigned long long inst_executed;
    unsigned long long inst_issued;
    unsigned long long cycles_elapsed;
    unsigned long long tensor_core_cycles;
    float dist_weight;
    float recon_weight;
    float distance_exponent;
    float quality_weight;
    float fitness_rank_exponent;
    float fitness_coherence_exponent;
    float fitness_coupling_exponent;
    float fitness_task_exponent;
    float fitness_gen_exponent;
    float fitness_efficiency_exponent;
    float baldwin_sensitivity;
    int coherence_window_size;
    float renyi_q;

    float flow_beta_A;
    float flow_n;
    float flow_s;
    float flow_alpha_min;
    float flow_alpha_max;
    float flow_sharpness;
    float flow_resource_dt;

    MultiHeadCAState* ca_state;
};

__device__ constexpr uint64_t fnv1a_hash(const char* str) {
    uint64_t hash = 14695981039346656037ULL;
    int cycles = 0;
    while (*str && cycles < MAX_KERNEL_CYCLES) {
        hash ^= (uint64_t)(*str++);
        hash *= 1099511628211ULL;
        cycles++;
    }
    return hash;
}



__device__ __forceinline__ int derive_param_slot(uint64_t genome_hash, const char* param_id) {
    uint64_t id_hash = fnv1a_hash(param_id);

    uint64_t combined = genome_hash ^ id_hash;
    combined ^= (combined >> 33);
    combined *= HASH_MIX_CONSTANT_A;
    combined ^= (combined >> 33);
    combined *= HASH_MIX_CONSTANT_B;
    combined ^= (combined >> 33);

    int slot = (int)(combined % GENOME_SIZE);
    DEVICE_VALIDATE_GENOME_SLOT(slot);
    return slot;
}

__device__ __forceinline__ float genome_to_bootstrap_param(
    const float* genome,
    const float* epigenetic,
    int primary_slot,
    float min_val,
    float max_val
) {
    DEVICE_VALIDATE_PTR(genome);
    DEVICE_VALIDATE_GENOME_SLOT(primary_slot);
    float base_val = genome[primary_slot];
    DEVICE_VALIDATE_FINITE(base_val);

    int epi_slot = (primary_slot * 0x9e3779b9 + 0x7f4a7c15) % GENOME_SIZE;
    DEVICE_VALIDATE_GENOME_SLOT(epi_slot);
    float epigenetic_sensitivity = (genome[epi_slot] + 1.0f) * 0.5f;

    int epi_bounds_slot = (primary_slot * 0x3c9f82a5 + 0x1ce4e5b9) % GENOME_SIZE;
    DEVICE_VALIDATE_GENOME_SLOT(epi_bounds_slot);
    float epi_min = genome[epi_bounds_slot] * 0.5f;
    float epi_max = 2.0f + genome[epi_bounds_slot];

    float epigenetic_factor = 1.0f;
    if (epigenetic != nullptr) {
        epigenetic_factor = 1.0f + epigenetic_sensitivity * epigenetic[primary_slot];
        epigenetic_factor = fminf(fmaxf(epigenetic_factor, epi_min), epi_max);
        DEVICE_VALIDATE_EPIGENETIC(epigenetic_factor, epi_min, epi_max);
    }

    int enhancer_1_slot = (primary_slot * 0x51b3e1f4 + 0x3f2a9c71) % GENOME_SIZE;
    int enhancer_2_slot = (primary_slot * 0x7a2e914c + 0x5d8b4e3a) % GENOME_SIZE;
    int enhancer_3_slot = (primary_slot * 0x2f1a8c9d + 0x9b7e5f2c) % GENOME_SIZE;
    DEVICE_VALIDATE_GENOME_SLOT(enhancer_1_slot);
    DEVICE_VALIDATE_GENOME_SLOT(enhancer_2_slot);
    DEVICE_VALIDATE_GENOME_SLOT(enhancer_3_slot);

    float internal_modulation = 0.1f * (genome[enhancer_1_slot] + genome[enhancer_2_slot] + genome[enhancer_3_slot]);

    float expressed_val = base_val * epigenetic_factor * (1.0f + internal_modulation);
    DEVICE_VALIDATE_FINITE(expressed_val);

    float normalized = tanhf(expressed_val);
    normalized = (normalized + 1.0f) * 0.5f;
    DEVICE_VALIDATE_PROBABILITY(normalized);

    float result = min_val + (max_val - min_val) * normalized;
    DEVICE_VALIDATE_FINITE(result);
    return result;
}

struct InitContext {
    float metabolic;
    float stress;
    float morphogen;

    __device__ __forceinline__ void derive_from_genome(uint64_t genome_hash, const float* genome, const float* epigenetic) {
        int ctx_metabolic_slot = derive_param_slot(genome_hash, "init_context_metabolic");
        int ctx_stress_slot = derive_param_slot(genome_hash, "init_context_stress");
        int ctx_morphogen_slot = derive_param_slot(genome_hash, "init_context_morphogen");

        int metabolic_min_slot = derive_param_slot(genome_hash, "init_context_metabolic_min");
        int metabolic_max_slot = derive_param_slot(genome_hash, "init_context_metabolic_max");
        int stress_min_slot = derive_param_slot(genome_hash, "init_context_stress_min");
        int stress_max_slot = derive_param_slot(genome_hash, "init_context_stress_max");
        int morphogen_min_slot = derive_param_slot(genome_hash, "init_context_morphogen_min");
        int morphogen_max_slot = derive_param_slot(genome_hash, "init_context_morphogen_max");

        float metabolic_min = (genome[metabolic_min_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float metabolic_max_raw = (genome[metabolic_max_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float metabolic_max = metabolic_min + metabolic_max_raw * (NORMALIZED_MAX - metabolic_min);

        float stress_min = (genome[stress_min_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float stress_max_raw = (genome[stress_max_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float stress_max = stress_min + stress_max_raw * (NORMALIZED_MAX - stress_min);

        float morphogen_min = (genome[morphogen_min_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float morphogen_max_raw = (genome[morphogen_max_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float morphogen_max = morphogen_min + morphogen_max_raw * (NORMALIZED_MAX - morphogen_min);

        metabolic = genome_to_bootstrap_param(genome, epigenetic, ctx_metabolic_slot, metabolic_min, metabolic_max);
        stress = genome_to_bootstrap_param(genome, epigenetic, ctx_stress_slot, stress_min, stress_max);
        morphogen = genome_to_bootstrap_param(genome, epigenetic, ctx_morphogen_slot, morphogen_min, morphogen_max);
    }
};

struct BehavioralDimensions {
    int hw_dim;
    int task_dim;
    int gen_dim;

    __device__ __forceinline__ void derive_from_genome(uint64_t genome_hash, const float* genome, const float* epigenetic) {
        int hw_dim_slot = derive_param_slot(genome_hash, "behavioral_dim_hw");
        int task_dim_slot = derive_param_slot(genome_hash, "behavioral_dim_task");
        int gen_dim_slot = derive_param_slot(genome_hash, "behavioral_dim_gen");

        int hw_min_slot = derive_param_slot(genome_hash, "behavioral_dim_hw_min");
        int hw_max_slot = derive_param_slot(genome_hash, "behavioral_dim_hw_max");
        int task_min_slot = derive_param_slot(genome_hash, "behavioral_dim_task_min");
        int task_max_slot = derive_param_slot(genome_hash, "behavioral_dim_task_max");
        int gen_min_slot = derive_param_slot(genome_hash, "behavioral_dim_gen_min");
        int gen_max_slot = derive_param_slot(genome_hash, "behavioral_dim_gen_max");

        float hw_min = (genome[hw_min_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float hw_max_raw = (genome[hw_max_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float hw_max = hw_min + hw_max_raw * (NORMALIZED_MAX - hw_min);

        float task_min = (genome[task_min_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float task_max_raw = (genome[task_max_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float task_max = task_min + task_max_raw * (NORMALIZED_MAX - task_min);

        float gen_min = (genome[gen_min_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float gen_max_raw = (genome[gen_max_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float gen_max = gen_min + gen_max_raw * (NORMALIZED_MAX - gen_min);

        float hw_norm = genome_to_bootstrap_param(genome, epigenetic, hw_dim_slot, hw_min, hw_max);
        float task_norm = genome_to_bootstrap_param(genome, epigenetic, task_dim_slot, task_min, task_max);
        float gen_norm = genome_to_bootstrap_param(genome, epigenetic, gen_dim_slot, gen_min, gen_max);

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
    int enhancer_4_slot = (primary_slot * 0x4b7c2e91 + 0x6a5d3f1c) % GENOME_SIZE;  
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
            enhancer_4 * context_complexity +    
            enhancer_5 * context_niche +         
            enhancer_6 * context_learning +      
            enhancer_7 * context_performance     
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
