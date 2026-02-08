#ifndef GENOME_PARAMS_CUH
#define GENOME_PARAMS_CUH

#include "cuda_primitives.cuh"
#include "../memory/genome_ops.cuh"





struct ArchitectureParams {
    int num_heads;
    int channels;
    int hidden_dim;
    int head_dim;
    int grid_size;
    float ca_gate_center;  
};





struct PoolInitParams {
    float initial_hunger;
    float mutation_scale;
    float mutation_levy_alpha;
    float diversity_normalization;

    __device__ __forceinline__ void derive_from_genome(const float* genome, const float* epigenetic) {
        initial_hunger = genome_to_bootstrap_param(genome, epigenetic, GenomeParamTable::initial_hunger, INITIAL_HUNGER_MIN, INITIAL_HUNGER_MAX);
        mutation_scale = genome_to_bootstrap_param(genome, epigenetic, GenomeParamTable::genome_mutation_scale, GENOME_MUTATION_SCALE_MIN, GENOME_MUTATION_SCALE_MAX);
        mutation_levy_alpha = genome_to_bootstrap_param(genome, epigenetic, GenomeParamTable::mutation_levy_alpha, CHEMOTAXIS_LEVY_ALPHA_MIN, CHEMOTAXIS_LEVY_ALPHA_MAX);
        diversity_normalization = genome_to_bootstrap_param(genome, epigenetic, GenomeParamTable::diversity_normalization, DIVERSITY_NORMALIZATION_MIN, DIVERSITY_NORMALIZATION_MAX);
    }
};





struct ChemotaxisParams {
    __device__ __forceinline__ int get_theta_slot() { return GenomeParamTable::chemotaxis_theta; }
    __device__ __forceinline__ int get_sigma_slot() { return GenomeParamTable::chemotaxis_sigma; }
    __device__ __forceinline__ int get_sensitivity_threshold_slot() { return GenomeParamTable::chemotaxis_sensitivity_threshold; }
    __device__ __forceinline__ int get_exploration_growth_slot() { return GenomeParamTable::chemotaxis_exploration_growth; }
    __device__ __forceinline__ int get_sensitivity_decay_slot() { return GenomeParamTable::chemotaxis_sensitivity_decay; }
    __device__ __forceinline__ int get_exploration_decay_slot() { return GenomeParamTable::chemotaxis_exploration_decay; }
    __device__ __forceinline__ int get_sensitivity_growth_slot() { return GenomeParamTable::chemotaxis_sensitivity_growth; }
    __device__ __forceinline__ int get_min_sensitivity_slot() { return GenomeParamTable::chemotaxis_min_sensitivity_clamp; }
    __device__ __forceinline__ int get_max_sensitivity_slot() { return GenomeParamTable::chemotaxis_max_sensitivity_clamp; }
    __device__ __forceinline__ int get_min_exploration_slot() { return GenomeParamTable::chemotaxis_min_exploration_clamp; }
    __device__ __forceinline__ int get_max_exploration_slot() { return GenomeParamTable::chemotaxis_max_exploration_clamp; }
    __device__ __forceinline__ int get_gradient_mix_weight_slot() { return GenomeParamTable::chemotaxis_gradient_mix_weight; }
    __device__ __forceinline__ int get_memory_decay_rate_slot() { return GenomeParamTable::chemotaxis_memory_decay_rate; }
    __device__ __forceinline__ int get_attractor_sigma_slot() { return GenomeParamTable::chemotaxis_attractor_sigma; }
    __device__ __forceinline__ int get_behavioral_field_sigma_slot() { return GenomeParamTable::chemotaxis_behavioral_field_sigma; }
    __device__ __forceinline__ int get_agent_embedding_scale_slot() { return GenomeParamTable::chemotaxis_agent_embedding_scale; }
    __device__ __forceinline__ int get_init_exploration_slot() { return GenomeParamTable::chemotaxis_init_exploration; }
    __device__ __forceinline__ int get_init_sensitivity_slot() { return GenomeParamTable::chemotaxis_init_sensitivity; }
    __device__ __forceinline__ int get_agent_source_sigma_slot() { return GenomeParamTable::chemotaxis_agent_source_sigma; }
    __device__ __forceinline__ int get_agent_source_strength_slot() { return GenomeParamTable::chemotaxis_agent_source_strength; }
    __device__ __forceinline__ int get_chemical_decay_slot() { return GenomeParamTable::chemotaxis_chemical_decay; }
    __device__ __forceinline__ int get_hurst_exponent_slot() { return GenomeParamTable::chemotaxis_hurst_exponent; }
    __device__ __forceinline__ int get_levy_alpha_slot() { return GenomeParamTable::chemotaxis_levy_alpha; }

    __device__ __forceinline__ float get_theta(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::chemotaxis_theta, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, CHEMOTAXIS_THETA_BASE_MIN, CHEMOTAXIS_THETA_BASE_MAX);
    }

    __device__ __forceinline__ float get_sigma(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::chemotaxis_sigma, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, CHEMOTAXIS_SIGMA_BASE_MIN, CHEMOTAXIS_SIGMA_BASE_MAX);
    }

    __device__ __forceinline__ float get_sensitivity_threshold(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::chemotaxis_sensitivity_threshold, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, SENSITIVITY_THRESHOLD_BASE_MIN, SENSITIVITY_THRESHOLD_BASE_MAX);
    }

    __device__ __forceinline__ float get_exploration_growth(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::chemotaxis_exploration_growth, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, EXPLORATION_GROWTH_BASE_MIN, EXPLORATION_GROWTH_BASE_MAX);
    }

    __device__ __forceinline__ float get_sensitivity_decay(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::chemotaxis_sensitivity_decay, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, SENSITIVITY_DECAY_BASE_MIN, SENSITIVITY_DECAY_BASE_MAX);
    }

    __device__ __forceinline__ float get_exploration_decay(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::chemotaxis_exploration_decay, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, EXPLORATION_DECAY_BASE_MIN, EXPLORATION_DECAY_BASE_MAX);
    }

    __device__ __forceinline__ float get_sensitivity_growth(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::chemotaxis_sensitivity_growth, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, SENSITIVITY_GROWTH_BASE_MIN, SENSITIVITY_GROWTH_BASE_MAX);
    }

    __device__ __forceinline__ float get_min_sensitivity_clamp(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::chemotaxis_min_sensitivity_clamp, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, MIN_SENSITIVITY_CLAMP_BASE_MIN, MIN_SENSITIVITY_CLAMP_BASE_MAX);
    }

    __device__ __forceinline__ float get_max_sensitivity_clamp(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::chemotaxis_max_sensitivity_clamp, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, MAX_SENSITIVITY_CLAMP_BASE_MIN, MAX_SENSITIVITY_CLAMP_BASE_MAX);
    }

    __device__ __forceinline__ float get_min_exploration_clamp(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::chemotaxis_min_exploration_clamp, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, MIN_EXPLORATION_CLAMP_BASE_MIN, MIN_EXPLORATION_CLAMP_BASE_MAX);
    }

    __device__ __forceinline__ float get_max_exploration_clamp(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::chemotaxis_max_exploration_clamp, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, MAX_EXPLORATION_CLAMP_BASE_MIN, MAX_EXPLORATION_CLAMP_BASE_MAX);
    }

    __device__ __forceinline__ float get_gradient_mix_weight(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::chemotaxis_gradient_mix_weight, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, GRADIENT_MIX_WEIGHT_BASE_MIN, GRADIENT_MIX_WEIGHT_BASE_MAX);
    }

    __device__ __forceinline__ float get_memory_decay_rate(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::chemotaxis_memory_decay_rate, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, MEMORY_DECAY_RATE_BASE_MIN, MEMORY_DECAY_RATE_BASE_MAX);
    }

    __device__ __forceinline__ float get_attractor_sigma(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::chemotaxis_attractor_sigma, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, ATTRACTOR_SIGMA_BASE_MIN, ATTRACTOR_SIGMA_BASE_MAX);
    }

    __device__ __forceinline__ float get_behavioral_field_sigma(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::chemotaxis_behavioral_field_sigma, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, BEHAVIORAL_FIELD_SIGMA_BASE_MIN, BEHAVIORAL_FIELD_SIGMA_BASE_MAX);
    }

    __device__ __forceinline__ float get_agent_embedding_scale(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::chemotaxis_agent_embedding_scale, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, AGENT_EMBEDDING_SCALE_BASE_MIN, AGENT_EMBEDDING_SCALE_BASE_MAX);
    }

    __device__ __forceinline__ float get_init_exploration(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::chemotaxis_init_exploration, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, INIT_EXPLORATION_BASE_MIN, INIT_EXPLORATION_BASE_MAX);
    }

    __device__ __forceinline__ float get_init_sensitivity(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::chemotaxis_init_sensitivity, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, INIT_SENSITIVITY_BASE_MIN, INIT_SENSITIVITY_BASE_MAX);
    }

    __device__ __forceinline__ float get_agent_source_sigma(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::chemotaxis_agent_source_sigma, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, AGENT_SOURCE_SIGMA_BASE_MIN, AGENT_SOURCE_SIGMA_BASE_MAX);
    }

    __device__ __forceinline__ float get_agent_source_strength(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::chemotaxis_agent_source_strength, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, AGENT_SOURCE_STRENGTH_BASE_MIN, AGENT_SOURCE_STRENGTH_BASE_MAX);
    }

    __device__ __forceinline__ float get_chemical_decay(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::chemotaxis_chemical_decay, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, CHEMICAL_DECAY_BASE_MIN, CHEMICAL_DECAY_BASE_MAX);
    }

    __device__ __forceinline__ float get_hurst_exponent(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::chemotaxis_hurst_exponent, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, CHEMOTAXIS_HURST_EXPONENT_MIN, CHEMOTAXIS_HURST_EXPONENT_MAX);
    }

    __device__ __forceinline__ float get_levy_alpha(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::chemotaxis_levy_alpha, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, CHEMOTAXIS_LEVY_ALPHA_MIN, CHEMOTAXIS_LEVY_ALPHA_MAX);
    }
};

struct CAParams {
    __device__ __forceinline__ float get_warp_ca_growth_rate(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::warp_ca_growth_rate, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, WARP_CA_GROWTH_RATE_MIN, WARP_CA_GROWTH_RATE_MAX);
    }
};

struct TrainingParams {
    __device__ __forceinline__ float get_adam_beta1(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::adam_beta1, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, ADAM_BETA1_MIN, ADAM_BETA1_MAX);
    }

    __device__ __forceinline__ float get_adam_beta2(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::adam_beta2, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, ADAM_BETA2_MIN, ADAM_BETA2_MAX);
    }

    __device__ __forceinline__ float get_adam_epsilon(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::adam_epsilon, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, ADAM_EPSILON_MIN, ADAM_EPSILON_MAX);
    }

    __device__ __forceinline__ float get_gradient_clip_norm(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::gradient_clip_norm, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, GRADIENT_CLIP_MIN, GRADIENT_CLIP_MAX);
    }

    __device__ __forceinline__ float get_learning_rate(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::learning_rate, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LEARNING_RATE_MIN, LEARNING_RATE_MAX);
    }

    __device__ __forceinline__ int get_batch_size(const float* genome, const float* gradients) {
        float batch_size_norm = genome_to_bootstrap_param(genome, gradients, GenomeParamTable::batch_size, NORMALIZED_MIN, NORMALIZED_MAX);
        return BATCH_SIZE_MIN + (int)(batch_size_norm * (BATCH_SIZE_MAX - BATCH_SIZE_MIN));
    }

    __device__ __forceinline__ float get_flow_lenia_lr(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::flow_lenia_lr, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, FLOW_LENIA_LR_MIN, FLOW_LENIA_LR_MAX);
    }

    __device__ __forceinline__ float get_gradient_fitness_weight(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::gradient_fitness_weight, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, GRADIENT_FITNESS_WEIGHT_MIN, GRADIENT_FITNESS_WEIGHT_MAX);
    }

    __device__ __forceinline__ float get_coherence_fitness_weight(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::coherence_fitness_weight, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, COHERENCE_FITNESS_WEIGHT_MIN, COHERENCE_FITNESS_WEIGHT_MAX);
    }

    __device__ __forceinline__ float get_behavioral_learning_rate(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, GenomeParamTable::behavioral_learning_rate, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, BEHAVIORAL_LEARNING_RATE_MIN, BEHAVIORAL_LEARNING_RATE_MAX);
    }
};

__global__ void clear_buffer_kernel(float* buffer, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        buffer[idx] = 0.0f;
    }
}


#endif
