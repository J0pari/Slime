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

    __device__ __forceinline__ void derive_from_genome(uint64_t genome_hash, const float* genome, const float* epigenetic) {
        int hunger_slot = derive_param_slot(genome_hash, "initial_hunger");
        int mutation_slot = derive_param_slot(genome_hash, "genome_mutation_scale");
        int levy_alpha_slot = derive_param_slot(genome_hash, "mutation_levy_alpha");
        int diversity_slot = derive_param_slot(genome_hash, "diversity_normalization");

        int hunger_min_slot = derive_param_slot(genome_hash, "initial_hunger_min");
        int hunger_max_slot = derive_param_slot(genome_hash, "initial_hunger_max");
        int mutation_min_slot = derive_param_slot(genome_hash, "genome_mutation_scale_min");
        int mutation_max_slot = derive_param_slot(genome_hash, "genome_mutation_scale_max");
        int diversity_min_slot = derive_param_slot(genome_hash, "diversity_normalization_min");
        int diversity_max_slot = derive_param_slot(genome_hash, "diversity_normalization_max");

        float hunger_min = (genome[hunger_min_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float hunger_max_raw = (genome[hunger_max_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float hunger_max = hunger_min + hunger_max_raw * (NORMALIZED_MAX - hunger_min);

        float mutation_min = (genome[mutation_min_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float mutation_max_raw = (genome[mutation_max_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float mutation_max = mutation_min + mutation_max_raw * (NORMALIZED_MAX - mutation_min);

        float diversity_min = (genome[diversity_min_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float diversity_max_raw = (genome[diversity_max_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float diversity_max = diversity_min + diversity_max_raw * (NORMALIZED_MAX - diversity_min);

        initial_hunger = genome_to_bootstrap_param(genome, epigenetic, hunger_slot, hunger_min, hunger_max);
        mutation_scale = genome_to_bootstrap_param(genome, epigenetic, mutation_slot, mutation_min, mutation_max);
        mutation_levy_alpha = genome_to_bootstrap_param(genome, epigenetic, levy_alpha_slot, CHEMOTAXIS_LEVY_ALPHA_MIN, CHEMOTAXIS_LEVY_ALPHA_MAX);
        diversity_normalization = genome_to_bootstrap_param(genome, epigenetic, diversity_slot, diversity_min, diversity_max);
    }
};





struct ChemotaxisParams {
    int theta_slot, sigma_slot, sensitivity_threshold_slot;
    int exploration_growth_slot, sensitivity_decay_slot, exploration_decay_slot, sensitivity_growth_slot;
    int min_sensitivity_slot, max_sensitivity_slot, min_exploration_slot, max_exploration_slot;
    int gradient_mix_weight_slot, memory_decay_rate_slot;
    int attractor_sigma_slot, behavioral_field_sigma_slot, agent_embedding_scale_slot;
    int init_exploration_slot, init_sensitivity_slot;
    int agent_source_sigma_slot, agent_source_strength_slot, chemical_decay_slot;
    int hurst_exponent_slot, levy_alpha_slot;

    __device__ __forceinline__ void derive_from_genome_hash(uint64_t genome_hash) {
        theta_slot = derive_param_slot(genome_hash, "chemotaxis_theta");
        sigma_slot = derive_param_slot(genome_hash, "chemotaxis_sigma");
        sensitivity_threshold_slot = derive_param_slot(genome_hash, "chemotaxis_sensitivity_threshold");
        exploration_growth_slot = derive_param_slot(genome_hash, "chemotaxis_exploration_growth");
        sensitivity_decay_slot = derive_param_slot(genome_hash, "chemotaxis_sensitivity_decay");
        exploration_decay_slot = derive_param_slot(genome_hash, "chemotaxis_exploration_decay");
        sensitivity_growth_slot = derive_param_slot(genome_hash, "chemotaxis_sensitivity_growth");
        min_sensitivity_slot = derive_param_slot(genome_hash, "chemotaxis_min_sensitivity_clamp");
        max_sensitivity_slot = derive_param_slot(genome_hash, "chemotaxis_max_sensitivity_clamp");
        min_exploration_slot = derive_param_slot(genome_hash, "chemotaxis_min_exploration_clamp");
        max_exploration_slot = derive_param_slot(genome_hash, "chemotaxis_max_exploration_clamp");
        gradient_mix_weight_slot = derive_param_slot(genome_hash, "chemotaxis_gradient_mix_weight");
        memory_decay_rate_slot = derive_param_slot(genome_hash, "chemotaxis_memory_decay_rate");
        attractor_sigma_slot = derive_param_slot(genome_hash, "chemotaxis_attractor_sigma");
        behavioral_field_sigma_slot = derive_param_slot(genome_hash, "chemotaxis_behavioral_field_sigma");
        agent_embedding_scale_slot = derive_param_slot(genome_hash, "chemotaxis_agent_embedding_scale");
        init_exploration_slot = derive_param_slot(genome_hash, "chemotaxis_init_exploration");
        init_sensitivity_slot = derive_param_slot(genome_hash, "chemotaxis_init_sensitivity");
        agent_source_sigma_slot = derive_param_slot(genome_hash, "chemotaxis_agent_source_sigma");
        agent_source_strength_slot = derive_param_slot(genome_hash, "chemotaxis_agent_source_strength");
        chemical_decay_slot = derive_param_slot(genome_hash, "chemotaxis_chemical_decay");
        hurst_exponent_slot = derive_param_slot(genome_hash, "chemotaxis_hurst_exponent");
        levy_alpha_slot = derive_param_slot(genome_hash, "chemotaxis_levy_alpha");
    }

    __device__ __forceinline__ float get_theta(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, theta_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, CHEMOTAXIS_THETA_BASE_MIN, CHEMOTAXIS_THETA_BASE_MAX);
    }

    __device__ __forceinline__ float get_sigma(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, sigma_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, CHEMOTAXIS_SIGMA_BASE_MIN, CHEMOTAXIS_SIGMA_BASE_MAX);
    }

    __device__ __forceinline__ float get_sensitivity_threshold(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, sensitivity_threshold_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, SENSITIVITY_THRESHOLD_BASE_MIN, SENSITIVITY_THRESHOLD_BASE_MAX);
    }

    __device__ __forceinline__ float get_exploration_growth(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, exploration_growth_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, EXPLORATION_GROWTH_BASE_MIN, EXPLORATION_GROWTH_BASE_MAX);
    }

    __device__ __forceinline__ float get_sensitivity_decay(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, sensitivity_decay_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, SENSITIVITY_DECAY_BASE_MIN, SENSITIVITY_DECAY_BASE_MAX);
    }

    __device__ __forceinline__ float get_exploration_decay(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, exploration_decay_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, EXPLORATION_DECAY_BASE_MIN, EXPLORATION_DECAY_BASE_MAX);
    }

    __device__ __forceinline__ float get_sensitivity_growth(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, sensitivity_growth_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, SENSITIVITY_GROWTH_BASE_MIN, SENSITIVITY_GROWTH_BASE_MAX);
    }

    __device__ __forceinline__ float get_min_sensitivity_clamp(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, min_sensitivity_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, MIN_SENSITIVITY_CLAMP_BASE_MIN, MIN_SENSITIVITY_CLAMP_BASE_MAX);
    }

    __device__ __forceinline__ float get_max_sensitivity_clamp(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, max_sensitivity_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, MAX_SENSITIVITY_CLAMP_BASE_MIN, MAX_SENSITIVITY_CLAMP_BASE_MAX);
    }

    __device__ __forceinline__ float get_min_exploration_clamp(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, min_exploration_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, MIN_EXPLORATION_CLAMP_BASE_MIN, MIN_EXPLORATION_CLAMP_BASE_MAX);
    }

    __device__ __forceinline__ float get_max_exploration_clamp(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, max_exploration_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, MAX_EXPLORATION_CLAMP_BASE_MIN, MAX_EXPLORATION_CLAMP_BASE_MAX);
    }

    __device__ __forceinline__ float get_gradient_mix_weight(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, gradient_mix_weight_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, GRADIENT_MIX_WEIGHT_BASE_MIN, GRADIENT_MIX_WEIGHT_BASE_MAX);
    }

    __device__ __forceinline__ float get_memory_decay_rate(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, memory_decay_rate_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, MEMORY_DECAY_RATE_BASE_MIN, MEMORY_DECAY_RATE_BASE_MAX);
    }

    __device__ __forceinline__ float get_attractor_sigma(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, attractor_sigma_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, ATTRACTOR_SIGMA_BASE_MIN, ATTRACTOR_SIGMA_BASE_MAX);
    }

    __device__ __forceinline__ float get_behavioral_field_sigma(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, behavioral_field_sigma_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, BEHAVIORAL_FIELD_SIGMA_BASE_MIN, BEHAVIORAL_FIELD_SIGMA_BASE_MAX);
    }

    __device__ __forceinline__ float get_agent_embedding_scale(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, agent_embedding_scale_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, AGENT_EMBEDDING_SCALE_BASE_MIN, AGENT_EMBEDDING_SCALE_BASE_MAX);
    }

    __device__ __forceinline__ float get_init_exploration(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, init_exploration_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, INIT_EXPLORATION_BASE_MIN, INIT_EXPLORATION_BASE_MAX);
    }

    __device__ __forceinline__ float get_init_sensitivity(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, init_sensitivity_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, INIT_SENSITIVITY_BASE_MIN, INIT_SENSITIVITY_BASE_MAX);
    }

    __device__ __forceinline__ float get_agent_source_sigma(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, agent_source_sigma_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, AGENT_SOURCE_SIGMA_BASE_MIN, AGENT_SOURCE_SIGMA_BASE_MAX);
    }

    __device__ __forceinline__ float get_agent_source_strength(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, agent_source_strength_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, AGENT_SOURCE_STRENGTH_BASE_MIN, AGENT_SOURCE_STRENGTH_BASE_MAX);
    }

    __device__ __forceinline__ float get_chemical_decay(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, chemical_decay_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, CHEMICAL_DECAY_BASE_MIN, CHEMICAL_DECAY_BASE_MAX);
    }

    __device__ __forceinline__ float get_hurst_exponent(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, hurst_exponent_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, CHEMOTAXIS_HURST_EXPONENT_MIN, CHEMOTAXIS_HURST_EXPONENT_MAX);
    }

    __device__ __forceinline__ float get_levy_alpha(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, levy_alpha_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, CHEMOTAXIS_LEVY_ALPHA_MIN, CHEMOTAXIS_LEVY_ALPHA_MAX);
    }
};

struct CAParams {
    int warp_ca_growth_rate_slot;

    __device__ __forceinline__ void derive_from_genome_hash(uint64_t genome_hash) {
        warp_ca_growth_rate_slot = derive_param_slot(genome_hash, "warp_ca_growth_rate");
    }

    __device__ __forceinline__ float get_warp_ca_growth_rate(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, warp_ca_growth_rate_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, WARP_CA_GROWTH_RATE_MIN, WARP_CA_GROWTH_RATE_MAX);
    }
};

struct TrainingParams {
    int adam_beta1_slot;
    int adam_beta2_slot;
    int adam_epsilon_slot;
    int gradient_clip_slot;
    int learning_rate_slot;
    int batch_size_slot;
    int flow_lenia_lr_slot;

    int adam_epsilon_min_slot;
    int adam_epsilon_max_slot;
    int batch_size_min_slot;
    int batch_size_max_slot;

    __device__ __forceinline__ void derive_from_genome_hash(uint64_t genome_hash) {
        adam_beta1_slot = derive_param_slot(genome_hash, "adam_beta1");
        adam_beta2_slot = derive_param_slot(genome_hash, "adam_beta2");
        adam_epsilon_slot = derive_param_slot(genome_hash, "adam_epsilon");
        gradient_clip_slot = derive_param_slot(genome_hash, "gradient_clip_norm");
        learning_rate_slot = derive_param_slot(genome_hash, "learning_rate");
        batch_size_slot = derive_param_slot(genome_hash, "batch_size");
        flow_lenia_lr_slot = derive_param_slot(genome_hash, "flow_lenia_lr");

        adam_epsilon_min_slot = derive_param_slot(genome_hash, "adam_epsilon_min");
        adam_epsilon_max_slot = derive_param_slot(genome_hash, "adam_epsilon_max");
        batch_size_min_slot = derive_param_slot(genome_hash, "batch_size_min");
        batch_size_max_slot = derive_param_slot(genome_hash, "batch_size_max");
    }

    __device__ __forceinline__ float get_adam_beta1(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, adam_beta1_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, ADAM_BETA1_MIN, ADAM_BETA1_MAX);
    }

    __device__ __forceinline__ float get_adam_beta2(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, adam_beta2_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, ADAM_BETA2_MIN, ADAM_BETA2_MAX);
    }

    __device__ __forceinline__ float get_adam_epsilon(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        float eps_min = (genome[adam_epsilon_min_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float eps_max_raw = (genome[adam_epsilon_max_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float eps_max = eps_min + eps_max_raw * (NORMALIZED_MAX - eps_min);
        return genome_to_param(genome, gradients, adam_epsilon_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, eps_min, eps_max);
    }

    __device__ __forceinline__ float get_gradient_clip_norm(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, gradient_clip_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, GRADIENT_CLIP_MIN, GRADIENT_CLIP_MAX);
    }

    __device__ __forceinline__ float get_learning_rate(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, learning_rate_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, LEARNING_RATE_MIN, LEARNING_RATE_MAX);
    }

    __device__ __forceinline__ int get_batch_size(const float* genome, const float* gradients) {
        float bs_min = (genome[batch_size_min_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float bs_max_raw = (genome[batch_size_max_slot] + GENOME_TO_UNIT_OFFSET) * GENOME_TO_UNIT_SCALE;
        float bs_max = bs_min + bs_max_raw * (NORMALIZED_MAX - bs_min);
        float batch_size_norm = genome_to_bootstrap_param(genome, gradients, batch_size_slot, bs_min, bs_max);
        return BATCH_SIZE_MIN + (int)(batch_size_norm * (BATCH_SIZE_MAX - BATCH_SIZE_MIN));
    }

    __device__ __forceinline__ float get_flow_lenia_lr(const float* genome, const float* gradients, float ctx_metabolic, float ctx_stress, float ctx_morphogen, float ctx_complexity, float ctx_niche, float ctx_learning, float ctx_performance) {
        return genome_to_param(genome, gradients, flow_lenia_lr_slot, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance, FLOW_LENIA_LR_MIN, FLOW_LENIA_LR_MAX);
    }
};

__global__ void clear_buffer_kernel(float* buffer, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        buffer[idx] = 0.0f;
    }
}


#endif
