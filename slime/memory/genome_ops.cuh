#ifndef GENOME_OPS_CUH
#define GENOME_OPS_CUH

#include "../config/config.cu"
#include "../debug/param_validator.cu"
#include "../debug/provenance.cuh"
#include <cuda_runtime.h>

namespace GenomeParamTable {
    constexpr int BLOCK_SIZE = 8;

    
    constexpr int ARCHITECTURE_BLOCK = 0;
    constexpr int num_heads = ARCHITECTURE_BLOCK + 0;
    constexpr int head_dim = ARCHITECTURE_BLOCK + 1;
    constexpr int grid_size = ARCHITECTURE_BLOCK + 2;
    constexpr int channels = ARCHITECTURE_BLOCK + 3;
    constexpr int hidden_dim = ARCHITECTURE_BLOCK + 4;
    constexpr int ca_gate_center = ARCHITECTURE_BLOCK + 5;
    constexpr int pool_capacity = ARCHITECTURE_BLOCK + 6;
    constexpr int initial_pool_size = ARCHITECTURE_BLOCK + 7;

    
    constexpr int OPTIMIZER_BLOCK = 1 * BLOCK_SIZE;
    constexpr int adam_beta1 = OPTIMIZER_BLOCK + 0;
    constexpr int adam_beta2 = OPTIMIZER_BLOCK + 1;
    constexpr int adam_epsilon = OPTIMIZER_BLOCK + 2;
    constexpr int learning_rate = OPTIMIZER_BLOCK + 3;
    constexpr int gradient_clip_norm = OPTIMIZER_BLOCK + 4;
    constexpr int batch_size = OPTIMIZER_BLOCK + 5;
    constexpr int flow_lenia_lr = OPTIMIZER_BLOCK + 6;
    constexpr int behavioral_learning_rate = OPTIMIZER_BLOCK + 7;

    
    constexpr int FLOW_LENIA_BLOCK = 2 * BLOCK_SIZE;
    constexpr int flow_lenia_s = FLOW_LENIA_BLOCK + 0;
    constexpr int flow_lenia_beta_A = FLOW_LENIA_BLOCK + 1;
    constexpr int flow_lenia_n = FLOW_LENIA_BLOCK + 2;
    constexpr int flow_alpha_min = FLOW_LENIA_BLOCK + 3;
    constexpr int flow_alpha_max = FLOW_LENIA_BLOCK + 4;
    constexpr int flow_sharpness = FLOW_LENIA_BLOCK + 5;
    constexpr int flow_resource_dt = FLOW_LENIA_BLOCK + 6;

    
    constexpr int CHEMOTAXIS_BLOCK = 3 * BLOCK_SIZE;
    constexpr int chemotaxis_theta = CHEMOTAXIS_BLOCK + 0;
    constexpr int chemotaxis_sigma = CHEMOTAXIS_BLOCK + 1;
    constexpr int chemotaxis_sensitivity_threshold = CHEMOTAXIS_BLOCK + 2;
    constexpr int chemotaxis_exploration_growth = CHEMOTAXIS_BLOCK + 3;
    constexpr int chemotaxis_sensitivity_decay = CHEMOTAXIS_BLOCK + 4;
    constexpr int chemotaxis_exploration_decay = CHEMOTAXIS_BLOCK + 5;
    constexpr int chemotaxis_sensitivity_growth = CHEMOTAXIS_BLOCK + 6;
    constexpr int chemotaxis_hurst_exponent = CHEMOTAXIS_BLOCK + 7;

    constexpr int CHEMOTAXIS_BLOCK_2 = 4 * BLOCK_SIZE;
    constexpr int chemotaxis_levy_alpha = CHEMOTAXIS_BLOCK_2 + 0;
    constexpr int chemotaxis_min_sensitivity_clamp = CHEMOTAXIS_BLOCK_2 + 1;
    constexpr int chemotaxis_max_sensitivity_clamp = CHEMOTAXIS_BLOCK_2 + 2;
    constexpr int chemotaxis_min_exploration_clamp = CHEMOTAXIS_BLOCK_2 + 3;
    constexpr int chemotaxis_max_exploration_clamp = CHEMOTAXIS_BLOCK_2 + 4;
    constexpr int chemotaxis_gradient_mix_weight = CHEMOTAXIS_BLOCK_2 + 5;
    constexpr int chemotaxis_memory_decay_rate = CHEMOTAXIS_BLOCK_2 + 6;
    constexpr int chemotaxis_attractor_sigma = CHEMOTAXIS_BLOCK_2 + 7;

    constexpr int CHEMOTAXIS_BLOCK_3 = 5 * BLOCK_SIZE;
    constexpr int chemotaxis_behavioral_field_sigma = CHEMOTAXIS_BLOCK_3 + 0;
    constexpr int chemotaxis_agent_embedding_scale = CHEMOTAXIS_BLOCK_3 + 1;
    constexpr int chemotaxis_init_exploration = CHEMOTAXIS_BLOCK_3 + 2;
    constexpr int chemotaxis_init_sensitivity = CHEMOTAXIS_BLOCK_3 + 3;
    constexpr int chemotaxis_agent_source_sigma = CHEMOTAXIS_BLOCK_3 + 4;
    constexpr int chemotaxis_agent_source_strength = CHEMOTAXIS_BLOCK_3 + 5;
    constexpr int chemotaxis_chemical_decay = CHEMOTAXIS_BLOCK_3 + 6;
    constexpr int chemotaxis_dt = CHEMOTAXIS_BLOCK_3 + 7;

    
    constexpr int FITNESS_BLOCK = 6 * BLOCK_SIZE;
    constexpr int gradient_fitness_weight = FITNESS_BLOCK + 0;
    constexpr int coherence_fitness_weight = FITNESS_BLOCK + 1;
    constexpr int fitness_rank_exponent = FITNESS_BLOCK + 2;
    constexpr int fitness_coherence_exponent = FITNESS_BLOCK + 3;
    constexpr int fitness_coupling_exponent = FITNESS_BLOCK + 4;
    constexpr int fitness_task_exponent = FITNESS_BLOCK + 5;
    constexpr int fitness_gen_exponent = FITNESS_BLOCK + 6;
    constexpr int fitness_efficiency_exponent = FITNESS_BLOCK + 7;

    
    constexpr int POOL_BLOCK = 7 * BLOCK_SIZE;
    constexpr int initial_hunger = POOL_BLOCK + 0;
    constexpr int genome_mutation_scale = POOL_BLOCK + 1;
    constexpr int mutation_levy_alpha = POOL_BLOCK + 2;
    constexpr int diversity_normalization = POOL_BLOCK + 3;
    constexpr int warp_ca_growth_rate = POOL_BLOCK + 4;
    constexpr int baldwin_sensitivity = POOL_BLOCK + 5;
    constexpr int coherence_window_size = POOL_BLOCK + 6;
    constexpr int renyi_q = POOL_BLOCK + 7;

    
    constexpr int CONTEXT_BLOCK = 8 * BLOCK_SIZE;
    constexpr int init_context_metabolic = CONTEXT_BLOCK + 0;
    constexpr int init_context_stress = CONTEXT_BLOCK + 1;
    constexpr int init_context_morphogen = CONTEXT_BLOCK + 2;
    constexpr int context_stress_numerator = CONTEXT_BLOCK + 3;
    constexpr int embed_ctx_metabolic = CONTEXT_BLOCK + 4;
    constexpr int embed_ctx_stress = CONTEXT_BLOCK + 5;
    constexpr int embed_ctx_morphogen = CONTEXT_BLOCK + 6;

    
    constexpr int BEHAVIORAL_BLOCK = 9 * BLOCK_SIZE;
    constexpr int behavioral_dim_hw = BEHAVIORAL_BLOCK + 0;
    constexpr int behavioral_dim_task = BEHAVIORAL_BLOCK + 1;
    constexpr int behavioral_dim_gen = BEHAVIORAL_BLOCK + 2;

    
    constexpr int DIRESA_BLOCK = 10 * BLOCK_SIZE;
    constexpr int dist_weight = DIRESA_BLOCK + 0;
    constexpr int recon_weight = DIRESA_BLOCK + 1;
    constexpr int distance_exponent = DIRESA_BLOCK + 2;
    constexpr int quality_weight = DIRESA_BLOCK + 3;
    constexpr int diresa_num_replicas = DIRESA_BLOCK + 4;
    constexpr int diresa_hidden1 = DIRESA_BLOCK + 5;
    constexpr int diresa_hidden2 = DIRESA_BLOCK + 6;
    constexpr int diresa_batch_size = DIRESA_BLOCK + 7;

    
    constexpr int DIRESA_BLOCK_2 = 11 * BLOCK_SIZE;
    constexpr int diresa_anneal_step = DIRESA_BLOCK_2 + 0;
    constexpr int diresa_cov_target = DIRESA_BLOCK_2 + 1;
    constexpr int diresa_ctx_metabolic = DIRESA_BLOCK_2 + 2;
    constexpr int diresa_ctx_stress = DIRESA_BLOCK_2 + 3;
    constexpr int diresa_ctx_morphogen = DIRESA_BLOCK_2 + 4;
    constexpr int diresa_temp_base = DIRESA_BLOCK_2 + 5;
    constexpr int diresa_temp_scale = DIRESA_BLOCK_2 + 6;
    constexpr int diresa_gradient_clip = DIRESA_BLOCK_2 + 7;

    
    constexpr int DIRESA_BLOCK_3 = 36 * BLOCK_SIZE;
    constexpr int diresa_ctx_complexity = DIRESA_BLOCK_3 + 0;
    constexpr int diresa_ctx_niche = DIRESA_BLOCK_3 + 1;
    constexpr int diresa_ctx_learning = DIRESA_BLOCK_3 + 2;
    constexpr int diresa_ctx_performance = DIRESA_BLOCK_3 + 3;

    
    constexpr int LIFECYCLE_BLOCK = 12 * BLOCK_SIZE;
    constexpr int lifecycle_coherence_stressed = LIFECYCLE_BLOCK + 0;
    constexpr int lifecycle_coherence_recover = LIFECYCLE_BLOCK + 1;
    constexpr int lifecycle_stress_accum_rate = LIFECYCLE_BLOCK + 2;
    constexpr int lifecycle_stress_decay_rate = LIFECYCLE_BLOCK + 3;
    constexpr int lifecycle_stress_threshold = LIFECYCLE_BLOCK + 4;
    constexpr int lifecycle_fitness_multiplier = LIFECYCLE_BLOCK + 5;
    constexpr int lifecycle_gradient_stagnation = LIFECYCLE_BLOCK + 6;
    constexpr int lifecycle_dormant_stress_mult = LIFECYCLE_BLOCK + 7;

    constexpr int LIFECYCLE_BLOCK_2 = 13 * BLOCK_SIZE;
    constexpr int lifecycle_fitness_threshold_center = LIFECYCLE_BLOCK_2 + 0;
    constexpr int lifecycle_fitness_threshold_steepness = LIFECYCLE_BLOCK_2 + 1;
    constexpr int lifecycle_sigmoid_threshold = LIFECYCLE_BLOCK_2 + 2;
    constexpr int lifecycle_archive_threshold_center = LIFECYCLE_BLOCK_2 + 3;
    constexpr int lifecycle_archive_threshold_steepness = LIFECYCLE_BLOCK_2 + 4;
    constexpr int lifecycle_density_multiplier = LIFECYCLE_BLOCK_2 + 5;
    constexpr int lifecycle_stress_fitness_penalty = LIFECYCLE_BLOCK_2 + 6;
    constexpr int lifecycle_fitness_culling_mult = LIFECYCLE_BLOCK_2 + 7;

    constexpr int LIFECYCLE_BLOCK_3 = 14 * BLOCK_SIZE;
    constexpr int lifecycle_boost_threshold_center = LIFECYCLE_BLOCK_3 + 0;
    constexpr int lifecycle_boost_threshold_steepness = LIFECYCLE_BLOCK_3 + 1;
    constexpr int lifecycle_crisis_fitness_mult = LIFECYCLE_BLOCK_3 + 2;
    constexpr int lifecycle_crisis_coherence = LIFECYCLE_BLOCK_3 + 3;
    constexpr int lifecycle_crisis_threshold_center = LIFECYCLE_BLOCK_3 + 4;
    constexpr int lifecycle_crisis_threshold_steepness = LIFECYCLE_BLOCK_3 + 5;
    constexpr int lifecycle_elite_fitness_inherit = LIFECYCLE_BLOCK_3 + 6;
    constexpr int lifecycle_elite_coherence_reset = LIFECYCLE_BLOCK_3 + 7;

    constexpr int LIFECYCLE_BLOCK_4 = 15 * BLOCK_SIZE;
    constexpr int lifecycle_boost_fitness_ratio = LIFECYCLE_BLOCK_4 + 0;
    constexpr int lifecycle_coherence_boost = LIFECYCLE_BLOCK_4 + 1;
    constexpr int lifecycle_crisis_active_ratio = LIFECYCLE_BLOCK_4 + 2;
    constexpr int lifecycle_fitness_inherit_center = LIFECYCLE_BLOCK_4 + 3;
    constexpr int lifecycle_fitness_inherit_steepness = LIFECYCLE_BLOCK_4 + 4;

    
    constexpr int MEMORY_BLOCK = 16 * BLOCK_SIZE;
    constexpr int memory_decay_factor = MEMORY_BLOCK + 0;
    constexpr int memory_importance = MEMORY_BLOCK + 1;
    constexpr int memory_decay_threshold = MEMORY_BLOCK + 2;
    constexpr int memory_consolidation_threshold = MEMORY_BLOCK + 3;
    constexpr int memory_flow_lenia_dt = MEMORY_BLOCK + 4;
    constexpr int memory_default_decay_rate = MEMORY_BLOCK + 5;

    
    constexpr int CHEM_INIT_BLOCK = 18 * BLOCK_SIZE;
    constexpr int chem_diffusivity = CHEM_INIT_BLOCK + 0;
    constexpr int chem_reaction_order = CHEM_INIT_BLOCK + 1;
    constexpr int chem_reaction_rate = CHEM_INIT_BLOCK + 2;
    constexpr int chem_decay_rate = CHEM_INIT_BLOCK + 3;
    constexpr int chem_init_base_value = CHEM_INIT_BLOCK + 4;
    constexpr int chem_init_genome_influence = CHEM_INIT_BLOCK + 5;
    constexpr int chem_init_noise_scale = CHEM_INIT_BLOCK + 6;


    constexpr int AGENT_BLOCK = 20 * BLOCK_SIZE;
    constexpr int max_agent_velocity = AGENT_BLOCK + 0;

    constexpr int RESOURCE_BLOCK = 22 * BLOCK_SIZE;
    constexpr int resource_density_initial = RESOURCE_BLOCK + 0;
    constexpr int resource_density_noise = RESOURCE_BLOCK + 1;

    
    constexpr int SPAWN_BLOCK = 24 * BLOCK_SIZE;
    constexpr int spawn_fitness_threshold = SPAWN_BLOCK + 0;
    constexpr int spawn_ctx_metabolic = SPAWN_BLOCK + 1;
    constexpr int spawn_ctx_stress = SPAWN_BLOCK + 2;
    constexpr int spawn_ctx_morphogen = SPAWN_BLOCK + 3;
    constexpr int mutation_ctx_metabolic = SPAWN_BLOCK + 4;
    constexpr int mutation_ctx_stress = SPAWN_BLOCK + 5;
    constexpr int mutation_ctx_morphogen = SPAWN_BLOCK + 6;
    constexpr int metalearning_mutation_rate = SPAWN_BLOCK + 7;

    
    constexpr int CURRICULUM_BLOCK = 26 * BLOCK_SIZE;
    constexpr int curriculum_accuracy_threshold = CURRICULUM_BLOCK + 0;
    constexpr int curriculum_diversity_threshold = CURRICULUM_BLOCK + 1;
    constexpr int curriculum_min_generations = CURRICULUM_BLOCK + 2;

    
    constexpr int CONVERGENCE_BLOCK = 28 * BLOCK_SIZE;
    constexpr int convergence_fitness_threshold = CONVERGENCE_BLOCK + 0;
    constexpr int convergence_coherence_threshold = CONVERGENCE_BLOCK + 1;

    
    constexpr int DELTA_BLOCK = 30 * BLOCK_SIZE;
    constexpr int delta_threshold = DELTA_BLOCK + 0;


    constexpr int VORONOI_BLOCK = 31 * BLOCK_SIZE;
    constexpr int weight_delta_threshold = VORONOI_BLOCK + 2;

    
    constexpr int DIVERSITY_BLOCK = 32 * BLOCK_SIZE;
    constexpr int diversity_sample_count = DIVERSITY_BLOCK + 0;

    
    constexpr int EMA_BLOCK = 33 * BLOCK_SIZE;
    constexpr int accuracy_ema_smoothing = EMA_BLOCK + 0;

    
    constexpr int FITNESS_INHERIT_BLOCK = 34 * BLOCK_SIZE;
    constexpr int fitness_inherit_center = FITNESS_INHERIT_BLOCK + 0;
    constexpr int fitness_inherit_steepness = FITNESS_INHERIT_BLOCK + 1;

    
    constexpr int BOUNDS_BLOCK = 35 * BLOCK_SIZE;

    
    constexpr int BOUNDS_BLOCK_2 = 37 * BLOCK_SIZE;
    constexpr int memory_decay_factor_min = BOUNDS_BLOCK_2 + 0;
    constexpr int memory_decay_factor_max = BOUNDS_BLOCK_2 + 1;
    constexpr int memory_importance_min = BOUNDS_BLOCK_2 + 2;
    constexpr int memory_importance_max = BOUNDS_BLOCK_2 + 3;
    constexpr int agent_embedding_scale_min = BOUNDS_BLOCK_2 + 4;
    constexpr int agent_embedding_scale_max = BOUNDS_BLOCK_2 + 5;
    constexpr int init_exploration_min = BOUNDS_BLOCK_2 + 6;
    constexpr int init_exploration_max = BOUNDS_BLOCK_2 + 7;

    constexpr int BOUNDS_BLOCK_3 = 38 * BLOCK_SIZE;
    constexpr int init_sensitivity_min = BOUNDS_BLOCK_3 + 0;
    constexpr int init_sensitivity_max = BOUNDS_BLOCK_3 + 1;
    constexpr int levy_alpha_min = BOUNDS_BLOCK_3 + 2;
    constexpr int levy_alpha_max = BOUNDS_BLOCK_3 + 3;
    constexpr int resource_density_initial_min = BOUNDS_BLOCK_3 + 4;
    constexpr int resource_density_initial_max = BOUNDS_BLOCK_3 + 5;
    constexpr int resource_density_noise_min = BOUNDS_BLOCK_3 + 6;
    constexpr int resource_density_noise_max = BOUNDS_BLOCK_3 + 7;

    constexpr int BOUNDS_BLOCK_4 = 39 * BLOCK_SIZE;
    constexpr int chem_init_base_value_min = BOUNDS_BLOCK_4 + 0;
    constexpr int chem_init_base_value_max = BOUNDS_BLOCK_4 + 1;
    constexpr int chem_init_genome_influence_min = BOUNDS_BLOCK_4 + 2;
    constexpr int chem_init_genome_influence_max = BOUNDS_BLOCK_4 + 3;
    constexpr int chem_init_noise_scale_min = BOUNDS_BLOCK_4 + 4;
    constexpr int chem_init_noise_scale_max = BOUNDS_BLOCK_4 + 5;
    constexpr int diversity_normalization_min = BOUNDS_BLOCK_4 + 6;
    constexpr int diversity_normalization_max = BOUNDS_BLOCK_4 + 7;

    constexpr int BOUNDS_BLOCK_5 = 40 * BLOCK_SIZE;
    constexpr int diversity_sample_count_min = BOUNDS_BLOCK_5 + 0;
    constexpr int diversity_sample_count_max = BOUNDS_BLOCK_5 + 1;
    constexpr int diversity_sample_count_base = BOUNDS_BLOCK_5 + 2;
    constexpr int diversity_sample_count_range = BOUNDS_BLOCK_5 + 3;
    constexpr int genome_mutation_scale_min = BOUNDS_BLOCK_5 + 4;
    constexpr int genome_mutation_scale_max = BOUNDS_BLOCK_5 + 5;
    constexpr int initial_hunger_min = BOUNDS_BLOCK_5 + 6;
    constexpr int initial_hunger_max = BOUNDS_BLOCK_5 + 7;

    constexpr int BOUNDS_BLOCK_6 = 41 * BLOCK_SIZE;
    constexpr int weight_delta_threshold_min = BOUNDS_BLOCK_6 + 0;
    constexpr int weight_delta_threshold_max = BOUNDS_BLOCK_6 + 1;
    constexpr int convergence_fitness_min = BOUNDS_BLOCK_6 + 2;
    constexpr int convergence_fitness_max = BOUNDS_BLOCK_6 + 3;
    constexpr int convergence_coherence_min = BOUNDS_BLOCK_6 + 4;
    constexpr int convergence_coherence_max = BOUNDS_BLOCK_6 + 5;
    constexpr int genome_slot = BOUNDS_BLOCK_6 + 6;

    constexpr int EPIGENETIC_SENSITIVITY_BLOCK = 64 * BLOCK_SIZE;
    constexpr int ENHANCER_BLOCK = 80 * BLOCK_SIZE;
    constexpr int CONTEXT_COUPLING_BLOCK = 96 * BLOCK_SIZE;

    constexpr int TOTAL_TYPED_SLOTS = 100 * BLOCK_SIZE;
}

__device__ __forceinline__ int get_epigenetic_sensitivity_slot(int primary_slot) {
    return GenomeParamTable::EPIGENETIC_SENSITIVITY_BLOCK + (primary_slot % GenomeParamTable::BLOCK_SIZE);
}

__device__ __forceinline__ int get_enhancer_slot(int primary_slot, int enhancer_idx) {
    return GenomeParamTable::ENHANCER_BLOCK + (primary_slot % GenomeParamTable::BLOCK_SIZE) * 8 + enhancer_idx;
}

__device__ __forceinline__ int get_context_coupling_slot(int primary_slot) {
    return GenomeParamTable::CONTEXT_COUPLING_BLOCK + (primary_slot % GenomeParamTable::BLOCK_SIZE);
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

    int epi_slot = get_epigenetic_sensitivity_slot(primary_slot);
    DEVICE_VALIDATE_GENOME_SLOT(epi_slot);
    float epigenetic_sensitivity = genome_slot_to_unit(genome, epi_slot);

    int epi_bounds_slot = GenomeParamTable::BOUNDS_BLOCK + (primary_slot % GenomeParamTable::BLOCK_SIZE);
    DEVICE_VALIDATE_GENOME_SLOT(epi_bounds_slot);
    float raw_epi_lo = genome[epi_bounds_slot] * 0.5f;
    float raw_epi_hi = 2.0f + genome[epi_bounds_slot];
    float epi_min = fminf(raw_epi_lo, raw_epi_hi);
    float epi_max = fmaxf(raw_epi_lo, raw_epi_hi);

    float epigenetic_factor = 1.0f;
    if (epigenetic != nullptr) {
        epigenetic_factor = 1.0f + epigenetic_sensitivity * epigenetic[primary_slot];
        epigenetic_factor = fminf(fmaxf(epigenetic_factor, epi_min), epi_max);
        DEVICE_VALIDATE_EPIGENETIC(epigenetic_factor, epi_min, epi_max);
    }

    int enhancer_1_slot = get_enhancer_slot(primary_slot, 0);
    int enhancer_2_slot = get_enhancer_slot(primary_slot, 1);
    int enhancer_3_slot = get_enhancer_slot(primary_slot, 2);
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

    __device__ __forceinline__ void derive_from_genome(const float* genome, const float* epigenetic) {
        int metabolic_bounds_slot = GenomeParamTable::BOUNDS_BLOCK + 0;
        int stress_bounds_slot = GenomeParamTable::BOUNDS_BLOCK + 1;
        int morphogen_bounds_slot = GenomeParamTable::BOUNDS_BLOCK + 2;

        float metabolic_min = genome_slot_to_unit(genome, metabolic_bounds_slot);
        float metabolic_max = metabolic_min + genome_slot_to_unit(genome, metabolic_bounds_slot + 3) * (NORMALIZED_MAX - metabolic_min);

        float stress_min = genome_slot_to_unit(genome, stress_bounds_slot);
        float stress_max = stress_min + genome_slot_to_unit(genome, stress_bounds_slot + 3) * (NORMALIZED_MAX - stress_min);

        float morphogen_min = genome_slot_to_unit(genome, morphogen_bounds_slot);
        float morphogen_max = morphogen_min + genome_slot_to_unit(genome, morphogen_bounds_slot + 3) * (NORMALIZED_MAX - morphogen_min);

        metabolic = genome_to_bootstrap_param(genome, epigenetic, GenomeParamTable::init_context_metabolic, metabolic_min, metabolic_max);
        stress = genome_to_bootstrap_param(genome, epigenetic, GenomeParamTable::init_context_stress, stress_min, stress_max);
        morphogen = genome_to_bootstrap_param(genome, epigenetic, GenomeParamTable::init_context_morphogen, morphogen_min, morphogen_max);
    }
};

struct BehavioralDimensions {
    int hw_dim;
    int task_dim;
    int gen_dim;

    __device__ __forceinline__ void derive_from_genome() {
        hw_dim = BEHAVIORAL_DIM_HW;
        task_dim = BEHAVIORAL_DIM_TASK;
        gen_dim = BEHAVIORAL_DIM_GEN;
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
    float base_val = genome[primary_slot];

    int epi_slot = get_epigenetic_sensitivity_slot(primary_slot);
    float epigenetic_sensitivity = genome_slot_to_unit(genome, epi_slot);

    int epi_bounds_slot = GenomeParamTable::BOUNDS_BLOCK + (primary_slot % GenomeParamTable::BLOCK_SIZE);
    float raw_epi_lo2 = genome[epi_bounds_slot] * 0.5f;
    float raw_epi_hi2 = 2.0f + genome[epi_bounds_slot];
    float epi_min = fminf(raw_epi_lo2, raw_epi_hi2);
    float epi_max = fmaxf(raw_epi_lo2, raw_epi_hi2);

    float epigenetic_factor = 1.0f + epigenetic_sensitivity * epigenetic[primary_slot];
    epigenetic_factor = fminf(fmaxf(epigenetic_factor, epi_min), epi_max);

    float enhancer_1 = genome[get_enhancer_slot(primary_slot, 0)];
    float enhancer_2 = genome[get_enhancer_slot(primary_slot, 1)];
    float enhancer_3 = genome[get_enhancer_slot(primary_slot, 2)];
    float enhancer_4 = genome[get_enhancer_slot(primary_slot, 3)];
    float enhancer_5 = genome[get_enhancer_slot(primary_slot, 4)];
    float enhancer_6 = genome[get_enhancer_slot(primary_slot, 5)];
    float enhancer_7 = genome[get_enhancer_slot(primary_slot, 6)];

    int ctx_slot = get_context_coupling_slot(primary_slot);
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
