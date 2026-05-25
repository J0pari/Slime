
#ifndef TRAINING_TYPES_CU
#define TRAINING_TYPES_CU

#include "../config/config.cu"
#include "../core/organism.cu"
#include "../utils/genome_params.cuh"
#include <curand_kernel.h>

__device__ void init_unified_gradient_buffer_device(Organism* organism) {
    UnifiedGradientBuffer* grad_buf = organism->unified_grad_buffer;
    float* perception_grads = organism->tt_perception_grads;
    float* interaction_grads = organism->tt_interaction_grads;
    float* value_grads = organism->tt_value_grads;
    float* pooling_weight_grads = organism->tt_pooling_weight_grads;
    float* fc_weight_grads = organism->tt_fc_weight_grads;
    float* fc_bias_grads = organism->tt_fc_bias_grads;

    int entry_idx = blockIdx.x % organism->pool->capacity;
    PoolEntry* entry = &organism->pool->entries[entry_idx];
    int num_classes = organism->cls_num_classes;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        grad_buf->perception_grads = perception_grads;
        grad_buf->interaction_grads = interaction_grads;
        grad_buf->flow_projection_grads = value_grads;
        grad_buf->pooling_weight_grads = pooling_weight_grads;
        grad_buf->fc_weight_grads = fc_weight_grads;
        grad_buf->fc_bias_grads = fc_bias_grads;

        grad_buf->perception_size = entry->num_heads * entry->channels * entry->head_dim;
        grad_buf->interaction_size = entry->num_heads * entry->head_dim * entry->head_dim;
        grad_buf->flow_projection_size = entry->num_heads * 2 * entry->head_dim;
        grad_buf->num_classes = num_classes;
        grad_buf->num_features = entry->num_heads * POOLING_NUM_TILES * entry->channels;

        grad_buf->has_autodiff_grads = 0;
        grad_buf->has_backprop_grads = 0;
    }
}

__device__ void zero_unified_gradients_device(Organism* organism) {
    UnifiedGradientBuffer* grad_buf = organism->unified_grad_buffer;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = grad_buf->perception_size + grad_buf->interaction_size + grad_buf->flow_projection_size;

    if (idx < grad_buf->perception_size) {
        grad_buf->perception_grads[idx] = 0.0f;
    } else if (idx < grad_buf->perception_size + grad_buf->interaction_size) {
        grad_buf->interaction_grads[idx - grad_buf->perception_size] = 0.0f;
    } else if (idx < total) {
        grad_buf->flow_projection_grads[idx - grad_buf->perception_size - grad_buf->interaction_size] = 0.0f;
    }

    int class_total = grad_buf->num_features + grad_buf->num_classes * grad_buf->num_features + grad_buf->num_classes;
    if (idx < grad_buf->num_features) {
        grad_buf->pooling_weight_grads[idx] = 0.0f;
    }
    if (idx < grad_buf->num_classes * grad_buf->num_features) {
        grad_buf->fc_weight_grads[idx] = 0.0f;
    }
    if (idx < grad_buf->num_classes) {
        grad_buf->fc_bias_grads[idx] = 0.0f;
    }

    if (idx == 0) {
        grad_buf->has_autodiff_grads = 0;
        grad_buf->has_backprop_grads = 0;
    }
}

__device__ void init_ca_param_map_device(Organism* organism) {
    CAParameterMap* param_map = organism->param_map;
    int entry_idx = blockIdx.x % organism->pool->capacity;
    PoolEntry* entry = &organism->pool->entries[entry_idx];
    Architecture arch;
    arch.num_heads = entry->num_heads;
    arch.head_dim = entry->head_dim;
    arch.channels = entry->channels;
    arch.grid_size = entry->grid_size;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        param_map->perception_size = arch.num_heads * arch.channels * arch.head_dim;
        param_map->interaction_size = arch.num_heads * arch.head_dim * arch.head_dim;
        param_map->flow_projection_size = arch.num_heads * 2 * arch.head_dim;
        param_map->total_ca_params = param_map->perception_size + param_map->interaction_size + param_map->flow_projection_size;
        param_map->grid_size = arch.grid_size;
        param_map->channels = arch.channels;
        param_map->hidden_dim = arch.num_heads * arch.head_dim;
        param_map->total_params = param_map->total_ca_params;
    }

    int total_threads = blockDim.x * gridDim.x;
    for (int h = tid; h < arch.num_heads; h += total_threads) {
        int per_head = arch.channels * arch.head_dim + arch.head_dim * arch.head_dim + 2 * arch.head_dim;
        int offset = h * per_head;

        param_map->perception_start[h] = offset;
        offset += arch.channels * arch.head_dim;

        param_map->interaction_start[h] = offset;
        offset += arch.head_dim * arch.head_dim;

        param_map->flow_projection_start[h] = offset;

        param_map->head_param_offsets[h] = param_map->perception_start[h];
        param_map->head_param_counts[h] = per_head;
    }
}

__device__ void init_training_mode_device(Organism* organism) {
    int entry_idx = blockIdx.x;
    HybridTrainingMode* mode = organism->training_mode;
    int grid_size = organism->ca_grid_size;
    float* genome = &organism->workspace_genomes[entry_idx * GENOME_SIZE * 2];
    float* gradients = organism->gradients_buffer;
    float ctx_metabolic = organism->ctx_metabolic;
    float ctx_stress = organism->ctx_stress;
    float ctx_morphogen = organism->ctx_morphogen;
    float ctx_complexity = organism->ctx_complexity;
    float ctx_niche = organism->ctx_niche;
    float ctx_learning = organism->ctx_learning;
    float ctx_performance = organism->ctx_performance;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // All threads compute same values, distribute writes across threads
    TrainingParams training_params;
    float gradient_fitness_weight = training_params.get_gradient_fitness_weight(genome, gradients, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);
    float coherence_fitness_weight = training_params.get_coherence_fitness_weight(genome, gradients, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);
    int batch_size_val = training_params.get_batch_size();
    float learning_rate_val = training_params.get_learning_rate(genome, gradients, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);
    float gradient_clip_norm_val = training_params.get_gradient_clip_norm(genome, gradients, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);

    int num_fields = 8;
    for (int field = global_tid; field < num_fields; field += total_threads) {
        switch (field) {
            case 0: mode->use_gradients = true; break;
            case 1: mode->use_selection = true; break;
            case 2: mode->gradient_fitness_weight = gradient_fitness_weight; break;
            case 3: mode->coherence_fitness_weight = coherence_fitness_weight; break;
            case 4: mode->batch_size = batch_size_val; break;
            case 5: mode->learning_rate = learning_rate_val; break;
            case 6: mode->gradient_clip_norm = gradient_clip_norm_val; break;
            case 7: break;  // adam_timestep incremented in hybrid_lifecycle (last wave)
        }
    }
}

__device__ void init_classifier_device(Organism* organism) {
    int entry_idx = blockIdx.x;
    PoolEntry* entry = &organism->pool->entries[entry_idx];
    ClassificationHead* classifier = &organism->classifier[entry_idx];

    size_t workspace_offset = 0;
    for (int prev_idx = 0; prev_idx < entry_idx; prev_idx++) {
        PoolEntry* prev = &organism->pool->entries[prev_idx];
        int prev_input_dim = prev->num_heads * POOLING_NUM_TILES * prev->channels;
        int prev_num_classes = organism->classifier_num_classes;
        workspace_offset += prev_input_dim + (prev_input_dim * prev_num_classes) + prev_num_classes;
    }
    float* workspace = organism->classifier_workspace + workspace_offset;

    int input_dim = entry->num_heads * POOLING_NUM_TILES * entry->channels;
    int num_classes = organism->classifier_num_classes;
    unsigned int seed = organism->classifier_seed + entry_idx;

    int tid = threadIdx.x;

    // All threads compute identical pointer values - redundant but no sync needed
    classifier->pooling_weights = workspace;
    classifier->fc_weights = workspace + input_dim;
    classifier->fc_bias = workspace + input_dim + (input_dim * num_classes);

    cg::this_grid().sync();

    curandState state;
    curand_init(seed + tid, 0, 0, &state);

    for (int pi = tid; pi < input_dim; pi += blockDim.x) {
        classifier->pooling_weights[pi] = curand_normal(&state) * 0.1f;
    }

    float scale = sqrtf(2.0f / (input_dim + num_classes));
    for (int pi = tid; pi < input_dim * num_classes; pi += blockDim.x) {
        classifier->fc_weights[pi] = curand_normal(&state) * scale;
    }

    for (int pi = tid; pi < num_classes; pi += blockDim.x) {
        classifier->fc_bias[pi] = 0.0f;
    }

    cg::this_grid().sync();
}

__device__ void init_curriculum_device(Organism* organism) {
    int entry_idx = blockIdx.x;
    AdaptiveCurriculum* curriculum = organism->curriculum;
    float* genome = &organism->workspace_genomes[entry_idx * GENOME_SIZE * 2];
    float* gradients = organism->gradients_buffer;
    float ctx_metabolic = organism->ctx_metabolic;
    float ctx_stress = organism->ctx_stress;
    float ctx_morphogen = organism->ctx_morphogen;
    float ctx_complexity = organism->ctx_complexity;
    float ctx_niche = organism->ctx_niche;
    float ctx_learning = organism->ctx_learning;
    float ctx_performance = organism->ctx_performance;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // Parallel dataset stats init - each thread handles different datasets
    for (int i = global_tid; i < NUM_ACTIVE_DATASETS; i += total_threads) {
        curriculum->stats[i].dataset_id = ACTIVE_DATASET_IDS[i];
        curriculum->stats[i].population_mean_accuracy = 0.0f;
        curriculum->stats[i].population_best_accuracy = 0.0f;
        curriculum->stats[i].population_accuracy_variance = 0.0f;
        curriculum->stats[i].niche_diversity = 0.0f;
        curriculum->stats[i].num_generations_trained = 0;
        curriculum->stats[i].activation_threshold_met = (i == 0);  // Only first dataset starts active
    }

    // All threads compute thresholds
    int acc_slot = GenomeParamTable::curriculum_accuracy_threshold;
    int div_slot = GenomeParamTable::curriculum_diversity_threshold;
    int gen_slot = GenomeParamTable::curriculum_min_generations;

    float accuracy_threshold = genome_to_param(
        genome, gradients, acc_slot,
        ctx_metabolic, ctx_stress, ctx_morphogen,
        ctx_complexity, ctx_niche, ctx_learning, ctx_performance,
        CURRICULUM_ACCURACY_THRESHOLD_MIN, CURRICULUM_ACCURACY_THRESHOLD_MAX
    );

    float diversity_threshold = genome_to_param(
        genome, gradients, div_slot,
        ctx_metabolic, ctx_stress, ctx_morphogen,
        ctx_complexity, ctx_niche, ctx_learning, ctx_performance,
        CURRICULUM_DIVERSITY_THRESHOLD_MIN, CURRICULUM_DIVERSITY_THRESHOLD_MAX
    );

    float min_generations_threshold = genome_to_param(
        genome, gradients, gen_slot,
        ctx_metabolic, ctx_stress, ctx_morphogen,
        ctx_complexity, ctx_niche, ctx_learning, ctx_performance,
        CURRICULUM_MIN_GENERATIONS_MIN, CURRICULUM_MIN_GENERATIONS_MAX
    );

    // Distribute scalar writes
    int num_fields = 6;
    for (int field = global_tid; field < num_fields; field += total_threads) {
        switch (field) {
            case 0: curriculum->current_dataset_idx = 0; break;
            case 1: curriculum->num_datasets_completed = 0; break;
            case 2: curriculum->curriculum_progress = 0.0f; break;
            case 3: curriculum->accuracy_threshold = accuracy_threshold; break;
            case 4: curriculum->diversity_threshold = diversity_threshold; break;
            case 5: curriculum->min_generations_threshold = min_generations_threshold; break;
        }
    }
}

__device__ void update_curriculum_device(Organism* organism) {
    AdaptiveCurriculum* curriculum = organism->curriculum;
    float* pool_task_accuracies = organism->tt_pool_task_accuracies;
    float* voronoi_occupancy_histogram = organism->tt_voronoi_occupancy_histogram;
    int pool_size = organism->tt_pool_size;
    int num_voronoi_cells = organism->tt_num_voronoi_cells;
    int generation = organism->generation;

    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;
    int total_threads = gridDim.x * blockDim.x;

    __shared__ float shared_sum[BLOCK_SIZE];
    __shared__ float shared_max[BLOCK_SIZE];
    __shared__ float shared_var[BLOCK_SIZE];
    __shared__ float shared_entropy[BLOCK_SIZE];

    int current_idx = curriculum->current_dataset_idx;
    DatasetStats* current_stats = &curriculum->stats[current_idx];

    // Parallel reduction for sum and max accuracy
    float local_sum = 0.0f;
    float local_max = 0.0f;
    for (int i = global_tid; i < pool_size; i += total_threads) {
        float acc = pool_task_accuracies[i];
        local_sum += acc;
        local_max = fmaxf(local_max, acc);
    }

    // Block-level reduction for sum
    shared_sum[tid] = local_sum;
    shared_max[tid] = local_max;
    cg::this_grid().sync();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        cg::this_grid().sync();
    }

    float mean_acc = shared_sum[0] / pool_size;
    float best_acc = shared_max[0];

    // Parallel variance calculation
    float local_var = 0.0f;
    for (int i = global_tid; i < pool_size; i += total_threads) {
        float diff = pool_task_accuracies[i] - mean_acc;
        local_var += diff * diff;
    }

    shared_var[tid] = local_var;
    cg::this_grid().sync();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_var[tid] += shared_var[tid + s];
        }
        cg::this_grid().sync();
    }

    float var_acc = shared_var[0] / pool_size;

    // Parallel entropy calculation
    float local_entropy = 0.0f;
    float total_organisms = (float)pool_size;
    for (int i = global_tid; i < num_voronoi_cells; i += total_threads) {
        if (voronoi_occupancy_histogram[i] > 0) {
            float p = voronoi_occupancy_histogram[i] / total_organisms;
            local_entropy -= p * logf(p);
        }
    }

    shared_entropy[tid] = local_entropy;
    cg::this_grid().sync();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_entropy[tid] += shared_entropy[tid + s];
        }
        cg::this_grid().sync();
    }

    DEVICE_FATAL_IF(num_voronoi_cells <= 1, "update_curriculum: num_voronoi_cells must be > 1");

    float log_cells = logf((float)num_voronoi_cells);
    float diversity = shared_entropy[0] / log_cells;

    // Single thread writes final stats (deterministic)
    if (global_tid == 0) {
        current_stats->num_generations_trained = generation;
        current_stats->population_mean_accuracy = mean_acc;
        current_stats->population_best_accuracy = best_acc;
        current_stats->population_accuracy_variance = var_acc;
        current_stats->niche_diversity = diversity;

        bool acc_met = mean_acc >= curriculum->accuracy_threshold;
        bool div_met = diversity >= curriculum->diversity_threshold;
        bool gen_met = generation >= (int)curriculum->min_generations_threshold;

        if (acc_met && div_met && gen_met && curriculum->current_dataset_idx < NUM_ACTIVE_DATASETS - 1) {
            curriculum->num_datasets_completed++;
            curriculum->current_dataset_idx++;
            curriculum->stats[curriculum->current_dataset_idx].activation_threshold_met = true;
            curriculum->curriculum_progress = (float)(curriculum->current_dataset_idx + 1) / (float)NUM_ACTIVE_DATASETS;
        }
    }
}

#endif