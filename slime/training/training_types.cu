
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
    Architecture arch = organism->current_arch;
    int num_classes = organism->cls_num_classes;
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    grad_buf->perception_grads = perception_grads;
    grad_buf->interaction_grads = interaction_grads;
    grad_buf->value_grads = value_grads;
    grad_buf->pooling_weight_grads = pooling_weight_grads;
    grad_buf->fc_weight_grads = fc_weight_grads;
    grad_buf->fc_bias_grads = fc_bias_grads;

    grad_buf->perception_size = arch.num_heads * arch.channels * arch.head_dim;
    grad_buf->interaction_size = arch.num_heads * arch.head_dim * arch.head_dim;
    grad_buf->value_size = arch.num_heads * arch.head_dim * arch.channels;
    grad_buf->num_classes = num_classes;
    grad_buf->num_features = arch.num_heads * arch.channels;

    grad_buf->has_autodiff_grads = 0;
    grad_buf->has_backprop_grads = 0;
}

__device__ void zero_unified_gradients_device(Organism* organism) {
    UnifiedGradientBuffer* grad_buf = organism->unified_grad_buffer;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = grad_buf->perception_size + grad_buf->interaction_size + grad_buf->value_size;

    if (idx < grad_buf->perception_size) {
        grad_buf->perception_grads[idx] = 0.0f;
    } else if (idx < grad_buf->perception_size + grad_buf->interaction_size) {
        grad_buf->interaction_grads[idx - grad_buf->perception_size] = 0.0f;
    } else if (idx < total) {
        grad_buf->value_grads[idx - grad_buf->perception_size - grad_buf->interaction_size] = 0.0f;
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
    Architecture arch = Architecture::maxBounds();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0) {
        param_map->perception_size = arch.num_heads * arch.channels * arch.head_dim;
        param_map->interaction_size = arch.num_heads * arch.head_dim * arch.head_dim;
        param_map->value_size = arch.num_heads * arch.head_dim * arch.channels;
        param_map->total_ca_params = param_map->perception_size + param_map->interaction_size + param_map->value_size;
        param_map->grid_size = arch.grid_size;
        param_map->channels = arch.channels;
        param_map->hidden_dim = arch.head_dim;
        param_map->total_params = param_map->total_ca_params;
    }

    int total_threads = blockDim.x * gridDim.x;
    for (int h = tid; h < arch.num_heads; h += total_threads) {
        int offset = h * (arch.channels * arch.head_dim + arch.head_dim * arch.head_dim + arch.head_dim * arch.channels);

        param_map->perception_start[h] = offset;
        offset += arch.channels * arch.head_dim;

        param_map->interaction_start[h] = offset;
        offset += arch.head_dim * arch.head_dim;

        param_map->value_start[h] = offset;

        param_map->head_param_offsets[h] = param_map->perception_start[h];
        param_map->head_param_counts[h] = arch.channels * arch.head_dim + arch.head_dim * arch.head_dim + arch.head_dim * arch.channels;
    }
}

__device__ void init_training_mode_device(Organism* organism) {
    HybridTrainingMode* mode = organism->training;
    int grid_size = organism->ca_grid_size;
    float* batch_images = organism->tt_batch_images;
    int* batch_labels = organism->tt_batch_labels;
    float* genome = organism->genome;
    float* gradients = organism->output_gradients;
    uint64_t genome_hash = organism->genome_hash;
    float ctx_metabolic = organism->ctx_metabolic;
    float ctx_stress = organism->ctx_stress;
    float ctx_morphogen = organism->ctx_morphogen;
    float ctx_complexity = organism->ctx_complexity;
    float ctx_niche = organism->ctx_niche;
    float ctx_learning = organism->ctx_learning;
    float ctx_performance = organism->ctx_performance;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        TrainingParams training_params;

        mode->use_gradients = true;
        mode->use_selection = true;
        mode->gradient_fitness_weight = training_params.get_gradient_fitness_weight(genome, gradients, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);
        mode->coherence_fitness_weight = training_params.get_coherence_fitness_weight(genome, gradients, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);
        mode->batch_size = training_params.get_batch_size(genome, gradients);
        mode->learning_rate = training_params.get_learning_rate(genome, gradients, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);
        mode->gradient_clip_norm = training_params.get_gradient_clip_norm(genome, gradients, ctx_metabolic, ctx_stress, ctx_morphogen, ctx_complexity, ctx_niche, ctx_learning, ctx_performance);
        mode->adam_timestep = 1;

        mode->batch_images = batch_images;
        mode->batch_labels = batch_labels;
    }
}

__device__ void init_classifier_device(Organism* organism) {
    ClassificationHead* classifier = organism->classifier;
    float* workspace = organism->classifier_workspace;
    Architecture arch = Architecture::maxBounds();
    int input_dim = arch.num_heads * arch.channels;
    int num_classes = organism->classifier_num_classes;
    unsigned int seed = organism->classifier_seed;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) {
        classifier->pooling_weights = workspace;
        classifier->fc_weights = workspace + input_dim;
        classifier->fc_bias = workspace + input_dim + (input_dim * num_classes);
        __threadfence();
        classifier->pointers_ready = 1;
    }
    while (classifier->pointers_ready == 0) {
        __threadfence();
    }
    __syncthreads();

    curandState state;
    curand_init(seed + tid, 0, 0, &state);

    if (tid < input_dim) {
        float val = curand_normal(&state) * 0.1f;
        if (isnan(val) || isinf(val)) {
            return;
        }
        classifier->pooling_weights[tid] = val;
    }

    if (tid < input_dim * num_classes) {
        float scale = sqrtf(2.0f / (input_dim + num_classes));
        float val = curand_normal(&state) * scale;
        if (isnan(val) || isinf(val)) {
            return;
        }
        classifier->fc_weights[tid] = val;
    }

    if (tid < num_classes) {
        classifier->fc_bias[tid] = 0.0f;
    }

    __syncthreads();

    if (tid == 0) {
        for (int i = 0; i < input_dim && i < 10; i++) {
            if (isnan(classifier->pooling_weights[i])) {
                return;
            }
        }
        for (int i = 0; i < input_dim * num_classes && i < 10; i++) {
            if (isnan(classifier->fc_weights[i])) {
                return;
            }
        }
    }
}

__device__ void init_curriculum_device(Organism* organism) {
    AdaptiveCurriculum* curriculum = organism->curriculum;
    float* genome = organism->genome;
    float* gradients = organism->output_gradients;
    uint64_t genome_hash = organism->genome_hash;
    float ctx_metabolic = organism->ctx_metabolic;
    float ctx_stress = organism->ctx_stress;
    float ctx_morphogen = organism->ctx_morphogen;
    float ctx_complexity = organism->ctx_complexity;
    float ctx_niche = organism->ctx_niche;
    float ctx_learning = organism->ctx_learning;
    float ctx_performance = organism->ctx_performance;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < NUM_ACTIVE_DATASETS; i++) {
            curriculum->stats[i].dataset_id = ACTIVE_DATASET_IDS[i];
            curriculum->stats[i].population_mean_accuracy = 0.0f;
            curriculum->stats[i].population_best_accuracy = 0.0f;
            curriculum->stats[i].population_accuracy_variance = 0.0f;
            curriculum->stats[i].niche_diversity = 0.0f;
            curriculum->stats[i].num_generations_trained = 0;
            curriculum->stats[i].activation_threshold_met = false;
        }

        curriculum->current_dataset_idx = 0;
        curriculum->num_datasets_completed = 0;
        curriculum->curriculum_progress = 0.0f;
        curriculum->stats[0].activation_threshold_met = true;

        int acc_slot = GenomeParamTable::curriculum_accuracy_threshold;
        int div_slot = GenomeParamTable::curriculum_diversity_threshold;
        int gen_slot = GenomeParamTable::curriculum_min_generations;

        curriculum->accuracy_threshold = genome_to_param(
            genome, gradients, acc_slot,
            ctx_metabolic, ctx_stress, ctx_morphogen,
            ctx_complexity, ctx_niche, ctx_learning, ctx_performance,
            CURRICULUM_ACCURACY_THRESHOLD_MIN, CURRICULUM_ACCURACY_THRESHOLD_MAX
        );

        curriculum->diversity_threshold = genome_to_param(
            genome, gradients, div_slot,
            ctx_metabolic, ctx_stress, ctx_morphogen,
            ctx_complexity, ctx_niche, ctx_learning, ctx_performance,
            CURRICULUM_DIVERSITY_THRESHOLD_MIN, CURRICULUM_DIVERSITY_THRESHOLD_MAX
        );

        curriculum->min_generations_threshold = genome_to_param(
            genome, gradients, gen_slot,
            ctx_metabolic, ctx_stress, ctx_morphogen,
            ctx_complexity, ctx_niche, ctx_learning, ctx_performance,
            CURRICULUM_MIN_GENERATIONS_MIN, CURRICULUM_MIN_GENERATIONS_MAX
        );
    }
}

__device__ void update_curriculum_device(Organism* organism) {
    AdaptiveCurriculum* curriculum = organism->curriculum;
    float* pool_task_accuracies = organism->tt_pool_task_accuracies;
    float* voronoi_occupancy_histogram = organism->tt_voronoi_occupancy_histogram;
    int pool_size = organism->tt_pool_size;
    int num_voronoi_cells = organism->tt_num_voronoi_cells;
    int generation = organism->generation;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int current_idx = curriculum->current_dataset_idx;
        DatasetStats* current_stats = &curriculum->stats[current_idx];

        current_stats->num_generations_trained = generation;

        float sum_acc = 0.0f;
        float best_acc = 0.0f;
        for (int i = 0; i < pool_size; i++) {
            float acc = pool_task_accuracies[i];
            sum_acc += acc;
            if (acc > best_acc) best_acc = acc;
        }
        current_stats->population_mean_accuracy = sum_acc / pool_size;
        current_stats->population_best_accuracy = best_acc;

        float var_sum = 0.0f;
        for (int i = 0; i < pool_size; i++) {
            float diff = pool_task_accuracies[i] - current_stats->population_mean_accuracy;
            var_sum += diff * diff;
        }
        current_stats->population_accuracy_variance = var_sum / pool_size;

        float total_organisms = (float)pool_size;
        float entropy = 0.0f;
        for (int i = 0; i < num_voronoi_cells; i++) {
            if (voronoi_occupancy_histogram[i] > 0) {
                float p = voronoi_occupancy_histogram[i] / total_organisms;
                entropy -= p * logf(p);
            }
        }
        if (num_voronoi_cells <= 1) {
            return;
        }
        float log_cells = logf((float)num_voronoi_cells);
        current_stats->niche_diversity = entropy / log_cells;

        bool acc_met = current_stats->population_mean_accuracy >= curriculum->accuracy_threshold;
        bool div_met = current_stats->niche_diversity >= curriculum->diversity_threshold;
        bool gen_met = generation >= (int)curriculum->min_generations_threshold;

        if (acc_met && div_met && gen_met && curriculum->current_dataset_idx < NUM_ACTIVE_DATASETS - 1) {
            curriculum->num_datasets_completed++;
            curriculum->current_dataset_idx++;

            int next_dataset_id = ACTIVE_DATASET_IDS[curriculum->current_dataset_idx];
            curriculum->stats[curriculum->current_dataset_idx].activation_threshold_met = true;
            curriculum->curriculum_progress = (float)(curriculum->current_dataset_idx + 1) / (float)NUM_ACTIVE_DATASETS;
        }
    }
}

#endif