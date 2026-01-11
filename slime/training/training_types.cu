
#ifndef TRAINING_TYPES_CU
#define TRAINING_TYPES_CU

#include "../config/config.cu"
#include "../utils/genome_params.cuh"
#include <curand_kernel.h>

// Forward declaration to avoid circular dependency
struct DatasetDescriptor;

struct ClassificationHead {
    float* pooling_weights;
    float* fc_weights;
    float* fc_bias;
};

struct CAParameterMap {
    int perception_start[NUM_HEADS_MAX];
    int interaction_start[NUM_HEADS_MAX];
    int value_start[NUM_HEADS_MAX];

    int head_param_offsets[NUM_HEADS_MAX];
    int head_param_counts[NUM_HEADS_MAX];

    int perception_size;
    int interaction_size;
    int value_size;

    int total_params;
    int total_ca_params;

    int batch_size;
    int grid_size;
    int channels;
    int hidden_dim;
};

struct HybridTrainingMode {
    bool use_gradients;
    bool use_selection;
    float gradient_fitness_weight;
    float coherence_fitness_weight;
    float* batch_images;
    int* batch_labels;
    int batch_size;
    ClassificationHead* classifier;
    float learning_rate;
    float gradient_clip_norm;
    float* adam_m_perception;
    float* adam_v_perception;
    float* adam_m_interaction;
    float* adam_v_interaction;
    float* adam_m_value;
    float* adam_v_value;
    float* adam_m_policy;
    float* adam_v_policy;
    int adam_timestep;
    bool is_train_batch;
};

struct Dataset {
    const DatasetDescriptor* descriptor;  // Points to entry in DATASET_REGISTRY
    unsigned char* samples;               // Raw sample data (format determined by descriptor->encoding)
    unsigned char* labels;
    int num_samples;
    bool is_train;
};

struct DatasetStats {
    int dataset_id;
    float population_mean_accuracy;
    float population_best_accuracy;
    float population_accuracy_variance;
    float niche_diversity;  // Voronoi cell occupancy entropy
    int num_generations_trained;
    bool activation_threshold_met;
};

struct AdaptiveCurriculum {
    DatasetStats stats[NUM_ACTIVE_DATASETS];
    int current_dataset_idx;  // Index into ACTIVE_DATASET_IDS array
    int num_datasets_completed;
    float curriculum_progress;  // 0.0 to 1.0

    // Thresholds for dataset activation (genome-derived per organism)
    float accuracy_threshold;
    float diversity_threshold;
    float min_generations_threshold;
};

__global__ void init_ca_param_map_kernel(CAParameterMap* param_map, ArchitectureParams arch) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        param_map->perception_size = arch.num_heads * arch.channels * arch.head_dim;
        param_map->interaction_size = arch.num_heads * arch.head_dim * arch.head_dim;
        param_map->value_size = arch.num_heads * arch.head_dim * arch.channels;
        param_map->total_ca_params = param_map->perception_size + param_map->interaction_size + param_map->value_size;
        param_map->grid_size = arch.grid_size;
        param_map->channels = arch.channels;
        param_map->hidden_dim = arch.head_dim;

        int offset = 0;
        for (int h = 0; h < arch.num_heads; h++) {
            param_map->perception_start[h] = offset;
            offset += arch.channels * arch.head_dim;

            param_map->interaction_start[h] = offset;
            offset += arch.head_dim * arch.head_dim;

            param_map->value_start[h] = offset;
            offset += arch.head_dim * arch.channels;

            param_map->head_param_offsets[h] = param_map->perception_start[h];
            param_map->head_param_counts[h] = arch.channels * arch.head_dim + arch.head_dim * arch.head_dim + arch.head_dim * arch.channels;
        }
        param_map->total_params = offset;
    }
}

__global__ void init_training_mode_kernel(HybridTrainingMode* mode, int grid_size, float* batch_images, int* batch_labels) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        mode->use_gradients = true;
        mode->use_selection = true;
        mode->gradient_fitness_weight = 0.7f;
        mode->coherence_fitness_weight = 0.3f;
        mode->batch_size = 32;
        mode->learning_rate = 0.001f;
        mode->gradient_clip_norm = 1.0f;
        mode->adam_timestep = 1;

        // Use preallocated buffers instead of device cudaMalloc
        mode->batch_images = batch_images;
        mode->batch_labels = batch_labels;
    }
}

__global__ void init_classifier_kernel(ClassificationHead* classifier, int input_dim, int num_classes, unsigned int seed, float* workspace) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) {
        classifier->pooling_weights = workspace;
        classifier->fc_weights = workspace + input_dim;
        classifier->fc_bias = workspace + input_dim + (input_dim * num_classes);
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

__global__ void init_curriculum_kernel(
    AdaptiveCurriculum* curriculum,
    float* genome,
    float* gradients,
    uint64_t genome_hash,
    float ctx_metabolic,
    float ctx_stress,
    float ctx_morphogen,
    float ctx_complexity,
    float ctx_niche,
    float ctx_learning,
    float ctx_performance
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Initialize all dataset stats
        for (int i = 0; i < NUM_ACTIVE_DATASETS; i++) {
            curriculum->stats[i].dataset_id = ACTIVE_DATASET_IDS[i];
            curriculum->stats[i].population_mean_accuracy = 0.0f;
            curriculum->stats[i].population_best_accuracy = 0.0f;
            curriculum->stats[i].population_accuracy_variance = 0.0f;
            curriculum->stats[i].niche_diversity = 0.0f;
            curriculum->stats[i].num_generations_trained = 0;
            curriculum->stats[i].activation_threshold_met = false;
        }

        // Start with first dataset in active list
        curriculum->current_dataset_idx = 0;
        curriculum->num_datasets_completed = 0;
        curriculum->curriculum_progress = 0.0f;
        curriculum->stats[0].activation_threshold_met = true;

        // Genome-derive curriculum thresholds using full context
        int acc_slot = derive_param_slot(genome_hash, "curriculum_accuracy_threshold");
        int div_slot = derive_param_slot(genome_hash, "curriculum_diversity_threshold");
        int gen_slot = derive_param_slot(genome_hash, "curriculum_min_generations");

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

__global__ void update_curriculum_kernel(
    AdaptiveCurriculum* curriculum,
    float* pool_task_accuracies,
    float* voronoi_occupancy_histogram,
    int pool_size,
    int num_voronoi_cells,
    int generation
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int current_idx = curriculum->current_dataset_idx;
        DatasetStats* current_stats = &curriculum->stats[current_idx];

        current_stats->num_generations_trained = generation;

        // Compute population mean and best accuracy
        float sum_acc = 0.0f;
        float best_acc = 0.0f;
        for (int i = 0; i < pool_size; i++) {
            float acc = pool_task_accuracies[i];
            sum_acc += acc;
            if (acc > best_acc) best_acc = acc;
        }
        current_stats->population_mean_accuracy = sum_acc / pool_size;
        current_stats->population_best_accuracy = best_acc;

        // Compute accuracy variance
        float var_sum = 0.0f;
        for (int i = 0; i < pool_size; i++) {
            float diff = pool_task_accuracies[i] - current_stats->population_mean_accuracy;
            var_sum += diff * diff;
        }
        current_stats->population_accuracy_variance = var_sum / pool_size;

        // Compute niche diversity (Shannon entropy of Voronoi occupancy)
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

        // Check if thresholds met for current dataset
        bool acc_met = current_stats->population_mean_accuracy >= curriculum->accuracy_threshold;
        bool div_met = current_stats->niche_diversity >= curriculum->diversity_threshold;
        bool gen_met = generation >= (int)curriculum->min_generations_threshold;

        // If all thresholds met, progress to next dataset in active list
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