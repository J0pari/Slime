#ifndef TELEMETRY_PROBES_CU
#define TELEMETRY_PROBES_CU

#include "../config/config.cu"
#include "../memory/archive.cu"
#include "../memory/pool.cu"
#include <cuda_runtime.h>

struct GenomeComplexityMetrics {
    float delta_diversity;
    float hash_entropy;
    float parameter_variance;
    int unique_hashes;
    float avg_deltas_per_genome;
};

struct ArchiveTopologyMetrics {
    float density_variance;
    float novelty_gradient;
    float hash_clustering_coefficient;
    int occupied_cells;
    float quality_range;
};

struct DIRESAEvolutionMetrics {
    float behavioral_drift_rate;
    float hardware_feature_correlation;
    float gradient_magnitude_avg;
    int archive_injections;
};

struct TaskPerformanceMetrics {
    float accuracy;
    float train_accuracy;
    float test_accuracy;
    float loss;
    float classification_stability;
    int correct_predictions;
    int total_predictions;
};

struct MemoryAllocationMetrics {
    size_t total_gpu_allocated;
    size_t total_gpu_free;
    size_t total_gpu_capacity;
    size_t unified_memory_allocated;
    size_t archive_pools_size;
    size_t training_pools_size;
    size_t ca_state_size;
    size_t chemical_field_size;
    size_t behavioral_pools_size;
    size_t diresa_weights_size;
    size_t autodiff_tape_size;
};

struct TelemetryBuffer {
    GenomeComplexityMetrics genome_complexity;
    ArchiveTopologyMetrics archive_topology;
    DIRESAEvolutionMetrics diresa_evolution;
    TaskPerformanceMetrics task_performance;
    MemoryAllocationMetrics memory_allocation;
    int generation;
    bool valid;
};

__device__ void track_allocation(
    const char* label,
    void* ptr,
    size_t size_bytes,
    cudaError_t result,
    MemoryAllocationMetrics* metrics
) {
    float size_mb = size_bytes / (1024.0f * 1024.0f);

    if (result != cudaSuccess) {
        printf("[MEM ERROR] %s FAILED: %s (requested %.2f MB)\n",
               label, cudaGetErrorString(result), size_mb);
        return;
    }

    if (ptr == nullptr) {
        printf("[MEM ERROR] %s returned nullptr (requested %.2f MB)\n", label, size_mb);
        return;
    }

    printf("[MEM OK] %s: %.2f MB at %p\n", label, size_mb, ptr);
    metrics->total_gpu_allocated += size_bytes;
}

#define TRACKED_ALLOC(ptr, size, metrics, category_size) \
    do { \
        cudaError_t alloc_err = cudaMalloc(&(ptr), (size)); \
        track_allocation(#ptr, (void*)(ptr), (size), alloc_err, (metrics)); \
        if (alloc_err != cudaSuccess) { \
            printf("[FATAL] Allocation failed: %s\n", #ptr); \
            return; \
        } \
        (category_size) += (size); \
    } while(0)

__global__ void genome_complexity_probe_kernel(
    ComponentPool* pool,
    GenomeComplexityMetrics* metrics
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int active_count = Atomics::load_int(pool->active_count);
    if (active_count == 0) {
        metrics->delta_diversity = 0.0f;
        metrics->hash_entropy = 0.0f;
        metrics->parameter_variance = 0.0f;
        metrics->unique_hashes = 0;
        metrics->avg_deltas_per_genome = 0.0f;
        return;
    }

    uint64_t seen_hashes[MAX_POOL_SIZE];
    int unique_count = 0;
    float total_deltas = 0.0f;
    float hash_frequencies[MAX_POOL_SIZE];

    for (int i = 0; i < active_count && i < MAX_POOL_SIZE; i++) {
        if (!pool->entries[i].alive) continue;

        uint64_t hash = pool->entries[i].genome_hash;
        bool found = false;
        for (int j = 0; j < unique_count; j++) {
            if (seen_hashes[j] == hash) {
                hash_frequencies[j] += 1.0f;
                found = true;
                break;
            }
        }
        if (!found && unique_count < MAX_POOL_SIZE) {
            seen_hashes[unique_count] = hash;
            hash_frequencies[unique_count] = 1.0f;
            unique_count++;
        }

        total_deltas += pool->entries[i].num_deltas;
    }

    metrics->unique_hashes = unique_count;
    metrics->avg_deltas_per_genome = total_deltas / active_count;

    float entropy = 0.0f;
    for (int i = 0; i < unique_count; i++) {
        float p = hash_frequencies[i] / active_count;
        if (p > 0.0f) {
            entropy -= p * log2f(p);
        }
    }
    metrics->hash_entropy = entropy;

    metrics->delta_diversity = (float)unique_count / active_count;
    metrics->parameter_variance = 0.0f;
}

__global__ void archive_topology_probe_kernel(
    GPUElite* archive,
    int archive_size,
    VoronoiCell* voronoi_cells,
    int num_cells,
    ArchiveTopologyMetrics* metrics
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    if (archive_size == 0 || num_cells == 0) {
        metrics->density_variance = 0.0f;
        metrics->novelty_gradient = 0.0f;
        metrics->hash_clustering_coefficient = 0.0f;
        metrics->occupied_cells = 0;
        metrics->quality_range = 0.0f;
        return;
    }

    int occupied = 0;
    float sum_density = 0.0f;
    float sum_density_sq = 0.0f;
    float min_quality = 1e10f;
    float max_quality = -1e10f;

    for (int i = 0; i < num_cells; i++) {
        if (voronoi_cells[i].density > 0) {
            occupied++;
            float d = (float)voronoi_cells[i].density;
            sum_density += d;
            sum_density_sq += d * d;

            float q = voronoi_cells[i].quality_threshold;
            min_quality = fminf(min_quality, q);
            max_quality = fmaxf(max_quality, q);
        }
    }

    metrics->occupied_cells = occupied;
    metrics->quality_range = max_quality - min_quality;

    if (occupied > 0) {
        float mean_density = sum_density / occupied;
        float variance = (sum_density_sq / occupied) - (mean_density * mean_density);
        metrics->density_variance = sqrtf(variance);
    } else {
        metrics->density_variance = 0.0f;
    }

    metrics->novelty_gradient = (float)occupied / num_cells;
    metrics->hash_clustering_coefficient = 1.0f - metrics->novelty_gradient;
}

__global__ void diresa_evolution_probe_kernel(
    GPUElite* archive,
    int archive_size,
    DIRESAEvolutionMetrics* metrics
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    if (archive_size == 0) {
        metrics->behavioral_drift_rate = 0.0f;
        metrics->hardware_feature_correlation = 0.0f;
        metrics->gradient_magnitude_avg = 0.0f;
        metrics->archive_injections = 0;
        return;
    }

    float sum_drift = 0.0f;
    float sum_hw_corr = 0.0f;
    int recent_count = 0;

    for (int i = 0; i < archive_size && i < MAX_ARCHIVE_SIZE; i++) {
        if (archive[i].generation > 0 && recent_count < 100) {
            float drift = 0.0f;
            for (int d = 0; d < BEHAVIORAL_DIM_HW_MAX; d++) {
                drift += fabsf(archive[i].hw_coords[d]);
            }
            for (int d = 0; d < BEHAVIORAL_DIM_TASK_MAX; d++) {
                drift += fabsf(archive[i].task_coords[d]);
            }
            for (int d = 0; d < BEHAVIORAL_DIM_GEN_MAX; d++) {
                drift += fabsf(archive[i].gen_coords[d]);
            }
            int total_dims = BEHAVIORAL_DIM_HW_MAX + BEHAVIORAL_DIM_TASK_MAX + BEHAVIORAL_DIM_GEN_MAX;
            sum_drift += drift / total_dims;

            float hw_sum = 0.0f;
            for (int h = 0; h < (WMMA_TILE_DIM - 1); h++) {
                hw_sum += archive[i].hardware_features[h];
            }
            sum_hw_corr += hw_sum / (WMMA_TILE_DIM - 1);

            recent_count++;
        }
    }

    metrics->behavioral_drift_rate = recent_count > 0 ? sum_drift / recent_count : 0.0f;
    metrics->hardware_feature_correlation = recent_count > 0 ? sum_hw_corr / recent_count : 0.0f;
    metrics->gradient_magnitude_avg = 0.0f;
    metrics->archive_injections = recent_count;
}

__global__ void task_performance_probe_kernel(
    float* logits,
    int* labels,
    int batch_size,
    int num_classes,
    TaskPerformanceMetrics* metrics,
    bool is_train_batch
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int correct = 0;
    float total_loss = 0.0f;

    for (int b = 0; b < batch_size; b++) {
        int predicted_class = 0;
        float max_logit = logits[b * num_classes];

        for (int c = 1; c < num_classes; c++) {
            if (logits[b * num_classes + c] > max_logit) {
                max_logit = logits[b * num_classes + c];
                predicted_class = c;
            }
        }

        if (predicted_class == labels[b]) {
            correct++;
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            sum_exp += expf(logits[b * num_classes + c] - max_logit);
        }
        float log_sum_exp = max_logit + logf(sum_exp);
        total_loss -= (logits[b * num_classes + labels[b]] - log_sum_exp);
    }

    float computed_accuracy = (float)correct / batch_size;

    metrics->correct_predictions = correct;
    metrics->total_predictions = batch_size;
    metrics->accuracy = computed_accuracy;
    metrics->loss = total_loss / batch_size;
    metrics->classification_stability = 0.0f;

    if (is_train_batch) {
        metrics->train_accuracy = computed_accuracy;
    } else {
        metrics->test_accuracy = computed_accuracy;
    }
}

#endif
