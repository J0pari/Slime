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

__device__ void print_size(const char* prefix, const char* label, size_t size_bytes, const char* suffix) {
    if (size_bytes < BYTES_PER_KB) {
        printf("%s%s: %llu bytes%s\n", prefix, label, (unsigned long long)size_bytes, suffix);
    } else if (size_bytes < BYTES_PER_MB) {
        printf("%s%s: %.2f KB%s\n", prefix, label, size_bytes / (float)BYTES_PER_KB, suffix);
    } else {
        printf("%s%s: %.2f MB%s\n", prefix, label, size_bytes / (float)BYTES_PER_MB, suffix);
    }
}

__device__ void track_allocation(
    const char* label,
    void* ptr,
    size_t size_bytes,
    cudaError_t result,
    MemoryAllocationMetrics* metrics
) {
    if (result != cudaSuccess) {
        printf("[MEM ERROR] %s FAILED: %s (requested %llu bytes)\n",
               label, cudaGetErrorString(result), (unsigned long long)size_bytes);
        return;
    }

    if (ptr == nullptr) {
        printf("[MEM ERROR] %s returned nullptr (requested %llu bytes)\n", label, (unsigned long long)size_bytes);
        return;
    }

    if (size_bytes < BYTES_PER_KB) {
        printf("[MEM OK] %s: %llu bytes at %p\n", label, (unsigned long long)size_bytes, ptr);
    } else if (size_bytes < BYTES_PER_MB) {
        printf("[MEM OK] %s: %.2f KB at %p\n", label, size_bytes / (float)BYTES_PER_KB, ptr);
    } else {
        printf("[MEM OK] %s: %.2f MB at %p\n", label, size_bytes / (float)BYTES_PER_MB, ptr);
    }
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

        PoolEntry* e = &pool->entries[i];
        uint64_t hash = e->genome_hash;
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

        total_deltas += e->num_deltas;
    }

    int display_limit = active_count < 10 ? active_count : 10;
    printf("[POOL-DIVERSITY] active=%d unique=%d showing top %d organisms\n", active_count, unique_count, display_limit);
    for (int i = 0; i < display_limit; i++) {
        if (!pool->entries[i].alive) continue;
        PoolEntry* e = &pool->entries[i];
        printf("  [%3d] fit=%.6f task=%.4f gen_gap=%.4f hw_eff=%.4f age=%d deltas=%d\n",
               i, e->fitness, e->task_accuracy, e->generalization_gap, e->hardware_efficiency, e->age, e->num_deltas);
        printf("        exp(α=%.3f β=%.3f γ=%.3f δ=%.3f) arch(h=%d ch=%d grid=%d) diresa(h1=%d h2=%d)\n",
               e->fitness_task_exponent, e->fitness_gen_exponent, e->fitness_rank_exponent, e->fitness_efficiency_exponent,
               e->num_heads, e->channels, e->grid_size, e->diresa_hidden1, e->diresa_hidden2);
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

    int display_limit = archive_size < 10 ? archive_size : 10;
    printf("[ARCHIVE-ELITES] size=%d showing %d niche specializations\n", archive_size, display_limit);
    for (int i = 0; i < display_limit; i++) {
        GPUElite* e = &archive[i];
        printf("  [%3d] gen=%d fit=%.6f hw[%.3f,%.3f,%.3f,%.3f] task[%.3f,%.3f,%.3f,%.3f] gen[%.3f,%.3f]\n",
               i, e->generation, e->fitness,
               e->hw_coords[0], e->hw_coords[1], e->hw_coords[2], e->hw_coords[3],
               e->task_coords[0], e->task_coords[1], e->task_coords[2], e->task_coords[3],
               e->gen_coords[0], e->gen_coords[1]);
    }
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

    int display_limit = archive_size < 5 ? archive_size : 5;
    printf("[DIRESA-LATENT] showing %d/%d elite latent coordinates\n", display_limit, archive_size);

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

            if (i < display_limit) {
                printf("  [%3d] drift=%.4f hw_feat_avg=%.4f latent[%.3f,%.3f,%.3f,%.3f...]\n",
                       i, drift / total_dims, hw_sum / (WMMA_TILE_DIM - 1),
                       archive[i].latent_genome[0], archive[i].latent_genome[1],
                       archive[i].latent_genome[2], archive[i].latent_genome[3]);
            }

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
