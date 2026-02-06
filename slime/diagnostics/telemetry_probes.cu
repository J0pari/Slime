#ifndef TELEMETRY_PROBES_CU
#define TELEMETRY_PROBES_CU

#include "../config/config.cu"
#include "../memory/archive.cu"
#include "../memory/pool.cu"
#include "../utils/cuda_primitives.cuh"
#include "../core/ca_state.cuh"
#include "../core/chemotaxis.cu"
#include "../metrics/hardware_geometry.cu"
#include <cuda_runtime.h>
#include <cmath>

struct GenomeComplexityMetrics {
    float delta_diversity;
    float hash_entropy;
    float parameter_variance;
    int unique_hashes;
    float avg_deltas_per_genome;
};

struct ArchiveTopologyMetrics {
    int occupied_cells;
    int frontier_cells_gained;   
    int frontier_cells_lost;     
    int sparse_cell_count;       
    float niche_entropy;         
    float novelty_gradient;      

    float elite_fitness_best;
    float elite_fitness_mean;    
    float elite_fitness_delta;   
    float quality_floor;         
    float quality_mean;          
    float quality_range;

    float density_mean;          
    float density_max;           
    float density_variance;

    float hw_axis_min, hw_axis_max, hw_axis_mean;
    float task_axis_min, task_axis_max, task_axis_mean;
    float gen_axis_min, gen_axis_max, gen_axis_mean;

    float axis_corr_hw_task;
    float axis_corr_hw_gen;
    float axis_corr_task_gen;

    int total_population;
    int births_since_checkpoint;   
    int deaths_since_checkpoint;   

    float hash_clustering_coefficient;
};

struct DIRESAEvolutionMetrics {
    float recon_loss_hw;
    float recon_loss_task;
    float recon_loss_gen;
    float recon_loss_total;
    float behavioral_drift_rate;
    float latent_utilization;
    float compression_ratio;
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
    float avg_confidence;
    int correct_predictions;
    int total_predictions;
    int per_class_correct[NUM_CLASSES_MAX];
    int per_class_total[NUM_CLASSES_MAX];
};

struct PopulationMetrics {
    float total_accuracy;
    float total_generalization_gap;
    float total_hardware_efficiency;
    float total_fitness;
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
    size_t device_heap_limit;        
    size_t device_heap_allocated;    
};

struct TelemetryBuffer {
    GenomeComplexityMetrics genome_complexity;
    ArchiveTopologyMetrics archive_topology;
    DIRESAEvolutionMetrics diresa_evolution;
    TaskPerformanceMetrics task_performance;
    PopulationMetrics population_metrics;
    MemoryAllocationMetrics memory_allocation;
    int generation;
    bool valid;

    ArchiveTopologyMetrics last_checkpoint;
    int last_occupancy[MAX_CELLS];
    int last_total_spawned;
    int last_total_culled;
};

__device__ void track_allocation(
    void* ptr,
    size_t size_bytes,
    cudaError_t result,
    MemoryAllocationMetrics* metrics
) {
    if (result != cudaSuccess || ptr == nullptr) {
        return;
    }
    metrics->total_gpu_allocated += size_bytes;
}

#define TRACKED_ALLOC(ptr, size, metrics, category_size) \
    do { \
        size_t heap_limit = Atomics::load_size(&(metrics)->device_heap_limit); \
        size_t heap_used = Atomics::load_size(&(metrics)->device_heap_allocated); \
        size_t heap_free = heap_limit - heap_used; \
        if (heap_free < (size)) { \
            return; \
        } \
        cudaError_t alloc_err = cudaMalloc(&(ptr), (size)); \
        if (alloc_err != cudaSuccess || (ptr) == nullptr) { \
            return; \
        } \
        Atomics::add_size(&(metrics)->device_heap_allocated, (size)); \
        track_allocation((void*)(ptr), (size), alloc_err, (metrics)); \
        (category_size) += (size); \
    } while(0)

__device__ void genome_complexity_probe(
    ComponentPool* pool,
    GenomeComplexityMetrics* metrics
) {
    int capacity = pool->capacity;
    DEVICE_FATAL_IF(capacity == 0, "genome_complexity_probe: pool capacity is 0");

    uint64_t seen_hashes[POOL_CAPACITY_MAX];
    int unique_count = 0;
    float total_deltas = 0.0f;
    float hash_frequencies[POOL_CAPACITY_MAX];
    int alive_count = 0;

    for (int i = 0; i < capacity; i++) {
        if (!pool->alive_flags[i]) continue;
        alive_count++;

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
        if (!found && unique_count < POOL_CAPACITY_MAX) {
            seen_hashes[unique_count] = hash;
            hash_frequencies[unique_count] = 1.0f;
            unique_count++;
        }

        total_deltas += e->num_deltas;
    }

    DEVICE_FATAL_IF(alive_count == 0, "genome_complexity_probe: no alive entries");

    int display_count = 0;
    for (int i = 0; i < capacity && display_count < 10; i++) {
        if (!pool->alive_flags[i]) continue;
        PoolEntry* e = &pool->entries[i];
        display_count++;
    }

    metrics->unique_hashes = unique_count;
    metrics->avg_deltas_per_genome = total_deltas / alive_count;

    float entropy = 0.0f;
    for (int i = 0; i < unique_count; i++) {
        float p = hash_frequencies[i] / alive_count;
        if (p > 0.0f) {
            entropy -= p * log2f(p);
        }
    }
    metrics->hash_entropy = entropy;

    metrics->delta_diversity = (float)unique_count / alive_count;
    metrics->parameter_variance = NAN;
}

__device__ void archive_topology_probe(
    GPUElite* archive,
    int archive_size,
    VoronoiCell* voronoi_cells,
    int num_cells,
    ArchiveTopologyMetrics* metrics,
    ArchiveTopologyMetrics* prev_metrics,  
    int* prev_occupied_flags,              
    int hw_dim, int task_dim, int gen_dim
) {

    DEVICE_FATAL_IF(archive_size == 0, "archive_topology_probe: archive_size is 0");
    DEVICE_FATAL_IF(num_cells == 0, "archive_topology_probe: num_cells is 0");

    int occupied = 0;
    int sparse_count = 0;
    int frontier_gained = 0;
    int frontier_lost = 0;
    int total_pop = 0;
    float sum_density = 0.0f;
    float sum_density_sq = 0.0f;
    int max_density = 0;

    float min_quality = 1e10f;
    float max_quality = -1e10f;
    float sum_quality = 0.0f;
    float best_fitness = -1e10f;
    float sum_fitness = 0.0f;

    float hw_min = 1e10f, hw_max = -1e10f, hw_sum = 0.0f;
    float task_min = 1e10f, task_max = -1e10f, task_sum = 0.0f;
    float gen_min = 1e10f, gen_max = -1e10f, gen_sum = 0.0f;

    for (int i = 0; i < num_cells; i++) {
        int dens = voronoi_cells[i].density;
        int prev_dens = (prev_occupied_flags && i < num_cells) ? prev_occupied_flags[i] : 0;

        if (dens > 0) {
            occupied++;
            total_pop += dens;
            float d = (float)dens;
            sum_density += d;
            sum_density_sq += d * d;

            if (dens < 3) sparse_count++;  
            if (prev_dens == 0) frontier_gained++;  
            if (dens > max_density) max_density = dens;

            float q = voronoi_cells[i].quality_threshold;
            min_quality = fminf(min_quality, q);
            max_quality = fmaxf(max_quality, q);
            sum_quality += q;

            float hw_mag = 0.0f, task_mag = 0.0f, gen_mag = 0.0f;
            for (int d = 0; d < hw_dim && voronoi_cells[i].hw_centroid; d++) {
                hw_mag += voronoi_cells[i].hw_centroid[d] * voronoi_cells[i].hw_centroid[d];
            }
            for (int d = 0; d < task_dim && voronoi_cells[i].task_centroid; d++) {
                task_mag += voronoi_cells[i].task_centroid[d] * voronoi_cells[i].task_centroid[d];
            }
            for (int d = 0; d < gen_dim && voronoi_cells[i].gen_centroid; d++) {
                gen_mag += voronoi_cells[i].gen_centroid[d] * voronoi_cells[i].gen_centroid[d];
            }
            hw_mag = sqrtf(hw_mag); task_mag = sqrtf(task_mag); gen_mag = sqrtf(gen_mag);

            hw_min = fminf(hw_min, hw_mag); hw_max = fmaxf(hw_max, hw_mag); hw_sum += hw_mag;
            task_min = fminf(task_min, task_mag); task_max = fmaxf(task_max, task_mag); task_sum += task_mag;
            gen_min = fminf(gen_min, gen_mag); gen_max = fmaxf(gen_max, gen_mag); gen_sum += gen_mag;
        } else if (prev_dens > 0) {
            frontier_lost++;  
        }

        if (prev_occupied_flags) prev_occupied_flags[i] = dens;
    }

    DEVICE_FATAL_IF(occupied <= 0, "archive_topology_probe: no occupied cells after scan");

    for (int i = 0; i < archive_size; i++) {
        float f = archive->fitness[i];
        if (f > best_fitness) best_fitness = f;
        sum_fitness += f;
    }

    float entropy = 0.0f;
    if (total_pop > 0) {
        for (int i = 0; i < num_cells; i++) {
            if (voronoi_cells[i].density > 0) {
                float p = (float)voronoi_cells[i].density / total_pop;
                entropy -= p * log2f(p);
            }
        }
    }

    metrics->occupied_cells = occupied;
    metrics->frontier_cells_gained = frontier_gained;
    metrics->frontier_cells_lost = frontier_lost;
    metrics->sparse_cell_count = sparse_count;
    metrics->niche_entropy = entropy;
    metrics->novelty_gradient = (float)occupied / num_cells;

    DEVICE_FATAL_IF(archive_size <= 0, "archive_topology_probe: archive_size must be positive");
    DEVICE_FATAL_IF(!prev_metrics, "archive_topology_probe: prev_metrics required");

    metrics->elite_fitness_best = best_fitness;
    metrics->elite_fitness_mean = sum_fitness / archive_size;
    metrics->elite_fitness_delta = best_fitness - prev_metrics->elite_fitness_best;
    metrics->quality_floor = min_quality;
    metrics->quality_mean = sum_quality / occupied;
    metrics->quality_range = max_quality - min_quality;

    metrics->density_mean = sum_density / occupied;
    metrics->density_max = (float)max_density;

    metrics->hw_axis_min = hw_min;
    metrics->hw_axis_max = hw_max;
    metrics->hw_axis_mean = hw_sum / occupied;
    metrics->task_axis_min = task_min;
    metrics->task_axis_max = task_max;
    metrics->task_axis_mean = task_sum / occupied;
    metrics->gen_axis_min = gen_min;
    metrics->gen_axis_max = gen_max;
    metrics->gen_axis_mean = gen_sum / occupied;

    float sum_hw_task = 0.0f, sum_hw_gen = 0.0f, sum_task_gen = 0.0f;
    float sum_hw_sq = 0.0f, sum_task_sq = 0.0f, sum_gen_sq = 0.0f;
    int corr_n = 0;
    for (int i = 0; i < num_cells; i++) {
        if (voronoi_cells[i].density > 0) {
            float hw_mag = 0.0f, task_mag = 0.0f, gen_mag = 0.0f;
            for (int d = 0; d < hw_dim && voronoi_cells[i].hw_centroid; d++) {
                hw_mag += voronoi_cells[i].hw_centroid[d] * voronoi_cells[i].hw_centroid[d];
            }
            for (int d = 0; d < task_dim && voronoi_cells[i].task_centroid; d++) {
                task_mag += voronoi_cells[i].task_centroid[d] * voronoi_cells[i].task_centroid[d];
            }
            for (int d = 0; d < gen_dim && voronoi_cells[i].gen_centroid; d++) {
                gen_mag += voronoi_cells[i].gen_centroid[d] * voronoi_cells[i].gen_centroid[d];
            }
            hw_mag = sqrtf(hw_mag); task_mag = sqrtf(task_mag); gen_mag = sqrtf(gen_mag);
            sum_hw_task += hw_mag * task_mag;
            sum_hw_gen += hw_mag * gen_mag;
            sum_task_gen += task_mag * gen_mag;
            sum_hw_sq += hw_mag * hw_mag;
            sum_task_sq += task_mag * task_mag;
            sum_gen_sq += gen_mag * gen_mag;
            corr_n++;
        }
    }
    DEVICE_FATAL_IF(corr_n <= 1, "archive_topology_probe: corr_n <= 1, insufficient data for correlation");
    float hw_mean = hw_sum / corr_n;
    float task_mean = task_sum / corr_n;
    float gen_mean = gen_sum / corr_n;
    float hw_var = sum_hw_sq / corr_n - hw_mean * hw_mean;
    float task_var = sum_task_sq / corr_n - task_mean * task_mean;
    float gen_var = sum_gen_sq / corr_n - gen_mean * gen_mean;
    DEVICE_FATAL_IF(hw_var <= 0.0f, "archive_topology_probe: hw_var <= 0");
    DEVICE_FATAL_IF(task_var <= 0.0f, "archive_topology_probe: task_var <= 0");
    DEVICE_FATAL_IF(gen_var <= 0.0f, "archive_topology_probe: gen_var <= 0");
    float hw_std = sqrtf(hw_var);
    float task_std = sqrtf(task_var);
    float gen_std = sqrtf(gen_var);
    float cov_hw_task = sum_hw_task / corr_n - hw_mean * task_mean;
    float cov_hw_gen = sum_hw_gen / corr_n - hw_mean * gen_mean;
    float cov_task_gen = sum_task_gen / corr_n - task_mean * gen_mean;
    metrics->axis_corr_hw_task = cov_hw_task / (hw_std * task_std);
    metrics->axis_corr_hw_gen = cov_hw_gen / (hw_std * gen_std);
    metrics->axis_corr_task_gen = cov_task_gen / (task_std * gen_std);

    metrics->total_population = total_pop;

    DEVICE_FATAL_IF(occupied <= 0, "archive_topology_probe: occupied <= 0");
    float mean_density = sum_density / occupied;
    float variance = (sum_density_sq / occupied) - (mean_density * mean_density);
    DEVICE_FATAL_IF(variance < 0.0f, "archive_topology_probe: negative variance");
    metrics->density_variance = sqrtf(variance);
    metrics->hash_clustering_coefficient = 1.0f - metrics->novelty_gradient;
}

__device__ void diresa_evolution_probe(
    ComponentPool* pool,
    DIRESAEvolutionMetrics* metrics
) {
    int capacity = pool->capacity;
    DEVICE_FATAL_IF(capacity == 0, "diresa_evolution_probe: pool capacity is 0");

    float sum_recon_hw = 0.0f;
    float sum_recon_task = 0.0f;
    float sum_recon_gen = 0.0f;
    float sum_recon_total = 0.0f;
    float sum_drift = 0.0f;
    float sum_latent_util = 0.0f;
    float sum_compression = 0.0f;
    float sum_hw_corr = 0.0f;
    float sum_grad_mag = 0.0f;
    int alive_count = 0;

    for (int i = 0; i < capacity; i++) {
        if (!pool->alive_flags[i]) continue;
        PoolEntry* e = &pool->entries[i];

        sum_recon_hw += e->recon_loss_hw;
        sum_recon_task += e->recon_loss_task;
        sum_recon_gen += e->recon_loss_gen;
        sum_recon_total += e->recon_loss_total;
        sum_drift += e->behavioral_drift_rate;
        sum_latent_util += e->latent_utilization;
        sum_compression += e->compression_ratio;
        sum_hw_corr += e->hardware_feature_correlation;
        sum_grad_mag += e->gradient_magnitude;
        alive_count++;
    }

    DEVICE_FATAL_IF(alive_count == 0, "diresa_evolution_probe: no alive entries");
    metrics->recon_loss_hw = sum_recon_hw / alive_count;
    metrics->recon_loss_task = sum_recon_task / alive_count;
    metrics->recon_loss_gen = sum_recon_gen / alive_count;
    metrics->recon_loss_total = sum_recon_total / alive_count;
    metrics->behavioral_drift_rate = sum_drift / alive_count;
    metrics->latent_utilization = sum_latent_util / alive_count;
    metrics->compression_ratio = sum_compression / alive_count;
    metrics->hardware_feature_correlation = sum_hw_corr / alive_count;
    metrics->gradient_magnitude_avg = sum_grad_mag / alive_count;
    metrics->archive_injections = alive_count;
}

__device__ void task_performance_probe(
    ComponentPool* pool,
    TaskPerformanceMetrics* metrics
) {
    int capacity = pool->capacity;
    DEVICE_FATAL_IF(capacity == 0, "task_performance_probe: pool capacity is 0");

    float sum_accuracy = 0.0f;
    float sum_train_accuracy = 0.0f;
    float sum_test_accuracy = 0.0f;
    float sum_loss = 0.0f;
    float sum_stability = 0.0f;
    float sum_confidence = 0.0f;
    int sum_per_class_correct[NUM_CLASSES_MAX];
    int sum_per_class_total[NUM_CLASSES_MAX];
    for (int c = 0; c < NUM_CLASSES_MAX; c++) {
        sum_per_class_correct[c] = 0;
        sum_per_class_total[c] = 0;
    }
    int alive_count = 0;

    for (int i = 0; i < capacity; i++) {
        if (!pool->alive_flags[i]) continue;
        PoolEntry* e = &pool->entries[i];

        sum_accuracy += e->task_accuracy;
        sum_train_accuracy += e->train_accuracy;
        sum_test_accuracy += e->test_accuracy;
        sum_loss += e->task_loss;
        sum_stability += e->classification_stability;
        sum_confidence += e->avg_confidence;
        for (int c = 0; c < NUM_CLASSES_MAX; c++) {
            sum_per_class_correct[c] += e->per_class_correct[c];
            sum_per_class_total[c] += e->per_class_total[c];
        }
        alive_count++;
    }

    DEVICE_FATAL_IF(alive_count == 0, "task_performance_probe: no alive entries");
    metrics->accuracy = sum_accuracy / alive_count;
    metrics->train_accuracy = sum_train_accuracy / alive_count;
    metrics->test_accuracy = sum_test_accuracy / alive_count;
    metrics->loss = sum_loss / alive_count;
    metrics->classification_stability = sum_stability / alive_count;
    metrics->avg_confidence = sum_confidence / alive_count;
    metrics->correct_predictions = (int)(sum_accuracy * alive_count);
    metrics->total_predictions = alive_count;
    for (int c = 0; c < NUM_CLASSES_MAX; c++) {
        metrics->per_class_correct[c] = sum_per_class_correct[c];
        metrics->per_class_total[c] = sum_per_class_total[c];
    }
}

__device__ void population_metrics_probe(
    ComponentPool* pool,
    PopulationMetrics* metrics
) {
    int capacity = pool->capacity;
    DEVICE_FATAL_IF(capacity == 0, "population_metrics_probe: pool capacity is 0");

    float sum_accuracy = 0.0f;
    float sum_gen_gap = 0.0f;
    float sum_hw_eff = 0.0f;
    float sum_fitness = 0.0f;
    int alive_count = 0;

    for (int i = 0; i < capacity; i++) {
        if (!pool->alive_flags[i]) continue;
        PoolEntry* e = &pool->entries[i];

        sum_accuracy += e->task_accuracy;
        sum_gen_gap += e->generalization_gap;
        sum_hw_eff += e->hardware_efficiency;
        sum_fitness += e->fitness;
        alive_count++;
    }

    DEVICE_FATAL_IF(alive_count == 0, "population_metrics_probe: no alive entries");
    metrics->total_accuracy = sum_accuracy / alive_count;
    metrics->total_generalization_gap = sum_gen_gap / alive_count;
    metrics->total_hardware_efficiency = sum_hw_eff / alive_count;
    metrics->total_fitness = sum_fitness / alive_count;
}

__device__ void populate_audit_buffer(
    AuditBuffer* audit,
    int generation,
    float* logits,
    int* labels,
    float* batch_images,
    int batch_size,
    int num_classes,
    float* ca_concentration,
    int grid_size,
    float train_accuracy,
    float test_accuracy,
    TelemetryBuffer* telemetry,
    ComponentPool* pool,
    ChemicalField* chemical_field,
    MultiHeadCAState* ca_state,
    HardwareGeometry* hardware_geom
) {
    printf("V:audit_entry gen=%d logits=%p labels=%p imgs=%p batch=%d n_cls=%d ca=%p grid=%d\n",
           generation, (void*)logits, (void*)labels, (void*)batch_images,
           batch_size, num_classes, (void*)ca_concentration, grid_size);

    audit->ready = 0;
    __threadfence_system();

    audit->generation = generation;
    audit->batch_size = batch_size;
    audit->num_classes = num_classes;
    audit->grid_size = grid_size;
    printf("V:audit_cp1 batch=%d classes=%d grid=%d\n", batch_size, num_classes, grid_size);

    int samples_to_copy = (batch_size < AUDIT_SAMPLE_COUNT) ? batch_size : AUDIT_SAMPLE_COUNT;

    for (int s = 0; s < samples_to_copy; s++) {
        int pred = 0;
        float max_logit = logits[s * num_classes];
        for (int c = 1; c < num_classes; c++) {
            if (logits[s * num_classes + c] > max_logit) {
                max_logit = logits[s * num_classes + c];
                pred = c;
            }
        }

        audit->sample_labels[s] = labels[s];
        audit->sample_predictions[s] = pred;

        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            sum_exp += expf(logits[s * num_classes + c] - max_logit);
        }
        audit->sample_confidences[s] = 1.0f / sum_exp;

        for (int c = 0; c < num_classes; c++) {
            audit->sample_logits[s * NUM_CLASSES_MAX + c] = logits[s * num_classes + c];
        }
    }
    printf("V:audit_cp2 samples_copied=%d\n", samples_to_copy);

    int correct = 0;
    float total_loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        int pred = 0;
        float max_logit = logits[b * num_classes];
        for (int c = 1; c < num_classes; c++) {
            if (logits[b * num_classes + c] > max_logit) {
                max_logit = logits[b * num_classes + c];
                pred = c;
            }
        }
        if (pred == labels[b]) correct++;

        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            sum_exp += expf(logits[b * num_classes + c] - max_logit);
        }
        total_loss -= (logits[b * num_classes + labels[b]] - max_logit - logf(sum_exp));
    }

    audit->correct_count = correct;
    audit->accuracy = (float)correct / batch_size;
    audit->loss = total_loss / batch_size;
    audit->train_accuracy = train_accuracy;
    audit->test_accuracy = test_accuracy;
    audit->generalization_gap = fabsf(train_accuracy - test_accuracy);
    printf("V:audit_cp3 correct=%d/%d acc=%.4f\n", correct, batch_size, audit->accuracy);

    if (batch_images) {
        int img_size = grid_size * grid_size;
        for (int s = 0; s < samples_to_copy; s++) {
            for (int p = 0; p < img_size; p++) {
                float val = batch_images[s * img_size + p];
                val = (val < 0.0f) ? 0.0f : ((val > 1.0f) ? 1.0f : val);
                audit->sample_images[s * img_size + p] = (unsigned char)(val * 255.0f);
            }
        }
    }
    printf("V:audit_cp4 images_done\n");

    if (ca_concentration && pool && pool->entries[0].channels > 0) {
        int snap_grid = 64;
        int channels = pool->entries[0].channels;
        for (int y = 0; y < snap_grid && y < grid_size; y++) {
            for (int x = 0; x < snap_grid && x < grid_size; x++) {
                int cell_idx = y * grid_size + x;
                int dst_idx = y * snap_grid + x;
                audit->ca_snapshot[dst_idx] = ca_concentration[cell_idx * channels + 0];
            }
        }
    }
    printf("V:audit_cp5 ca_done\n");

    DEVICE_FATAL_IF(!pool, "populate_audit_buffer: pool is null");
    audit->pool_alive_count = pool->active_count.load(cuda::memory_order_relaxed);
    audit->pool_capacity = pool->capacity;

    for (int i = 0; i < pool->capacity && i < POOL_CAPACITY_MAX; i++) {
        PoolEntry* e = &pool->entries[i];
        audit->pool_entry_alive[i] = e->alive ? 1 : 0;
        audit->pool_entry_fitness[i] = e->fitness;
        audit->pool_entry_hunger[i] = e->hunger;
        audit->pool_entry_age[i] = e->age;
        audit->pool_entry_num_deltas[i] = e->num_deltas;
        audit->pool_entry_genome_hash[i] = e->genome_hash;
    }
    for (int i = pool->capacity; i < POOL_CAPACITY_MAX; i++) {
        audit->pool_entry_alive[i] = 0;
        audit->pool_entry_fitness[i] = 0.0f;
        audit->pool_entry_hunger[i] = 0.0f;
        audit->pool_entry_age[i] = 0;
        audit->pool_entry_num_deltas[i] = 0;
        audit->pool_entry_genome_hash[i] = 0;
    }
    printf("V:audit_cp6 pool_done\n");

    DEVICE_FATAL_IF(telemetry == nullptr, "populate_audit_buffer: telemetry is null");
    DEVICE_FATAL_IF(!telemetry->valid, "populate_audit_buffer: telemetry not valid");

    audit->archive_occupied_cells = telemetry->archive_topology.occupied_cells;
    audit->frontier_cells_gained = telemetry->archive_topology.frontier_cells_gained;
    audit->frontier_cells_lost = telemetry->archive_topology.frontier_cells_lost;
    audit->sparse_cell_count = telemetry->archive_topology.sparse_cell_count;
    audit->niche_entropy = telemetry->archive_topology.niche_entropy;
    audit->novelty_gradient = telemetry->archive_topology.novelty_gradient;

    audit->elite_fitness_best = telemetry->archive_topology.elite_fitness_best;
    audit->elite_fitness_mean = telemetry->archive_topology.elite_fitness_mean;
    audit->elite_fitness_delta = telemetry->archive_topology.elite_fitness_delta;
    audit->quality_floor = telemetry->archive_topology.quality_floor;
    audit->quality_mean = telemetry->archive_topology.quality_mean;
    audit->quality_range = telemetry->archive_topology.quality_range;

    audit->density_mean = telemetry->archive_topology.density_mean;
    audit->density_max = telemetry->archive_topology.density_max;
    audit->density_variance = telemetry->archive_topology.density_variance;

    audit->hw_axis_min = telemetry->archive_topology.hw_axis_min;
    audit->hw_axis_max = telemetry->archive_topology.hw_axis_max;
    audit->hw_axis_mean = telemetry->archive_topology.hw_axis_mean;
    audit->task_axis_min = telemetry->archive_topology.task_axis_min;
    audit->task_axis_max = telemetry->archive_topology.task_axis_max;
    audit->task_axis_mean = telemetry->archive_topology.task_axis_mean;
    audit->gen_axis_min = telemetry->archive_topology.gen_axis_min;
    audit->gen_axis_max = telemetry->archive_topology.gen_axis_max;
    audit->gen_axis_mean = telemetry->archive_topology.gen_axis_mean;

    audit->total_population = telemetry->archive_topology.total_population;
    audit->births_this_gen = telemetry->archive_topology.births_since_checkpoint;
    audit->deaths_this_gen = telemetry->archive_topology.deaths_since_checkpoint;

    audit->diresa_recon_loss_hw = telemetry->diresa_evolution.recon_loss_hw;
    audit->diresa_recon_loss_task = telemetry->diresa_evolution.recon_loss_task;
    audit->diresa_recon_loss_gen = telemetry->diresa_evolution.recon_loss_gen;
    audit->diresa_recon_loss_total = telemetry->diresa_evolution.recon_loss_total;
    audit->diresa_behavioral_drift = telemetry->diresa_evolution.behavioral_drift_rate;
    audit->diresa_latent_utilization = telemetry->diresa_evolution.latent_utilization;

    audit->genome_unique_hashes = telemetry->genome_complexity.unique_hashes;
    audit->genome_hash_entropy = telemetry->genome_complexity.hash_entropy;
    audit->genome_avg_deltas = telemetry->genome_complexity.avg_deltas_per_genome;

    for (int c = 0; c < NUM_CLASSES_MAX; c++) {
        audit->per_class_correct[c] = (float)telemetry->task_performance.per_class_correct[c];
        audit->per_class_total[c] = (float)telemetry->task_performance.per_class_total[c];
    }

    audit->axis_corr_hw_task = telemetry->archive_topology.axis_corr_hw_task;
    audit->axis_corr_hw_gen = telemetry->archive_topology.axis_corr_hw_gen;
    audit->axis_corr_task_gen = telemetry->archive_topology.axis_corr_task_gen;
    audit->hash_clustering_coefficient = telemetry->archive_topology.hash_clustering_coefficient;

    audit->memory_gpu_allocated = telemetry->memory_allocation.total_gpu_allocated;
    audit->memory_gpu_free = telemetry->memory_allocation.total_gpu_free;
    audit->memory_ca_state_size = telemetry->memory_allocation.ca_state_size;
    audit->memory_chemical_field_size = telemetry->memory_allocation.chemical_field_size;
    audit->memory_archive_size = telemetry->memory_allocation.archive_pools_size;

    DEVICE_FATAL_IF(pool == nullptr, "populate_audit_buffer: pool is null");
    DEVICE_FATAL_IF(!pool->entries[0].alive, "populate_audit_buffer: pool entry 0 not alive");
    PoolEntry* e0 = &pool->entries[0];
    audit->fitness_alpha = e0->fitness_task_exponent;
    audit->fitness_beta = e0->fitness_gen_exponent;
    audit->fitness_gamma = e0->fitness_rank_exponent;
    audit->fitness_delta = e0->fitness_efficiency_exponent;

    DEVICE_FATAL_IF(hardware_geom == nullptr, "populate_audit_buffer: hardware_geom is null");
    audit->hw_warp_divergence_entropy = hardware_geom->warp_divergence_entropy;
    audit->hw_warp_convergence_rate = hardware_geom->warp_convergence_rate;
    audit->hw_active_thread_fraction = hardware_geom->active_thread_fraction;
    audit->hw_memory_coalescing_efficiency = hardware_geom->memory_coalescing_efficiency;
    audit->hw_cache_line_utilization = hardware_geom->cache_line_utilization;
    audit->hw_tensor_core_usage = hardware_geom->tensor_core_usage;
    audit->hw_instruction_throughput = hardware_geom->instruction_throughput;
    audit->hw_occupancy_variance = hardware_geom->occupancy_variance;
    audit->hw_arithmetic_intensity = hardware_geom->arithmetic_intensity;
    audit->hw_memory_bandwidth_saturation = hardware_geom->memory_bandwidth_saturation;

    DEVICE_FATAL_IF(chemical_field == nullptr, "populate_audit_buffer: chemical_field is null");
    DEVICE_FATAL_IF(chemical_field->concentration == nullptr, "populate_audit_buffer: chemical_field concentration is null");
    DEVICE_FATAL_IF(chemical_field->gradient_x == nullptr, "populate_audit_buffer: chemical_field gradient_x is null");
    DEVICE_FATAL_IF(chemical_field->gradient_y == nullptr, "populate_audit_buffer: chemical_field gradient_y is null");
    DEVICE_FATAL_IF(chemical_field->sources == nullptr, "populate_audit_buffer: chemical_field sources is null");
    DEVICE_FATAL_IF(chemical_field->decay_factors == nullptr, "populate_audit_buffer: chemical_field decay_factors is null");
    int total_cells = grid_size * grid_size;
    float conc_sum = 0.0f, conc_max = 0.0f;
    float grad_mag_sum = 0.0f;
    float source_sum = 0.0f;
    float decay_sum = 0.0f;
    for (int i = 0; i < total_cells; i++) {
        float c = chemical_field->concentration[i];
        conc_sum += c;
        if (c > conc_max) conc_max = c;
        float gx = chemical_field->gradient_x[i];
        float gy = chemical_field->gradient_y[i];
        grad_mag_sum += sqrtf(gx * gx + gy * gy);
        source_sum += chemical_field->sources[i];
        decay_sum += chemical_field->decay_factors[i];
    }
    audit->chemical_concentration_mean = conc_sum / total_cells;
    audit->chemical_concentration_max = conc_max;
    audit->chemical_gradient_magnitude_mean = grad_mag_sum / total_cells;
    audit->chemical_source_activity = source_sum / total_cells;
    audit->chemical_decay_rate_mean = decay_sum / total_cells;

    DEVICE_FATAL_IF(ca_state == nullptr, "populate_audit_buffer: ca_state is null");
    DEVICE_FATAL_IF(pool->entries[0].channels <= 0, "populate_audit_buffer: pool entry 0 channels <= 0");
    DEVICE_FATAL_IF(ca_state->ca_concentration == nullptr, "populate_audit_buffer: ca_concentration is null");
    DEVICE_FATAL_IF(ca_state->affinity_reduced == nullptr, "populate_audit_buffer: affinity_reduced is null");
    DEVICE_FATAL_IF(ca_state->flow_field == nullptr, "populate_audit_buffer: flow_field is null");
    int channels = pool->entries[0].channels;
    float mass_total = 0.0f;
    float affinity_sum = 0.0f;
    float flow_mag_sum = 0.0f;
    for (int i = 0; i < total_cells * channels; i++) {
        mass_total += ca_state->ca_concentration[i];
    }
    for (int i = 0; i < total_cells; i++) {
        affinity_sum += ca_state->affinity_reduced[i];
    }
    for (int i = 0; i < total_cells; i++) {
        float fx = ca_state->flow_field[i * 2];
        float fy = ca_state->flow_field[i * 2 + 1];
        flow_mag_sum += sqrtf(fx * fx + fy * fy);
    }
    if (generation > 0) {
        float prev_mass = audit->flow_lenia_mass_total;
        DEVICE_FATAL_IF(prev_mass <= 0.0f, "populate_audit_buffer: prev mass invalid");
        audit->flow_lenia_mass_conservation_error = fabsf(mass_total - prev_mass) / prev_mass;
    }
    audit->flow_lenia_mass_total = mass_total;
    audit->flow_lenia_affinity_mean = affinity_sum / total_cells;
    audit->flow_lenia_flow_magnitude_mean = flow_mag_sum / total_cells;

    __threadfence_system();
    audit->ready = 1;
    __threadfence_system();

    printf("V:audit_done gen=%d correct=%d/%d acc=%.4f loss=%.4f train=%.4f test=%.4f pool=%d/%d\n",
           generation, audit->correct_count, batch_size, audit->accuracy, audit->loss,
           train_accuracy, test_accuracy, audit->pool_alive_count, audit->pool_capacity);
}

#endif
