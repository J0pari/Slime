// Core - must come first
#include "../slime/config/config.cu"
#include "../slime/core/organism.cu"

// Utilities and debug
#include "../slime/debug/device_trace.cu"
#include "../slime/debug/param_validator.cu"

// Memory management
#include "../slime/memory/archive.cu"
#include "../slime/memory/pool.cu"
#include "../slime/memory/tubes.cu"
#include "../slime/memory/parallel_compaction.cu"

// Core computations
#include "../slime/core/pseudopod.cu"
#include "../slime/core/pseudopod_tensor.cu"
#include "../slime/core/chemotaxis.cu"
#include "../slime/core/correlation_matrix.cu"

// Compute
#include "../slime/compute/tensor_core_ca.cu"
#include "../slime/compute/warp_ca.cu"

// Learning
#include "../slime/learning/autodiff.cu"
#include "../slime/learning/diresa.cu"

// Training
#include "../slime/training/training_types.cu"
#include "../slime/training/losses.cu"
#include "../slime/training/classification.cu"
#include "../slime/training/optimizer.cu"
#include "../slime/training/autodiff_integration.cu"
#include "../slime/training/gradient_fitness.cu"
#include "../slime/training/hybrid_lifecycle.cu"

// Data
#include "../slime/data/dataset_loader.cu"

// Lifecycle
#include "../slime/lifecycle/genealogy.cu"
#include "../slime/lifecycle/archive_sampling.cu"
#include "../slime/lifecycle/lifecycle_stages.cu"

// Metrics and diagnostics
#include "../slime/metrics/hardware_geometry.cu"
#include "../slime/diagnostics/telemetry_probes.cu"
#include "../slime/diagnostics/report_generator.cu"
#include "../slime/diagnostics/audit_writer.cu"

// Runtime - must come last
#include "../slime/runtime.cu"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <thread>
#include <chrono>
#include <atomic>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#define MKDIR(path) mkdir(path, 0755)
#endif

#define CUDA_ALLOC_CHECK(ptr, size, name) do { \
    cudaError_t _err = cudaMalloc(&(ptr), (size)); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "FATAL: %s cudaMalloc(%zu bytes) failed: %s\n", \
                (name), (size_t)(size), cudaGetErrorString(_err)); \
        char _fail_path[512]; \
        snprintf(_fail_path, sizeof(_fail_path), "%s/alloc_failure.csv", session_dir); \
        FILE* _ff = fopen(_fail_path, "w"); \
        if (_ff) { \
            size_t _free, _total; \
            cudaMemGetInfo(&_free, &_total); \
            fprintf(_ff, "failed_allocation,%s\n", (name)); \
            fprintf(_ff, "requested_bytes,%zu\n", (size_t)(size)); \
            fprintf(_ff, "requested_mb,%.2f\n", (size_t)(size) / (float)BYTES_PER_MB); \
            fprintf(_ff, "gpu_free_mb,%zu\n", _free / BYTES_PER_MB); \
            fprintf(_ff, "gpu_total_mb,%zu\n", _total / BYTES_PER_MB); \
            fprintf(_ff, "error_code,%d\n", (int)_err); \
            fprintf(_ff, "error_string,%s\n", cudaGetErrorString(_err)); \
            fclose(_ff); \
        } \
        return 1; \
    } \
} while(0)

int main() {
    printf("[H1] Entry\n"); fflush(stdout);
    cudaSetDevice(0);
    printf("[H2] Device set\n"); fflush(stdout);

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[H3] Mem: %zu MB free\n", free_mem / BYTES_PER_MB); fflush(stdout);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    printf("[H9] Device has %d SMs, max %d threads/block, max %d threads/SM\n",
        props.multiProcessorCount, props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor); fflush(stdout);

    cudaDeviceSetLimit(cudaLimitMallocHeapSize, DEVICE_MALLOC_HEAP_MB * BYTES_PER_MB);
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[H11] heap_limit=%zu MB, %zu MB free\n", DEVICE_MALLOC_HEAP_MB, free_mem / BYTES_PER_MB); fflush(stdout);

    cudaError_t err = cudaSuccess;

    Dataset* d_datasets[NUM_ACTIVE_DATASETS];
    Dataset* d_test_datasets[NUM_ACTIVE_DATASETS];
    for (int i = 0; i < NUM_ACTIVE_DATASETS; i++) {
        int dataset_id = HOST_ACTIVE_DATASET_IDS[i];
        err = load_dataset_from_registry(dataset_id, true, &d_datasets[i]);
        if (err != cudaSuccess) {
            printf("[H-ERR] Train dataset %d load failed: %s\n", dataset_id, cudaGetErrorString(err));
            return 1;
        }
        err = load_dataset_from_registry(dataset_id, false, &d_test_datasets[i]);
        if (err != cudaSuccess) {
            printf("[H-ERR] Test dataset %d load failed: %s\n", dataset_id, cudaGetErrorString(err));
            return 1;
        }
    }
    printf("[H9] %d train+test dataset pairs loaded\n", NUM_ACTIVE_DATASETS); fflush(stdout);

    Dataset** d_dataset_array;
    cudaError_t malloc_err = cudaMalloc(&d_dataset_array, sizeof(Dataset*) * NUM_ACTIVE_DATASETS);
    if (malloc_err != cudaSuccess) {
        fprintf(stderr, "FATAL [main]: d_dataset_array cudaMalloc failed: %s\n", cudaGetErrorString(malloc_err));
        return 1;
    }
    cudaMemcpy(d_dataset_array, d_datasets, sizeof(Dataset*) * NUM_ACTIVE_DATASETS, cudaMemcpyHostToDevice);

    Dataset** d_test_dataset_array;
    malloc_err = cudaMalloc(&d_test_dataset_array, sizeof(Dataset*) * NUM_ACTIVE_DATASETS);
    if (malloc_err != cudaSuccess) {
        fprintf(stderr, "FATAL [main]: d_test_dataset_array cudaMalloc failed: %s\n", cudaGetErrorString(malloc_err));
        return 1;
    }
    cudaMemcpy(d_test_dataset_array, d_test_datasets, sizeof(Dataset*) * NUM_ACTIVE_DATASETS, cudaMemcpyHostToDevice);

    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[H22] Mem after loading datasets: %zu MB free\n", free_mem / BYTES_PER_MB); fflush(stdout);


    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[H23] Mem before cudaFuncGetAttributes: %zu MB free\n", free_mem / BYTES_PER_MB); fflush(stdout);

    cudaFuncAttributes attr;
    err = cudaFuncGetAttributes(&attr, persistent_evolution_kernel);

    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[H24b] Mem after cudaFuncGetAttributes: %zu MB free (err=%d)\n", free_mem / BYTES_PER_MB, (int)err); fflush(stdout);
    printf("[H24c] Consumed by cudaFuncGetAttributes: %zu MB\n", (4398 - free_mem / BYTES_PER_MB)); fflush(stdout);

    printf("[H24d] Kernel attrs: localSize=%zu sharedSize=%zu constSize=%zu maxThreads=%d regs=%d\n",
        attr.localSizeBytes, attr.sharedSizeBytes, attr.constSizeBytes,
        attr.maxThreadsPerBlock, attr.numRegs); fflush(stdout);

    printf("[H25] Device props from earlier: SMs=%d maxThreads/SM=%d\n",
        props.multiProcessorCount, props.maxThreadsPerMultiProcessor); fflush(stdout);

    cudaMemGetInfo(&free_mem, &total_mem);

    cudaMemGetInfo(&free_mem, &total_mem);

    cudaMemGetInfo(&free_mem, &total_mem);

    cudaMemGetInfo(&free_mem, &total_mem);

    AuditBuffer* h_audit = nullptr;
    AuditBuffer* d_audit = nullptr;
    err = cudaHostAlloc(&h_audit, sizeof(AuditBuffer), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("[H-ERR] cudaHostAlloc audit buffer failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaHostGetDevicePointer(&d_audit, h_audit, 0);
    if (err != cudaSuccess) {
        printf("[H-ERR] cudaHostGetDevicePointer audit buffer failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    state_export_buffer_init_sentinel(h_audit);
    printf("[H-AUDIT] Mapped audit buffer: host=%p device=%p size=%zu\n",
           (void*)h_audit, (void*)d_audit, sizeof(AuditBuffer));

    time_t now = time(nullptr);
    struct tm* t = localtime(&now);
    char session_dir[256];
    snprintf(session_dir, sizeof(session_dir), "diagnostics/run_%04d%02d%02d_%02d%02d%02d",
             t->tm_year + 1900, t->tm_mon + 1, t->tm_mday,
             t->tm_hour, t->tm_min, t->tm_sec);

    MKDIR("diagnostics");
    MKDIR(session_dir);

    char samples_dir[256], ca_dir[256], pool_dir[256], chem_dir[256];
    snprintf(samples_dir, sizeof(samples_dir), "%s/samples", session_dir);
    snprintf(ca_dir, sizeof(ca_dir), "%s/ca_states", session_dir);
    snprintf(pool_dir, sizeof(pool_dir), "%s/pool_states", session_dir);
    snprintf(chem_dir, sizeof(chem_dir), "%s/chemical_fields", session_dir);
    MKDIR(samples_dir);
    MKDIR(ca_dir);
    MKDIR(pool_dir);
    MKDIR(chem_dir);

    char manifest_path[256];
    snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.csv", session_dir);
    printf("[H-SESSION] Output directory: %s\n", session_dir); fflush(stdout);

    OrganismPreallocatedBuffers* buffers;
    err = cudaMalloc(&buffers, sizeof(OrganismPreallocatedBuffers));
    if (err != cudaSuccess) {
        printf("[H-ERR] buffers cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    OrganismPreallocatedBuffers buffers_host;

    CUDA_ALLOC_CHECK(buffers_host.pool, sizeof(ComponentPool), "pool");
    CUDA_ALLOC_CHECK(buffers_host.pool_entries, sizeof(PoolEntry) * POOL_CAPACITY_MAX, "pool_entries");
    CUDA_ALLOC_CHECK(buffers_host.pool_alive_indices, sizeof(int) * POOL_CAPACITY_MAX, "pool_alive_indices");
    CUDA_ALLOC_CHECK(buffers_host.pool_alive_flags, sizeof(bool) * POOL_CAPACITY_MAX, "pool_alive_flags");
    CUDA_ALLOC_CHECK(buffers_host.pool_fitness_values, sizeof(float) * POOL_CAPACITY_MAX, "pool_fitness_values");
    CUDA_ALLOC_CHECK(buffers_host.pool_compaction_flags, sizeof(int) * POOL_CAPACITY_MAX, "pool_compaction_flags");
    CUDA_ALLOC_CHECK(buffers_host.pool_compaction_scan, sizeof(int) * POOL_CAPACITY_MAX, "pool_compaction_scan");
    CUDA_ALLOC_CHECK(buffers_host.pool_compaction_recursive_workspace, sizeof(int) * POOL_CAPACITY_MAX, "pool_compaction_recursive_workspace");
    CUDA_ALLOC_CHECK(buffers_host.archive, sizeof(GPUElite), "archive");
    CUDA_ALLOC_CHECK(buffers_host.archive_hash_table_keys, sizeof(uint64_t) * GENOME_HASH_TABLE_SIZE, "archive_hash_table_keys");
    CUDA_ALLOC_CHECK(buffers_host.archive_hash_table_values, sizeof(int) * GENOME_HASH_TABLE_SIZE, "archive_hash_table_values");
    CUDA_ALLOC_CHECK(buffers_host.voronoi_cells, sizeof(VoronoiCell) * POOL_CAPACITY_MAX, "voronoi_cells");
    CUDA_ALLOC_CHECK(buffers_host.behavioral_agents, sizeof(BehavioralState) * POOL_CAPACITY_MAX, "behavioral_agents");
    CUDA_ALLOC_CHECK(buffers_host.delta_indices_buffer, sizeof(uint16_t) * GENOME_SIZE * POOL_CAPACITY_MAX, "delta_indices_buffer");
    CUDA_ALLOC_CHECK(buffers_host.delta_values_buffer, sizeof(float) * GENOME_SIZE * POOL_CAPACITY_MAX, "delta_values_buffer");
    CUDA_ALLOC_CHECK(buffers_host.gradients_buffer, sizeof(float) * GENOME_SIZE * POOL_CAPACITY_MAX, "gradients_buffer");
    CUDA_ALLOC_CHECK(buffers_host.behavioral_hw_coords_buffer, sizeof(float) * POOL_CAPACITY_MAX * BEHAVIORAL_DIM_HW, "behavioral_hw_coords_buffer");
    CUDA_ALLOC_CHECK(buffers_host.behavioral_task_coords_buffer, sizeof(float) * POOL_CAPACITY_MAX * BEHAVIORAL_DIM_TASK, "behavioral_task_coords_buffer");
    CUDA_ALLOC_CHECK(buffers_host.behavioral_gen_coords_buffer, sizeof(float) * POOL_CAPACITY_MAX * BEHAVIORAL_DIM_GEN, "behavioral_gen_coords_buffer");
    CUDA_ALLOC_CHECK(buffers_host.archive_fitness, sizeof(float) * MAX_ARCHIVE_SIZE, "archive_fitness");
    CUDA_ALLOC_CHECK(buffers_host.archive_coherence, sizeof(float) * MAX_ARCHIVE_SIZE, "archive_coherence");
    CUDA_ALLOC_CHECK(buffers_host.archive_effective_rank, sizeof(float) * MAX_ARCHIVE_SIZE, "archive_effective_rank");
    CUDA_ALLOC_CHECK(buffers_host.archive_genome_hash, sizeof(uint64_t) * MAX_ARCHIVE_SIZE, "archive_genome_hash");
    CUDA_ALLOC_CHECK(buffers_host.archive_parent_ids, sizeof(uint32_t) * MAX_ARCHIVE_SIZE * PARENT_COUNT, "archive_parent_ids");
    CUDA_ALLOC_CHECK(buffers_host.archive_generation, sizeof(uint16_t) * MAX_ARCHIVE_SIZE, "archive_generation");
    CUDA_ALLOC_CHECK(buffers_host.archive_fitness_input_hash, sizeof(uint64_t) * MAX_ARCHIVE_SIZE, "archive_fitness_input_hash");
    CUDA_ALLOC_CHECK(buffers_host.archive_fitness_computed_at_generation, sizeof(int) * MAX_ARCHIVE_SIZE, "archive_fitness_computed_at_generation");
    CUDA_ALLOC_CHECK(buffers_host.archive_hw_coords, sizeof(float) * MAX_ARCHIVE_SIZE * BEHAVIORAL_DIM_HW, "archive_hw_coords");
    CUDA_ALLOC_CHECK(buffers_host.archive_task_coords, sizeof(float) * MAX_ARCHIVE_SIZE * BEHAVIORAL_DIM_TASK, "archive_task_coords");
    CUDA_ALLOC_CHECK(buffers_host.archive_gen_coords, sizeof(float) * MAX_ARCHIVE_SIZE * BEHAVIORAL_DIM_GEN, "archive_gen_coords");
    CUDA_ALLOC_CHECK(buffers_host.archive_latent_genome, sizeof(float) * MAX_ARCHIVE_SIZE * GENOME_LATENT_DIM_MAX, "archive_latent_genome");
    CUDA_ALLOC_CHECK(buffers_host.archive_hardware_features, sizeof(float) * MAX_ARCHIVE_SIZE * (WMMA_TILE_DIM - 1), "archive_hardware_features");
    CUDA_ALLOC_CHECK(buffers_host.archive_task_performance, sizeof(float) * MAX_ARCHIVE_SIZE, "archive_task_performance");
    CUDA_ALLOC_CHECK(buffers_host.archive_per_class_accuracy, sizeof(float) * MAX_ARCHIVE_SIZE * NUM_CLASSES_MAX, "archive_per_class_accuracy");
    CUDA_ALLOC_CHECK(buffers_host.hw_coords_pool, sizeof(float) * POOL_CAPACITY_MAX * BEHAVIORAL_DIM_HW, "hw_coords_pool");
    CUDA_ALLOC_CHECK(buffers_host.task_coords_pool, sizeof(float) * POOL_CAPACITY_MAX * BEHAVIORAL_DIM_TASK, "task_coords_pool");
    CUDA_ALLOC_CHECK(buffers_host.gen_coords_pool, sizeof(float) * POOL_CAPACITY_MAX * BEHAVIORAL_DIM_GEN, "gen_coords_pool");
    CUDA_ALLOC_CHECK(buffers_host.voronoi_hw_centroid_buffer, sizeof(float) * POOL_CAPACITY_MAX * BEHAVIORAL_DIM_HW, "voronoi_hw_centroid_buffer");
    CUDA_ALLOC_CHECK(buffers_host.voronoi_task_centroid_buffer, sizeof(float) * POOL_CAPACITY_MAX * BEHAVIORAL_DIM_TASK, "voronoi_task_centroid_buffer");
    CUDA_ALLOC_CHECK(buffers_host.voronoi_gen_centroid_buffer, sizeof(float) * POOL_CAPACITY_MAX * BEHAVIORAL_DIM_GEN, "voronoi_gen_centroid_buffer");
    CUDA_ALLOC_CHECK(buffers_host.telemetry, sizeof(TelemetryBuffer), "telemetry");
    CUDA_ALLOC_CHECK(buffers_host.ca_state_pool, sizeof(MultiHeadCAState) * POOL_CAPACITY_MAX, "ca_state_pool");
    CUDA_ALLOC_CHECK(buffers_host.chemical_field, sizeof(ChemicalField), "chemical_field");
    CUDA_ALLOC_CHECK(buffers_host.chemical_field_history, sizeof(TemporalTube), "chemical_field_history");
    CUDA_ALLOC_CHECK(buffers_host.chemical_field_history_entries, sizeof(MemoryEntry) * MAX_HISTORY_LENGTH, "chemical_field_history_entries");
    CUDA_ALLOC_CHECK(buffers_host.history_data_buffer, sizeof(float) * CA_FIELD_SIZE * MAX_HISTORY_LENGTH, "history_data_buffer");
    CUDA_ALLOC_CHECK(buffers_host.all_ca_weights, sizeof(half) * POOL_CAPACITY_MAX * CA_WEIGHTS_PER_ENTRY_STRIDE, "all_ca_weights");
    CUDA_ALLOC_CHECK(buffers_host.all_ca_state, sizeof(float) * POOL_CAPACITY_MAX * CA_STATE_STRIDE, "all_ca_state");
    CUDA_ALLOC_CHECK(buffers_host.all_chem_fields, sizeof(float) * CHEM_FIELD_COUNT * CA_FIELD_SIZE, "all_chem_fields");
    CUDA_ALLOC_CHECK(buffers_host.all_rd_fields, sizeof(float) * RD_FIELD_COUNT * CA_FIELD_SIZE, "all_rd_fields");
    CUDA_ALLOC_CHECK(buffers_host.shared_workspace, sizeof(float) * POOL_CAPACITY_MAX * POOL_CAPACITY_MAX, "shared_workspace");
    CUDA_ALLOC_CHECK(buffers_host.lifecycle_states, sizeof(LocalOrganismState<BLOCK_SIZE>) * ((POOL_CAPACITY_MAX + BLOCK_SIZE - 1) / BLOCK_SIZE), "lifecycle_states");
    
    CUDA_ALLOC_CHECK(buffers_host.diresa_genome_weights, sizeof(DIRESAWeights) * NUM_TEMPERING_REPLICAS_MAX, "diresa_genome_weights");
    CUDA_ALLOC_CHECK(buffers_host.diresa_genome_weight_pool, sizeof(float) * NUM_TEMPERING_REPLICAS_MAX * DIRESA_GENOME_STRIDE, "diresa_genome_weight_pool");
    
    CUDA_ALLOC_CHECK(buffers_host.per_entry_diresa_task_weights, sizeof(DIRESAWeights) * POOL_CAPACITY_MAX, "per_entry_diresa_task_weights");
    CUDA_ALLOC_CHECK(buffers_host.per_entry_diresa_hw_weights, sizeof(DIRESAWeights) * POOL_CAPACITY_MAX, "per_entry_diresa_hw_weights");
    CUDA_ALLOC_CHECK(buffers_host.per_entry_diresa_gen_weights, sizeof(DIRESAWeights) * POOL_CAPACITY_MAX, "per_entry_diresa_gen_weights");
    CUDA_ALLOC_CHECK(buffers_host.per_entry_diresa_task_weight_pool, sizeof(float) * POOL_CAPACITY_MAX * DIRESA_TASK_STRIDE_PER_ENTRY, "per_entry_diresa_task_weight_pool");
    CUDA_ALLOC_CHECK(buffers_host.per_entry_diresa_hw_weight_pool, sizeof(float) * POOL_CAPACITY_MAX * DIRESA_HW_STRIDE, "per_entry_diresa_hw_weight_pool");
    CUDA_ALLOC_CHECK(buffers_host.per_entry_diresa_gen_weight_pool, sizeof(float) * POOL_CAPACITY_MAX * DIRESA_GEN_STRIDE, "per_entry_diresa_gen_weight_pool");
    CUDA_ALLOC_CHECK(buffers_host.fp32_ca_workspace, sizeof(float) * POOL_CAPACITY_MAX * CA_FIELD_SIZE * (NUM_HEADS + 1) * HEAD_DIM, "fp32_ca_workspace");
    CUDA_ALLOC_CHECK(buffers_host.fp16_ca_workspace, sizeof(half) * POOL_CAPACITY_MAX * CA_FIELD_SIZE * (CHANNELS + HEAD_DIM), "fp16_ca_workspace");
    CUDA_ALLOC_CHECK(buffers_host.latent_genome_pool, sizeof(float) * MAX_ARCHIVE_SIZE * GENOME_LATENT_DIM_MAX, "latent_genome_pool");
    CUDA_ALLOC_CHECK(buffers_host.behavioral_field_pool, sizeof(float) * POOL_CAPACITY_MAX * CA_FIELD_SIZE * BEHAVIORAL_DIM_TOTAL, "behavioral_field_pool");
    CUDA_ALLOC_CHECK(buffers_host.behavioral_gradient_pool, sizeof(float) * POOL_CAPACITY_MAX * CA_FIELD_SIZE * BEHAVIORAL_DIM_TOTAL, "behavioral_gradient_pool");
    CUDA_ALLOC_CHECK(buffers_host.memory_data_pool, sizeof(float) * POOL_CAPACITY_MAX * (BEHAVIORAL_DIM_TOTAL + AGENT_SPATIAL_DIMS), "memory_data_pool");
    CUDA_ALLOC_CHECK(buffers_host.prediction_error_history, sizeof(float) * TELEMETRY_DETAILED, "prediction_error_history");
    CUDA_ALLOC_CHECK(buffers_host.trace_buffer, sizeof(TraceBuffer), "trace_buffer");
    CUDA_ALLOC_CHECK(buffers_host.trace_array, sizeof(ExecutionTrace) * TRACE_CAPACITY * POOL_CAPACITY_MAX, "trace_array");
    CUDA_ALLOC_CHECK(buffers_host.hardware_geom, sizeof(HardwareGeometry), "hardware_geom");
    CUDA_ALLOC_CHECK(buffers_host.delta_indices_pool, sizeof(uint16_t) * POOL_CAPACITY_MAX * MAX_DELTAS_PER_ENTRY, "delta_indices_pool");
    CUDA_ALLOC_CHECK(buffers_host.delta_values_pool, sizeof(float) * POOL_CAPACITY_MAX * MAX_DELTAS_PER_ENTRY, "delta_values_pool");
    CUDA_ALLOC_CHECK(buffers_host.delta_counts_pool, sizeof(uint16_t) * POOL_CAPACITY_MAX, "delta_counts_pool");
    CUDA_ALLOC_CHECK(buffers_host.memory_compaction_valid_flags, sizeof(int) * MAX_HISTORY_LENGTH, "memory_compaction_valid_flags");
    CUDA_ALLOC_CHECK(buffers_host.memory_compaction_scan, sizeof(int) * MAX_HISTORY_LENGTH, "memory_compaction_scan");
    CUDA_ALLOC_CHECK(buffers_host.memory_compaction_recursive_workspace, sizeof(int) * MAX_HISTORY_LENGTH, "memory_compaction_recursive_workspace");
    CUDA_ALLOC_CHECK(buffers_host.memory_compaction_buffer, sizeof(MemoryEntry) * MAX_HISTORY_LENGTH, "memory_compaction_buffer");
    CUDA_ALLOC_CHECK(buffers_host.fitness_rank_pool, sizeof(float) * POOL_CAPACITY_MAX, "fitness_rank_pool");
    CUDA_ALLOC_CHECK(buffers_host.fitness_coherence_pool, sizeof(float) * POOL_CAPACITY_MAX, "fitness_coherence_pool");
    CUDA_ALLOC_CHECK(buffers_host.fitness_history, sizeof(float) * 2 * POOL_CAPACITY_MAX, "fitness_history");
    CUDA_ALLOC_CHECK(buffers_host.coherence_history, sizeof(float) * 2 * POOL_CAPACITY_MAX, "coherence_history");
    CUDA_ALLOC_CHECK(buffers_host.effective_rank_history, sizeof(float) * POOL_CAPACITY_MAX, "effective_rank_history");
    CUDA_ALLOC_CHECK(buffers_host.ad_tape_entries_pool, sizeof(TapeEntry) * MAX_TAPE_SIZE, "ad_tape_entries_pool");
    CUDA_ALLOC_CHECK(buffers_host.ad_tape_values_pool, sizeof(float) * MAX_TAPE_VALUES, "ad_tape_values_pool");
    CUDA_ALLOC_CHECK(buffers_host.ad_tape_grads_pool, sizeof(float) * MAX_TAPE_VALUES, "ad_tape_grads_pool");
    CUDA_ALLOC_CHECK(buffers_host.ad_tape_levels_pool, sizeof(int) * MAX_TAPE_VALUES, "ad_tape_levels_pool");
    CUDA_ALLOC_CHECK(buffers_host.param_map, sizeof(CAParameterMap), "param_map");
    CUDA_ALLOC_CHECK(buffers_host.perception_activations_saved, sizeof(float) * WAVE_SIZE * BATCH_SIZE * NUM_HEADS * CA_FIELD_SIZE * HEAD_DIM, "perception_activations_saved");
    CUDA_ALLOC_CHECK(buffers_host.interaction_activations_saved, sizeof(float) * WAVE_SIZE * BATCH_SIZE * NUM_HEADS * CA_FIELD_SIZE * HEAD_DIM, "interaction_activations_saved");
    CUDA_ALLOC_CHECK(buffers_host.pre_gelu_values_saved, sizeof(float) * WAVE_SIZE * BATCH_SIZE * NUM_HEADS * CA_FIELD_SIZE * HEAD_DIM, "pre_gelu_values_saved");
    CUDA_ALLOC_CHECK(buffers_host.lifecycle_phase_counts, sizeof(int) * 8, "lifecycle_phase_counts");
    CUDA_ALLOC_CHECK(buffers_host.gradient_features_pool, sizeof(float) * POOL_CAPACITY_MAX * BATCH_SIZE * NUM_HEADS * CHANNELS, "gradient_features_pool");
    CUDA_ALLOC_CHECK(buffers_host.gradient_logits_pool, sizeof(float) * POOL_CAPACITY_MAX * BATCH_SIZE * NUM_CLASSES_MAX, "gradient_logits_pool");
    CUDA_ALLOC_CHECK(buffers_host.gradient_loss_pool, sizeof(float) * POOL_CAPACITY_MAX, "gradient_loss_pool");
    CUDA_ALLOC_CHECK(buffers_host.gradient_logit_grads_pool, sizeof(float) * POOL_CAPACITY_MAX * BATCH_SIZE * NUM_CLASSES_MAX, "gradient_logit_grads_pool");
    CUDA_ALLOC_CHECK(buffers_host.gradient_magnitudes_pool, sizeof(float) * POOL_CAPACITY_MAX * NUM_HEADS, "gradient_magnitudes_pool");
    CUDA_ALLOC_CHECK(buffers_host.pooling_weights_grad, sizeof(float) * NUM_HEADS * CHANNELS * POOL_CAPACITY_MAX, "pooling_weights_grad");
    CUDA_ALLOC_CHECK(buffers_host.fc_weights_grad, sizeof(float) * NUM_CLASSES_MAX * NUM_HEADS * CHANNELS * POOL_CAPACITY_MAX, "fc_weights_grad");
    CUDA_ALLOC_CHECK(buffers_host.fc_bias_grad, sizeof(float) * NUM_CLASSES_MAX * POOL_CAPACITY_MAX, "fc_bias_grad");
    CUDA_ALLOC_CHECK(buffers_host.features_grad, sizeof(float) * POOL_CAPACITY_MAX * BATCH_SIZE * NUM_HEADS * CHANNELS, "features_grad");
    constexpr size_t ADAM_CA_ENTRY_SIZE =
        (NUM_HEADS * CHANNELS * HEAD_DIM) +
        (NUM_HEADS * HEAD_DIM * HEAD_DIM) +
        (NUM_HEADS * HEAD_DIM * CHANNELS) +
        (NUM_CLASSES_MAX * NUM_HEADS * CHANNELS);
    CUDA_ALLOC_CHECK(buffers_host.adam_m_ca_pool, sizeof(float) * ADAM_CA_ENTRY_SIZE * POOL_CAPACITY_MAX, "adam_m_ca_pool");
    CUDA_ALLOC_CHECK(buffers_host.adam_v_ca_pool, sizeof(float) * ADAM_CA_ENTRY_SIZE * POOL_CAPACITY_MAX, "adam_v_ca_pool");
    CUDA_ALLOC_CHECK(buffers_host.adam_m_pooling, sizeof(float) * NUM_HEADS * CHANNELS * POOL_CAPACITY_MAX, "adam_m_pooling");
    CUDA_ALLOC_CHECK(buffers_host.adam_v_pooling, sizeof(float) * NUM_HEADS * CHANNELS * POOL_CAPACITY_MAX, "adam_v_pooling");
    CUDA_ALLOC_CHECK(buffers_host.adam_m_fc_weights, sizeof(float) * NUM_CLASSES_MAX * NUM_HEADS * CHANNELS * POOL_CAPACITY_MAX, "adam_m_fc_weights");
    CUDA_ALLOC_CHECK(buffers_host.adam_v_fc_weights, sizeof(float) * NUM_CLASSES_MAX * NUM_HEADS * CHANNELS * POOL_CAPACITY_MAX, "adam_v_fc_weights");
    CUDA_ALLOC_CHECK(buffers_host.adam_m_fc_bias, sizeof(float) * NUM_CLASSES_MAX * POOL_CAPACITY_MAX, "adam_m_fc_bias");
    CUDA_ALLOC_CHECK(buffers_host.adam_v_fc_bias, sizeof(float) * NUM_CLASSES_MAX * POOL_CAPACITY_MAX, "adam_v_fc_bias");
    CUDA_ALLOC_CHECK(buffers_host.batch_ca_states_pool, sizeof(float) * WAVE_SIZE * BATCH_SIZE * CA_FIELD_SIZE * CHANNELS, "batch_ca_states_pool");
    CUDA_ALLOC_CHECK(buffers_host.batch_ca_input_grads, sizeof(float) * WAVE_SIZE * BATCH_SIZE * CA_FIELD_SIZE * CHANNELS, "batch_ca_input_grads");
    CUDA_ALLOC_CHECK(buffers_host.batched_ca_output, sizeof(float) * WAVE_SIZE * BATCH_SIZE * NUM_HEADS * CA_FIELD_SIZE * CHANNELS, "batched_ca_output");
    CUDA_ALLOC_CHECK(buffers_host.batch_affinity_reduced, sizeof(float) * WAVE_SIZE * BATCH_SIZE * CA_FIELD_SIZE, "batch_affinity_reduced");
    CUDA_ALLOC_CHECK(buffers_host.batch_flow_field, sizeof(float) * WAVE_SIZE * BATCH_SIZE * CA_FIELD_SIZE * 2, "batch_flow_field");
    CUDA_ALLOC_CHECK(buffers_host.batch_reintegration_buffer, sizeof(float) * WAVE_SIZE * BATCH_SIZE * CA_FIELD_SIZE * CHANNELS, "batch_reintegration_buffer");
    CUDA_ALLOC_CHECK(buffers_host.batch_prev_concentration, sizeof(float) * WAVE_SIZE * BATCH_SIZE * CA_FIELD_SIZE * CHANNELS, "batch_prev_concentration");
    CUDA_ALLOC_CHECK(buffers_host.batch_labels_pool, sizeof(int) * BATCH_SIZE, "batch_labels_pool");
    CUDA_ALLOC_CHECK(buffers_host.batch_images_pool, sizeof(float) * BATCH_SIZE * CA_FIELD_SIZE * 3, "batch_images_pool");
    CUDA_ALLOC_CHECK(buffers_host.task_loss_pool, sizeof(float), "task_loss_pool");
    CUDA_ALLOC_CHECK(buffers_host.reg_loss_pool, sizeof(float), "reg_loss_pool");
    CUDA_ALLOC_CHECK(buffers_host.rank_loss_pool, sizeof(float), "rank_loss_pool");
    CUDA_ALLOC_CHECK(buffers_host.coherence_loss_pool, sizeof(float), "coherence_loss_pool");
    CUDA_ALLOC_CHECK(buffers_host.diversity_loss_pool, sizeof(float), "diversity_loss_pool");
    CUDA_ALLOC_CHECK(buffers_host.total_loss_pool, sizeof(float), "total_loss_pool");
    CUDA_ALLOC_CHECK(buffers_host.training_mode, sizeof(HybridTrainingMode), "training_mode");
    CUDA_ALLOC_CHECK(buffers_host.classifier, sizeof(ClassificationHead) * POOL_CAPACITY_MAX, "classifier");
    cudaMemset(buffers_host.classifier, 0, sizeof(ClassificationHead) * POOL_CAPACITY_MAX);
    constexpr int CLASSIFIER_INPUT_DIM_MAX = NUM_HEADS * CHANNELS;
    CUDA_ALLOC_CHECK(buffers_host.classifier_workspace, sizeof(float) * (CLASSIFIER_INPUT_DIM_MAX + (CLASSIFIER_INPUT_DIM_MAX * NUM_CLASSES_MAX) + NUM_CLASSES_MAX) * POOL_CAPACITY_MAX, "classifier_workspace");
    CUDA_ALLOC_CHECK(buffers_host.curriculum, sizeof(AdaptiveCurriculum), "curriculum");
    CUDA_ALLOC_CHECK(buffers_host.voronoi_occupancy_histogram, sizeof(float) * MAX_CELLS, "voronoi_occupancy_histogram");
    CUDA_ALLOC_CHECK(buffers_host.pool_task_accuracies, sizeof(float) * POOL_CAPACITY_MAX, "pool_task_accuracies");
    CUDA_ALLOC_CHECK(buffers_host.organism, sizeof(Organism), "organism");
    CUDA_ALLOC_CHECK(buffers_host.reduction_workspace, sizeof(float) * ((CA_FIELD_SIZE * CHANNELS + BLOCK_SIZE - 1) / BLOCK_SIZE), "reduction_workspace");
    CUDA_ALLOC_CHECK(buffers_host.rng_states, sizeof(curandState) * POOL_CAPACITY_MAX, "rng_states");
    CUDA_ALLOC_CHECK(buffers_host.phase_barrier_counter, sizeof(int), "phase_barrier_counter");
    CUDA_ALLOC_CHECK(buffers_host.phase_barrier_generation, sizeof(int), "phase_barrier_generation");
    CUDA_ALLOC_CHECK(buffers_host.memory_params, sizeof(MemoryUpdateParams), "memory_params");
    CUDA_ALLOC_CHECK(buffers_host.organism_workspace_genomes, sizeof(float) * BLOCK_SIZE * SPAWN_WS_COUNT * GENOME_SIZE, "organism_workspace_genomes");
    CUDA_ALLOC_CHECK(buffers_host.behavioral_features_buffer, sizeof(float) * POOL_CAPACITY_MAX * BEHAVIORAL_DIM_TOTAL, "behavioral_features_buffer");
    CUDA_ALLOC_CHECK(buffers_host.behavioral_embedding_weights, sizeof(float) * BEHAVIORAL_DIM_TOTAL * BEHAVIORAL_DIM_TOTAL, "behavioral_embedding_weights");
    CUDA_ALLOC_CHECK(buffers_host.behavioral_reconstruction_error, sizeof(float), "behavioral_reconstruction_error");
    CUDA_ALLOC_CHECK(buffers_host.grad_concentration_buffer, sizeof(float) * CA_FIELD_SIZE, "grad_concentration_buffer");
    CUDA_ALLOC_CHECK(buffers_host.ca_output_grad_buffer, sizeof(float) * WAVE_SIZE * BATCH_SIZE * NUM_HEADS * CA_FIELD_SIZE * CHANNELS, "ca_output_grad_buffer");
    CUDA_ALLOC_CHECK(buffers_host.dL_dperception_buffer, sizeof(float) * WAVE_SIZE * BATCH_SIZE * NUM_HEADS * CA_FIELD_SIZE * HEAD_DIM, "dL_dperception_buffer");
    CUDA_ALLOC_CHECK(buffers_host.dL_dinteraction_buffer, sizeof(float) * WAVE_SIZE * BATCH_SIZE * NUM_HEADS * CA_FIELD_SIZE * HEAD_DIM, "dL_dinteraction_buffer");
    CUDA_ALLOC_CHECK(buffers_host.component_workspace_genomes_buffer, sizeof(float) * GENOME_SIZE * 2, "component_workspace_genomes_buffer");
    CUDA_ALLOC_CHECK(buffers_host.behavioral_workspace_genomes_buffer, sizeof(float) * GENOME_SIZE * 2, "behavioral_workspace_genomes_buffer");
    CUDA_ALLOC_CHECK(buffers_host.inherit_child_indices, sizeof(int) * POOL_CAPACITY_MAX, "inherit_child_indices");
    CUDA_ALLOC_CHECK(buffers_host.inherit_parent_indices, sizeof(int) * POOL_CAPACITY_MAX, "inherit_parent_indices");
    CUDA_ALLOC_CHECK(buffers_host.num_pending_inherits, sizeof(int), "num_pending_inherits");
    
    constexpr size_t BACKWARD_WS_TOTAL_SIZE = BACKWARD_WS_FP16_A_SIZE + BACKWARD_WS_FP16_B_SIZE +
        BACKWARD_WS_DW_SIZE + BACKWARD_WS_DI_SIZE + BACKWARD_WS_W_T_SIZE +
        BACKWARD_WS_IM2COL_SIZE + BACKWARD_WS_DPREGELU_SIZE;
    CUDA_ALLOC_CHECK(buffers_host.backward_workspace, BACKWARD_WS_TOTAL_SIZE, "backward_workspace");

    // Set self-referential pointers before copying to device
    buffers_host.buffers = buffers;
    buffers_host.organism = buffers;

    cudaMemcpy(buffers, &buffers_host, sizeof(OrganismPreallocatedBuffers), cudaMemcpyHostToDevice);

    // Use cooperative launch for grid-wide synchronization (cg::this_grid().sync())
    unsigned int seed = (unsigned int)time(nullptr);
    void* kernelArgs[] = {
        &seed,
        &d_dataset_array,
        &d_test_dataset_array,
        &buffers,
        &d_audit
    };

    // Check if cooperative launch is supported
    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, device);
    if (!supportsCoopLaunch) {
        printf("[H-ERR] Device does not support cooperative launch\n");
        return 1;
    }
    printf("[H29] Cooperative launch supported\n"); fflush(stdout);

    // Calculate max blocks for cooperative launch
    int numBlocksPerSm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, persistent_evolution_kernel, BLOCK_SIZE, 0);
    int maxCoopBlocks = numBlocksPerSm * props.multiProcessorCount;
    int numBlocks = (POOL_CAPACITY_MAX < maxCoopBlocks) ? POOL_CAPACITY_MAX : maxCoopBlocks;
    printf("[H29b] Cooperative launch: %d blocks (max %d), %d threads/block\n",
           numBlocks, maxCoopBlocks, BLOCK_SIZE); fflush(stdout);

    size_t shared_mem_bytes = BLOCK_SIZE * sizeof(float);
    err = cudaLaunchCooperativeKernel(
        (void*)persistent_evolution_kernel,
        dim3(numBlocks),
        dim3(BLOCK_SIZE),
        kernelArgs,
        shared_mem_bytes,
        0   // stream
    );
    printf("[H30] cudaLaunchCooperativeKernel=%d (%s)\n", (int)err, cudaGetErrorString(err)); fflush(stdout);
    if (err != cudaSuccess) {
        printf("[H-ERR] Cooperative launch failed\n");
        return 1;
    }

    printf("[H31] Kernel launched, polling audit buffer\n"); fflush(stdout);

    char state_json_path[256];
    snprintf(state_json_path, sizeof(state_json_path), "%s/state.jsonl", session_dir);
    FILE* state_json_file = fopen(state_json_path, "w");
    if (!state_json_file) {
        fprintf(stderr, "FATAL: could not open state.jsonl for writing\n");
        return 1;
    }

    StateExportBufferReader reader;
    reader.init(h_audit);

    auto start_time = std::chrono::steady_clock::now();
    int last_gen = -1;

    while (true) {
        TelemetryAuditEntry entry;
        RecordHeader hdr;

        while (reader.read_next(&entry, &hdr)) {
            auto now_time = std::chrono::steady_clock::now();
            double elapsed_sec = std::chrono::duration<double>(now_time - start_time).count();

            if (!state_export_field_valid(&entry, AUDIT_MASK_GENERATION)) {
                fprintf(stderr, "E_NOGEN seq=%llu\n", (unsigned long long)hdr.sequence_number);
                continue;
            }

            int gen = entry.generation;

            if (write_state_json(state_json_file, elapsed_sec, &entry) != 0) {
                fprintf(stderr, "E_JSON gen=%d\n", gen);
            }

            if (gen != last_gen) {
                last_gen = gen;

                printf("[AUDIT] gen=%d batch=%d acc=%.4f loss=%.4f correct=%d/%d seq=%llu (%.1fs)\n",
                       gen, entry.batch_size, entry.accuracy, entry.loss,
                       entry.correct_count, entry.batch_size,
                       (unsigned long long)hdr.sequence_number, elapsed_sec);

                if (write_sample_images(session_dir, gen, &entry) != 0) {
                    fprintf(stderr, "E_SAMPLES gen=%d\n", gen);
                }

                char ca_path[256];
                snprintf(ca_path, sizeof(ca_path), "%s/ca_states/gen%04d.pgm", session_dir, gen);
                if (write_ca_snapshot(ca_path, gen, &entry) != 0) {
                    fprintf(stderr, "E_CA gen=%d\n", gen);
                }

                char predictions_path[256];
                snprintf(predictions_path, sizeof(predictions_path), "%s/predictions_gen%04d.csv", session_dir, gen);
                if (write_predictions_csv(predictions_path, gen, &entry) != 0) {
                    fprintf(stderr, "E_PRED gen=%d\n", gen);
                }

                if (write_generation_summary(session_dir, gen, &entry) != 0) {
                    fprintf(stderr, "E_SUMMARY gen=%d\n", gen);
                }

                if (write_pool_state(session_dir, gen, &entry) != 0) {
                    fprintf(stderr, "E_POOL gen=%d\n", gen);
                }

                append_to_manifest(manifest_path, predictions_path, ca_path, elapsed_sec);
            }
        }

        uint64_t dropped = reader.get_dropped_count();
        uint64_t corrupted = reader.get_corrupted_count();
        if (dropped > 0 || corrupted > 0) {
            fprintf(stderr, "E_RING dropped=%llu corrupted=%llu\n",
                    (unsigned long long)dropped, (unsigned long long)corrupted);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    cudaFreeHost(h_audit);
    return 0;
}
