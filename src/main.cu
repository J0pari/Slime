#include "../slime/runtime.cu"
#include "../slime/diagnostics/report_generator.cu"
#include "../slime/diagnostics/audit_writer.cu"
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
            fprintf(_ff, "requested_mb,%.2f\n", (size_t)(size) / (1024.0 * 1024.0)); \
            fprintf(_ff, "gpu_free_mb,%zu\n", _free / (1024 * 1024)); \
            fprintf(_ff, "gpu_total_mb,%zu\n", _total / (1024 * 1024)); \
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

    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[H4] Mem before CDP limits: %zu MB free\n", free_mem / BYTES_PER_MB); fflush(stdout);

    printf("[H5] NOT setting sync_depth yet - will set just before kernel launch\n"); fflush(stdout);
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[H6] Mem with default CDP limits: %zu MB free\n", free_mem / BYTES_PER_MB); fflush(stdout);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    printf("[H9] Device has %d SMs, max %d threads/block, max %d threads/SM\n",
        props.multiProcessorCount, props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor); fflush(stdout);

    size_t max_possible_threads = props.multiProcessorCount * props.maxThreadsPerMultiProcessor;
    size_t predicted_stack_alloc = max_possible_threads * CDP_STACK_SIZE;
    printf("[H10] Predicted stack allocation: %zu threads * %d bytes = %zu MB\n",
        max_possible_threads, CDP_STACK_SIZE, predicted_stack_alloc / BYTES_PER_MB); fflush(stdout);

    printf("[H11] Setting device heap limit to %zu MB\n", DEVICE_MALLOC_HEAP_MB); fflush(stdout);
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, DEVICE_MALLOC_HEAP_MB * BYTES_PER_MB);
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[H11b] Mem after heap limit: %zu MB free\n", free_mem / BYTES_PER_MB); fflush(stdout);

    cudaError_t err = cudaSuccess;

    Dataset* d_datasets[NUM_ACTIVE_DATASETS];
    printf("[H8] Loading %d active datasets for curriculum...\n", NUM_ACTIVE_DATASETS); fflush(stdout);

    Dataset* d_test_datasets[NUM_ACTIVE_DATASETS];
    for (int i = 0; i < NUM_ACTIVE_DATASETS; i++) {
        int dataset_id = HOST_ACTIVE_DATASET_IDS[i];
        printf("[H8.%d] Loading train dataset %d...\n", i, dataset_id); fflush(stdout);
        err = load_dataset_from_registry(dataset_id, true, &d_datasets[i]);
        if (err != cudaSuccess) {
            printf("[H-ERR] Train dataset %d load failed: %s\n", dataset_id, cudaGetErrorString(err));
            return 1;
        }
        printf("[H8.%d] Loading test dataset %d...\n", i, dataset_id); fflush(stdout);
        err = load_dataset_from_registry(dataset_id, false, &d_test_datasets[i]);
        if (err != cudaSuccess) {
            printf("[H-ERR] Test dataset %d load failed: %s\n", dataset_id, cudaGetErrorString(err));
            return 1;
        }
    }
    printf("[H9] All %d train+test dataset pairs loaded\n", NUM_ACTIVE_DATASETS); fflush(stdout);

    // Allocate device array of dataset pointers (train)
    Dataset** d_dataset_array;
    cudaError_t malloc_err = cudaMalloc(&d_dataset_array, sizeof(Dataset*) * NUM_ACTIVE_DATASETS);
    if (malloc_err != cudaSuccess) {
        fprintf(stderr, "FATAL [main]: d_dataset_array cudaMalloc failed: %s\n", cudaGetErrorString(malloc_err));
        return 1;
    }
    cudaMemcpy(d_dataset_array, d_datasets, sizeof(Dataset*) * NUM_ACTIVE_DATASETS, cudaMemcpyHostToDevice);

    // Allocate device array of dataset pointers (test)
    Dataset** d_test_dataset_array;
    malloc_err = cudaMalloc(&d_test_dataset_array, sizeof(Dataset*) * NUM_ACTIVE_DATASETS);
    if (malloc_err != cudaSuccess) {
        fprintf(stderr, "FATAL [main]: d_test_dataset_array cudaMalloc failed: %s\n", cudaGetErrorString(malloc_err));
        return 1;
    }
    cudaMemcpy(d_test_dataset_array, d_test_datasets, sizeof(Dataset*) * NUM_ACTIVE_DATASETS, cudaMemcpyHostToDevice);

    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[H22] Mem after loading datasets: %zu MB free\n", free_mem / BYTES_PER_MB); fflush(stdout);

    size_t heap_limit, stack_limit, sync_limit, pending_limit;

    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[H23a] Mem before cudaDeviceGetLimit calls: %zu MB free\n", free_mem / BYTES_PER_MB); fflush(stdout);

    cudaDeviceGetLimit(&heap_limit, cudaLimitMallocHeapSize);
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[H23b] Mem after get heap_limit: %zu MB free\n", free_mem / BYTES_PER_MB); fflush(stdout);

    cudaDeviceGetLimit(&stack_limit, cudaLimitStackSize);
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[H23c] Mem after get stack_limit: %zu MB free\n", free_mem / BYTES_PER_MB); fflush(stdout);

    cudaDeviceGetLimit(&sync_limit, cudaLimitDevRuntimeSyncDepth);
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[H23d] Mem after get sync_limit: %zu MB free\n", free_mem / BYTES_PER_MB); fflush(stdout);

    cudaDeviceGetLimit(&pending_limit, cudaLimitDevRuntimePendingLaunchCount);
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[H23e] Mem after get pending_limit: %zu MB free\n", free_mem / BYTES_PER_MB); fflush(stdout);

    printf("[H23f] CDP Limits: heap=%zuMB stack=%zu sync_depth=%zu pending=%zu\n",
        heap_limit / BYTES_PER_MB, stack_limit, sync_limit, pending_limit); fflush(stdout);

    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[H24a] Mem before cudaFuncGetAttributes: %zu MB free\n", free_mem / BYTES_PER_MB); fflush(stdout);

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
    printf("[H26] Mem before setting sync_depth: %zu MB free\n", free_mem / BYTES_PER_MB); fflush(stdout);

    printf("[H28] Setting sync_depth=%d\n", CDP_SYNC_DEPTH); fflush(stdout);
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, CDP_SYNC_DEPTH);
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[H28b] Mem after sync_depth: %zu MB free\n", free_mem / BYTES_PER_MB); fflush(stdout);

    constexpr size_t CDP_PENDING_LAUNCH_COUNT = 32768;
    printf("[H28b2] Setting pending_launch_count=%zu\n", CDP_PENDING_LAUNCH_COUNT); fflush(stdout);
    cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, CDP_PENDING_LAUNCH_COUNT);
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[H28b3] Mem after pending_launch_count: %zu MB free\n", free_mem / BYTES_PER_MB); fflush(stdout);

    printf("[H28c] NOT setting global stack limit - let CUDA allocate per-kernel dynamically\n"); fflush(stdout);
    printf("[H28d] Each kernel will use only the stack it needs (default 1024, up to %d for init_organism)\n", CDP_STACK_SIZE); fflush(stdout);

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
    memset(h_audit, 0, sizeof(AuditBuffer));
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
    CUDA_ALLOC_CHECK(buffers_host.archive_hash_table_keys, sizeof(uint64_t) * 16384, "archive_hash_table_keys");
    CUDA_ALLOC_CHECK(buffers_host.archive_hash_table_values, sizeof(int) * 16384, "archive_hash_table_values");
    CUDA_ALLOC_CHECK(buffers_host.voronoi_cells, sizeof(VoronoiCell) * POOL_CAPACITY_MAX, "voronoi_cells");
    CUDA_ALLOC_CHECK(buffers_host.behavioral_agents, sizeof(BehavioralState) * POOL_CAPACITY_MAX, "behavioral_agents");
    CUDA_ALLOC_CHECK(buffers_host.delta_indices_buffer, sizeof(uint16_t) * GENOME_SIZE * POOL_CAPACITY_MAX, "delta_indices_buffer");
    CUDA_ALLOC_CHECK(buffers_host.delta_values_buffer, sizeof(float) * GENOME_SIZE * POOL_CAPACITY_MAX, "delta_values_buffer");
    CUDA_ALLOC_CHECK(buffers_host.gradients_buffer, sizeof(float) * GENOME_SIZE * POOL_CAPACITY_MAX, "gradients_buffer");
    CUDA_ALLOC_CHECK(buffers_host.behavioral_hw_coords_buffer, sizeof(float) * POOL_CAPACITY_MAX * BEHAVIORAL_DIM_HW_MAX, "behavioral_hw_coords_buffer");
    CUDA_ALLOC_CHECK(buffers_host.behavioral_task_coords_buffer, sizeof(float) * POOL_CAPACITY_MAX * BEHAVIORAL_DIM_TASK_MAX, "behavioral_task_coords_buffer");
    CUDA_ALLOC_CHECK(buffers_host.behavioral_gen_coords_buffer, sizeof(float) * POOL_CAPACITY_MAX * BEHAVIORAL_DIM_GEN_MAX, "behavioral_gen_coords_buffer");
    CUDA_ALLOC_CHECK(buffers_host.archive_fitness, sizeof(float) * MAX_ARCHIVE_SIZE, "archive_fitness");
    CUDA_ALLOC_CHECK(buffers_host.archive_coherence, sizeof(float) * MAX_ARCHIVE_SIZE, "archive_coherence");
    CUDA_ALLOC_CHECK(buffers_host.archive_effective_rank, sizeof(float) * MAX_ARCHIVE_SIZE, "archive_effective_rank");
    CUDA_ALLOC_CHECK(buffers_host.archive_genome_hash, sizeof(uint64_t) * MAX_ARCHIVE_SIZE, "archive_genome_hash");
    CUDA_ALLOC_CHECK(buffers_host.archive_parent_ids, sizeof(uint32_t) * MAX_ARCHIVE_SIZE * PARENT_COUNT, "archive_parent_ids");
    CUDA_ALLOC_CHECK(buffers_host.archive_generation, sizeof(uint16_t) * MAX_ARCHIVE_SIZE, "archive_generation");
    CUDA_ALLOC_CHECK(buffers_host.archive_hw_coords, sizeof(float) * MAX_ARCHIVE_SIZE * BEHAVIORAL_DIM_HW_MAX, "archive_hw_coords");
    CUDA_ALLOC_CHECK(buffers_host.archive_task_coords, sizeof(float) * MAX_ARCHIVE_SIZE * BEHAVIORAL_DIM_TASK_MAX, "archive_task_coords");
    CUDA_ALLOC_CHECK(buffers_host.archive_gen_coords, sizeof(float) * MAX_ARCHIVE_SIZE * BEHAVIORAL_DIM_GEN_MAX, "archive_gen_coords");
    CUDA_ALLOC_CHECK(buffers_host.archive_latent_genome, sizeof(float) * MAX_ARCHIVE_SIZE * GENOME_LATENT_DIM_MAX, "archive_latent_genome");
    CUDA_ALLOC_CHECK(buffers_host.archive_hardware_features, sizeof(float) * MAX_ARCHIVE_SIZE * (WMMA_TILE_DIM - 1), "archive_hardware_features");
    CUDA_ALLOC_CHECK(buffers_host.archive_task_performance, sizeof(float) * MAX_ARCHIVE_SIZE, "archive_task_performance");
    CUDA_ALLOC_CHECK(buffers_host.archive_per_class_accuracy, sizeof(float) * MAX_ARCHIVE_SIZE * NUM_CLASSES_MAX, "archive_per_class_accuracy");
    CUDA_ALLOC_CHECK(buffers_host.hw_coords_pool, sizeof(float) * POOL_CAPACITY_MAX * BEHAVIORAL_DIM_HW_MAX, "hw_coords_pool");
    CUDA_ALLOC_CHECK(buffers_host.task_coords_pool, sizeof(float) * POOL_CAPACITY_MAX * BEHAVIORAL_DIM_TASK_MAX, "task_coords_pool");
    CUDA_ALLOC_CHECK(buffers_host.gen_coords_pool, sizeof(float) * POOL_CAPACITY_MAX * BEHAVIORAL_DIM_GEN_MAX, "gen_coords_pool");
    CUDA_ALLOC_CHECK(buffers_host.voronoi_hw_centroid_buffer, sizeof(float) * POOL_CAPACITY_MAX * BEHAVIORAL_DIM_HW_MAX, "voronoi_hw_centroid_buffer");
    CUDA_ALLOC_CHECK(buffers_host.voronoi_task_centroid_buffer, sizeof(float) * POOL_CAPACITY_MAX * BEHAVIORAL_DIM_TASK_MAX, "voronoi_task_centroid_buffer");
    CUDA_ALLOC_CHECK(buffers_host.voronoi_gen_centroid_buffer, sizeof(float) * POOL_CAPACITY_MAX * BEHAVIORAL_DIM_GEN_MAX, "voronoi_gen_centroid_buffer");
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
    CUDA_ALLOC_CHECK(buffers_host.diresa_hw_weights, sizeof(DIRESAWeights) * NUM_TEMPERING_REPLICAS_MAX, "diresa_hw_weights");
    CUDA_ALLOC_CHECK(buffers_host.diresa_task_weights, sizeof(DIRESAWeights) * NUM_TEMPERING_REPLICAS_MAX, "diresa_task_weights");
    CUDA_ALLOC_CHECK(buffers_host.diresa_gen_weights, sizeof(DIRESAWeights) * NUM_TEMPERING_REPLICAS_MAX, "diresa_gen_weights");
    CUDA_ALLOC_CHECK(buffers_host.diresa_genome_weights, sizeof(DIRESAWeights) * NUM_TEMPERING_REPLICAS_MAX, "diresa_genome_weights");
    CUDA_ALLOC_CHECK(buffers_host.diresa_hw_weight_pool, sizeof(float) * NUM_TEMPERING_REPLICAS_MAX * DIRESA_HW_STRIDE, "diresa_hw_weight_pool");
    CUDA_ALLOC_CHECK(buffers_host.diresa_task_weight_pool, sizeof(float) * NUM_TEMPERING_REPLICAS_MAX * DIRESA_TASK_STRIDE, "diresa_task_weight_pool");
    CUDA_ALLOC_CHECK(buffers_host.diresa_gen_weight_pool, sizeof(float) * NUM_TEMPERING_REPLICAS_MAX * DIRESA_GEN_STRIDE, "diresa_gen_weight_pool");
    CUDA_ALLOC_CHECK(buffers_host.diresa_genome_weight_pool, sizeof(float) * NUM_TEMPERING_REPLICAS_MAX * DIRESA_GENOME_STRIDE, "diresa_genome_weight_pool");
    CUDA_ALLOC_CHECK(buffers_host.fp32_ca_workspace, sizeof(float) * POOL_CAPACITY_MAX * CA_FIELD_SIZE * (NUM_HEADS_MAX + 1) * HEAD_DIM_MAX, "fp32_ca_workspace");
    CUDA_ALLOC_CHECK(buffers_host.fp16_ca_workspace, sizeof(half) * POOL_CAPACITY_MAX * CA_FIELD_SIZE * (CHANNELS_MAX + HEAD_DIM_MAX), "fp16_ca_workspace");
    CUDA_ALLOC_CHECK(buffers_host.latent_genome_pool, sizeof(float) * MAX_ARCHIVE_SIZE * GENOME_LATENT_DIM_MAX, "latent_genome_pool");
    CUDA_ALLOC_CHECK(buffers_host.behavioral_field_pool, sizeof(float) * POOL_CAPACITY_MAX * CA_FIELD_SIZE * BEHAVIORAL_DIM_MAX, "behavioral_field_pool");
    CUDA_ALLOC_CHECK(buffers_host.behavioral_gradient_pool, sizeof(float) * POOL_CAPACITY_MAX * CA_FIELD_SIZE * BEHAVIORAL_DIM_MAX, "behavioral_gradient_pool");
    CUDA_ALLOC_CHECK(buffers_host.memory_data_pool, sizeof(float) * POOL_CAPACITY_MAX * (BEHAVIORAL_DIM_MAX + AGENT_SPATIAL_DIMS), "memory_data_pool");
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
    CUDA_ALLOC_CHECK(buffers_host.perception_activations_saved, sizeof(float) * BATCH_SIZE_MAX * NUM_HEADS_MAX * CA_FIELD_SIZE * HEAD_DIM_MAX, "perception_activations_saved");
    CUDA_ALLOC_CHECK(buffers_host.interaction_activations_saved, sizeof(float) * BATCH_SIZE_MAX * NUM_HEADS_MAX * CA_FIELD_SIZE * HEAD_DIM_MAX, "interaction_activations_saved");
    CUDA_ALLOC_CHECK(buffers_host.pre_gelu_values_saved, sizeof(float) * BATCH_SIZE_MAX * NUM_HEADS_MAX * CA_FIELD_SIZE * HEAD_DIM_MAX, "pre_gelu_values_saved");
    CUDA_ALLOC_CHECK(buffers_host.lifecycle_phase_counts, sizeof(int) * 8, "lifecycle_phase_counts");
    CUDA_ALLOC_CHECK(buffers_host.gradient_features_pool, sizeof(float) * BATCH_SIZE_MAX * HARDWARE_FEATURES_DIM, "gradient_features_pool");
    CUDA_ALLOC_CHECK(buffers_host.gradient_logits_pool, sizeof(float) * BATCH_SIZE_MAX * NUM_CLASSES_MAX, "gradient_logits_pool");
    CUDA_ALLOC_CHECK(buffers_host.gradient_loss_pool, sizeof(float) * BATCH_SIZE_MAX, "gradient_loss_pool");
    CUDA_ALLOC_CHECK(buffers_host.gradient_logit_grads_pool, sizeof(float) * BATCH_SIZE_MAX * NUM_CLASSES_MAX, "gradient_logit_grads_pool");
    CUDA_ALLOC_CHECK(buffers_host.gradient_magnitudes_pool, sizeof(float) * POOL_CAPACITY_MAX, "gradient_magnitudes_pool");
    CUDA_ALLOC_CHECK(buffers_host.pooling_weights_grad, sizeof(float) * CHANNELS_MAX, "pooling_weights_grad");
    CUDA_ALLOC_CHECK(buffers_host.fc_weights_grad, sizeof(float) * NUM_CLASSES_MAX * BEHAVIORAL_DIM_HW_MAX, "fc_weights_grad");
    CUDA_ALLOC_CHECK(buffers_host.fc_bias_grad, sizeof(float) * NUM_CLASSES_MAX, "fc_bias_grad");
    CUDA_ALLOC_CHECK(buffers_host.features_grad, sizeof(float) * BATCH_SIZE_MAX * BEHAVIORAL_DIM_HW_MAX, "features_grad");
    CUDA_ALLOC_CHECK(buffers_host.adam_m_perception_pool, sizeof(float) * NUM_HEADS_MAX * CHANNELS_MAX * HEAD_DIM_MAX, "adam_m_perception_pool");
    CUDA_ALLOC_CHECK(buffers_host.adam_v_perception_pool, sizeof(float) * NUM_HEADS_MAX * CHANNELS_MAX * HEAD_DIM_MAX, "adam_v_perception_pool");
    CUDA_ALLOC_CHECK(buffers_host.adam_m_interaction_pool, sizeof(float) * NUM_HEADS_MAX * HEAD_DIM_MAX * HEAD_DIM_MAX, "adam_m_interaction_pool");
    CUDA_ALLOC_CHECK(buffers_host.adam_v_interaction_pool, sizeof(float) * NUM_HEADS_MAX * HEAD_DIM_MAX * HEAD_DIM_MAX, "adam_v_interaction_pool");
    CUDA_ALLOC_CHECK(buffers_host.adam_m_value_pool, sizeof(float) * NUM_HEADS_MAX * HEAD_DIM_MAX * CHANNELS_MAX, "adam_m_value_pool");
    CUDA_ALLOC_CHECK(buffers_host.adam_v_value_pool, sizeof(float) * NUM_HEADS_MAX * HEAD_DIM_MAX * CHANNELS_MAX, "adam_v_value_pool");
    CUDA_ALLOC_CHECK(buffers_host.adam_m_classifier_pool, sizeof(float) * NUM_CLASSES_MAX * BEHAVIORAL_DIM_HW_MAX, "adam_m_classifier_pool");
    CUDA_ALLOC_CHECK(buffers_host.adam_v_classifier_pool, sizeof(float) * NUM_CLASSES_MAX * BEHAVIORAL_DIM_HW_MAX, "adam_v_classifier_pool");
    CUDA_ALLOC_CHECK(buffers_host.adam_m_pooling, sizeof(float) * CHANNELS_MAX, "adam_m_pooling");
    CUDA_ALLOC_CHECK(buffers_host.adam_v_pooling, sizeof(float) * CHANNELS_MAX, "adam_v_pooling");
    CUDA_ALLOC_CHECK(buffers_host.adam_m_fc_weights, sizeof(float) * NUM_CLASSES_MAX * BEHAVIORAL_DIM_HW_MAX, "adam_m_fc_weights");
    CUDA_ALLOC_CHECK(buffers_host.adam_v_fc_weights, sizeof(float) * NUM_CLASSES_MAX * BEHAVIORAL_DIM_HW_MAX, "adam_v_fc_weights");
    CUDA_ALLOC_CHECK(buffers_host.adam_m_fc_bias, sizeof(float) * NUM_CLASSES_MAX, "adam_m_fc_bias");
    CUDA_ALLOC_CHECK(buffers_host.adam_v_fc_bias, sizeof(float) * NUM_CLASSES_MAX, "adam_v_fc_bias");
    CUDA_ALLOC_CHECK(buffers_host.batch_ca_states_pool, sizeof(float) * BATCH_SIZE_MAX * CA_FIELD_SIZE * CHANNELS_MAX, "batch_ca_states_pool");
    CUDA_ALLOC_CHECK(buffers_host.batch_ca_input_grads, sizeof(float) * BATCH_SIZE_MAX * CA_FIELD_SIZE * CHANNELS_MAX, "batch_ca_input_grads");
    CUDA_ALLOC_CHECK(buffers_host.batched_ca_output, sizeof(float) * BATCH_SIZE_MAX * NUM_HEADS_MAX * CA_FIELD_SIZE * HEAD_DIM_MAX, "batched_ca_output");
    CUDA_ALLOC_CHECK(buffers_host.batch_affinity_reduced, sizeof(float) * BATCH_SIZE_MAX * CA_FIELD_SIZE, "batch_affinity_reduced");
    CUDA_ALLOC_CHECK(buffers_host.batch_flow_field, sizeof(float) * BATCH_SIZE_MAX * CA_FIELD_SIZE * 2, "batch_flow_field");
    CUDA_ALLOC_CHECK(buffers_host.batch_reintegration_buffer, sizeof(float) * BATCH_SIZE_MAX * CA_FIELD_SIZE * CHANNELS_MAX, "batch_reintegration_buffer");
    CUDA_ALLOC_CHECK(buffers_host.batch_prev_concentration, sizeof(float) * BATCH_SIZE_MAX * CA_FIELD_SIZE * CHANNELS_MAX, "batch_prev_concentration");
    CUDA_ALLOC_CHECK(buffers_host.batch_labels_pool, sizeof(int) * BATCH_SIZE_MAX, "batch_labels_pool");
    CUDA_ALLOC_CHECK(buffers_host.task_loss_pool, sizeof(float), "task_loss_pool");
    CUDA_ALLOC_CHECK(buffers_host.reg_loss_pool, sizeof(float), "reg_loss_pool");
    CUDA_ALLOC_CHECK(buffers_host.rank_loss_pool, sizeof(float), "rank_loss_pool");
    CUDA_ALLOC_CHECK(buffers_host.coherence_loss_pool, sizeof(float), "coherence_loss_pool");
    CUDA_ALLOC_CHECK(buffers_host.diversity_loss_pool, sizeof(float), "diversity_loss_pool");
    CUDA_ALLOC_CHECK(buffers_host.total_loss_pool, sizeof(float), "total_loss_pool");
    CUDA_ALLOC_CHECK(buffers_host.training_mode, sizeof(HybridTrainingMode), "training_mode");
    CUDA_ALLOC_CHECK(buffers_host.classifier, sizeof(ClassificationHead), "classifier");
    // Classifier input_dim = num_heads * channels (from spatial pooling of CA output)
    constexpr int CLASSIFIER_INPUT_DIM_MAX = NUM_HEADS_MAX * CHANNELS_MAX;
    CUDA_ALLOC_CHECK(buffers_host.classifier_workspace, sizeof(float) * (CLASSIFIER_INPUT_DIM_MAX + (CLASSIFIER_INPUT_DIM_MAX * NUM_CLASSES_MAX) + NUM_CLASSES_MAX), "classifier_workspace");
    CUDA_ALLOC_CHECK(buffers_host.curriculum, sizeof(AdaptiveCurriculum), "curriculum");
    CUDA_ALLOC_CHECK(buffers_host.voronoi_occupancy_histogram, sizeof(float) * MAX_CELLS, "voronoi_occupancy_histogram");
    CUDA_ALLOC_CHECK(buffers_host.pool_task_accuracies, sizeof(float) * POOL_CAPACITY_MAX, "pool_task_accuracies");
    CUDA_ALLOC_CHECK(buffers_host.organism, sizeof(Organism), "organism");
    CUDA_ALLOC_CHECK(buffers_host.reduction_workspace, sizeof(float) * ((CA_FIELD_SIZE * CHANNELS_MAX + BLOCK_SIZE - 1) / BLOCK_SIZE), "reduction_workspace");
    CUDA_ALLOC_CHECK(buffers_host.rng_states, sizeof(curandState) * POOL_CAPACITY_MAX, "rng_states");
    CUDA_ALLOC_CHECK(buffers_host.memory_params, sizeof(MemoryUpdateParams), "memory_params");
    CUDA_ALLOC_CHECK(buffers_host.organism_workspace_genomes, sizeof(float) * BLOCK_SIZE * SPAWN_WS_COUNT * GENOME_SIZE, "organism_workspace_genomes");
    CUDA_ALLOC_CHECK(buffers_host.behavioral_features_buffer, sizeof(float) * POOL_CAPACITY_MAX * BEHAVIORAL_DIM_MAX, "behavioral_features_buffer");
    CUDA_ALLOC_CHECK(buffers_host.behavioral_embedding_weights, sizeof(float) * BEHAVIORAL_DIM_MAX * BEHAVIORAL_DIM_MAX, "behavioral_embedding_weights");
    CUDA_ALLOC_CHECK(buffers_host.behavioral_reconstruction_error, sizeof(float), "behavioral_reconstruction_error");
    CUDA_ALLOC_CHECK(buffers_host.ca_output_grad_buffer, sizeof(float) * CA_FIELD_SIZE * CHANNELS_MAX, "ca_output_grad_buffer");
    CUDA_ALLOC_CHECK(buffers_host.dL_dperception_buffer, sizeof(float) * NUM_HEADS_MAX * CA_FIELD_SIZE * HEAD_DIM_MAX, "dL_dperception_buffer");
    CUDA_ALLOC_CHECK(buffers_host.dL_dinteraction_buffer, sizeof(float) * NUM_HEADS_MAX * CA_FIELD_SIZE * HEAD_DIM_MAX, "dL_dinteraction_buffer");
    CUDA_ALLOC_CHECK(buffers_host.component_workspace_genomes_buffer, sizeof(float) * GENOME_SIZE * 2, "component_workspace_genomes_buffer");
    CUDA_ALLOC_CHECK(buffers_host.behavioral_workspace_genomes_buffer, sizeof(float) * GENOME_SIZE * 2, "behavioral_workspace_genomes_buffer");
    CUDA_ALLOC_CHECK(buffers_host.weight_inherit_child_indices, sizeof(int) * POOL_CAPACITY_MAX, "weight_inherit_child_indices");
    CUDA_ALLOC_CHECK(buffers_host.weight_inherit_parent_indices, sizeof(int) * POOL_CAPACITY_MAX, "weight_inherit_parent_indices");
    CUDA_ALLOC_CHECK(buffers_host.weight_inherit_num_pending, sizeof(int), "weight_inherit_num_pending");
    // Backward pass workspace buffers (sized for BACKWARD_CHUNK_SAMPLES)
    CUDA_ALLOC_CHECK(buffers_host.backward_ws_fp16_a, BACKWARD_WS_FP16_A_SIZE, "backward_ws_fp16_a");
    CUDA_ALLOC_CHECK(buffers_host.backward_ws_fp16_b, BACKWARD_WS_FP16_B_SIZE, "backward_ws_fp16_b");
    CUDA_ALLOC_CHECK(buffers_host.backward_ws_dW, BACKWARD_WS_DW_SIZE, "backward_ws_dW");
    CUDA_ALLOC_CHECK(buffers_host.backward_ws_dI, BACKWARD_WS_DI_SIZE, "backward_ws_dI");
    CUDA_ALLOC_CHECK(buffers_host.backward_ws_W_T, BACKWARD_WS_W_T_SIZE, "backward_ws_W_T");
    CUDA_ALLOC_CHECK(buffers_host.backward_ws_im2col, BACKWARD_WS_IM2COL_SIZE, "backward_ws_im2col");
    CUDA_ALLOC_CHECK(buffers_host.backward_ws_dpregelu, BACKWARD_WS_DPREGELU_SIZE, "backward_ws_dpregelu");

    cudaMemcpy(buffers, &buffers_host, sizeof(OrganismPreallocatedBuffers), cudaMemcpyHostToDevice);

    printf("[H29] Launching persistent_evolution_kernel<<<1,1>>>\n"); fflush(stdout);
    persistent_evolution_kernel<<<1, 1>>>(
        (unsigned int)time(nullptr),
        d_dataset_array,
        d_test_dataset_array,
        buffers,
        d_audit
    );
    err = cudaGetLastError();
    printf("[H30] cudaGetLastError=%d (%s)\n", (int)err, cudaGetErrorString(err)); fflush(stdout);
    if (err != cudaSuccess) {
        printf("[H-ERR] Launch failed\n");
        return 1;
    }

    printf("[H31] Kernel launched, polling audit buffer\n"); fflush(stdout);

    auto start_time = std::chrono::steady_clock::now();
    int last_gen = -1;

    // Host only reads audit buffer, never writes - kernel runs autonomously
    while (true) {
        std::atomic_thread_fence(std::memory_order_acquire);

        if (h_audit->ready) {
            int gen = h_audit->generation;
            if (gen != last_gen) {
                last_gen = gen;

                auto now_time = std::chrono::steady_clock::now();
                double elapsed_sec = std::chrono::duration<double>(now_time - start_time).count();

                printf("[AUDIT] gen=%d batch=%d acc=%.4f loss=%.4f correct=%d/%d (%.1fs)\n",
                       gen, h_audit->batch_size, h_audit->accuracy, h_audit->loss,
                       h_audit->correct_count, h_audit->batch_size, elapsed_sec);

                // Write sample images using audit_writer
                if (write_sample_images(session_dir, gen, h_audit) != 0) {
                    fprintf(stderr, "FATAL: write_sample_images failed\n");
                }

                // Write CA snapshot
                char ca_path[256];
                snprintf(ca_path, sizeof(ca_path), "%s/ca_states/gen%04d.pgm", session_dir, gen);
                if (write_ca_snapshot(ca_path, gen, h_audit) != 0) {
                    fprintf(stderr, "FATAL: write_ca_snapshot failed\n");
                }

                // Write predictions CSV
                char predictions_path[256];
                snprintf(predictions_path, sizeof(predictions_path), "%s/predictions_gen%04d.csv", session_dir, gen);
                if (write_predictions_csv(predictions_path, gen, h_audit) != 0) {
                    fprintf(stderr, "FATAL: write_predictions_csv failed\n");
                }

                // Write generation summary to metrics.csv
                if (write_generation_summary(session_dir, gen, h_audit) != 0) {
                    fprintf(stderr, "FATAL: write_generation_summary failed\n");
                }

                // Write pool state
                if (write_pool_state(session_dir, gen, h_audit) != 0) {
                    fprintf(stderr, "FATAL: write_pool_state failed\n");
                }

                // Append to manifest
                append_to_manifest(manifest_path, predictions_path, ca_path, elapsed_sec);
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    cudaFreeHost(h_audit);
    return 0;
}
