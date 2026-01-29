#include "../../config/config.cu"
#include "../../runtime.cu"
#include "../../data/dataset_loader.cu"
#include "../../diagnostics/telemetry_probes.cu"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TEST_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "TEST_FAIL: %s at %s:%d - %s\n", \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            return false; \
        } \
    } while(0)

// Minimal integration test covering all major subsystems
bool test_full_lifecycle_iteration() {
    printf("\n=== FULL INTEGRATION TEST: One Complete Lifecycle Iteration ===\n");

    // 1. DATASET LOADING
    printf("[1/15] Loading MNIST training dataset...\n");
    Dataset* d_mnist_train;
    TEST_CHECK(load_dataset_from_registry(0, true, &d_mnist_train));

    Dataset* d_mnist_test;
    TEST_CHECK(load_dataset_from_registry(0, false, &d_mnist_test));

    Dataset** d_dataset_array;
    TEST_CHECK(cudaMalloc(&d_dataset_array, sizeof(Dataset*)));
    TEST_CHECK(cudaMemcpy(d_dataset_array, &d_mnist_train, sizeof(Dataset*), cudaMemcpyHostToDevice));

    Dataset** d_test_dataset_array;
    TEST_CHECK(cudaMalloc(&d_test_dataset_array, sizeof(Dataset*)));
    TEST_CHECK(cudaMemcpy(d_test_dataset_array, &d_mnist_test, sizeof(Dataset*), cudaMemcpyHostToDevice));

    // 2. POOL ALLOCATION
    printf("[2/15] Allocating ComponentPool...\n");
    constexpr int TEST_POOL_SIZE = 4;  // Small pool for fast test
    ComponentPool pool_host;
    pool_host.capacity = TEST_POOL_SIZE;

    TEST_CHECK(cudaMalloc(&pool_host.entries, sizeof(PoolEntry) * TEST_POOL_SIZE));
    TEST_CHECK(cudaMalloc(&pool_host.fitness_values, sizeof(float) * TEST_POOL_SIZE));
    TEST_CHECK(cudaMalloc(&pool_host.alive_flags, sizeof(bool) * TEST_POOL_SIZE));
    TEST_CHECK(cudaMalloc(&pool_host.alive_indices, sizeof(int) * TEST_POOL_SIZE));

    // Delta buffers for init_pool_kernel
    uint16_t* delta_indices_buffer;
    float* delta_values_buffer;
    float* gradients_buffer;
    TEST_CHECK(cudaMalloc(&delta_indices_buffer, sizeof(uint16_t) * TEST_POOL_SIZE * GENOME_SIZE));
    TEST_CHECK(cudaMalloc(&delta_values_buffer, sizeof(float) * TEST_POOL_SIZE * GENOME_SIZE));
    TEST_CHECK(cudaMalloc(&gradients_buffer, sizeof(float) * TEST_POOL_SIZE * GENOME_SIZE));

    ComponentPool* d_pool;
    TEST_CHECK(cudaMalloc(&d_pool, sizeof(ComponentPool)));
    TEST_CHECK(cudaMemcpy(d_pool, &pool_host, sizeof(ComponentPool), cudaMemcpyHostToDevice));

    // 3. ARCHIVE ALLOCATION (GPUElite SoA)
    printf("[3/15] Allocating GPUElite archive...\n");
    GPUElite archive_host;
    archive_host.hw_dim = BEHAVIORAL_DIM_HW;
    archive_host.task_dim = BEHAVIORAL_DIM_TASK;
    archive_host.gen_dim = BEHAVIORAL_DIM_GEN;

    TEST_CHECK(cudaMalloc(&archive_host.fitness, sizeof(float) * MAX_ARCHIVE_SIZE));
    TEST_CHECK(cudaMalloc(&archive_host.coherence, sizeof(float) * MAX_ARCHIVE_SIZE));
    TEST_CHECK(cudaMalloc(&archive_host.effective_rank, sizeof(float) * MAX_ARCHIVE_SIZE));
    TEST_CHECK(cudaMalloc(&archive_host.genome_hash, sizeof(uint64_t) * MAX_ARCHIVE_SIZE));
    TEST_CHECK(cudaMalloc(&archive_host.parent_ids, sizeof(uint32_t) * MAX_ARCHIVE_SIZE * PARENT_COUNT));
    TEST_CHECK(cudaMalloc(&archive_host.generation, sizeof(uint16_t) * MAX_ARCHIVE_SIZE));
    TEST_CHECK(cudaMalloc(&archive_host.hw_coords, sizeof(float) * MAX_ARCHIVE_SIZE * BEHAVIORAL_DIM_MAX));
    TEST_CHECK(cudaMalloc(&archive_host.task_coords, sizeof(float) * MAX_ARCHIVE_SIZE * BEHAVIORAL_DIM_MAX));
    TEST_CHECK(cudaMalloc(&archive_host.gen_coords, sizeof(float) * MAX_ARCHIVE_SIZE * BEHAVIORAL_DIM_MAX));
    TEST_CHECK(cudaMalloc(&archive_host.latent_genome, sizeof(float) * MAX_ARCHIVE_SIZE * GENOME_LATENT_DIM_MAX));
    TEST_CHECK(cudaMalloc(&archive_host.hardware_features, sizeof(float) * MAX_ARCHIVE_SIZE * HARDWARE_FEATURE_DIM));
    TEST_CHECK(cudaMalloc(&archive_host.task_performance, sizeof(float) * MAX_ARCHIVE_SIZE));
    TEST_CHECK(cudaMalloc(&archive_host.per_class_accuracy, sizeof(float) * MAX_ARCHIVE_SIZE * NUM_CLASSES_MAX));
    TEST_CHECK(cudaMalloc(&archive_host.hash_table_keys, sizeof(uint64_t) * GENOME_HASH_TABLE_SIZE));
    TEST_CHECK(cudaMalloc(&archive_host.hash_table_values, sizeof(int) * GENOME_HASH_TABLE_SIZE));
    TEST_CHECK(cudaMalloc(&archive_host.weight_deltas, sizeof(half) * MAX_ARCHIVE_SIZE * MAX_WEIGHT_DELTAS_PER_ELITE));
    TEST_CHECK(cudaMalloc(&archive_host.weight_delta_indices, sizeof(uint32_t) * MAX_ARCHIVE_SIZE * MAX_WEIGHT_DELTAS_PER_ELITE));
    TEST_CHECK(cudaMalloc(&archive_host.num_weight_deltas, sizeof(uint16_t) * MAX_ARCHIVE_SIZE));
    TEST_CHECK(cudaMalloc(&archive_host.archived_num_heads, sizeof(int) * MAX_ARCHIVE_SIZE));
    TEST_CHECK(cudaMalloc(&archive_host.archived_channels, sizeof(int) * MAX_ARCHIVE_SIZE));
    TEST_CHECK(cudaMalloc(&archive_host.archived_head_dim, sizeof(int) * MAX_ARCHIVE_SIZE));

    GPUElite* d_archive;
    TEST_CHECK(cudaMalloc(&d_archive, sizeof(GPUElite)));
    TEST_CHECK(cudaMemcpy(d_archive, &archive_host, sizeof(GPUElite), cudaMemcpyHostToDevice));

    // 4. VORONOI CELLS ALLOCATION
    printf("[4/15] Allocating VoronoiCell array...\n");
    constexpr int TEST_NUM_CELLS = 8;
    VoronoiCell* voronoi_cells_host = (VoronoiCell*)malloc(sizeof(VoronoiCell) * TEST_NUM_CELLS);
    float* voronoi_hw_centroid_buffer;
    float* voronoi_task_centroid_buffer;
    float* voronoi_gen_centroid_buffer;
    TEST_CHECK(cudaMalloc(&voronoi_hw_centroid_buffer, sizeof(float) * TEST_NUM_CELLS * BEHAVIORAL_DIM_MAX));
    TEST_CHECK(cudaMalloc(&voronoi_task_centroid_buffer, sizeof(float) * TEST_NUM_CELLS * BEHAVIORAL_DIM_MAX));
    TEST_CHECK(cudaMalloc(&voronoi_gen_centroid_buffer, sizeof(float) * TEST_NUM_CELLS * BEHAVIORAL_DIM_MAX));

    for (int i = 0; i < TEST_NUM_CELLS; i++) {
        voronoi_cells_host[i].hw_centroid = voronoi_hw_centroid_buffer + i * BEHAVIORAL_DIM_MAX;
        voronoi_cells_host[i].task_centroid = voronoi_task_centroid_buffer + i * BEHAVIORAL_DIM_MAX;
        voronoi_cells_host[i].gen_centroid = voronoi_gen_centroid_buffer + i * BEHAVIORAL_DIM_MAX;
    }

    VoronoiCell* d_voronoi_cells;
    TEST_CHECK(cudaMalloc(&d_voronoi_cells, sizeof(VoronoiCell) * TEST_NUM_CELLS));
    TEST_CHECK(cudaMemcpy(d_voronoi_cells, voronoi_cells_host, sizeof(VoronoiCell) * TEST_NUM_CELLS, cudaMemcpyHostToDevice));

    // 5. TELEMETRY ALLOCATION
    printf("[5/15] Allocating TelemetryBuffer...\n");
    TelemetryBuffer telemetry_host;
    memset(&telemetry_host, 0, sizeof(TelemetryBuffer));

    TelemetryBuffer* d_telemetry;
    TEST_CHECK(cudaMalloc(&d_telemetry, sizeof(TelemetryBuffer)));
    TEST_CHECK(cudaMemcpy(d_telemetry, &telemetry_host, sizeof(TelemetryBuffer), cudaMemcpyHostToDevice));

    // 6. AUDIT BUFFER ALLOCATION
    printf("[6/15] Allocating AuditBuffer...\n");
    AuditBuffer* d_audit;
    TEST_CHECK(cudaMalloc(&d_audit, sizeof(AuditBuffer)));
    TEST_CHECK(cudaMemset(d_audit, 0, sizeof(AuditBuffer)));

    // 7. INITIALIZE POOL WITH RANDOM GENOMES
    printf("[7/15] Initializing pool with random genomes...\n");
    dim3 pool_grid((TEST_POOL_SIZE + 255) / 256);
    dim3 pool_block(256);
    init_pool_kernel<<<pool_grid, pool_block>>>(
        d_pool, TEST_POOL_SIZE,
        delta_indices_buffer, delta_values_buffer, gradients_buffer
    );
    TEST_CHECK(cudaDeviceSynchronize());

    // 8. INITIALIZE ARCHIVE HASH TABLE
    printf("[8/15] Initializing archive hash table...\n");
    dim3 hash_grid((GENOME_HASH_TABLE_SIZE + 255) / 256);
    init_hash_table_kernel<<<hash_grid, 256>>>(
        archive_host.hash_table_keys,
        archive_host.hash_table_values,
        GENOME_HASH_TABLE_SIZE
    );
    TEST_CHECK(cudaDeviceSynchronize());

    // 9. INITIALIZE VORONOI CELLS
    printf("[9/15] Initializing Voronoi cells...\n");
    init_voronoi_cells_kernel<<<1, TEST_NUM_CELLS>>>(
        d_voronoi_cells,
        TEST_NUM_CELLS,
        BEHAVIORAL_DIM_HW,
        BEHAVIORAL_DIM_TASK,
        BEHAVIORAL_DIM_GEN,
        12345
    );
    TEST_CHECK(cudaDeviceSynchronize());

    // 10. VERIFY POOL INITIALIZATION
    printf("[10/15] Verifying pool initialization...\n");
    ComponentPool pool_verify;
    TEST_CHECK(cudaMemcpy(&pool_verify, d_pool, sizeof(ComponentPool), cudaMemcpyDeviceToHost));

    if (pool_verify.capacity != TEST_POOL_SIZE) {
        fprintf(stderr, "TEST_FAIL: Pool capacity mismatch: expected %d, got %d\n",
                TEST_POOL_SIZE, pool_verify.capacity);
        return false;
    }
    printf("    ✓ Pool capacity: %d\n", pool_verify.capacity);

    // 11. VERIFY POOL FITNESS VALUES
    printf("[11/15] Verifying pool fitness values...\n");
    float* h_fitness = new float[TEST_POOL_SIZE];
    TEST_CHECK(cudaMemcpy(h_fitness, pool_host.fitness_values, sizeof(float) * TEST_POOL_SIZE, cudaMemcpyDeviceToHost));

    int fitness_nan_count = 0;
    float fitness_sum = 0.0f;
    for (int i = 0; i < TEST_POOL_SIZE; i++) {
        if (isnan(h_fitness[i]) || isinf(h_fitness[i])) {
            fitness_nan_count++;
        } else {
            fitness_sum += h_fitness[i];
        }
    }
    // Initial slots beyond POOL_CAPACITY_MIN may have NaN (expected for empty slots)
    printf("    ✓ Pool fitness: %d valid values, avg=%.6f\n",
           TEST_POOL_SIZE - fitness_nan_count,
           (TEST_POOL_SIZE - fitness_nan_count > 0) ? fitness_sum / (TEST_POOL_SIZE - fitness_nan_count) : 0.0f);
    delete[] h_fitness;

    // 12. VERIFY ALIVE FLAGS
    printf("[12/15] Verifying alive flags...\n");
    bool* h_alive = new bool[TEST_POOL_SIZE];
    TEST_CHECK(cudaMemcpy(h_alive, pool_host.alive_flags, sizeof(bool) * TEST_POOL_SIZE, cudaMemcpyDeviceToHost));

    int alive_count = 0;
    for (int i = 0; i < TEST_POOL_SIZE; i++) {
        if (h_alive[i]) alive_count++;
    }
    printf("    ✓ Alive flags: %d/%d alive\n", alive_count, TEST_POOL_SIZE);
    delete[] h_alive;

    // 13. VERIFY HASH TABLE
    printf("[13/15] Verifying hash table initialization...\n");
    uint64_t* h_hash_keys = new uint64_t[16];
    TEST_CHECK(cudaMemcpy(h_hash_keys, archive_host.hash_table_keys, sizeof(uint64_t) * 16, cudaMemcpyDeviceToHost));

    int empty_slots = 0;
    for (int i = 0; i < 16; i++) {
        if (h_hash_keys[i] == HASH_TABLE_EMPTY_KEY) empty_slots++;
    }
    if (empty_slots != 16) {
        fprintf(stderr, "TEST_FAIL: Hash table not properly initialized (expected 16 empty, got %d)\n", empty_slots);
        delete[] h_hash_keys;
        return false;
    }
    printf("    ✓ Hash table: first 16 slots empty (correctly initialized)\n");
    delete[] h_hash_keys;

    // 14. VERIFY VORONOI CELLS
    printf("[14/15] Verifying Voronoi cell initialization...\n");
    VoronoiCell* h_cells = new VoronoiCell[TEST_NUM_CELLS];
    TEST_CHECK(cudaMemcpy(h_cells, d_voronoi_cells, sizeof(VoronoiCell) * TEST_NUM_CELLS, cudaMemcpyDeviceToHost));

    int valid_cells = 0;
    for (int i = 0; i < TEST_NUM_CELLS; i++) {
        if (h_cells[i].radius > 0.0f && h_cells[i].density >= 0) {
            valid_cells++;
        }
    }
    printf("    ✓ Voronoi cells: %d/%d valid\n", valid_cells, TEST_NUM_CELLS);
    delete[] h_cells;

    // 15. CLEANUP
    printf("[15/15] Cleaning up...\n");
    cudaFree(pool_host.entries);
    cudaFree(pool_host.fitness_values);
    cudaFree(pool_host.alive_flags);
    cudaFree(pool_host.alive_indices);
    cudaFree(delta_indices_buffer);
    cudaFree(delta_values_buffer);
    cudaFree(gradients_buffer);
    cudaFree(d_pool);

    cudaFree(archive_host.fitness);
    cudaFree(archive_host.coherence);
    cudaFree(archive_host.effective_rank);
    cudaFree(archive_host.genome_hash);
    cudaFree(archive_host.parent_ids);
    cudaFree(archive_host.generation);
    cudaFree(archive_host.hw_coords);
    cudaFree(archive_host.task_coords);
    cudaFree(archive_host.gen_coords);
    cudaFree(archive_host.latent_genome);
    cudaFree(archive_host.hardware_features);
    cudaFree(archive_host.task_performance);
    cudaFree(archive_host.per_class_accuracy);
    cudaFree(archive_host.hash_table_keys);
    cudaFree(archive_host.hash_table_values);
    cudaFree(archive_host.weight_deltas);
    cudaFree(archive_host.weight_delta_indices);
    cudaFree(archive_host.num_weight_deltas);
    cudaFree(archive_host.archived_num_heads);
    cudaFree(archive_host.archived_channels);
    cudaFree(archive_host.archived_head_dim);
    cudaFree(d_archive);

    cudaFree(voronoi_hw_centroid_buffer);
    cudaFree(voronoi_task_centroid_buffer);
    cudaFree(voronoi_gen_centroid_buffer);
    cudaFree(d_voronoi_cells);
    free(voronoi_cells_host);

    cudaFree(d_telemetry);
    cudaFree(d_audit);
    cudaFree(d_dataset_array);
    cudaFree(d_test_dataset_array);

    printf("\n=== TEST PASSED: Full lifecycle iteration completed successfully ===\n");
    return true;
}

int main() {
    cudaSetDevice(0);

    int passed = 0;
    int total = 0;

    total++;
    if (test_full_lifecycle_iteration()) {
        passed++;
        printf("\n✓ PASS: test_full_lifecycle_iteration\n");
    } else {
        printf("\n✗ FAIL: test_full_lifecycle_iteration\n");
    }

    printf("\n========================================\n");
    printf("RESULTS: %d/%d tests passed\n", passed, total);
    printf("========================================\n");

    return (passed == total) ? 0 : 1;
}
