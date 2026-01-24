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
    pool_host.size = TEST_POOL_SIZE;
    pool_host.generation = 0;

    TEST_CHECK(cudaMalloc(&pool_host.entries, sizeof(PoolEntry) * TEST_POOL_SIZE));
    TEST_CHECK(cudaMalloc(&pool_host.fitness_values, sizeof(float) * TEST_POOL_SIZE));
    TEST_CHECK(cudaMalloc(&pool_host.alive_flags, sizeof(bool) * TEST_POOL_SIZE));

    ComponentPool* d_pool;
    TEST_CHECK(cudaMalloc(&d_pool, sizeof(ComponentPool)));
    TEST_CHECK(cudaMemcpy(d_pool, &pool_host, sizeof(ComponentPool), cudaMemcpyHostToDevice));

    // 3. ARCHIVE ALLOCATION
    printf("[3/15] Allocating BehavioralArchive...\n");
    BehavioralArchive archive_host;
    archive_host.capacity = ARCHIVE_CAPACITY;
    archive_host.size = 0;
    archive_host.generation = 0;

    TEST_CHECK(cudaMalloc(&archive_host.elites, sizeof(GPUElite) * ARCHIVE_CAPACITY));
    TEST_CHECK(cudaMalloc(&archive_host.cells, sizeof(ArchiveCell) * ARCHIVE_GRID_DIM * ARCHIVE_GRID_DIM * ARCHIVE_GRID_DIM));

    BehavioralArchive* d_archive;
    TEST_CHECK(cudaMalloc(&d_archive, sizeof(BehavioralArchive)));
    TEST_CHECK(cudaMemcpy(d_archive, &archive_host, sizeof(BehavioralArchive), cudaMemcpyHostToDevice));

    // 4. SHARED BUFFERS ALLOCATION
    printf("[4/15] Allocating SharedBuffers...\n");
    SharedBuffers buffers_host;

    TEST_CHECK(cudaMalloc(&buffers_host.all_chem_fields, sizeof(float) * CHEM_FIELD_COUNT * CA_FIELD_SIZE));
    TEST_CHECK(cudaMalloc(&buffers_host.all_rd_fields, sizeof(float) * RD_FIELD_COUNT * CA_FIELD_SIZE));
    TEST_CHECK(cudaMalloc(&buffers_host.all_ca_concentration, sizeof(float) * CA_CONCENTRATION_SIZE));
    TEST_CHECK(cudaMalloc(&buffers_host.all_ca_output, sizeof(float) * CA_OUTPUT_SIZE));
    TEST_CHECK(cudaMalloc(&buffers_host.all_ca_affinity, sizeof(float) * CA_AFFINITY_SIZE));
    TEST_CHECK(cudaMalloc(&buffers_host.all_ca_flow, sizeof(float) * CA_FLOW_SIZE));
    TEST_CHECK(cudaMalloc(&buffers_host.all_ca_reintegration, sizeof(float) * CA_REINTEGRATION_SIZE));
    TEST_CHECK(cudaMalloc(&buffers_host.spawn_workspace, sizeof(float) * GENOME_SIZE * SPAWN_WS_COUNT));

    SharedBuffers* d_buffers;
    TEST_CHECK(cudaMalloc(&d_buffers, sizeof(SharedBuffers)));
    TEST_CHECK(cudaMemcpy(d_buffers, &buffers_host, sizeof(SharedBuffers), cudaMemcpyHostToDevice));

    // 5. TRAINING MODE ALLOCATION
    printf("[5/15] Allocating TrainingMode...\n");
    TrainingMode training_host;
    training_host.enabled = true;
    training_host.batch_size = BATCH_SIZE;
    training_host.learning_rate = LEARNING_RATE;
    training_host.num_datasets = 1;
    training_host.dataset_array = d_dataset_array;
    training_host.test_dataset_array = d_test_dataset_array;
    training_host.current_dataset_idx = 0;

    TEST_CHECK(cudaMalloc(&training_host.ca_input_batch, sizeof(float) * BATCH_SIZE * CHANNELS_MAX * CA_FIELD_SIZE));
    TEST_CHECK(cudaMalloc(&training_host.ca_target_batch, sizeof(int) * BATCH_SIZE));
    TEST_CHECK(cudaMalloc(&training_host.ca_logits_batch, sizeof(float) * BATCH_SIZE * NUM_CLASSES));
    TEST_CHECK(cudaMalloc(&training_host.ca_loss_batch, sizeof(float) * BATCH_SIZE));
    TEST_CHECK(cudaMalloc(&training_host.ca_grad_output, sizeof(float) * BATCH_SIZE * NUM_HEADS_MAX * HEAD_DIM_MAX * CA_FIELD_SIZE));
    TEST_CHECK(cudaMalloc(&training_host.tape, sizeof(AutodiffTape)));

    TrainingMode* d_training;
    TEST_CHECK(cudaMalloc(&d_training, sizeof(TrainingMode)));
    TEST_CHECK(cudaMemcpy(d_training, &training_host, sizeof(TrainingMode), cudaMemcpyHostToDevice));

    // 6. TELEMETRY ALLOCATION
    printf("[6/15] Allocating Telemetry...\n");
    Telemetry telemetry_host;
    memset(&telemetry_host, 0, sizeof(Telemetry));

    Telemetry* d_telemetry;
    TEST_CHECK(cudaMalloc(&d_telemetry, sizeof(Telemetry)));
    TEST_CHECK(cudaMemcpy(d_telemetry, &telemetry_host, sizeof(Telemetry), cudaMemcpyHostToDevice));

    // 7. INITIALIZE POOL WITH RANDOM GENOMES
    printf("[7/15] Initializing pool with random genomes...\n");
    dim3 pool_grid(TEST_POOL_SIZE);
    dim3 pool_block(256);
    initialize_pool_kernel<<<pool_grid, pool_block>>>(d_pool, 12345);
    TEST_CHECK(cudaDeviceSynchronize());

    // 8. INITIALIZE ARCHIVE
    printf("[8/15] Initializing archive...\n");
    initialize_archive_kernel<<<1, 256>>>(d_archive);
    TEST_CHECK(cudaDeviceSynchronize());

    // 9. RUN ONE FULL LIFECYCLE ITERATION
    printf("[9/15] Executing persistent_evolution_kernel (1 iteration)...\n");

    EvolutionConfig config;
    config.pool = d_pool;
    config.archive = d_archive;
    config.buffers = d_buffers;
    config.training = d_training;
    config.telemetry = d_telemetry;
    config.max_iterations = 1;  // Just one iteration for test
    config.seed = 42;

    EvolutionConfig* d_config;
    TEST_CHECK(cudaMalloc(&d_config, sizeof(EvolutionConfig)));
    TEST_CHECK(cudaMemcpy(d_config, &config, sizeof(EvolutionConfig), cudaMemcpyHostToDevice));

    TEST_CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, CDP_SYNC_DEPTH));
    TEST_CHECK(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, CDP_PENDING_LAUNCH_COUNT));
    TEST_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, CDP_STACK_SIZE));

    persistent_evolution_kernel<<<1, 1>>>(d_config);
    TEST_CHECK(cudaDeviceSynchronize());

    // 10. VERIFY NO NAN IN CHEMICAL FIELDS
    printf("[10/15] Verifying chemical fields have no NaN...\n");
    float* h_chem_check = new float[CA_FIELD_SIZE];
    TEST_CHECK(cudaMemcpy(h_chem_check, buffers_host.all_chem_fields, sizeof(float) * CA_FIELD_SIZE, cudaMemcpyDeviceToHost));

    int chem_nan_count = 0;
    for (int i = 0; i < CA_FIELD_SIZE; i++) {
        if (isnan(h_chem_check[i]) || isinf(h_chem_check[i])) {
            chem_nan_count++;
        }
    }
    if (chem_nan_count > 0) {
        fprintf(stderr, "TEST_FAIL: Chemical field has %d NaN/Inf values\n", chem_nan_count);
        delete[] h_chem_check;
        return false;
    }
    delete[] h_chem_check;
    printf("    ✓ Chemical concentration: 0 NaN\n");

    // 11. VERIFY RESOURCE GRADIENTS ARE NON-ZERO
    printf("[11/15] Verifying resource gradients are computed...\n");
    float* h_grad_x = new float[CA_FIELD_SIZE];
    float* h_grad_y = new float[CA_FIELD_SIZE];

    // Resource gradients are at offsets RD_RESOURCE_GRADIENT_X and RD_RESOURCE_GRADIENT_Y
    TEST_CHECK(cudaMemcpy(h_grad_x, buffers_host.all_rd_fields + CA_FIELD_SIZE * RD_RESOURCE_GRADIENT_X,
                          sizeof(float) * CA_FIELD_SIZE, cudaMemcpyDeviceToHost));
    TEST_CHECK(cudaMemcpy(h_grad_y, buffers_host.all_rd_fields + CA_FIELD_SIZE * RD_RESOURCE_GRADIENT_Y,
                          sizeof(float) * CA_FIELD_SIZE, cudaMemcpyDeviceToHost));

    int grad_nan_count = 0;
    int grad_nonzero_count = 0;
    for (int i = 0; i < CA_FIELD_SIZE; i++) {
        if (isnan(h_grad_x[i]) || isinf(h_grad_x[i]) || isnan(h_grad_y[i]) || isinf(h_grad_y[i])) {
            grad_nan_count++;
        }
        if (fabsf(h_grad_x[i]) > 1e-6f || fabsf(h_grad_y[i]) > 1e-6f) {
            grad_nonzero_count++;
        }
    }

    if (grad_nan_count > 0) {
        fprintf(stderr, "TEST_FAIL: Resource gradients have %d NaN/Inf values\n", grad_nan_count);
        delete[] h_grad_x;
        delete[] h_grad_y;
        return false;
    }

    if (grad_nonzero_count == 0) {
        fprintf(stderr, "TEST_FAIL: All resource gradients are zero (not being computed)\n");
        delete[] h_grad_x;
        delete[] h_grad_y;
        return false;
    }

    printf("    ✓ Resource gradient_x: 0 NaN, %d/%d non-zero\n", grad_nonzero_count, CA_FIELD_SIZE);
    printf("    ✓ Resource gradient_y: 0 NaN, %d/%d non-zero\n", grad_nonzero_count, CA_FIELD_SIZE);
    delete[] h_grad_x;
    delete[] h_grad_y;

    // 12. VERIFY CA OUTPUT HAS NO NAN
    printf("[12/15] Verifying CA output has no NaN...\n");
    float* h_ca_output = new float[CA_OUTPUT_SIZE];
    TEST_CHECK(cudaMemcpy(h_ca_output, buffers_host.all_ca_output, sizeof(float) * CA_OUTPUT_SIZE, cudaMemcpyDeviceToHost));

    int ca_nan_count = 0;
    for (int i = 0; i < CA_OUTPUT_SIZE; i++) {
        if (isnan(h_ca_output[i]) || isinf(h_ca_output[i])) {
            ca_nan_count++;
        }
    }
    if (ca_nan_count > 0) {
        fprintf(stderr, "TEST_FAIL: CA output has %d NaN/Inf values\n", ca_nan_count);
        delete[] h_ca_output;
        return false;
    }
    delete[] h_ca_output;
    printf("    ✓ CA output: 0 NaN\n");

    // 13. VERIFY TELEMETRY WAS UPDATED
    printf("[13/15] Verifying telemetry was written...\n");
    Telemetry h_telemetry;
    TEST_CHECK(cudaMemcpy(&h_telemetry, d_telemetry, sizeof(Telemetry), cudaMemcpyDeviceToHost));

    if (h_telemetry.genome_complexity.hash_entropy == 0.0f &&
        h_telemetry.archive_topology.novelty_gradient == 0.0f &&
        h_telemetry.task_performance.accuracy == 0.0f) {
        fprintf(stderr, "TEST_FAIL: Telemetry appears uninitialized (all zeros)\n");
        return false;
    }
    printf("    ✓ Telemetry hash_entropy: %.6f\n", h_telemetry.genome_complexity.hash_entropy);
    printf("    ✓ Telemetry novelty_gradient: %.6f\n", h_telemetry.archive_topology.novelty_gradient);
    printf("    ✓ Telemetry accuracy: %.6f\n", h_telemetry.task_performance.accuracy);

    // 14. VERIFY POOL FITNESS VALUES
    printf("[14/15] Verifying pool fitness values...\n");
    float* h_fitness = new float[TEST_POOL_SIZE];
    TEST_CHECK(cudaMemcpy(h_fitness, pool_host.fitness_values, sizeof(float) * TEST_POOL_SIZE, cudaMemcpyDeviceToHost));

    int fitness_nan_count = 0;
    for (int i = 0; i < TEST_POOL_SIZE; i++) {
        if (isnan(h_fitness[i]) || isinf(h_fitness[i])) {
            fitness_nan_count++;
        }
    }
    if (fitness_nan_count > 0) {
        fprintf(stderr, "TEST_FAIL: Pool has %d NaN/Inf fitness values\n", fitness_nan_count);
        delete[] h_fitness;
        return false;
    }
    printf("    ✓ Pool fitness: 0 NaN, avg=%.6f\n",
           (h_fitness[0] + h_fitness[1] + h_fitness[2] + h_fitness[3]) / 4.0f);
    delete[] h_fitness;

    // 15. CLEANUP
    printf("[15/15] Cleaning up...\n");
    cudaFree(pool_host.entries);
    cudaFree(pool_host.fitness_values);
    cudaFree(pool_host.alive_flags);
    cudaFree(d_pool);

    cudaFree(archive_host.elites);
    cudaFree(archive_host.cells);
    cudaFree(d_archive);

    cudaFree(buffers_host.all_chem_fields);
    cudaFree(buffers_host.all_rd_fields);
    cudaFree(buffers_host.all_ca_concentration);
    cudaFree(buffers_host.all_ca_output);
    cudaFree(buffers_host.all_ca_affinity);
    cudaFree(buffers_host.all_ca_flow);
    cudaFree(buffers_host.all_ca_reintegration);
    cudaFree(buffers_host.spawn_workspace);
    cudaFree(d_buffers);

    cudaFree(training_host.ca_input_batch);
    cudaFree(training_host.ca_target_batch);
    cudaFree(training_host.ca_logits_batch);
    cudaFree(training_host.ca_loss_batch);
    cudaFree(training_host.ca_grad_output);
    cudaFree(training_host.tape);
    cudaFree(d_training);

    cudaFree(d_telemetry);
    cudaFree(d_dataset_array);
    cudaFree(d_test_dataset_array);
    cudaFree(d_config);

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
