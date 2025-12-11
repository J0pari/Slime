#include "../slime/runtime.cu"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

    for (int i = 0; i < NUM_ACTIVE_DATASETS; i++) {
        int dataset_id = HOST_ACTIVE_DATASET_IDS[i];
        printf("[H8.%d] Loading dataset %d...\n", i, dataset_id); fflush(stdout);
        err = load_dataset_from_registry(dataset_id, true, &d_datasets[i]);
        if (err != cudaSuccess) {
            printf("[H-ERR] Dataset %d load failed: %s\n", dataset_id, cudaGetErrorString(err));
            return 1;
        }
    }
    printf("[H9] All %d datasets loaded\n", NUM_ACTIVE_DATASETS); fflush(stdout);

    // Allocate device array of dataset pointers
    Dataset** d_dataset_array;
    cudaMalloc(&d_dataset_array, sizeof(Dataset*) * NUM_ACTIVE_DATASETS);
    cudaMemcpy(d_dataset_array, d_datasets, sizeof(Dataset*) * NUM_ACTIVE_DATASETS, cudaMemcpyHostToDevice);

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

    printf("[H28c] NOT setting global stack limit - let CUDA allocate per-kernel dynamically\n"); fflush(stdout);
    printf("[H28d] Each kernel will use only the stack it needs (default 1024, up to %d for init_organism)\n", CDP_STACK_SIZE); fflush(stdout);

    printf("[H29] Launching persistent_evolution_kernel<<<1,1>>>\n"); fflush(stdout);
    persistent_evolution_kernel<<<1, 1>>>(
        (unsigned int)time(nullptr),
        d_dataset_array
    );
    err = cudaGetLastError();
    printf("[H30] cudaGetLastError=%d (%s)\n", (int)err, cudaGetErrorString(err)); fflush(stdout);
    if (err != cudaSuccess) {
        printf("[H-ERR] Launch failed\n");
        return 1;
    }

    printf("[H31] Kernel launched successfully, entering infinite loop\n"); fflush(stdout);
    while (1) { cudaDeviceSynchronize(); }
}