#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("GPU Memory:\n");
    printf("  Total: %.2f MB\n", total_mem / (1024.0 * 1024.0));
    printf("  Free: %.2f MB\n", free_mem / (1024.0 * 1024.0));
    printf("  Used: %.2f MB\n", (total_mem - free_mem) / (1024.0 * 1024.0));
    return 0;
}
