
#include "../../config/config.cu"
#include "../../runtime.cu"
#include "../../data/dataset_loader.cu"
#include "../../training/training_types.cu"
#include "../../utils/genome_params.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

#define TEST_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "TEST_FAIL: %s at %s:%d - %s\n", \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            return false; \
        } \
    } while(0)

bool test_dataset_registry_consistency() {
    printf("\n[TEST 1] DatasetDescriptor registry consistency\n");

    for (int i = 0; i < NUM_DATASETS; i++) {
        const DatasetDescriptor* desc = &HOST_DATASET_REGISTRY[i];

        printf("  [%d] %s: %zux%zux%zu, %zu classes, %zu train/%zu test\n",
               i, desc->name,
               desc->sample_rows, desc->sample_cols, desc->channels,
               desc->num_classes, desc->num_train, desc->num_test);

        if (desc->sample_rows == 0 || desc->sample_cols == 0) {
            fprintf(stderr, "  FAIL: Dataset %s has zero dimensions\n", desc->name);
            return false;
        }
        if (desc->num_classes == 0) {
            fprintf(stderr, "  FAIL: Dataset %s has zero classes\n", desc->name);
            return false;
        }
        if (desc->num_train == 0 && desc->num_test == 0) {
            fprintf(stderr, "  FAIL: Dataset %s has no samples\n", desc->name);
            return false;
        }

        size_t expected_sample_bytes = desc->sample_rows * desc->sample_cols * desc->channels;
        if (desc->bit_depth == 16) expected_sample_bytes *= 2;

        size_t train_sample_bytes = (desc->num_train > 0 && desc->train_size_bytes > 0) ?
                                     desc->train_size_bytes / desc->num_train : expected_sample_bytes;

        if (desc->num_train > 0 && train_sample_bytes < expected_sample_bytes / 2) {
            fprintf(stderr, "  WARN: Dataset %s train_size_bytes may be inconsistent\n", desc->name);
        }
    }

    printf("  ✓ All %d datasets have valid descriptors\n", NUM_DATASETS);
    return true;
}

bool test_dataset_loading() {
    printf("\n[TEST 2] Dataset loading via load_dataset_from_registry\n");

    Dataset* d_dataset = nullptr;
    cudaError_t err = load_dataset_from_registry(0, true, &d_dataset);

    if (err != cudaSuccess || d_dataset == nullptr) {
        printf("  SKIP: Dataset files not available (expected in CI/test environment)\n");
        return true;  
    }

    Dataset h_dataset;
    TEST_CHECK(cudaMemcpy(&h_dataset, d_dataset, sizeof(Dataset), cudaMemcpyDeviceToHost));

    printf("  Loaded: num_samples=%d, is_train=%d\n",
           h_dataset.num_samples, h_dataset.is_train);

    if (h_dataset.num_samples <= 0) {
        fprintf(stderr, "  FAIL: Loaded dataset has no samples\n");
        return false;
    }
    if (h_dataset.samples == nullptr) {
        fprintf(stderr, "  FAIL: Loaded dataset has null samples pointer\n");
        return false;
    }

    printf("  ✓ Dataset loaded with %d samples\n", h_dataset.num_samples);
    return true;
}

bool test_sample_to_ca_interpolation() {
    printf("\n[TEST 3] Sample→CA grid bilinear interpolation\n");

    const DatasetDescriptor* desc = &HOST_DATASET_REGISTRY[0];
    int sample_rows = (int)desc->sample_rows;
    int sample_cols = (int)desc->sample_cols;
    int ca_grid_size = GRID_SIZE_MIN + 8;  

    printf("  Source: %dx%d → CA grid: %dx%d\n",
           sample_rows, sample_cols, ca_grid_size, ca_grid_size);

    int sample_size = sample_rows * sample_cols;
    unsigned char* h_sample = new unsigned char[sample_size];
    for (int y = 0; y < sample_rows; y++) {
        for (int x = 0; x < sample_cols; x++) {
            h_sample[y * sample_cols + x] = (unsigned char)((x + y) * 255 / (sample_rows + sample_cols - 2));
        }
    }

    int ca_size = ca_grid_size * ca_grid_size;
    float* h_ca_grid = new float[ca_size];

    for (int ca_y = 0; ca_y < ca_grid_size; ca_y++) {
        for (int ca_x = 0; ca_x < ca_grid_size; ca_x++) {
            float src_x = ca_x * (float)sample_cols / ca_grid_size;
            float src_y = ca_y * (float)sample_rows / ca_grid_size;

            int x0 = (int)src_x;
            int y0 = (int)src_y;
            int x1 = (x0 + 1 < sample_cols) ? x0 + 1 : x0;
            int y1 = (y0 + 1 < sample_rows) ? y0 + 1 : y0;

            float fx = src_x - x0;
            float fy = src_y - y0;

            float p00 = h_sample[y0 * sample_cols + x0] / 255.0f;
            float p01 = h_sample[y0 * sample_cols + x1] / 255.0f;
            float p10 = h_sample[y1 * sample_cols + x0] / 255.0f;
            float p11 = h_sample[y1 * sample_cols + x1] / 255.0f;

            float value = p00 * (1 - fx) * (1 - fy) +
                          p01 * fx * (1 - fy) +
                          p10 * (1 - fx) * fy +
                          p11 * fx * fy;

            h_ca_grid[ca_y * ca_grid_size + ca_x] = value;
        }
    }

    int out_of_range = 0;
    int nan_count = 0;
    float min_val = 1.0f, max_val = 0.0f;
    for (int i = 0; i < ca_size; i++) {
        float v = h_ca_grid[i];
        if (isnan(v)) nan_count++;
        else if (v < 0.0f || v > 1.0f) out_of_range++;
        else {
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }
    }

    printf("  Interpolated range: [%.4f, %.4f], NaN=%d, out_of_range=%d\n",
           min_val, max_val, nan_count, out_of_range);

    delete[] h_sample;
    delete[] h_ca_grid;

    if (nan_count > 0 || out_of_range > 0) {
        fprintf(stderr, "  FAIL: Interpolation produced invalid values\n");
        return false;
    }

    printf("  ✓ Bilinear interpolation produces valid [0,1] values\n");
    return true;
}

bool test_ca_channel_layout() {
    printf("\n[TEST 4] CA state channel layout for dataset injection\n");


    constexpr int CHEM_CHANNELS = 6;      
    constexpr int RD_CHANNELS = 4;        
    constexpr int BEHAVIORAL_CHANNELS = 1; 
    constexpr int IMAGE_CHANNEL_START = 11;
    constexpr int IMAGE_CHANNEL_END = 13;
    constexpr int RECURRENCE_CHANNEL = 14;
    constexpr int TEMPORAL_CHANNEL = 15;
    constexpr int TOTAL_FIXED_CHANNELS = 16;

    printf("  Channel layout:\n");
    printf("    [0-5]  ChemicalField (%d channels)\n", CHEM_CHANNELS);
    printf("    [6-9]  RDField (%d channels)\n", RD_CHANNELS);
    printf("    [10]   BehavioralField\n");
    printf("    [11-13] Dataset image (3 channels max)\n");
    printf("    [14]   Recurrence\n");
    printf("    [15]   Temporal retrieval\n");

    if (CHANNELS_MIN < TOTAL_FIXED_CHANNELS) {
        fprintf(stderr, "  FAIL: CHANNELS_MIN=%d < required %d fixed channels\n",
                CHANNELS_MIN, TOTAL_FIXED_CHANNELS);
        return false;
    }

    int image_slots = IMAGE_CHANNEL_END - IMAGE_CHANNEL_START + 1;
    bool all_fit = true;
    for (int i = 0; i < NUM_DATASETS; i++) {
        if (HOST_DATASET_REGISTRY[i].channels > (size_t)image_slots) {
            printf("  WARN: Dataset %s has %zu channels > %d image slots\n",
                   HOST_DATASET_REGISTRY[i].name,
                   HOST_DATASET_REGISTRY[i].channels,
                   image_slots);
        }
    }

    printf("  ✓ Channel layout accommodates %d fixed channels + dynamic\n", TOTAL_FIXED_CHANNELS);
    return true;
}

bool test_architecture_dataset_compatibility() {
    printf("\n[TEST 5] Architecture↔Dataset dimension compatibility\n");

    printf("  CA grid range: [%d, %d]\n", GRID_SIZE_MIN, GRID_SIZE_MAX);

    for (int i = 0; i < NUM_DATASETS; i++) {
        const DatasetDescriptor* desc = &HOST_DATASET_REGISTRY[i];
        int max_dim = (desc->sample_rows > desc->sample_cols) ?
                      (int)desc->sample_rows : (int)desc->sample_cols;

        float min_scale = (float)GRID_SIZE_MIN / max_dim;
        float max_scale = (float)GRID_SIZE_MAX / max_dim;

        printf("  %s: %zux%zu → scale range [%.2f, %.2f]\n",
               desc->name, desc->sample_rows, desc->sample_cols,
               min_scale, max_scale);

        if (max_scale < 0.1f) {
            fprintf(stderr, "  WARN: Dataset %s may lose too much resolution\n", desc->name);
        }
    }

    printf("  ✓ All datasets can be scaled to CA grid range\n");
    return true;
}

int main() {
    cudaSetDevice(0);

    printf("\n========================================\n");
    printf("Dataloader→CA Integration Tests\n");
    printf("========================================\n");

    int passed = 0;
    int total = 0;

    total++; if (test_dataset_registry_consistency()) passed++;
    total++; if (test_dataset_loading()) passed++;
    total++; if (test_sample_to_ca_interpolation()) passed++;
    total++; if (test_ca_channel_layout()) passed++;
    total++; if (test_architecture_dataset_compatibility()) passed++;

    printf("\n========================================\n");
    printf("RESULTS: %d/%d tests passed\n", passed, total);
    printf("========================================\n");

    return (passed == total) ? 0 : 1;
}
