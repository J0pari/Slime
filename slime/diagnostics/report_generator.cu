
#ifndef REPORT_GENERATOR_CU
#define REPORT_GENERATOR_CU

#include "../config/config.cu"
#include "../utils/tile_ops.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <wincrypt.h>
#pragma comment(lib, "advapi32.lib")

__host__ void compute_sha256(const void* data, size_t len, uint8_t hash[SHA256_HASH_SIZE]) {
    HCRYPTPROV hProv = 0;
    HCRYPTHASH hHash = 0;

    if (!CryptAcquireContext(&hProv, NULL, NULL, PROV_RSA_AES, CRYPT_VERIFYCONTEXT)) {
        printf("ERROR: CryptAcquireContext failed\n");
        memset(hash, 0, SHA256_HASH_SIZE);
        return;
    }

    if (!CryptCreateHash(hProv, CALG_SHA_256, 0, 0, &hHash)) {
        printf("ERROR: CryptCreateHash failed\n");
        CryptReleaseContext(hProv, 0);
        memset(hash, 0, SHA256_HASH_SIZE);
        return;
    }

    if (!CryptHashData(hHash, (const BYTE*)data, (DWORD)len, 0)) {
        printf("ERROR: CryptHashData failed\n");
        CryptDestroyHash(hHash);
        CryptReleaseContext(hProv, 0);
        memset(hash, 0, SHA256_HASH_SIZE);
        return;
    }

    DWORD hashLen = SHA256_HASH_SIZE;
    if (!CryptGetHashParam(hHash, HP_HASHVAL, hash, &hashLen, 0)) {
        printf("ERROR: CryptGetHashParam failed\n");
        memset(hash, 0, SHA256_HASH_SIZE);
    }

    CryptDestroyHash(hHash);
    CryptReleaseContext(hProv, 0);
}
#else
#include <openssl/sha.h>

__host__ void compute_sha256(const void* data, size_t len, uint8_t hash[SHA256_HASH_SIZE]) {
    SHA256((const unsigned char*)data, len, hash);
}
#endif

__host__ void dump_raw_buffer(const char* filename, const void* data, size_t size_bytes, const char* manifest_path) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        printf("ERROR: Cannot write %s\n", filename);
        return;
    }

    size_t written = fwrite(data, 1, size_bytes, f);
    fclose(f);

    if (written != size_bytes) {
        printf("ERROR: Wrote %zu bytes, expected %zu for %s\n", written, size_bytes, filename);
        return;
    }

    uint8_t hash[SHA256_HASH_SIZE];
    compute_sha256(data, size_bytes, hash);

    FILE* manifest = fopen(manifest_path, "a");
    if (manifest) {
        fprintf(manifest, "%s,%zu,", filename, size_bytes);
        for (int i = 0; i < SHA256_HASH_SIZE; i++) {
            fprintf(manifest, "%02x", hash[i]);
        }
        fprintf(manifest, "\n");
        fclose(manifest);
    }
}

__host__ void dump_sample_raw(const char* base_path, int generation, unsigned char* h_image,
                             int sample_rows, int sample_cols, int channels, const char* manifest) {
    char raw_file[PATH_BUFFER_SIZE], pgm_file[PATH_BUFFER_SIZE];
    int sample_size = sample_rows * sample_cols * channels;

    snprintf(raw_file, sizeof(raw_file), "%s/sample_%04d.raw", base_path, generation);
    snprintf(pgm_file, sizeof(pgm_file), "%s/sample_%04d.pgm", base_path, generation);

    dump_raw_buffer(raw_file, h_image, sample_size, manifest);

    FILE* f = fopen(pgm_file, "wb");
    if (f) {
        fprintf(f, "P5\n%d %d\n255\n", sample_cols, sample_rows);
        fwrite(h_image, 1, sample_size, f);
        fclose(f);
    }
}

__host__ void dump_ca_raw(const char* base_path, int generation, float* h_ca_state, int grid_size, int channels, const char* manifest) {
    char raw_file[PATH_BUFFER_SIZE], pgm_file[PATH_BUFFER_SIZE];
    size_t ca_size = grid_size * grid_size * channels * sizeof(float);

    snprintf(raw_file, sizeof(raw_file), "%s/ca_%04d.raw", base_path, generation);
    snprintf(pgm_file, sizeof(pgm_file), "%s/ca_%04d.pgm", base_path, generation);

    dump_raw_buffer(raw_file, h_ca_state, ca_size, manifest);

    FILE* f = fopen(pgm_file, "wb");
    if (f) {
        fprintf(f, "P5\n%d %d\n255\n", grid_size, grid_size);
        for (int i = 0; i < grid_size * grid_size; i++) {
            float val = h_ca_state[i * channels];
            float clamped = (val < 0.0f) ? 0.0f : ((val > 1.0f) ? 1.0f : val);
            unsigned char pixel = (unsigned char)(clamped * UINT8_MAX);
            fwrite(&pixel, 1, 1, f);
        }
        fclose(f);
    }
}

__host__ void dump_logits_raw(const char* base_path, int generation, float* h_logits, int num_classes, const char* manifest) {
    char filename[PATH_BUFFER_SIZE];
    snprintf(filename, sizeof(filename), "%s/logits_%04d.raw", base_path, generation);
    dump_raw_buffer(filename, h_logits, num_classes * sizeof(float), manifest);
}

__host__ void dump_label_raw(const char* base_path, int generation, int label, const char* manifest) {
    char filename[PATH_BUFFER_SIZE];
    snprintf(filename, sizeof(filename), "%s/label_%04d.raw", base_path, generation);
    dump_raw_buffer(filename, &label, sizeof(int), manifest);
}

__host__ void dump_gradients_raw(const char* base_path, int generation, float* h_gradients, size_t num_params, const char* manifest) {
    char filename[PATH_BUFFER_SIZE];
    snprintf(filename, sizeof(filename), "%s/gradients_%04d.raw", base_path, generation);
    dump_raw_buffer(filename, h_gradients, num_params * sizeof(float), manifest);
}

__host__ void dump_rd_fields_raw(const char* base_path, int generation, float* h_u_field, float* h_v_field, int grid_size, const char* manifest) {
    char u_file[PATH_BUFFER_SIZE], v_file[PATH_BUFFER_SIZE];
    size_t field_size = grid_size * grid_size * sizeof(float);

    snprintf(u_file, sizeof(u_file), "%s/rd_u_%04d.raw", base_path, generation);
    snprintf(v_file, sizeof(v_file), "%s/rd_v_%04d.raw", base_path, generation);

    dump_raw_buffer(u_file, h_u_field, field_size, manifest);
    dump_raw_buffer(v_file, h_v_field, field_size, manifest);
}

__host__ void dump_resource_flow_raw(const char* base_path, int generation, float* h_resource_density, float* h_fitness_landscape, int grid_size, const char* manifest) {
    char resource_file[PATH_BUFFER_SIZE], fitness_file[PATH_BUFFER_SIZE];
    size_t field_size = grid_size * grid_size * sizeof(float);

    snprintf(resource_file, sizeof(resource_file), "%s/resources_%04d.raw", base_path, generation);
    snprintf(fitness_file, sizeof(fitness_file), "%s/fitness_%04d.raw", base_path, generation);

    dump_raw_buffer(resource_file, h_resource_density, field_size, manifest);
    dump_raw_buffer(fitness_file, h_fitness_landscape, field_size, manifest);
}

__host__ void dump_lifecycle_histogram_raw(const char* base_path, int generation, int* h_phase_counts, const char* manifest) {
    char filename[PATH_BUFFER_SIZE];
    snprintf(filename, sizeof(filename), "%s/lifecycle_%04d.raw", base_path, generation);
    dump_raw_buffer(filename, h_phase_counts, 4 * sizeof(int), manifest);
}

__host__ void dump_performance_metrics_raw(const char* base_path, int generation, float* h_metrics, int num_metrics, const char* manifest) {
    char filename[PATH_BUFFER_SIZE];
    snprintf(filename, sizeof(filename), "%s/perf_%04d.raw", base_path, generation);
    dump_raw_buffer(filename, h_metrics, num_metrics * sizeof(float), manifest);
}

#endif
