
#ifndef DATASET_LOADER_CU
#define DATASET_LOADER_CU

#include "../config/config.cu"
#include "../training/training_types.cu"
#include "../utils/cuda_primitives.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cufft.h>

enum DatasetFormat {
    FORMAT_IDX_UBYTE,
    FORMAT_NPZ,
    FORMAT_CIFAR_BIN,
    FORMAT_WAV_METADATA,
    FORMAT_WAV_DIRS,
    FORMAT_TXT_TIMESERIES,
    FORMAT_WFDB,
    FORMAT_DAT_BINARY,
    FORMAT_TARGZ_IMAGES
};

enum FeatureEncoding {
    ENCODING_SPATIAL_2D,        
    ENCODING_SPECTRAL_AUDIO,    
    ENCODING_TEMPORAL_1D        
};

enum DatasetModality {
    MODALITY_VISION,
    MODALITY_AUDIO,
    MODALITY_TIMESERIES,
    MODALITY_MEDICAL
};

struct DatasetDescriptor {
    const char* name;
    DatasetFormat format;
    DatasetModality modality;
    FeatureEncoding encoding;
    const char* base_path;

    size_t sample_rows;
    size_t sample_cols;
    size_t channels;
    size_t num_classes;

    size_t num_train;
    size_t num_test;

    size_t train_size_bytes;
    size_t test_size_bytes;

    bool has_separate_test;
    bool needs_preprocessing;

    int n_fft;
    int hop_length;
    int n_mels;

    bool preserve_stereo;
    int bit_depth;
    bool use_multi_resolution;
    int pyramid_levels;
    int hilbert_order;
};

#define DATASET_REGISTRY_INIT { \
    { \
        "MNIST", \
        FORMAT_IDX_UBYTE, \
        MODALITY_VISION, \
        ENCODING_SPATIAL_2D, \
        "../data/vision/mnist", \
        MNIST_ROWS, MNIST_COLS, MNIST_CHANNELS, MNIST_CLASSES, \
        MNIST_TRAIN_SAMPLES, MNIST_TEST_SAMPLES, \
        SafeSize::vision_2d(MNIST_TRAIN_SAMPLES, MNIST_ROWS, MNIST_COLS), SafeSize::vision_2d(MNIST_TEST_SAMPLES, MNIST_ROWS, MNIST_COLS), \
        true, false, \
        0, 0, 0, \
        false, BIT_DEPTH_8, false, 0, 0 \
    }, \
    { \
        "Fashion-MNIST", \
        FORMAT_IDX_UBYTE, \
        MODALITY_VISION, \
        ENCODING_SPATIAL_2D, \
        "../data/vision/fashion-mnist", \
        FASHION_MNIST_ROWS, FASHION_MNIST_COLS, FASHION_MNIST_CHANNELS, FASHION_MNIST_CLASSES, \
        FASHION_MNIST_TRAIN_SAMPLES, FASHION_MNIST_TEST_SAMPLES, \
        SafeSize::vision_2d(FASHION_MNIST_TRAIN_SAMPLES, FASHION_MNIST_ROWS, FASHION_MNIST_COLS), SafeSize::vision_2d(FASHION_MNIST_TEST_SAMPLES, FASHION_MNIST_ROWS, FASHION_MNIST_COLS), \
        true, false, \
        0, 0, 0, \
        false, BIT_DEPTH_8, false, 0, 0 \
    }, \
    { \
        "CIFAR-10", \
        FORMAT_CIFAR_BIN, \
        MODALITY_VISION, \
        ENCODING_SPATIAL_2D, \
        "../data/vision/cifar-10/cifar-10-batches-bin", \
        CIFAR10_ROWS, CIFAR10_COLS, CIFAR10_CHANNELS, CIFAR10_CLASSES, \
        CIFAR10_TRAIN_SAMPLES, CIFAR10_TEST_SAMPLES, \
        SafeSize::vision_3d(CIFAR10_TRAIN_SAMPLES, CIFAR10_ROWS, CIFAR10_COLS, CIFAR10_CHANNELS), SafeSize::vision_3d(CIFAR10_TEST_SAMPLES, CIFAR10_ROWS, CIFAR10_COLS, CIFAR10_CHANNELS), \
        true, false, \
        0, 0, 0, \
        false, BIT_DEPTH_8, false, 0, 0 \
    }, \
    { \
        "PathMNIST", \
        FORMAT_NPZ, \
        MODALITY_MEDICAL, \
        ENCODING_SPATIAL_2D, \
        "../data/vision/pathmnist", \
        PATHMNIST_ROWS, PATHMNIST_COLS, PATHMNIST_CHANNELS, PATHMNIST_CLASSES, \
        PATHMNIST_TRAIN_SAMPLES, PATHMNIST_TEST_SAMPLES, \
        SafeSize::vision_3d(PATHMNIST_TRAIN_SAMPLES, PATHMNIST_ROWS, PATHMNIST_COLS, PATHMNIST_CHANNELS), SafeSize::vision_3d(PATHMNIST_TEST_SAMPLES, PATHMNIST_ROWS, PATHMNIST_COLS, PATHMNIST_CHANNELS), \
        true, false, \
        0, 0, 0, \
        false, BIT_DEPTH_8, false, 0, 0 \
    }, \
    { \
        "ESC-50", \
        FORMAT_WAV_METADATA, \
        MODALITY_AUDIO, \
        ENCODING_SPECTRAL_AUDIO, \
        "../data/audio/esc-50/ESC-50-master", \
        AUDIO_TIME_LONG, AUDIO_N_MELS, AUDIO_SPEC_CHANNELS, ESC50_CLASSES, \
        ESC50_TRAIN_SAMPLES, ESC50_TEST_SAMPLES, \
        SafeSize::audio_spectral(ESC50_TRAIN_SAMPLES, AUDIO_TIME_LONG, AUDIO_N_MELS, AUDIO_SPEC_CHANNELS), SafeSize::audio_spectral(ESC50_TEST_SAMPLES, AUDIO_TIME_LONG, AUDIO_N_MELS, AUDIO_SPEC_CHANNELS), \
        false, true, \
        AUDIO_N_FFT_LARGE, AUDIO_HOP_MEDIUM, AUDIO_N_MELS, \
        false, BIT_DEPTH_16, false, 0, 0 \
    }, \
    { \
        "Speech-Commands", \
        FORMAT_WAV_DIRS, \
        MODALITY_AUDIO, \
        ENCODING_SPECTRAL_AUDIO, \
        "../data/audio/speech-commands", \
        AUDIO_TIME_SHORT, AUDIO_N_MELS, AUDIO_SPEC_CHANNELS, SPEECH_COMMANDS_CLASSES, \
        SPEECH_COMMANDS_TRAIN_SAMPLES, SPEECH_COMMANDS_TEST_SAMPLES, \
        SafeSize::audio_spectral(SPEECH_COMMANDS_TRAIN_SAMPLES, AUDIO_TIME_SHORT, AUDIO_N_MELS, AUDIO_SPEC_CHANNELS), SafeSize::audio_spectral(SPEECH_COMMANDS_TEST_SAMPLES, AUDIO_TIME_SHORT, AUDIO_N_MELS, AUDIO_SPEC_CHANNELS), \
        false, false, \
        AUDIO_N_FFT_SMALL, AUDIO_HOP_SMALL, AUDIO_N_MELS, \
        false, BIT_DEPTH_16, false, 0, 0 \
    }, \
    { \
        "UrbanSound8K", \
        FORMAT_WAV_METADATA, \
        MODALITY_AUDIO, \
        ENCODING_SPECTRAL_AUDIO, \
        "../data/audio/urbansound8k/UrbanSound8K", \
        AUDIO_TIME_MEDIUM, AUDIO_N_MELS, AUDIO_SPEC_CHANNELS, URBANSOUND8K_CLASSES, \
        URBANSOUND8K_TRAIN_SAMPLES, URBANSOUND8K_TEST_SAMPLES, \
        SafeSize::audio_spectral(URBANSOUND8K_TRAIN_SAMPLES, AUDIO_TIME_MEDIUM, AUDIO_N_MELS, AUDIO_SPEC_CHANNELS), SafeSize::audio_spectral(URBANSOUND8K_TEST_SAMPLES, AUDIO_TIME_MEDIUM, AUDIO_N_MELS, AUDIO_SPEC_CHANNELS), \
        false, true, \
        AUDIO_N_FFT_LARGE, AUDIO_HOP_LARGE, AUDIO_N_MELS, \
        false, BIT_DEPTH_16, false, 0, 0 \
    }, \
    { \
        "UCI-HAR", \
        FORMAT_TXT_TIMESERIES, \
        MODALITY_TIMESERIES, \
        ENCODING_TEMPORAL_1D, \
        "../data/timeseries/uci-har/UCI HAR Dataset", \
        UCIHAR_TIMESTEPS, UCIHAR_FEATURES, UCIHAR_CHANNELS, UCIHAR_CLASSES, \
        UCIHAR_TRAIN_SAMPLES, UCIHAR_TEST_SAMPLES, \
        SafeSize::timeseries_multi(UCIHAR_TRAIN_SAMPLES, UCIHAR_TIMESTEPS, UCIHAR_FEATURES), SafeSize::timeseries_multi(UCIHAR_TEST_SAMPLES, UCIHAR_TIMESTEPS, UCIHAR_FEATURES), \
        true, false, \
        0, 0, 0, \
        false, BIT_DEPTH_16, false, 0, 0 \
    }, \
    { \
        "MIT-BIH-ECG", \
        FORMAT_WFDB, \
        MODALITY_TIMESERIES, \
        ENCODING_TEMPORAL_1D, \
        "../data/timeseries/ecg-heartbeat", \
        MITBIH_TIMESTEPS, MITBIH_CHANNELS, MITBIH_CHANNELS, MITBIH_CLASSES, \
        MITBIH_TRAIN_SAMPLES, MITBIH_TEST_SAMPLES, \
        SafeSize::timeseries_single(MITBIH_TRAIN_SAMPLES, MITBIH_TIMESTEPS), SafeSize::timeseries_single(MITBIH_TEST_SAMPLES, MITBIH_TIMESTEPS), \
        true, false, \
        0, 0, 0, \
        false, BIT_DEPTH_16, false, 0, 0 \
    }, \
    { \
        "OPPORTUNITY", \
        FORMAT_DAT_BINARY, \
        MODALITY_TIMESERIES, \
        ENCODING_TEMPORAL_1D, \
        "../data/timeseries/gesture-recognition/OpportunityUCIDataset/dataset", \
        OPPORTUNITY_TIMESTEPS, OPPORTUNITY_FEATURES, OPPORTUNITY_CHANNELS, OPPORTUNITY_CLASSES, \
        OPPORTUNITY_TRAIN_SAMPLES, OPPORTUNITY_TEST_SAMPLES, \
        SafeSize::timeseries_multi(OPPORTUNITY_TRAIN_SAMPLES, OPPORTUNITY_TIMESTEPS, OPPORTUNITY_FEATURES), SafeSize::timeseries_multi(OPPORTUNITY_TEST_SAMPLES, OPPORTUNITY_TIMESTEPS, OPPORTUNITY_FEATURES), \
        true, false, \
        0, 0, 0, \
        false, BIT_DEPTH_16, false, 0, 0 \
    }, \
    { \
        "ChestX-ray14", \
        FORMAT_TARGZ_IMAGES, \
        MODALITY_MEDICAL, \
        ENCODING_SPATIAL_2D, \
        "../data/medical/chestxray14", \
        CHESTXRAY_ROWS, CHESTXRAY_COLS, CHESTXRAY_CHANNELS, CHESTXRAY_CLASSES, \
        CHESTXRAY_TRAIN_SAMPLES, CHESTXRAY_TEST_SAMPLES, \
        SafeSize::vision_2d(CHESTXRAY_TRAIN_SAMPLES, CHESTXRAY_ROWS, CHESTXRAY_COLS), SafeSize::vision_2d(CHESTXRAY_TEST_SAMPLES, CHESTXRAY_ROWS, CHESTXRAY_COLS), \
        true, false, \
        0, 0, 0, \
        false, BIT_DEPTH_16, true, CHESTXRAY_PYRAMID_LEVELS, CHESTXRAY_HILBERT_ORDER \
    }, \
    { \
        "RetinaMNIST", \
        FORMAT_NPZ, \
        MODALITY_MEDICAL, \
        ENCODING_SPATIAL_2D, \
        "../data/medical/retinamnist", \
        RETINAMNIST_ROWS, RETINAMNIST_COLS, RETINAMNIST_CHANNELS, RETINAMNIST_CLASSES, \
        RETINAMNIST_TRAIN_SAMPLES, RETINAMNIST_TEST_SAMPLES, \
        SafeSize::vision_3d(RETINAMNIST_TRAIN_SAMPLES, RETINAMNIST_ROWS, RETINAMNIST_COLS, RETINAMNIST_CHANNELS), SafeSize::vision_3d(RETINAMNIST_TEST_SAMPLES, RETINAMNIST_ROWS, RETINAMNIST_COLS, RETINAMNIST_CHANNELS), \
        true, false, \
        0, 0, 0, \
        false, BIT_DEPTH_8, false, 0, 0 \
    } \
}

__device__ __constant__ DatasetDescriptor DATASET_REGISTRY[NUM_DATASETS] = DATASET_REGISTRY_INIT;

static const DatasetDescriptor HOST_DATASET_REGISTRY[NUM_DATASETS] = DATASET_REGISTRY_INIT;


__device__ void hilbert_index_to_xy(int index, int order, int* x, int* y) {
    int n = 1 << order;
    *x = 0;
    *y = 0;
    for (int s = 1; s < n; s *= 2) {
        int rx = 1 & (index / 2);
        int ry = 1 & (index ^ rx);
        if (ry == 0) {
            if (rx == 1) {
                *x = s - 1 - *x;
                *y = s - 1 - *y;
            }
            int t = *x;
            *x = *y;
            *y = t;
        }
        *x += s * rx;
        *y += s * ry;
        index /= 4;
    }
}

__device__ int hilbert_xy_to_index(int x, int y, int order) {
    int n = 1 << order;
    int index = 0;
    for (int s = n / 2; s > 0; s /= 2) {
        int rx = (x & s) > 0;
        int ry = (y & s) > 0;
        index += s * s * ((3 * rx) ^ ry);
        if (ry == 0) {
            if (rx == 1) {
                x = n - 1 - x;
                y = n - 1 - y;
            }
            int t = x;
            x = y;
            y = t;
        }
    }
    return index;
}

__global__ void apply_window_kernel(
    const float* waveform,
    float* windowed,
    int window_start,
    int window_size
) {
    int t = threadIdx.x + blockIdx.x * blockDim.x;
    if (t >= window_size) return;

    float hann = 0.5f * (1.0f - __cosf(TAU * t / (window_size - 1)));
    windowed[t] = waveform[window_start + t] * hann;
}

__global__ void extract_magnitude_phase_kernel(
    const cufftComplex* fft_out,
    float* magnitude,
    float* phase,
    int n_bins
) {
    int bin = threadIdx.x + blockIdx.x * blockDim.x;
    if (bin >= n_bins) return;

    float real = fft_out[bin].x;
    float imag = fft_out[bin].y;

    magnitude[bin] = sqrtf(real * real + imag * imag);
    phase[bin] = atan2f(imag, real);
}

__global__ void compute_phase_velocity_kernel(
    const float* phase_curr,
    const float* phase_prev,
    float* phase_velocity,
    int n_bins,
    float hop_length,
    float sample_rate
) {
    int bin = threadIdx.x + blockIdx.x * blockDim.x;
    if (bin >= n_bins) return;

    float phase_diff = phase_curr[bin] - phase_prev[bin];

    if (isnan(phase_diff) || isinf(phase_diff)) {
        printf("FATAL [phase_velocity]: phase_diff=%f bin=%d phase_curr=%f phase_prev=%f\n",
               phase_diff, bin, phase_curr[bin], phase_prev[bin]);
        return;
    }

    while (phase_diff > TAU * 0.5f) phase_diff -= TAU;
    while (phase_diff < -TAU * 0.5f) phase_diff += TAU;

    if (hop_length <= 0.0f) {
        printf("FATAL [phase_velocity]: hop_length=%f\n", hop_length);
        return;
    }
    phase_velocity[bin] = phase_diff * sample_rate / hop_length;
}

__global__ void mel_filterbank_kernel(
    const float* magnitude,
    const float* phase,
    const float* phase_velocity,
    float* mel_magnitude,
    float* mel_phase,
    float* mel_phase_velocity,
    int n_bins,
    int n_mels,
    int sample_rate,
    int n_fft
) {
    int mel_bin = blockIdx.x;
    if (mel_bin >= n_mels) return;

    float mel_low = 0.0f;
    float mel_high = 2595.0f * log10f(1.0f + (sample_rate / 2.0f) / 700.0f);
    float mel_bin_width = (mel_high - mel_low) / (n_mels + 1);

    float mel_center = mel_low + (mel_bin + 1) * mel_bin_width;
    float mel_left = mel_center - mel_bin_width;
    float mel_right = mel_center + mel_bin_width;

    float freq_center = 700.0f * (powf(10.0f, mel_center / 2595.0f) - 1.0f);
    float freq_left = 700.0f * (powf(10.0f, mel_left / 2595.0f) - 1.0f);
    float freq_right = 700.0f * (powf(10.0f, mel_right / 2595.0f) - 1.0f);

    float weighted_mag = 0.0f;
    float weighted_phase_real = 0.0f;
    float weighted_phase_imag = 0.0f;
    float weighted_pv = 0.0f;
    float total_weight = 0.0f;

    for (int freq_bin = threadIdx.x; freq_bin < n_bins; freq_bin += blockDim.x) {
        float freq = (float)freq_bin * sample_rate / n_fft;

        float weight = 0.0f;
        if (freq >= freq_left && freq <= freq_center) {
            weight = (freq - freq_left) / (freq_center - freq_left);
        } else if (freq > freq_center && freq <= freq_right) {
            weight = (freq_right - freq) / (freq_right - freq_center);
        }

        if (weight > 0.0f) {
            weighted_mag += weight * magnitude[freq_bin];
            weighted_phase_real += weight * __cosf(phase[freq_bin]);
            weighted_phase_imag += weight * __sinf(phase[freq_bin]);
            weighted_pv += weight * phase_velocity[freq_bin];
            total_weight += weight;
        }
    }

    weighted_mag = warp_reduce_sum(weighted_mag);
    weighted_phase_real = warp_reduce_sum(weighted_phase_real);
    weighted_phase_imag = warp_reduce_sum(weighted_phase_imag);
    weighted_pv = warp_reduce_sum(weighted_pv);
    total_weight = warp_reduce_sum(total_weight);

    if (threadIdx.x == 0) {
        mel_magnitude[mel_bin] = log10f(fmaxf(weighted_mag, safe_epsilon(1.0f)));
        mel_phase[mel_bin] = atan2f(weighted_phase_imag, weighted_phase_real);
        mel_phase_velocity[mel_bin] = safe_div(weighted_pv, total_weight);
    }
}

extern "C" __global__ void sample_batch_kernel(
    Dataset* dataset,
    HybridTrainingMode* training,
    int batch_size,
    int offset,
    int grid_size
) {
    int idx = blockIdx.x;
    if (idx >= batch_size) return;

    unsigned char* all_images = dataset->samples;
    unsigned char* all_labels = dataset->labels;
    float* batch_images = training->batch_images;
    int* batch_labels = training->batch_labels;
    int dataset_size = dataset->num_samples;

    int src_idx = (offset + idx) % dataset_size;

    batch_labels[idx] = all_labels[src_idx];

    int tid = threadIdx.x;
    int pixels_per_thread = (grid_size * grid_size + blockDim.x - 1) / blockDim.x;

    int sample_rows = dataset->descriptor->sample_rows;
    int sample_cols = dataset->descriptor->sample_cols;
    int src_channels = dataset->descriptor->channels;
    int sample_size = sample_rows * sample_cols * src_channels;

    for (int p = 0; p < pixels_per_thread; p++) {
        int pixel_idx = tid * pixels_per_thread + p;
        if (pixel_idx >= grid_size * grid_size) break;

        int out_y = pixel_idx / grid_size;
        int out_x = pixel_idx % grid_size;

        float src_y = out_y * (float)sample_rows / grid_size;
        float src_x = out_x * (float)sample_cols / grid_size;

        int y0 = (int)src_y;
        int x0 = (int)src_x;
        int y1 = min(y0 + 1, sample_rows - 1);
        int x1 = min(x0 + 1, sample_cols - 1);

        float fy = src_y - y0;
        float fx = src_x - x0;

        int batch_stride = grid_size * grid_size;

        if (src_channels >= 3) {
            for (int c = 0; c < 3; c++) {
                int channel_offset = c * sample_rows * sample_cols;
                float tl = all_images[src_idx * sample_size + channel_offset + y0 * sample_cols + x0] / (float)UINT8_MAX;
                float tr = all_images[src_idx * sample_size + channel_offset + y0 * sample_cols + x1] / (float)UINT8_MAX;
                float bl = all_images[src_idx * sample_size + channel_offset + y1 * sample_cols + x0] / (float)UINT8_MAX;
                float br = all_images[src_idx * sample_size + channel_offset + y1 * sample_cols + x1] / (float)UINT8_MAX;
                batch_images[idx * batch_stride * 3 + c * batch_stride + pixel_idx] = Interpolation::bilinear(tl, tr, bl, br, fx, fy);
            }
        } else {
            float tl = all_images[src_idx * sample_size + y0 * sample_cols + x0] / (float)UINT8_MAX;
            float tr = all_images[src_idx * sample_size + y0 * sample_cols + x1] / (float)UINT8_MAX;
            float bl = all_images[src_idx * sample_size + y1 * sample_cols + x0] / (float)UINT8_MAX;
            float br = all_images[src_idx * sample_size + y1 * sample_cols + x1] / (float)UINT8_MAX;
            float3 vg = Interpolation::bilinear_with_grad(tl, tr, bl, br, fx, fy);
            batch_images[idx * batch_stride * 3 + 0 * batch_stride + pixel_idx] = vg.x;
            batch_images[idx * batch_stride * 3 + 1 * batch_stride + pixel_idx] = vg.y;
            batch_images[idx * batch_stride * 3 + 2 * batch_stride + pixel_idx] = vg.z;
        }
    }
}


__global__ void inject_sample_to_ca_kernel(
    float* ca_state,
    int batch_size,
    int channels,
    int grid_size,
    float* chem_concentration,
    float* chem_gradient_x,
    float* chem_gradient_y,
    float* chem_laplacian,
    float* chem_sources,
    float* chem_decay_factors,
    float* rd_resource_density,
    float* rd_fitness_landscape,
    float* rd_resource_gradient_x,
    float* rd_resource_gradient_y,
    float* behavioral_field,
    float* batch_images,
    int image_channels,
    float* prev_concentration,
    float* attractor_field
) {
    int batch_idx = blockIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || y >= grid_size || x >= grid_size) return;

    if (ca_state == nullptr) {
        if (threadIdx.x == 0 && threadIdx.y == 0 && batch_idx == 0) {
            printf("FATAL [inject_sample_to_ca]: ca_state is NULL\n");
        }
        return;
    }

    int spatial_idx = y * grid_size + x;
    int base_idx = batch_idx * grid_size * grid_size * channels + spatial_idx * channels;

    ca_state[base_idx + 0] = chem_concentration[spatial_idx];
    ca_state[base_idx + 1] = chem_gradient_x[spatial_idx];
    ca_state[base_idx + 2] = chem_gradient_y[spatial_idx];
    ca_state[base_idx + 3] = chem_laplacian[spatial_idx];
    ca_state[base_idx + 4] = chem_sources[spatial_idx];
    ca_state[base_idx + 5] = chem_decay_factors[spatial_idx];

    ca_state[base_idx + 6] = rd_resource_density[spatial_idx];
    ca_state[base_idx + 7] = rd_fitness_landscape[spatial_idx];
    ca_state[base_idx + 8] = rd_resource_gradient_x[spatial_idx];
    ca_state[base_idx + 9] = rd_resource_gradient_y[spatial_idx];

    ca_state[base_idx + 10] = behavioral_field[spatial_idx];

    int batch_stride = grid_size * grid_size;
    int img_base = batch_idx * batch_stride * 3;
    ca_state[base_idx + 11] = batch_images[img_base + 0 * batch_stride + spatial_idx];
    ca_state[base_idx + 12] = batch_images[img_base + 1 * batch_stride + spatial_idx];
    ca_state[base_idx + 13] = batch_images[img_base + 2 * batch_stride + spatial_idx];

    int prev_idx = batch_idx * grid_size * grid_size * channels + spatial_idx * channels;
    ca_state[base_idx + 14] = prev_concentration[prev_idx + 0];

    ca_state[base_idx + 15] = attractor_field[spatial_idx];
}

__host__ bool read_binary_blob(const char* path, void* buffer, size_t size, size_t skip_bytes) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[read_binary_blob] ERROR: Failed to open file: %s\n", path);
        return false;
    }
    if (skip_bytes > 0) fseek(f, skip_bytes, SEEK_SET);
    size_t read = fread(buffer, 1, size, f);
    fclose(f);
    if (read != size) {
        fprintf(stderr, "[read_binary_blob] ERROR: Read %zu bytes but expected %zu from %s\n", read, size, path);
    }
    return read == size;
}


__host__ cudaError_t load_dataset_from_registry(
    int dataset_id,
    bool is_train,
    Dataset** out_dataset
) {
    if (dataset_id < 0 || dataset_id >= NUM_DATASETS) {
        fprintf(stderr, "[load_dataset] Invalid dataset_id=%d\n", dataset_id);
        return cudaErrorInvalidValue;
    }

    const DatasetDescriptor* h_descriptor = &HOST_DATASET_REGISTRY[dataset_id];

    printf("[load_dataset] Loading %s (%s split)\n", h_descriptor->name, is_train ? "train" : "test");

    Dataset* h_dataset = (Dataset*)malloc(sizeof(Dataset));
    if (!h_dataset) {
        fprintf(stderr, "[load_dataset] FATAL: malloc failed for Dataset\n");
        return cudaErrorMemoryAllocation;
    }
    h_dataset->is_train = is_train;
    h_dataset->num_samples = is_train ? h_descriptor->num_train : h_descriptor->num_test;

    size_t data_size = is_train ? h_descriptor->train_size_bytes : h_descriptor->test_size_bytes;

    unsigned char* h_samples = (unsigned char*)malloc(data_size);
    if (!h_samples) {
        fprintf(stderr, "[load_dataset] FATAL: malloc failed for samples (%zu bytes)\n", data_size);
        free(h_dataset);
        return cudaErrorMemoryAllocation;
    }
    unsigned char* h_labels = (unsigned char*)malloc(h_dataset->num_samples);
    if (!h_labels) {
        fprintf(stderr, "[load_dataset] FATAL: malloc failed for labels (%d bytes)\n", h_dataset->num_samples);
        free(h_samples); free(h_dataset);
        return cudaErrorMemoryAllocation;
    }

    if (h_descriptor->format == FORMAT_IDX_UBYTE) {
        char img_path[512], lbl_path[512];
        snprintf(img_path, sizeof(img_path), "%s/%s-images-idx3-ubyte", h_descriptor->base_path, is_train ? "train" : "t10k");
        snprintf(lbl_path, sizeof(lbl_path), "%s/%s-labels-idx1-ubyte", h_descriptor->base_path, is_train ? "train" : "t10k");
        printf("H:load_dataset img_path=%s lbl_path=%s\n", img_path, lbl_path);
        fflush(stdout);

        if (!read_binary_blob(img_path, h_samples, data_size, 16) ||
            !read_binary_blob(lbl_path, h_labels, h_dataset->num_samples, 8)) {
            fprintf(stderr, "[load_dataset] Failed to read IDX files\n");
            free(h_samples); free(h_labels); free(h_dataset);
            return cudaErrorUnknown;
        }
    }
    else if (h_descriptor->format == FORMAT_CIFAR_BIN) {
        unsigned char* temp_batch = (unsigned char*)malloc(10000 * 3073);
        if (!temp_batch) {
            fprintf(stderr, "[load_dataset] FATAL: malloc failed for temp_batch (%d bytes)\n", 10000 * 3073);
            free(h_samples); free(h_labels); free(h_dataset);
            return cudaErrorMemoryAllocation;
        }

        if (is_train) {
            for (int batch = 0; batch < 5; batch++) {
                char path[512];
                snprintf(path, sizeof(path), "%s/data_batch_%d.bin", h_descriptor->base_path, batch + 1);

                if (!read_binary_blob(path, temp_batch, 10000 * 3073, 0)) {
                    fprintf(stderr, "[load_dataset] Failed to read CIFAR batch %d\n", batch + 1);
                    free(temp_batch); free(h_samples); free(h_labels); free(h_dataset);
                    return cudaErrorUnknown;
                }

                for (int i = 0; i < 10000; i++) {
                    int idx = batch * 10000 + i;
                    h_labels[idx] = temp_batch[i * 3073];
                    memcpy(&h_samples[idx * 3072], &temp_batch[i * 3073 + 1], 3072);
                }
            }
        } else {
            char path[512];
            snprintf(path, sizeof(path), "%s/test_batch.bin", h_descriptor->base_path);

            if (!read_binary_blob(path, temp_batch, 10000 * 3073, 0)) {
                fprintf(stderr, "[load_dataset] Failed to read CIFAR test batch\n");
                free(temp_batch); free(h_samples); free(h_labels); free(h_dataset);
                return cudaErrorUnknown;
            }

            for (int i = 0; i < 10000; i++) {
                h_labels[i] = temp_batch[i * 3073];
                memcpy(&h_samples[i * 3072], &temp_batch[i * 3073 + 1], 3072);
            }
        }

        free(temp_batch);
    }
    else if (h_descriptor->format == FORMAT_NPZ) {
        char samples_path[512], labels_path[512];
        snprintf(samples_path, sizeof(samples_path), "%s/%s-images.bin", h_descriptor->base_path, is_train ? "train" : "test");
        snprintf(labels_path, sizeof(labels_path), "%s/%s-labels.bin", h_descriptor->base_path, is_train ? "train" : "test");

        if (!read_binary_blob(samples_path, h_samples, data_size, 0) ||
            !read_binary_blob(labels_path, h_labels, h_dataset->num_samples, 0)) {
            fprintf(stderr, "[load_dataset] Failed to read pre-extracted binary files: %s, %s\n", samples_path, labels_path);
            free(h_samples); free(h_labels); free(h_dataset);
            return cudaErrorUnknown;
        }
    }
    else if (h_descriptor->format == FORMAT_WAV_METADATA || h_descriptor->format == FORMAT_WAV_DIRS) {
        char samples_path[512], labels_path[512];
        snprintf(samples_path, sizeof(samples_path), "%s/%s-spectrograms.bin", h_descriptor->base_path, is_train ? "train" : "test");
        snprintf(labels_path, sizeof(labels_path), "%s/%s-labels.bin", h_descriptor->base_path, is_train ? "train" : "test");

        if (!read_binary_blob(samples_path, h_samples, data_size, 0) ||
            !read_binary_blob(labels_path, h_labels, h_dataset->num_samples, 0)) {
            fprintf(stderr, "[load_dataset] Failed to read binary files: %s, %s\n", samples_path, labels_path);
            free(h_samples); free(h_labels); free(h_dataset);
            return cudaErrorUnknown;
        }
    }
    else if (h_descriptor->format == FORMAT_TXT_TIMESERIES) {
        char samples_path[512], labels_path[512];
        snprintf(samples_path, sizeof(samples_path), "%s/%s-samples.bin", h_descriptor->base_path, is_train ? "train" : "test");
        snprintf(labels_path, sizeof(labels_path), "%s/%s-labels.bin", h_descriptor->base_path, is_train ? "train" : "test");

        if (!read_binary_blob(samples_path, h_samples, data_size, 0) ||
            !read_binary_blob(labels_path, h_labels, h_dataset->num_samples, 0)) {
            fprintf(stderr, "[load_dataset] Failed to read binary files: %s, %s\n", samples_path, labels_path);
            free(h_samples); free(h_labels); free(h_dataset);
            return cudaErrorUnknown;
        }
    }
    else if (h_descriptor->format == FORMAT_WFDB || h_descriptor->format == FORMAT_DAT_BINARY || h_descriptor->format == FORMAT_TARGZ_IMAGES) {
        char samples_path[512], labels_path[512];
        snprintf(samples_path, sizeof(samples_path), "%s/%s-samples.bin", h_descriptor->base_path, is_train ? "train" : "test");
        snprintf(labels_path, sizeof(labels_path), "%s/%s-labels.bin", h_descriptor->base_path, is_train ? "train" : "test");

        if (!read_binary_blob(samples_path, h_samples, data_size, 0) ||
            !read_binary_blob(labels_path, h_labels, h_dataset->num_samples, 0)) {
            fprintf(stderr, "[load_dataset] Failed to read binary files: %s, %s\n", samples_path, labels_path);
            free(h_samples); free(h_labels); free(h_dataset);
            return cudaErrorUnknown;
        }
    }
    else {
        fprintf(stderr, "[load_dataset] Unknown format %d\n", h_descriptor->format);
        free(h_samples); free(h_labels); free(h_dataset);
        return cudaErrorNotSupported;
    }

    cudaError_t err = cudaMalloc(&h_dataset->samples, data_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "[load_dataset] FATAL: samples cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_samples); free(h_labels); free(h_dataset);
        return err;
    }
    err = cudaMalloc(&h_dataset->labels, h_dataset->num_samples);
    if (err != cudaSuccess) {
        fprintf(stderr, "[load_dataset] FATAL: labels cudaMalloc failed: %s\n", cudaGetErrorString(err));
        cudaFree(h_dataset->samples);
        free(h_samples); free(h_labels); free(h_dataset);
        return err;
    }
    err = cudaMemcpy(h_dataset->samples, h_samples, data_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[load_dataset] FATAL: samples cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(h_dataset->samples);
        cudaFree(h_dataset->labels);
        free(h_samples); free(h_labels); free(h_dataset);
        return err;
    }
    err = cudaMemcpy(h_dataset->labels, h_labels, h_dataset->num_samples, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[load_dataset] FATAL: labels cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(h_dataset->samples);
        cudaFree(h_dataset->labels);
        free(h_samples); free(h_labels); free(h_dataset);
        return err;
    }

    free(h_samples);
    free(h_labels);

    Dataset* d_dataset;
    err = cudaMalloc(&d_dataset, sizeof(Dataset));
    if (err != cudaSuccess) {
        fprintf(stderr, "[load_dataset] FATAL: d_dataset cudaMalloc failed: %s\n", cudaGetErrorString(err));
        cudaFree(h_dataset->samples);
        cudaFree(h_dataset->labels);
        free(h_dataset);
        return err;
    }

    const DatasetDescriptor* d_descriptor_ptr = nullptr;
    err = cudaGetSymbolAddress((void**)&d_descriptor_ptr, DATASET_REGISTRY);
    if (err != cudaSuccess) {
        fprintf(stderr, "[load_dataset] FATAL: cudaGetSymbolAddress failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_dataset);
        cudaFree(h_dataset->samples);
        cudaFree(h_dataset->labels);
        free(h_dataset);
        return err;
    }
    d_descriptor_ptr += dataset_id;
    h_dataset->descriptor = d_descriptor_ptr;

    err = cudaMemcpy(d_dataset, h_dataset, sizeof(Dataset), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[load_dataset] FATAL: d_dataset cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_dataset);
        cudaFree(h_dataset->samples);
        cudaFree(h_dataset->labels);
        free(h_dataset);
        return err;
    }

    *out_dataset = d_dataset;

    printf("[load_dataset] Loaded %d samples, %zu bytes\n", h_dataset->num_samples, data_size);

    free(h_dataset);

    return cudaGetLastError();
}

#endif
