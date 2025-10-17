#include <cuda_runtime.h>
#include <cuda/std/atomic>
#include <cuda/std/complex>
#include <cufft.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>

namespace cg = cooperative_groups;
using complex = cuda::std::complex<float>;

constexpr int FIELD_DIM = 256;
constexpr int SPECTRUM_BANDS = 32;
constexpr int WAVE_MODES = 16;
constexpr int INTERFERENCE_ORDERS = 8;

struct SpectralField {
    complex* fourier_space[SPECTRUM_BANDS];
    float* real_space[SPECTRUM_BANDS];
    float* phase_velocities;
    float* group_velocities;
    float* dispersion_relations;
    float* nonlinear_coupling;
    cufftHandle fft_plan;
    cuda::std::atomic<float> total_energy;
};

struct ChemotaxisField {
    SpectralField* nutrient_spectrum;
    SpectralField* signal_spectrum;
    SpectralField* gradient_spectrum;

    float* behavioral_manifold;
    float* geodesic_distances;
    float* curvature_tensor;
    float* connection_coefficients;

    complex* wave_packets[WAVE_MODES];
    float* interference_pattern;
    float* mode_amplitudes;
    float* mode_phases;

    cuda::std::atomic<float> field_coherence;
    cuda::std::atomic<float> total_flux;
    int timestep;
};

struct WavePacket {
    complex amplitude;
    float k_vector[3];
    float omega;
    float group_velocity[3];
    float spread_rate;
    float nonlinearity;
};

__device__ complex dispersion_relation(float kx, float ky, float kz, int band) {
    float k_mag = sqrtf(kx * kx + ky * ky + kz * kz);

    float omega_linear = k_mag * (1.0f + 0.1f * band);

    float omega_nonlinear = 0.1f * k_mag * k_mag / (1.0f + k_mag);

    float omega_dispersive = sinf(k_mag * 0.5f) / (k_mag + 0.1f);

    float omega = omega_linear + omega_nonlinear * cosf((float)band) + omega_dispersive;

    float damping = expf(-k_mag * k_mag / (100.0f * (1.0f + band)));

    return complex(omega * damping, -0.01f * k_mag * (1.0f + 0.1f * band));
}

__global__ void compute_spectral_decomposition_kernel(ChemotaxisField* field, float* input) {
    cg::grid_group grid = cg::this_grid();

    int tid = grid.thread_rank();
    int x = tid % FIELD_DIM;
    int y = (tid / FIELD_DIM) % FIELD_DIM;
    int z = tid / (FIELD_DIM * FIELD_DIM);

    if (x >= FIELD_DIM || y >= FIELD_DIM || z >= FIELD_DIM) return;

    float value = input[tid];

    __shared__ float band_energies[SPECTRUM_BANDS];
    __shared__ complex band_phases[SPECTRUM_BANDS];

    if (threadIdx.x < SPECTRUM_BANDS) {
        band_energies[threadIdx.x] = 0.0f;
        band_phases[threadIdx.x] = complex(0.0f, 0.0f);
    }
    __syncthreads();

    for (int band = 0; band < SPECTRUM_BANDS; band++) {
        float kx = 2.0f * M_PI * x / FIELD_DIM;
        float ky = 2.0f * M_PI * y / FIELD_DIM;
        float kz = 2.0f * M_PI * z / FIELD_DIM;

        float k_mag = sqrtf(kx * kx + ky * ky + kz * kz);

        float band_center = (float)band * 2.0f * M_PI / SPECTRUM_BANDS;
        float band_width = 2.0f * M_PI / (2.0f * SPECTRUM_BANDS);

        float band_filter = expf(-powf(k_mag - band_center, 2.0f) / (2.0f * band_width * band_width));

        complex phase = complex(cosf(kx * x + ky * y + kz * z),
                               sinf(kx * x + ky * y + kz * z));

        complex spectral_component = complex(value, 0.0f) * phase * band_filter;

        field->nutrient_spectrum->fourier_space[band][tid] = spectral_component;

        atomicAdd(&band_energies[band], abs(spectral_component));

        complex dispersion = dispersion_relation(kx, ky, kz, band);
        band_phases[band] += spectral_component * dispersion;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int b = 0; b < SPECTRUM_BANDS; b++) {
            total += band_energies[b];
        }
        field->nutrient_spectrum->total_energy.store(total);
    }
}

__global__ void wave_packet_propagation_kernel(ChemotaxisField* field, int mode_id) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= FIELD_DIM * FIELD_DIM * FIELD_DIM) return;

    int x = tid % FIELD_DIM;
    int y = (tid / FIELD_DIM) % FIELD_DIM;
    int z = tid / (FIELD_DIM * FIELD_DIM);

    float rx = (float)x - FIELD_DIM / 2.0f;
    float ry = (float)y - FIELD_DIM / 2.0f;
    float rz = (float)z - FIELD_DIM / 2.0f;

    complex* packet = field->wave_packets[mode_id];

    float kx = field->mode_amplitudes[mode_id * 3];
    float ky = field->mode_amplitudes[mode_id * 3 + 1];
    float kz = field->mode_amplitudes[mode_id * 3 + 2];

    float omega = field->mode_phases[mode_id];
    float t = (float)field->timestep * 0.01f;

    float group_vx = kx / (sqrtf(kx * kx + ky * ky + kz * kz) + 0.1f);
    float group_vy = ky / (sqrtf(kx * kx + ky * ky + kz * kz) + 0.1f);
    float group_vz = kz / (sqrtf(kx * kx + ky * ky + kz * kz) + 0.1f);

    float x0 = group_vx * t;
    float y0 = group_vy * t;
    float z0 = group_vz * t;

    float spread = 1.0f + 0.1f * t;

    float envelope = expf(-((rx - x0) * (rx - x0) + (ry - y0) * (ry - y0) + (rz - z0) * (rz - z0)) /
                          (2.0f * spread * spread)) / powf(spread, 1.5f);

    float phase = kx * rx + ky * ry + kz * rz - omega * t;

    complex wave = complex(envelope * cosf(phase), envelope * sinf(phase));

    complex dispersion_factor = dispersion_relation(kx, ky, kz, mode_id % SPECTRUM_BANDS);
    wave *= dispersion_factor;

    packet[tid] = wave;

    complex nonlinear_term = wave * conj(wave) * wave * 0.01f;
    packet[tid] += nonlinear_term;
}

__global__ void compute_interference_pattern_kernel(ChemotaxisField* field) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= FIELD_DIM * FIELD_DIM * FIELD_DIM) return;

    complex total_field = complex(0.0f, 0.0f);

    #pragma unroll
    for (int mode = 0; mode < WAVE_MODES; mode++) {
        complex wave = field->wave_packets[mode][tid];

        float mode_weight = field->mode_amplitudes[mode];
        total_field += wave * mode_weight;
    }

    float intensity = abs(total_field) * abs(total_field);

    field->interference_pattern[tid] = intensity;

    for (int order = 2; order <= INTERFERENCE_ORDERS; order++) {
        complex higher_order = complex(1.0f, 0.0f);
        for (int n = 0; n < order; n++) {
            higher_order *= total_field;
        }

        float bessel_weight = j0f((float)order * abs(total_field));

        field->interference_pattern[tid] += bessel_weight * abs(higher_order) / (float)order;
    }

    int x = tid % FIELD_DIM;
    int y = (tid / FIELD_DIM) % FIELD_DIM;
    int z = tid / (FIELD_DIM * FIELD_DIM);

    if (x > 0 && x < FIELD_DIM - 1 &&
        y > 0 && y < FIELD_DIM - 1 &&
        z > 0 && z < FIELD_DIM - 1) {

        float laplacian = 0.0f;
        laplacian += field->interference_pattern[tid + 1] - 2.0f * field->interference_pattern[tid] +
                    field->interference_pattern[tid - 1];
        laplacian += field->interference_pattern[tid + FIELD_DIM] - 2.0f * field->interference_pattern[tid] +
                    field->interference_pattern[tid - FIELD_DIM];
        laplacian += field->interference_pattern[tid + FIELD_DIM * FIELD_DIM] -
                    2.0f * field->interference_pattern[tid] +
                    field->interference_pattern[tid - FIELD_DIM * FIELD_DIM];

        field->interference_pattern[tid] += 0.01f * laplacian;
    }
}

__global__ void compute_geodesic_flow_kernel(ChemotaxisField* field, float* positions, int n_agents) {
    int agent_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (agent_id >= n_agents) return;

    float x = positions[agent_id * 3];
    float y = positions[agent_id * 3 + 1];
    float z = positions[agent_id * 3 + 2];

    int ix = (int)x;
    int iy = (int)y;
    int iz = (int)z;

    float fx = x - ix;
    float fy = y - iy;
    float fz = z - iz;

    float gradient[3] = {0.0f, 0.0f, 0.0f};

    for (int dz = 0; dz <= 1; dz++) {
        for (int dy = 0; dy <= 1; dy++) {
            for (int dx = 0; dx <= 1; dx++) {
                int nx = clamp(ix + dx, 0, FIELD_DIM - 1);
                int ny = clamp(iy + dy, 0, FIELD_DIM - 1);
                int nz = clamp(iz + dz, 0, FIELD_DIM - 1);

                float weight = (dx ? fx : 1.0f - fx) *
                              (dy ? fy : 1.0f - fy) *
                              (dz ? fz : 1.0f - fz);

                int field_idx = nz * FIELD_DIM * FIELD_DIM + ny * FIELD_DIM + nx;

                if (dx == 1 && ix < FIELD_DIM - 1) {
                    gradient[0] += weight * (field->interference_pattern[field_idx] -
                                            field->interference_pattern[field_idx - 1]);
                }
                if (dy == 1 && iy < FIELD_DIM - 1) {
                    gradient[1] += weight * (field->interference_pattern[field_idx] -
                                            field->interference_pattern[field_idx - FIELD_DIM]);
                }
                if (dz == 1 && iz < FIELD_DIM - 1) {
                    gradient[2] += weight * (field->interference_pattern[field_idx] -
                                            field->interference_pattern[field_idx - FIELD_DIM * FIELD_DIM]);
                }
            }
        }
    }

    int manifold_idx = agent_id * 10;

    float metric_tensor[9];
    for (int i = 0; i < 9; i++) {
        metric_tensor[i] = field->behavioral_manifold[manifold_idx + i];
    }

    float christoffel[27];
    for (int i = 0; i < 27; i++) {
        christoffel[i] = field->connection_coefficients[agent_id * 27 + i];
    }

    float geodesic_acceleration[3] = {0.0f, 0.0f, 0.0f};

    #pragma unroll
    for (int i = 0; i < 3; i++) {
        #pragma unroll
        for (int j = 0; j < 3; j++) {
            #pragma unroll
            for (int k = 0; k < 3; k++) {
                geodesic_acceleration[i] -= christoffel[i * 9 + j * 3 + k] *
                                           gradient[j] * gradient[k];
            }
        }
    }

    float covariant_gradient[3];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        covariant_gradient[i] = 0.0f;
        #pragma unroll
        for (int j = 0; j < 3; j++) {
            covariant_gradient[i] += metric_tensor[i * 3 + j] *
                                    (gradient[j] + geodesic_acceleration[j]);
        }
    }

    float curvature_idx = agent_id * 81;
    float riemann_term[3] = {0.0f, 0.0f, 0.0f};

    #pragma unroll
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    riemann_term[i] += field->curvature_tensor[curvature_idx + i * 27 + j * 9 + k * 3 + l] *
                                      gradient[j] * gradient[k] * gradient[l];
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 3; i++) {
        positions[agent_id * 3 + i] += 0.01f * (covariant_gradient[i] + 0.1f * riemann_term[i]);

        positions[agent_id * 3 + i] = fmodf(positions[agent_id * 3 + i] + FIELD_DIM, FIELD_DIM);
    }

    field->geodesic_distances[agent_id] += sqrtf(covariant_gradient[0] * covariant_gradient[0] +
                                                 covariant_gradient[1] * covariant_gradient[1] +
                                                 covariant_gradient[2] * covariant_gradient[2]) * 0.01f;
}

__global__ void mode_coupling_kernel(ChemotaxisField* field) {
    int mode1 = blockIdx.x;
    int mode2 = threadIdx.x;

    if (mode1 >= WAVE_MODES || mode2 >= WAVE_MODES) return;

    __shared__ complex correlation_matrix[WAVE_MODES][WAVE_MODES];

    complex correlation = complex(0.0f, 0.0f);

    for (int i = 0; i < FIELD_DIM * FIELD_DIM * FIELD_DIM; i += blockDim.x * gridDim.x) {
        int idx = i + threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < FIELD_DIM * FIELD_DIM * FIELD_DIM) {
            complex w1 = field->wave_packets[mode1][idx];
            complex w2 = field->wave_packets[mode2][idx];
            correlation += w1 * conj(w2);
        }
    }

    correlation_matrix[mode1][mode2] = correlation;
    __syncthreads();

    if (mode2 == 0) {
        for (int i = 0; i < WAVE_MODES; i++) {
            for (int j = i + 1; j < WAVE_MODES; j++) {
                complex coupling = correlation_matrix[i][j];

                float phase_diff = arg(coupling);
                float amplitude_product = abs(coupling);

                if (amplitude_product > 0.1f) {
                    float energy_transfer = 0.01f * amplitude_product * sinf(phase_diff);

                    atomicAdd(&field->mode_amplitudes[i], -energy_transfer);
                    atomicAdd(&field->mode_amplitudes[j], energy_transfer);

                    float phase_lock = 0.1f * amplitude_product * cosf(phase_diff);
                    atomicAdd(&field->mode_phases[i], phase_lock);
                    atomicAdd(&field->mode_phases[j], -phase_lock);
                }
            }
        }
    }
}

__global__ void update_spectral_field_kernel(ChemotaxisField* field) {
    cg::grid_group grid = cg::this_grid();

    for (int band = 0; band < SPECTRUM_BANDS; band++) {
        cufftExecC2C(field->nutrient_spectrum->fft_plan,
                    (cufftComplex*)field->nutrient_spectrum->fourier_space[band],
                    (cufftComplex*)field->signal_spectrum->fourier_space[band],
                    CUFFT_FORWARD);

        __syncthreads();

        int tid = grid.thread_rank();
        if (tid < FIELD_DIM * FIELD_DIM * FIELD_DIM) {
            complex spectral = field->signal_spectrum->fourier_space[band][tid];

            float kx = 2.0f * M_PI * (tid % FIELD_DIM) / FIELD_DIM;
            float ky = 2.0f * M_PI * ((tid / FIELD_DIM) % FIELD_DIM) / FIELD_DIM;
            float kz = 2.0f * M_PI * (tid / (FIELD_DIM * FIELD_DIM)) / FIELD_DIM;

            complex dispersion = dispersion_relation(kx, ky, kz, band);

            float dt = 0.01f;
            complex evolution = exp(complex(0.0f, -real(dispersion) * dt)) *
                              exp(complex(-imag(dispersion) * dt, 0.0f));

            field->signal_spectrum->fourier_space[band][tid] = spectral * evolution;

            complex nonlinear = spectral * conj(spectral) * spectral *
                              field->nutrient_spectrum->nonlinear_coupling[band];
            field->signal_spectrum->fourier_space[band][tid] += nonlinear * dt;
        }

        __syncthreads();

        cufftExecC2C(field->signal_spectrum->fft_plan,
                    (cufftComplex*)field->signal_spectrum->fourier_space[band],
                    (cufftComplex*)field->signal_spectrum->fourier_space[band],
                    CUFFT_INVERSE);

        __syncthreads();

        if (tid < FIELD_DIM * FIELD_DIM * FIELD_DIM) {
            float normalization = 1.0f / (FIELD_DIM * FIELD_DIM * FIELD_DIM);
            field->signal_spectrum->real_space[band][tid] =
                real(field->signal_spectrum->fourier_space[band][tid]) * normalization;
        }
    }

    grid.sync();

    if (grid.thread_rank() == 0) {
        field->timestep++;
    }
}

__global__ void deposit_nutrient_kernel(ChemotaxisField* field, float3 location,
                                       float concentration, float* nutrient_type) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= FIELD_DIM * FIELD_DIM * FIELD_DIM) return;

    int x = tid % FIELD_DIM;
    int y = (tid / FIELD_DIM) % FIELD_DIM;
    int z = tid / (FIELD_DIM * FIELD_DIM);

    float dx = (float)x - location.x;
    float dy = (float)y - location.y;
    float dz = (float)z - location.z;

    dx = fminf(fabsf(dx), FIELD_DIM - fabsf(dx));
    dy = fminf(fabsf(dy), FIELD_DIM - fabsf(dy));
    dz = fminf(fabsf(dz), FIELD_DIM - fabsf(dz));

    float dist_sq = dx * dx + dy * dy + dz * dz;

    float deposit = concentration * expf(-dist_sq / 100.0f);

    for (int band = 0; band < SPECTRUM_BANDS; band++) {
        float band_weight = nutrient_type[band];

        atomicAdd(&field->nutrient_spectrum->real_space[band][tid], deposit * band_weight);
    }

    atomicAdd(&field->nutrient_spectrum->total_energy, deposit * concentration);
}

__global__ void forage_gradient_kernel(ChemotaxisField* field, float3 position,
                                      float hunger, float3* gradient_out) {
    __shared__ float local_gradients[3][8];

    int tid = threadIdx.x;

    if (tid < 8) {
        int dx = tid & 1;
        int dy = (tid >> 1) & 1;
        int dz = (tid >> 2) & 1;

        int x = clamp((int)position.x + dx, 0, FIELD_DIM - 1);
        int y = clamp((int)position.y + dy, 0, FIELD_DIM - 1);
        int z = clamp((int)position.z + dz, 0, FIELD_DIM - 1);

        int idx = z * FIELD_DIM * FIELD_DIM + y * FIELD_DIM + x;

        float total_signal = 0.0f;
        for (int band = 0; band < SPECTRUM_BANDS; band++) {
            float spectral_weight = expf(-(float)(band - SPECTRUM_BANDS/2) *
                                        (float)(band - SPECTRUM_BANDS/2) / (hunger * 10.0f + 1.0f));
            total_signal += field->signal_spectrum->real_space[band][idx] * spectral_weight;
        }

        total_signal += field->interference_pattern[idx] * (1.0f - hunger);

        local_gradients[0][tid] = (dx ? total_signal : -total_signal);
        local_gradients[1][tid] = (dy ? total_signal : -total_signal);
        local_gradients[2][tid] = (dz ? total_signal : -total_signal);
    }
    __syncthreads();

    if (tid == 0) {
        gradient_out->x = (local_gradients[0][1] - local_gradients[0][0] +
                          local_gradients[0][3] - local_gradients[0][2] +
                          local_gradients[0][5] - local_gradients[0][4] +
                          local_gradients[0][7] - local_gradients[0][6]) / 4.0f;

        gradient_out->y = (local_gradients[1][2] - local_gradients[1][0] +
                          local_gradients[1][3] - local_gradients[1][1] +
                          local_gradients[1][6] - local_gradients[1][4] +
                          local_gradients[1][7] - local_gradients[1][5]) / 4.0f;

        gradient_out->z = (local_gradients[2][4] - local_gradients[2][0] +
                          local_gradients[2][5] - local_gradients[2][1] +
                          local_gradients[2][6] - local_gradients[2][2] +
                          local_gradients[2][7] - local_gradients[2][3]) / 4.0f;

        float grad_mag = sqrtf(gradient_out->x * gradient_out->x +
                              gradient_out->y * gradient_out->y +
                              gradient_out->z * gradient_out->z);

        if (grad_mag > 0.001f) {
            gradient_out->x /= grad_mag;
            gradient_out->y /= grad_mag;
            gradient_out->z /= grad_mag;
        }

        gradient_out->x *= hunger;
        gradient_out->y *= hunger;
        gradient_out->z *= hunger;
    }
}

__global__ void compute_field_coherence_kernel(ChemotaxisField* field) {
    __shared__ float band_coherences[SPECTRUM_BANDS];

    int tid = threadIdx.x;
    int band = tid % SPECTRUM_BANDS;

    if (tid < SPECTRUM_BANDS) {
        band_coherences[band] = 0.0f;
    }
    __syncthreads();

    for (int i = tid; i < FIELD_DIM * FIELD_DIM * FIELD_DIM; i += blockDim.x) {
        complex spectral = field->signal_spectrum->fourier_space[band][i];
        float phase = arg(spectral);
        float amplitude = abs(spectral);

        float neighbor_phase_sum = 0.0f;
        int neighbor_count = 0;

        int x = i % FIELD_DIM;
        int y = (i / FIELD_DIM) % FIELD_DIM;
        int z = i / (FIELD_DIM * FIELD_DIM);

        for (int dz = -1; dz <= 1; dz++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;

                    int nx = (x + dx + FIELD_DIM) % FIELD_DIM;
                    int ny = (y + dy + FIELD_DIM) % FIELD_DIM;
                    int nz = (z + dz + FIELD_DIM) % FIELD_DIM;

                    int nidx = nz * FIELD_DIM * FIELD_DIM + ny * FIELD_DIM + nx;

                    complex neighbor = field->signal_spectrum->fourier_space[band][nidx];
                    float neighbor_phase = arg(neighbor);

                    neighbor_phase_sum += cosf(phase - neighbor_phase);
                    neighbor_count++;
                }
            }
        }

        float local_coherence = neighbor_phase_sum / neighbor_count;
        atomicAdd(&band_coherences[band], local_coherence * amplitude);
    }
    __syncthreads();

    if (tid == 0) {
        float total_coherence = 0.0f;
        for (int b = 0; b < SPECTRUM_BANDS; b++) {
            total_coherence += band_coherences[b];
        }
        total_coherence /= (SPECTRUM_BANDS * FIELD_DIM * FIELD_DIM * FIELD_DIM);

        field->field_coherence.store(total_coherence);
    }
}