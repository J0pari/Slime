// CAME optimizer and gradient aggregation
//
// Per cuda_engineering.md sections 4.3, 4.7, 8.
// aggregate_gradients_kernel: averages per-organism GradBuffers into d_mean_grad.
// came_step_kernel: confidence-adjusted momentum update on shared weights.

#ifndef COEVO_OPTIMIZER_CAME_CU
#define COEVO_OPTIMIZER_CAME_CU

#include "../autodiff/warp_tape.cu"
#include "came_math.cuh"

namespace slime::optimizer {

using autodiff::GradBuffers;
using autodiff::TOTAL_WEIGHTS;
using autodiff::OFF_INTER;
using autodiff::OFF_FLOW;
using autodiff::OFF_BMAP;
using autodiff::TelemetryScalars;

// CameHyperparams and CAME_DEFAULTS live in came_math.cuh (shared with the
// host unit tests); the came_step_scalar step itself is also defined there.

// CAME state: 4 arrays of TOTAL_WEIGHTS on device, plus step counter on host.
struct CameState {
    float* d_m;        // 1st moment
    float* d_v;        // 2nd moment
    float* d_c;        // confidence accumulator
    float* d_prev_u;   // previous update direction
    float* d_mean_grad; // averaged gradient buffer
    int step;
};

// ---- aggregate_gradients_kernel -----------------------------------------
// Per cuda_engineering.md section 4.7.
// Grid: <<<ceil(TOTAL_WEIGHTS/256), 256>>>
// Averages per-organism GradBuffers into a flat d_mean_grad buffer.

__global__ void aggregate_gradients_kernel(
    const GradBuffers* grads,
    float* mean_grad,
    int n_organisms)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= TOTAL_WEIGHTS) return;

    float sum = 0.f;
    for (int org = 0; org < n_organisms; ++org) {
        sum += grads[org].dW[i];
    }
    mean_grad[i] = sum / static_cast<float>(n_organisms);
}

// ---- came_step_kernel ---------------------------------------------------
// Per cuda_engineering.md section 4.3.
// Grid: <<<ceil(TOTAL_WEIGHTS/256), 256>>>
// CAME update per weight:
//   g = mean_grad[i]
//   m = beta1*m + (1-beta1)*g
//   v = beta2*v + (1-beta2)*g^2
//   u = m / (sqrt(v) + eps)
//   instability = (u - prev_u)^2
//   c = beta3*c + (1-beta3)*instability
//   confidence = 1 / (1 + c)
//   w -= lr * confidence * u + weight_decay * w
//   prev_u = u

__global__ void came_step_kernel(
    float* weights,
    const float* mean_grad,
    float* m,
    float* v,
    float* c,
    float* prev_u,
    float lr,
    float beta1,
    float beta2,
    float beta3,
    float epsilon,
    float weight_decay)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= TOTAL_WEIGHTS) return;

    const float g = mean_grad[i];
    CameScalarState s = { m[i], v[i], c[i], prev_u[i] };
    const CameScalarParams p = { lr, beta1, beta2, beta3, epsilon, weight_decay };

    float w = weights[i];
    came_step_scalar(&s, &w, g, p);

    m[i] = s.m;
    v[i] = s.v;
    c[i] = s.c;
    prev_u[i] = s.prev_u;
    weights[i] = w;
}

// ---- Host API -----------------------------------------------------------

inline bool allocate_came(CameState& state) {
    cudaError_t e = cudaMalloc(&state.d_m,         sizeof(float) * TOTAL_WEIGHTS);
    if (e == cudaSuccess) e = cudaMalloc(&state.d_v,         sizeof(float) * TOTAL_WEIGHTS);
    if (e == cudaSuccess) e = cudaMalloc(&state.d_c,         sizeof(float) * TOTAL_WEIGHTS);
    if (e == cudaSuccess) e = cudaMalloc(&state.d_prev_u,    sizeof(float) * TOTAL_WEIGHTS);
    if (e == cudaSuccess) e = cudaMalloc(&state.d_mean_grad, sizeof(float) * TOTAL_WEIGHTS);
    if (e == cudaSuccess) e = cudaMemset(state.d_m,       0, sizeof(float) * TOTAL_WEIGHTS);
    if (e == cudaSuccess) e = cudaMemset(state.d_v,       0, sizeof(float) * TOTAL_WEIGHTS);
    if (e == cudaSuccess) e = cudaMemset(state.d_c,       0, sizeof(float) * TOTAL_WEIGHTS);
    if (e == cudaSuccess) e = cudaMemset(state.d_prev_u,  0, sizeof(float) * TOTAL_WEIGHTS);
    if (e != cudaSuccess) {
        std::printf("[FATAL] CUDA CAME allocation failed: %s\n", cudaGetErrorString(e));
        return false;
    }
    state.step = 0;
    return true;
}

inline bool free_came(CameState& state) {
    cudaError_t e = cudaFree(state.d_m);
    if (e == cudaSuccess) e = cudaFree(state.d_v);
    if (e == cudaSuccess) e = cudaFree(state.d_c);
    if (e == cudaSuccess) e = cudaFree(state.d_prev_u);
    if (e == cudaSuccess) e = cudaFree(state.d_mean_grad);
    if (e != cudaSuccess) {
        std::printf("[WARN] CUDA CAME free failed: %s\n", cudaGetErrorString(e));
        return false;
    }
    return true;
}

inline void launch_aggregate_gradients(
    const GradBuffers* d_grads,
    float* d_mean_grad,
    int n_organisms,
    cudaStream_t stream)
{
    int grid = (TOTAL_WEIGHTS + 255) / 256;
    aggregate_gradients_kernel<<<grid, 256, 0, stream>>>(
        d_grads, d_mean_grad, n_organisms);
}

inline void launch_came_step(
    float* d_weights,
    CameState& state,
    const CameHyperparams& hp,
    cudaStream_t stream)
{
    int grid = (TOTAL_WEIGHTS + 255) / 256;
    came_step_kernel<<<grid, 256, 0, stream>>>(
        d_weights, state.d_mean_grad,
        state.d_m, state.d_v, state.d_c, state.d_prev_u,
        hp.lr, hp.beta1, hp.beta2, hp.beta3, hp.epsilon, hp.weight_decay);
    state.step++;
}

// ---- grad_norm_reduce_kernel ---------------------------------------------
// Per cuda_engineering.md section 8. Device-side L2 norm of d_mean_grad.
// Grid: <<<ceil(TOTAL_WEIGHTS/256), 256>>>
// Writes one float (the squared norm) to d_out. Host takes sqrt after read.
// Uses shared-memory tree reduction.

__global__ void grad_norm_reduce_kernel(const float* mean_grad,
                                        float* d_out, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    float val = 0.f;
    if (i < n) val = mean_grad[i] * mean_grad[i];
    sdata[tid] = val;
    __syncthreads();

    // Tree reduction in shared memory.
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Block 0 thread 0 writes partial sum; use atomicAdd for multi-block.
    if (tid == 0) atomicAdd(d_out, sdata[0]);
}

inline bool launch_grad_norm_reduce(const float* d_mean_grad,
                                    float* d_grad_norm,
                                    cudaStream_t stream) {
    // Zero the output scalar first.
    cudaError_t e = cudaMemsetAsync(d_grad_norm, 0, sizeof(float), stream);
    if (e != cudaSuccess) {
        std::printf("[FATAL] CUDA grad-norm memset failed: %s\n", cudaGetErrorString(e));
        return false;
    }
    int grid = (TOTAL_WEIGHTS + 255) / 256;
    grad_norm_reduce_kernel<<<grid, 256, 0, stream>>>(
        d_mean_grad, d_grad_norm, TOTAL_WEIGHTS);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::printf("[FATAL] CUDA grad-norm launch failed: %s\n", cudaGetErrorString(e));
        return false;
    }
    return true;
}

// ---- Numerical telemetry kernels (A-501) ---------------------------------
// Bank index for a flat weight index: 0 = W_perc, 1 = W_inter, 2 = W_flow,
// 3 = W_bmap (offsets from autodiff/warp_tape.cu).
__host__ __device__ inline int telemetry_bank(int i) {
    if (i < OFF_INTER) return 0;
    if (i < OFF_FLOW)  return 1;
    if (i < OFF_BMAP)  return 2;
    return 3;
}

// Per-bank squared norms of the mean gradient. The caller must zero the
// TelemetryScalars buffer before launch. Telemetry-only; float atomicAdd
// ordering is irrelevant.
__global__ void grad_bank_norm_kernel(const float* mean_grad,
                                      TelemetryScalars* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= TOTAL_WEIGHTS) return;
    float v = mean_grad[i];
    atomicAdd(&out->grad_norm_sq[telemetry_bank(i)], v * v);
}

// Per-bank squared norms of the shared weights and of the CAME normalized
// update direction (prev_u).
__global__ void weight_and_update_norm_kernel(const float* weights,
                                              const float* prev_u,
                                              TelemetryScalars* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= TOTAL_WEIGHTS) return;
    int b = telemetry_bank(i);
    float w = weights[i];
    float u = prev_u[i];
    atomicAdd(&out->weight_norm_sq[b], w * w);
    atomicAdd(&out->update_norm_sq[b], u * u);
}

// CAME instability/confidence statistics: mean and max of c, mean and min of
// confidence = 1/(1+c).
__global__ void came_stats_kernel(const float* c, TelemetryScalars* out) {
    __shared__ float s_sum_c[256];
    __shared__ float s_max_c[256];
    __shared__ float s_sum_conf[256];
    __shared__ float s_min_conf[256];
    int tid = threadIdx.x;

    float sum_c = 0.f, max_c = 0.f;
    float sum_conf = 0.f, min_conf = 1e30f;
    for (int i = tid; i < TOTAL_WEIGHTS; i += blockDim.x) {
        float ci = c[i];
        sum_c += ci;
        max_c = fmaxf(max_c, ci);
        float conf = 1.f / (1.f + ci);
        sum_conf += conf;
        min_conf = fminf(min_conf, conf);
    }
    s_sum_c[tid] = sum_c;
    s_max_c[tid] = max_c;
    s_sum_conf[tid] = sum_conf;
    s_min_conf[tid] = min_conf;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum_c[tid]   += s_sum_c[tid + s];
            s_max_c[tid]    = fmaxf(s_max_c[tid], s_max_c[tid + s]);
            s_sum_conf[tid] += s_sum_conf[tid + s];
            s_min_conf[tid] = fminf(s_min_conf[tid], s_min_conf[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        out->c_mean = s_sum_c[0] / static_cast<float>(TOTAL_WEIGHTS);
        out->c_max = s_max_c[0];
        out->conf_mean = s_sum_conf[0] / static_cast<float>(TOTAL_WEIGHTS);
        out->conf_min = s_min_conf[0];
    }
}

// Count nonfinite values across all shared optimizer state. Any nonzero count
// invalidates the run (hard abort in the host loop).
__global__ void nonfinite_scan_kernel(const float* weights,
                                      const float* m,
                                      const float* v,
                                      const float* c,
                                      const float* prev_u,
                                      TelemetryScalars* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= TOTAL_WEIGHTS) return;
    const float* arrays[5] = { weights, m, v, c, prev_u };
    int bad = 0;
    for (int a = 0; a < 5; ++a) {
        if (!isfinite(arrays[a][i])) bad++;
    }
    if (bad > 0) atomicAdd(&out->nonfinite_count, static_cast<float>(bad));
}

inline bool launch_telemetry_kernels(
    const float* d_mean_grad,
    const float* d_weights,
    const float* d_came_m,
    const float* d_came_v,
    const float* d_came_c,
    const float* d_came_prev_u,
    TelemetryScalars* d_tel,
    cudaStream_t stream)
{
    cudaError_t e = cudaMemsetAsync(d_tel, 0, sizeof(TelemetryScalars), stream);
    if (e != cudaSuccess) {
        std::printf("[FATAL] CUDA telemetry memset failed: %s\n", cudaGetErrorString(e));
        return false;
    }
    int grid = (TOTAL_WEIGHTS + 255) / 256;
    grad_bank_norm_kernel<<<grid, 256, 0, stream>>>(d_mean_grad, d_tel);
    weight_and_update_norm_kernel<<<grid, 256, 0, stream>>>(
        d_weights, d_came_prev_u, d_tel);
    came_stats_kernel<<<1, 256, 0, stream>>>(d_came_c, d_tel);
    nonfinite_scan_kernel<<<grid, 256, 0, stream>>>(
        d_weights, d_came_m, d_came_v, d_came_c, d_came_prev_u, d_tel);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::printf("[FATAL] CUDA telemetry launch failed: %s\n", cudaGetErrorString(e));
        return false;
    }
    return true;
}

// ---- Role gradient alignment (A-501) -------------------------------------
// One thread per shared weight; each thread sums the classifier and predictor
// contributions to that weight before a single set of atomics. Telemetry-only;
// float atomicAdd ordering is irrelevant. The caller must have zeroed the
// TelemetryScalars buffer already (launch_telemetry_kernels does).
__global__ void role_grad_alignment_kernel(
    const GradBuffers* grads,
    const nca::ForwardInputs* inputs,
    int n_organisms,
    TelemetryScalars* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= TOTAL_WEIGHTS) return;

    float sum_c = 0.f;
    float sum_p = 0.f;
    for (int org = 0; org < n_organisms; ++org) {
        float g = grads[org].dW[i];
        if (::canonical_role(inputs[org].role) == Role::Classifier) {
            sum_c += g;
        } else {
            sum_p += g;
        }
    }
    atomicAdd(&out->role_grad_dot, sum_c * sum_p);
    atomicAdd(&out->role_grad_norm_sq[0], sum_c * sum_c);
    atomicAdd(&out->role_grad_norm_sq[1], sum_p * sum_p);
}

inline bool launch_role_grad_alignment(
    const GradBuffers* d_grads,
    const nca::ForwardInputs* d_inputs,
    int n_organisms,
    TelemetryScalars* d_tel,
    cudaStream_t stream)
{
    int grid = (TOTAL_WEIGHTS + 255) / 256;
    role_grad_alignment_kernel<<<grid, 256, 0, stream>>>(
        d_grads, d_inputs, n_organisms, d_tel);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::printf("[FATAL] CUDA role grad alignment launch failed: %s\n",
                    cudaGetErrorString(e));
        return false;
    }
    return true;
}

}  // namespace slime::optimizer

#endif  // COEVO_OPTIMIZER_CAME_CU

