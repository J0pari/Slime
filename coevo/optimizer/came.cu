// Wave 1: CAME Optimizer + Gradient Aggregation
//
// Per cuda_engineering.md sections 4.3, 4.7, 8.
// aggregate_gradients_kernel: averages per-organism GradBuffers into d_mean_grad.
// came_step_kernel: confidence-adjusted momentum update on shared weights.

#ifndef COEVO_OPTIMIZER_CAME_CU
#define COEVO_OPTIMIZER_CAME_CU

#include "../autodiff/warp_tape.cu"

namespace slime::optimizer {

using autodiff::GradBuffers;
using autodiff::TOTAL_WEIGHTS;

// CAME hyperparameters (pinned by spec, cuda_engineering.md section 8).
struct CameHyperparams {
    float lr;
    float beta1;
    float beta2;
    float beta3;
    float epsilon;
    float weight_decay;
};

constexpr CameHyperparams CAME_DEFAULTS = {
    1e-3f,   // lr
    0.9f,    // beta1
    0.999f,  // beta2
    0.999f,  // beta3
    1e-8f,   // epsilon
    0.01f    // weight_decay
};

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

    float g = mean_grad[i];

    // Momentum update.
    float mi = beta1 * m[i] + (1.f - beta1) * g;
    m[i] = mi;

    // Variance update.
    float vi = beta2 * v[i] + (1.f - beta2) * g * g;
    v[i] = vi;

    // Update direction.
    float u = mi / (sqrtf(vi) + epsilon);

    // Instability tracking.
    float diff = u - prev_u[i];
    float instability = diff * diff;
    float ci = beta3 * c[i] + (1.f - beta3) * instability;
    c[i] = ci;

    // Confidence-adjusted step.
    float confidence = 1.f / (1.f + ci);
    weights[i] -= lr * confidence * u + weight_decay * weights[i];

    prev_u[i] = u;
}

// ---- Host API -----------------------------------------------------------

inline void allocate_came(CameState& state) {
    cudaMalloc(&state.d_m,         sizeof(float) * TOTAL_WEIGHTS);
    cudaMalloc(&state.d_v,         sizeof(float) * TOTAL_WEIGHTS);
    cudaMalloc(&state.d_c,         sizeof(float) * TOTAL_WEIGHTS);
    cudaMalloc(&state.d_prev_u,    sizeof(float) * TOTAL_WEIGHTS);
    cudaMalloc(&state.d_mean_grad, sizeof(float) * TOTAL_WEIGHTS);
    cudaMemset(state.d_m,       0, sizeof(float) * TOTAL_WEIGHTS);
    cudaMemset(state.d_v,       0, sizeof(float) * TOTAL_WEIGHTS);
    cudaMemset(state.d_c,       0, sizeof(float) * TOTAL_WEIGHTS);
    cudaMemset(state.d_prev_u,  0, sizeof(float) * TOTAL_WEIGHTS);
    state.step = 0;
}

inline void free_came(CameState& state) {
    cudaFree(state.d_m);
    cudaFree(state.d_v);
    cudaFree(state.d_c);
    cudaFree(state.d_prev_u);
    cudaFree(state.d_mean_grad);
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

inline void launch_grad_norm_reduce(const float* d_mean_grad,
                                    float* d_grad_norm,
                                    cudaStream_t stream) {
    // Zero the output scalar first.
    cudaMemsetAsync(d_grad_norm, 0, sizeof(float), stream);
    int grid = (TOTAL_WEIGHTS + 255) / 256;
    grad_norm_reduce_kernel<<<grid, 256, 0, stream>>>(
        d_mean_grad, d_grad_norm, TOTAL_WEIGHTS);
}

}  // namespace slime::optimizer

#endif  // COEVO_OPTIMIZER_CAME_CU
