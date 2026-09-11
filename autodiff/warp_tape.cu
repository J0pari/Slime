// Wave 1/2.5: Checkpointed Forward + Phase-Decomposed Batched Backward + BTRAJ Gather
//
// Per cuda_engineering.md sections 4.1, 4.2 (revised), 4.8, 10.
// forward_with_checkpoints: identical to forward_kernel but saves 4 checkpoints.
// backward: decomposed into batched sub-kernels (all orgs in parallel per launch).
// btraj_gather_kernel: gathers strided bmap_traj into contiguous buffer.

#ifndef COEVO_AUTODIFF_WARP_TAPE_CU
#define COEVO_AUTODIFF_WARP_TAPE_CU

#include "../nca/engine.cu"

namespace slime::autodiff {

using nca::OrganismState;
using nca::ForwardInputs;
using nca::PERC_DIM;
using nca::HIDDEN_DIM;
using nca::grid_idx;
using ::canonical_role;

// ---- Weight layout constants (mirror genome/codec.cu) --------------------
constexpr int W_INTER_SIZE = PERC_DIM * HIDDEN_DIM;       // 1536
constexpr int W_FLOW_SIZE  = HIDDEN_DIM * CA_CHANNELS;    // 512
constexpr int W_BMAP_SIZE  = CA_CHANNELS * BMAP_DIM;      // 512
constexpr int TOTAL_WEIGHTS = W_PERC_SIZE + W_INTER_SIZE + W_FLOW_SIZE + W_BMAP_SIZE; // 2587

static_assert(TOTAL_WEIGHTS == genome::TOTAL_WEIGHT_SLOTS,
              "autodiff and genome weight-space layouts must agree");

constexpr int CELLS = GRID_SIZE * GRID_SIZE;               // 4096
constexpr int GRID_ELEMS = CELLS * CA_CHANNELS;            // 65536
constexpr int PERC_ELEMS = CELLS * PERC_DIM;               // 196608

// Weight offsets into a flat buffer.
constexpr int OFF_PERC  = 0;
constexpr int OFF_INTER = W_PERC_SIZE;
constexpr int OFF_FLOW  = OFF_INTER + W_INTER_SIZE;
constexpr int OFF_BMAP  = OFF_FLOW + W_FLOW_SIZE;

// Backward sub-kernel config.
constexpr int BWD_THREADS = 256;  // threads per block (one block per organism)

// Checkpoint indices: steps 0, 16, 32, 48.
constexpr int NUM_CHECKPOINTS = 4;
constexpr int CHECKPOINT_INTERVAL = 16;

// ---- Structs -------------------------------------------------------------

// 4 FP16 grid snapshots per organism.
struct CheckpointBuffer {
    __half data[NUM_CHECKPOINTS][GRID_ELEMS];
};

// Per-organism weight gradients (FP32, flat layout matching weight buffer).
struct GradBuffers {
    float dW[TOTAL_WEIGHTS];
};

// Backward workspace: per-organism buffers for batched backward.
// Per cuda_engineering.md section 4.2 (revised) and section 10.
// All pointers address contiguous arrays of n_organisms sub-buffers.
// Organism `org` accesses index [org * GRID_ELEMS .. (org+1) * GRID_ELEMS).
struct BackwardWorkspace {
    float* d_state[2];          // [n_org * GRID_ELEMS] FP32 each (d_state_A, d_state_B)
    float* d_perc;              // [n_org * PERC_ELEMS] floats (per-org d_perc)
    __half* recomp[2];          // [n_org * GRID_ELEMS] FP16 each (re-forward curr/next)
    int n_organisms;            // stored for indexing
};

// ---- Numerical telemetry (A-501) ----------------------------------------
// Reduced per-generation scalars for gradient trust monitoring. Bank order:
// 0 = W_perc, 1 = W_inter, 2 = W_flow, 3 = W_bmap. Every field is written by
// device kernels; the host copies this struct and logs/hard-fails on it.
struct TelemetryScalars {
    float grad_norm_sq[4];    // per-bank squared norm of the mean gradient
    float weight_norm_sq[4];  // per-bank squared norm of the shared weights
    float update_norm_sq[4];  // per-bank squared norm of the CAME prev_u
    float c_mean;             // mean CAME instability accumulator
    float c_max;              // max CAME instability accumulator
    float conf_mean;          // mean confidence 1/(1+c)
    float conf_min;           // min confidence
    float nonfinite_count;    // nonfinite values across optimizer state
    float state_max_abs;      // max |x| over checkpointed + final FP16 states
    float state_near_max;     // count of state values with |x| > 60000
};

// ---- Loss functions (host-callable, proven in host tests) ----------------

// Cross-entropy classification loss. logits[0..n-1], target in [0,n).
// Writes softmax-minus-one-hot gradient into dlogits, loss into *loss_out.
inline void classifier_loss(const float* logits, int target, int n,
                            float* dlogits, float* loss_out) {
    float max_z = logits[0];
    for (int i = 1; i < n; ++i) if (logits[i] > max_z) max_z = logits[i];
    float Z = 0.f;
    float p[64];
    for (int i = 0; i < n; ++i) { p[i] = expf(logits[i] - max_z); Z += p[i]; }
    for (int i = 0; i < n; ++i) p[i] /= Z;
    if (loss_out) *loss_out = -logf(p[target] + 1e-30f);
    for (int i = 0; i < n; ++i) dlogits[i] = p[i] - ((i == target) ? 1.f : 0.f);
}

// Predictor MSE loss. prediction and target are BMAP_DIM floats.
// Writes d(MSE)/d(prediction) into dpred, loss into *loss_out.
inline void predictor_mse_loss(const float* prediction, const float* target,
                               float* dpred, float* loss_out) {
    float acc = 0.f;
    const float scale = 2.0f / static_cast<float>(BMAP_DIM);
    for (int d = 0; d < BMAP_DIM; ++d) {
        float diff = prediction[d] - target[d];
        acc += diff * diff;
        dpred[d] = scale * diff;
    }
    if (loss_out) *loss_out = acc / static_cast<float>(BMAP_DIM);
}

// ---- Device helpers for backward -----------------------------------------

__device__ inline float gelu_derivative(float x) {
    // d/dx GELU(x) using the Hendrycks-Gimpel tanh approximation.
    const float k = 0.7978845608f;     // sqrt(2/pi)
    float x3 = x * x * x;
    float inner = k * (x + 0.044715f * x3);
    float t = tanhf(inner);
    float sech2 = 1.f - t * t;
    float d_inner = k * (1.f + 3.f * 0.044715f * x * x);
    return 0.5f * (1.f + t) + 0.5f * x * sech2 * d_inner;
}

// Convert a flat cell index (0..4095) to (y, x).
__device__ inline void cell_yx(int cell, int& y, int& x) {
    y = cell / GRID_SIZE;
    x = cell % GRID_SIZE;
}

// ---- forward_with_checkpoints kernel ------------------------------------
// Per cuda_engineering.md section 4.1.
// Grid: <<<n_organisms, dim3(16,16)>>>. One block per organism.
// Identical to forward_kernel but saves 4 checkpoints at steps 0,16,32,48.
// eff_weights (nullable) selects per-organism flat weight banks (W_shared +
// genome delta); when null every organism uses the shared `weights` bank.

__global__ void forward_with_checkpoints_kernel(
    OrganismState* organisms,
    const ForwardInputs* inputs,
    const nca::rd::Coefficients* coeffs,
    const float* weights,          // [TOTAL_WEIGHTS] shared bank
    const float* eff_weights,      // [n_banks * TOTAL_WEIGHTS] or null
    CheckpointBuffer* checkpoints,
    int n_organisms)
{
    int org = blockIdx.x;
    if (org >= n_organisms) return;
    OrganismState* o = &organisms[org];
    const ForwardInputs& in = inputs[org];
    CheckpointBuffer& ckpt = checkpoints[org];

    const float* wbase = (eff_weights != nullptr)
        ? &eff_weights[static_cast<size_t>(org) * TOTAL_WEIGHTS] : weights;
    const float* W_perc  = wbase + OFF_PERC;
    const float* W_inter = wbase + OFF_INTER;
    const float* W_flow  = wbase + OFF_FLOW;
    const float* W_bmap  = wbase + OFF_BMAP;

    // Role-switched seeding.
    if (canonical_role(in.role) == Role::Classifier) {
        nca::seed_classifier_grid(o->grid, in.image_rgb, in.task_embedding);
    } else {
        nca::seed_predictor_grid(o->grid, in.target_bmap_32, in.task_embedding);
    }
    __syncthreads();

    // Save checkpoint 0 (initial state after seeding).
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * blockDim.y;
    for (int i = tid; i < GRID_ELEMS; i += nthreads) {
        ckpt.data[0][i] = o->grid[i];
    }
    __syncthreads();

    int steps[BTRAJ_SAMPLES];
    #pragma unroll
    for (int i = 0; i < BTRAJ_SAMPLES; ++i) steps[i] = d_BTRAJ_STEPS[i];

    __half* curr = o->grid;
    __half* next = o->scratch;
    int sample_idx = 0;

    for (int step = 1; step <= CA_STEPS; ++step) {
        nca::ca_step(curr, next, W_perc, W_inter, W_flow);
        if (coeffs != nullptr) {
            nca::rd::rd_step(curr, next, coeffs[org]);
        }
        __half* tmp = curr; curr = next; next = tmp;

        // Save checkpoints at steps 16, 32, 48 (indices 1, 2, 3).
        if (step == CHECKPOINT_INTERVAL || step == 2 * CHECKPOINT_INTERVAL ||
            step == 3 * CHECKPOINT_INTERVAL) {
            int ckpt_idx = step / CHECKPOINT_INTERVAL; // 1, 2, or 3
            for (int i = tid; i < GRID_ELEMS; i += nthreads) {
                ckpt.data[ckpt_idx][i] = curr[i];
            }
            __syncthreads();
        }

        if (sample_idx < BTRAJ_SAMPLES && step == steps[sample_idx]) {
            nca::project_bmap(curr, W_bmap, &o->bmap_traj[sample_idx * BMAP_DIM]);
            sample_idx++;
        }
    }

    // Leave final state in o->grid.
    if (curr != o->grid) {
        for (int i = tid; i < GRID_ELEMS; i += nthreads) {
            o->grid[i] = curr[i];
        }
    }
}

// ---- Batched backward sub-kernels ---------------------------------------
// Per cuda_engineering.md section 4.2 (revised).
// All sub-kernels use <<<n_organisms, 256>>>. One block per organism.
// Per-organism workspace buffers indexed by blockIdx.x.

// Sub-kernel 1: Zero grads and d_state_A for all organisms.
__global__ void bwd_zero_kernel(
    GradBuffers* grads,
    float* d_state_A,      // [n_org * GRID_ELEMS]
    int n_organisms)
{
    int org = blockIdx.x;
    if (org >= n_organisms) return;
    int tid = threadIdx.x;
    for (int i = tid; i < TOTAL_WEIGHTS; i += BWD_THREADS) {
        grads[org].dW[i] = 0.f;
    }
    float* my_dA = &d_state_A[org * GRID_ELEMS];
    for (int i = tid; i < GRID_ELEMS; i += BWD_THREADS) {
        my_dA[i] = 0.f;
    }
}

// Sub-kernel 2: Compute avg-pool summary of final grid state.
// Writes summary into first CA_CHANNELS floats of d_state_B[org].
__global__ void bwd_seed_avgpool_kernel(
    const OrganismState* organisms,
    float* d_state_B,      // [n_org * GRID_ELEMS] (scratch: first CH floats)
    int n_organisms)
{
    int org = blockIdx.x;
    if (org >= n_organisms) return;
    extern __shared__ float smem[];  // [CA_CHANNELS]

    int tid = threadIdx.x;
    for (int c = tid; c < CA_CHANNELS; c += BWD_THREADS) {
        smem[c] = 0.f;
    }
    __syncthreads();

    const __half* final_grid = organisms[org].grid;
    for (int cell = tid; cell < CELLS; cell += BWD_THREADS) {
        for (int c = 0; c < CA_CHANNELS; ++c) {
            atomicAdd(&smem[c], __half2float(final_grid[cell * CA_CHANNELS + c]));
        }
    }
    __syncthreads();

    float* g_summary = &d_state_B[org * GRID_ELEMS];
    for (int c = tid; c < CA_CHANNELS; c += BWD_THREADS) {
        g_summary[c] = smem[c] / static_cast<float>(CELLS);
    }
}

// Sub-kernel 3: Backprop through project_bmap, seed d_state_A.
// Reads summary from d_state_B[org][0..CH-1].
// eff_weights (nullable) selects the organism's own flat weight bank; the
// gradient dW_bmap is still accumulated into the per-organism GradBuffers and
// later aggregated across organisms (dL/dW_shared = sum_org dL_org/dW_eff_org
// because W_eff = W_shared + additive delta).
__global__ void bwd_seed_scatter_kernel(
    const float* weights,
    const float* eff_weights,    // [n_banks * TOTAL_WEIGHTS] or null
    const float* seed_grad,    // [n_org * BMAP_DIM]
    GradBuffers* grads,
    float* d_state_A,          // [n_org * GRID_ELEMS]
    const float* d_state_B,    // [n_org * GRID_ELEMS] (summary in first CH)
    int n_organisms)
{
    int org = blockIdx.x;
    if (org >= n_organisms) return;
    extern __shared__ float smem[];  // [CA_CHANNELS]

    int tid = threadIdx.x;
    const float* wbase = (eff_weights != nullptr)
        ? &eff_weights[static_cast<size_t>(org) * TOTAL_WEIGHTS] : weights;
    const float* W_bmap = wbase + OFF_BMAP;
    const float* org_seed = &seed_grad[org * BMAP_DIM];
    const float* g_summary = &d_state_B[org * GRID_ELEMS];

    // d_summary[c] = sum_d W_bmap[c*BMAP_DIM+d] * seed_grad[d]
    for (int c = tid; c < CA_CHANNELS; c += BWD_THREADS) {
        float acc = 0.f;
        for (int d = 0; d < BMAP_DIM; ++d) {
            acc += W_bmap[c * BMAP_DIM + d] * org_seed[d];
        }
        smem[c] = acc;
    }
    __syncthreads();

    // dW_bmap[c*BMAP_DIM+d] += summary[c] * seed_grad[d]
    float* dW_bmap = &grads[org].dW[OFF_BMAP];
    for (int i = tid; i < W_BMAP_SIZE; i += BWD_THREADS) {
        int c = i / BMAP_DIM;
        int d = i % BMAP_DIM;
        dW_bmap[i] += g_summary[c] * org_seed[d];
    }

    // Seed d_state_A: d_state[cell,c] += d_summary[c] / CELLS
    float inv_cells = 1.f / static_cast<float>(CELLS);
    float* my_dA = &d_state_A[org * GRID_ELEMS];
    for (int i = tid; i < GRID_ELEMS; i += BWD_THREADS) {
        int c = i % CA_CHANNELS;
        my_dA[i] += smem[c] * inv_cells;
    }
}

// Sub-kernel 4: Load checkpoint into recomp_curr.
__global__ void bwd_load_checkpoint_kernel(
    int seg,
    const CheckpointBuffer* checkpoints,
    __half* recomp_curr,       // [n_org * GRID_ELEMS]
    int n_organisms)
{
    int org = blockIdx.x;
    if (org >= n_organisms) return;
    int tid = threadIdx.x;
    __half* my_rc = &recomp_curr[org * GRID_ELEMS];
    for (int i = tid; i < GRID_ELEMS; i += BWD_THREADS) {
        my_rc[i] = checkpoints[org].data[seg][i];
    }
}

// Sub-kernel 5: One CA re-forward step on recomp workspace.
// NOTE: This re-forwards only the CA step (perception → hidden → flow → residual).
// When reaction-diffusion is active (coeffs != nullptr in the forward), the RD
// adjoint must also be replayed and differentiated here.  Today the main loop
// launches forward with coeffs = nullptr, so this omission is safe.  Enabling RD
// without adding the RD adjoint will produce biased gradients — the checkpoint
// states (saved after RD in the forward) will not match the re-forwarded states.
// The RD adjoint is deferred to the wave that enables reaction-diffusion.
__global__ void bwd_reforward_step_kernel(
    const float* weights,
    const float* eff_weights,    // [n_banks * TOTAL_WEIGHTS] or null
    const __half* recomp_curr, // [n_org * GRID_ELEMS]
    __half* recomp_next,       // [n_org * GRID_ELEMS]
    int n_organisms)
{
    int org = blockIdx.x;
    if (org >= n_organisms) return;
    int tid = threadIdx.x;
    const float* wbase = (eff_weights != nullptr)
        ? &eff_weights[static_cast<size_t>(org) * TOTAL_WEIGHTS] : weights;
    const float* W_perc  = wbase + OFF_PERC;
    const float* W_inter = wbase + OFF_INTER;
    const float* W_flow  = wbase + OFF_FLOW;

    const __half* rc = &recomp_curr[org * GRID_ELEMS];
    __half* rn = &recomp_next[org * GRID_ELEMS];

    for (int cell = tid; cell < CELLS; cell += BWD_THREADS) {
        int y, x;
        cell_yx(cell, y, x);

        float perc[PERC_DIM];
        nca::sample_neighborhood(rc, W_perc, y, x, perc);

        float hidden[HIDDEN_DIM];
        #pragma unroll
        for (int h = 0; h < HIDDEN_DIM; ++h) {
            float acc = 0.f;
            for (int p = 0; p < PERC_DIM; ++p) {
                acc += W_inter[p * HIDDEN_DIM + h] * perc[p];
            }
            hidden[h] = nca::gelu_approx(acc);
        }

        #pragma unroll
        for (int c = 0; c < CA_CHANNELS; ++c) {
            float acc = 0.f;
            for (int h = 0; h < HIDDEN_DIM; ++h) {
                acc += W_flow[h * CA_CHANNELS + c] * hidden[h];
            }
            float prev = __half2float(rc[cell * CA_CHANNELS + c]);
            float nxt = prev + acc;
            if (nxt >  65504.f) nxt =  65504.f;
            if (nxt < -65504.f) nxt = -65504.f;
            rn[cell * CA_CHANNELS + c] = __float2half(nxt);
        }
    }
}

// Sub-kernel 6 (Phase A): Weight gradient accumulation + d_perc computation.
// eff_weights (nullable) selects each organism's own weight bank.
__global__ void bwd_weight_grad_kernel(
    const float* weights,
    const float* eff_weights,    // [n_banks * TOTAL_WEIGHTS] or null
    const __half* recomp_curr, // [n_org * GRID_ELEMS] — recovered input state
    const float* d_state_A,    // [n_org * GRID_ELEMS] — d_state_next
    GradBuffers* grads,
    float* d_perc_buf,         // [n_org * PERC_ELEMS]
    int n_organisms)
{
    int org = blockIdx.x;
    if (org >= n_organisms) return;
    int tid = threadIdx.x;
    const float* wbase = (eff_weights != nullptr)
        ? &eff_weights[static_cast<size_t>(org) * TOTAL_WEIGHTS] : weights;
    const float* W_perc  = wbase + OFF_PERC;
    const float* W_inter = wbase + OFF_INTER;
    const float* W_flow  = wbase + OFF_FLOW;

    const __half* rc = &recomp_curr[org * GRID_ELEMS];
    const float* dA = &d_state_A[org * GRID_ELEMS];
    float* my_d_perc = &d_perc_buf[org * PERC_ELEMS];

    for (int cell = tid; cell < CELLS; cell += BWD_THREADS) {
        int y, x;
        cell_yx(cell, y, x);

        float perc[PERC_DIM];
        nca::sample_neighborhood(rc, W_perc, y, x, perc);

        float pre_hidden[HIDDEN_DIM];
        float hidden[HIDDEN_DIM];
        #pragma unroll
        for (int h = 0; h < HIDDEN_DIM; ++h) {
            float acc = 0.f;
            for (int p = 0; p < PERC_DIM; ++p) {
                acc += W_inter[p * HIDDEN_DIM + h] * perc[p];
            }
            pre_hidden[h] = acc;
            hidden[h] = nca::gelu_approx(acc);
        }

        float d_state[CA_CHANNELS];
        for (int c = 0; c < CA_CHANNELS; ++c) {
            d_state[c] = dA[cell * CA_CHANNELS + c];
        }

        // dW_flow
        float* dW_flow = &grads[org].dW[OFF_FLOW];
        for (int h = 0; h < HIDDEN_DIM; ++h) {
            for (int c = 0; c < CA_CHANNELS; ++c) {
                atomicAdd(&dW_flow[h * CA_CHANNELS + c], hidden[h] * d_state[c]);
            }
        }

        // d_hidden = W_flow^T * d_state
        float d_hidden[HIDDEN_DIM];
        for (int h = 0; h < HIDDEN_DIM; ++h) {
            float acc = 0.f;
            for (int c = 0; c < CA_CHANNELS; ++c) {
                acc += W_flow[h * CA_CHANNELS + c] * d_state[c];
            }
            d_hidden[h] = acc;
        }

        float d_pre_hidden[HIDDEN_DIM];
        for (int h = 0; h < HIDDEN_DIM; ++h) {
            d_pre_hidden[h] = d_hidden[h] * gelu_derivative(pre_hidden[h]);
        }

        // dW_inter
        float* dW_inter = &grads[org].dW[OFF_INTER];
        for (int p = 0; p < PERC_DIM; ++p) {
            for (int h = 0; h < HIDDEN_DIM; ++h) {
                atomicAdd(&dW_inter[p * HIDDEN_DIM + h], perc[p] * d_pre_hidden[h]);
            }
        }

        // d_perc = W_inter^T * d_pre_hidden
        float d_perc_local[PERC_DIM];
        for (int p = 0; p < PERC_DIM; ++p) {
            float acc = 0.f;
            for (int h = 0; h < HIDDEN_DIM; ++h) {
                acc += W_inter[p * HIDDEN_DIM + h] * d_pre_hidden[h];
            }
            d_perc_local[p] = acc;
        }

        // dW_perc
        float* dW_perc = &grads[org].dW[OFF_PERC];
        for (int f = 0; f < N_PERC_FILTERS; ++f) {
            for (int k = 0; k < 9; ++k) {
                int ky = (k / 3) - 1;
                int kx = (k % 3) - 1;
                int ny = (y + ky + GRID_SIZE) % GRID_SIZE;
                int nx = (x + kx + GRID_SIZE) % GRID_SIZE;
                float acc = 0.f;
                for (int c = 0; c < CA_CHANNELS; ++c) {
                    acc += d_perc_local[f * CA_CHANNELS + c] *
                           __half2float(rc[grid_idx(ny, nx, c)]);
                }
                atomicAdd(&dW_perc[f * 9 + k], acc);
            }
        }

        // Store d_perc for Phase B gather.
        for (int p = 0; p < PERC_DIM; ++p) {
            my_d_perc[cell * PERC_DIM + p] = d_perc_local[p];
        }
    }
}

// Sub-kernel 7 (Phase B): Stencil adjoint gather.
// eff_weights (nullable) selects each organism's own W_perc.
__global__ void bwd_stencil_gather_kernel(
    const float* weights,
    const float* eff_weights,    // [n_banks * TOTAL_WEIGHTS] or null
    const float* d_state_A,    // [n_org * GRID_ELEMS] — d_state_next (read)
    float* d_state_B,          // [n_org * GRID_ELEMS] — d_state_curr (write)
    const float* d_perc_buf,   // [n_org * PERC_ELEMS]
    int n_organisms)
{
    int org = blockIdx.x;
    if (org >= n_organisms) return;
    int tid = threadIdx.x;
    const float* wbase = (eff_weights != nullptr)
        ? &eff_weights[static_cast<size_t>(org) * TOTAL_WEIGHTS] : weights;
    const float* W_perc = wbase + OFF_PERC;

    const float* dA = &d_state_A[org * GRID_ELEMS];
    float* dB = &d_state_B[org * GRID_ELEMS];
    const float* my_d_perc = &d_perc_buf[org * PERC_ELEMS];

    for (int cell = tid; cell < CELLS; cell += BWD_THREADS) {
        int y, x;
        cell_yx(cell, y, x);

        for (int c = 0; c < CA_CHANNELS; ++c) {
            float d_from_stencil = 0.f;

            for (int f = 0; f < N_PERC_FILTERS; ++f) {
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int ny = (y - ky + GRID_SIZE) % GRID_SIZE;
                        int nx = (x - kx + GRID_SIZE) % GRID_SIZE;
                        int ncell = ny * GRID_SIZE + nx;
                        float w = W_perc[f * 9 + (ky + 1) * 3 + (kx + 1)];
                        d_from_stencil += w * my_d_perc[ncell * PERC_DIM + f * CA_CHANNELS + c];
                    }
                }
            }

            dB[cell * CA_CHANNELS + c] = dA[cell * CA_CHANNELS + c] + d_from_stencil;
        }
    }
}

// ---- btraj_gather_kernel ------------------------------------------------
// Per cuda_engineering.md section 4.8.
// Grid: <<<n_organisms, 128>>>
// Gathers strided bmap_traj from OrganismState[] into contiguous buffer.
__global__ void btraj_gather_kernel(const OrganismState* organisms,
                                    float* btraj_out,
                                    int n_organisms) {
    int org = blockIdx.x;
    if (org >= n_organisms) return;
    constexpr int BTRAJ_TOTAL = BTRAJ_SAMPLES * BMAP_DIM;
    for (int i = threadIdx.x; i < BTRAJ_TOTAL; i += blockDim.x) {
        btraj_out[org * BTRAJ_TOTAL + i] = organisms[org].bmap_traj[i];
    }
}

// ---- materialize_effective_weights_kernel -------------------------------
// W_eff[org] = W_shared + delta[org]. Runs once per generation after the CAME
// update and spawn wave. Thread 0 of each block walks the sparse delta list
// serially so the scatter is deterministic (no atomicAdd rounding races).
// Grid: <<<n_organisms, 256>>>.
__global__ void materialize_effective_weights_kernel(
    const float* shared_weights,              // [TOTAL_WEIGHTS]
    const genome::DeltaWeights* deltas,       // [n_organisms]
    float* eff_weights,                       // [n_organisms * TOTAL_WEIGHTS]
    int n_organisms)
{
    int org = blockIdx.x;
    if (org >= n_organisms) return;

    float* wbank = &eff_weights[static_cast<size_t>(org) * TOTAL_WEIGHTS];
    for (int i = threadIdx.x; i < TOTAL_WEIGHTS; i += blockDim.x) {
        wbank[i] = shared_weights[i];
    }
    __syncthreads();

    const genome::DeltaWeights& d = deltas[org];
    if (threadIdx.x == 0) {
        for (int k = 0; k < d.count; ++k) {
            uint32_t idx = d.indices[k] % genome::TOTAL_WEIGHT_SLOTS;
            wbank[idx] += d.values[k];
        }
    }
}

// ---- state_saturation_kernel --------------------------------------------
// Telemetry: max |state| and the count of values near the FP16 limit across
// the 4 checkpoints (steps 0/16/32/48) and the final grid (step 64) for every
// organism. Grid: <<<n_organisms, 256>>>. One block per organism.
__global__ void state_saturation_kernel(
    const CheckpointBuffer* checkpoints,
    const OrganismState* organisms,
    TelemetryScalars* out,
    int n_organisms)
{
    int org = blockIdx.x;
    if (org >= n_organisms) return;

    __shared__ float s_max[256];
    __shared__ int   s_cnt[256];
    int tid = threadIdx.x;
    float mx = 0.f;
    int cnt = 0;

    // Checkpoints 0..3.
    for (int seg = 0; seg < NUM_CHECKPOINTS; ++seg) {
        const __half* ck = checkpoints[org].data[seg];
        for (int i = tid; i < GRID_ELEMS; i += blockDim.x) {
            float v = fabsf(__half2float(ck[i]));
            mx = fmaxf(mx, v);
            if (v > 60000.f) cnt++;
        }
    }
    // Final grid (step 64).
    const __half* grid = organisms[org].grid;
    for (int i = tid; i < GRID_ELEMS; i += blockDim.x) {
        float v = fabsf(__half2float(grid[i]));
        mx = fmaxf(mx, v);
        if (v > 60000.f) cnt++;
    }

    s_max[tid] = mx;
    s_cnt[tid] = cnt;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
            s_cnt[tid] += s_cnt[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(reinterpret_cast<int*>(&out->state_max_abs),
                  __float_as_int(s_max[0]));
        atomicAdd(&out->state_near_max, static_cast<float>(s_cnt[0]));
    }
}

// ---- Host API -----------------------------------------------------------

inline void allocate_checkpoints(CheckpointBuffer** d_ckpt, int n_organisms) {
    cudaMalloc(d_ckpt, sizeof(CheckpointBuffer) * n_organisms);
}

inline void free_checkpoints(CheckpointBuffer* d_ckpt) {
    cudaFree(d_ckpt);
}

inline void allocate_grad_buffers(GradBuffers** d_grads, int n_organisms) {
    cudaMalloc(d_grads, sizeof(GradBuffers) * n_organisms);
}

inline void free_grad_buffers(GradBuffers* d_grads) {
    cudaFree(d_grads);
}

inline void allocate_backward_workspace(BackwardWorkspace& ws, int n_organisms) {
    ws.n_organisms = n_organisms;
    cudaMalloc(&ws.d_state[0], sizeof(float) * GRID_ELEMS * n_organisms);
    cudaMalloc(&ws.d_state[1], sizeof(float) * GRID_ELEMS * n_organisms);
    cudaMalloc(&ws.d_perc,     sizeof(float) * PERC_ELEMS * n_organisms);
    cudaMalloc(&ws.recomp[0],  sizeof(__half) * GRID_ELEMS * n_organisms);
    cudaMalloc(&ws.recomp[1],  sizeof(__half) * GRID_ELEMS * n_organisms);
}

inline void free_backward_workspace(BackwardWorkspace& ws) {
    cudaFree(ws.d_state[0]);
    cudaFree(ws.d_state[1]);
    cudaFree(ws.d_perc);
    cudaFree(ws.recomp[0]);
    cudaFree(ws.recomp[1]);
}

inline void launch_forward_with_checkpoints(
    OrganismState* d_organisms,
    const ForwardInputs* d_fwd_inputs,
    const nca::rd::Coefficients* d_coeffs,
    const float* d_weights,
    const float* d_eff_weights,      // [n_banks * TOTAL_WEIGHTS] or null
    CheckpointBuffer* d_checkpoints,
    int n_organisms,
    cudaStream_t stream)
{
    if (n_organisms <= 0) return;
    dim3 block(16, 16);
    dim3 grid(static_cast<unsigned>(n_organisms));
    forward_with_checkpoints_kernel<<<grid, block, 0, stream>>>(
        d_organisms, d_fwd_inputs, d_coeffs,
        d_weights, d_eff_weights,
        d_checkpoints, n_organisms);
}

inline void launch_materialize_effective_weights(
    const float* d_weights,
    const genome::DeltaWeights* d_deltas,
    float* d_eff_weights,
    int n_organisms,
    cudaStream_t stream)
{
    if (n_organisms <= 0) return;
    materialize_effective_weights_kernel<<<n_organisms, 256, 0, stream>>>(
        d_weights, d_deltas, d_eff_weights, n_organisms);
}

inline void launch_state_saturation(
    const CheckpointBuffer* d_checkpoints,
    const OrganismState* d_organisms,
    TelemetryScalars* d_tel,
    int n_organisms,
    cudaStream_t stream)
{
    if (n_organisms <= 0) return;
    state_saturation_kernel<<<n_organisms, 256, 0, stream>>>(
        d_checkpoints, d_organisms, d_tel, n_organisms);
}

inline void launch_backward_all(
    OrganismState* d_organisms,
    const float* d_weights,
    const float* d_eff_weights,      // [n_banks * TOTAL_WEIGHTS] or null
    const float* d_seed_grad,
    CheckpointBuffer* d_checkpoints,
    GradBuffers* d_grads,
    BackwardWorkspace& ws,
    int n_organisms,
    cudaStream_t stream)
{
    if (n_organisms <= 0) return;
    size_t smem = CA_CHANNELS * sizeof(float);
    int N = n_organisms;

    // Step 1: Zero grads and d_state_A.
    bwd_zero_kernel<<<N, BWD_THREADS, 0, stream>>>(
        d_grads, ws.d_state[0], N);

    // Step 2: Avg-pool summary of final grid state.
    bwd_seed_avgpool_kernel<<<N, BWD_THREADS, smem, stream>>>(
        d_organisms, ws.d_state[1], N);

    // Step 3: Backprop through project_bmap, seed d_state.
    bwd_seed_scatter_kernel<<<N, BWD_THREADS, smem, stream>>>(
        d_weights, d_eff_weights, d_seed_grad, d_grads,
        ws.d_state[0], ws.d_state[1], N);

    // Step 4: Reverse segments.
    // dA = d_state[0], dB = d_state[1]. Host swaps pointers.
    float* dA = ws.d_state[0];
    float* dB = ws.d_state[1];
    __half* rc = ws.recomp[0];
    __half* rn = ws.recomp[1];

    for (int seg = 3; seg >= 0; --seg) {
        for (int local_step = CHECKPOINT_INTERVAL; local_step >= 1; --local_step) {
            // Load checkpoint.
            bwd_load_checkpoint_kernel<<<N, BWD_THREADS, 0, stream>>>(
                seg, d_checkpoints, rc, N);

            // Re-forward (local_step - 1) steps.
            for (int fwd = 0; fwd < local_step - 1; ++fwd) {
                bwd_reforward_step_kernel<<<N, BWD_THREADS, 0, stream>>>(
                    d_weights, d_eff_weights, rc, rn, N);
                // Swap curr/next.
                __half* tmp = rc; rc = rn; rn = tmp;
            }

            // Phase A: weight grads + d_perc.
            bwd_weight_grad_kernel<<<N, BWD_THREADS, 0, stream>>>(
                d_weights, d_eff_weights, rc, dA, d_grads, ws.d_perc, N);

            // Phase B: stencil gather.
            bwd_stencil_gather_kernel<<<N, BWD_THREADS, 0, stream>>>(
                d_weights, d_eff_weights, dA, dB, ws.d_perc, N);

            // Swap dA/dB.
            float* tmp2 = dA; dA = dB; dB = tmp2;
        }
    }
}

inline void launch_btraj_gather(
    const OrganismState* d_organisms,
    float* d_btraj,
    int n_organisms,
    cudaStream_t stream)
{
    if (n_organisms <= 0) return;
    btraj_gather_kernel<<<n_organisms, 128, 0, stream>>>(
        d_organisms, d_btraj, n_organisms);
}

}  // namespace slime::autodiff

#endif  // COEVO_AUTODIFF_WARP_TAPE_CU
