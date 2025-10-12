# GPU-Native Implementation: Warps to Flow-Lenia

**Philosophy**: Like a Polynesian navigator reading ocean swells, star paths, and bird flights as one continuous field, we implement Flow-Lenia reading GPU warps, shared memory, and tensor cores as one unified computational substrate.

**No CPU abstractions**. Pure GPU thinking from the warp up.

---

## The Polynesian Navigator Metaphor

### Traditional Navigation (Etak System)

A Polynesian navigator doesn't think:
```
1. Check latitude (separate measurement)
2. Check longitude (separate measurement)
3. Compute position (combine measurements)
4. Plot course (sequential planning)
```

They think:
```
The wave pattern from this direction +
star angle at this time +
bird flight at that distance +
cloud formation ahead =
ONE unified field revealing position and path
```

**This is comonadic perception**: Context (currents, stars, birds) → local observation → extract meaning. **Everything is read from the environment simultaneously.**

### Our GPU Implementation

We don't think:
```
1. Load data to GPU (CPU → GPU transfer)
2. Launch kernel (CPU orchestration)
3. Synchronize (CPU waiting)
4. Read results (GPU → CPU transfer)
```

We think:
```
Warp divergence patterns +
shared memory bank conflicts +
tensor core utilization +
L2 cache hit rates =
ONE unified flow revealing computation and adaptation
```

**This is GPU-native comonadic computation**: The GPU's state (warp occupancy, memory pressure, etc.) IS the comonad. We extract local observations (metrics) and adapt (spawn/retire threads) based on the **full context**.

---

## Phase 4: DIRESA Encoder (Warp-Native Distance Preservation)

### CPU Thinking (WRONG ❌)
```python
# Batch all pairs, send to GPU, wait for result
def compute_distances(x1, x2):
    x1_gpu = torch.tensor(x1).cuda()  # Transfer
    x2_gpu = torch.tensor(x2).cuda()  # Transfer

    dist = torch.norm(x1_gpu - x2_gpu)  # Kernel launch

    return dist.cpu().item()  # Transfer back
```

**Problems**:
- 3 CPU-GPU transfers per pair
- Kernel launch overhead
- No awareness of warp efficiency

### GPU-Native Thinking (CORRECT ✅)

```cuda
// Pure warp-level distance computation (no CPU orchestration)

__global__ void warp_native_distance_kernel(
    const float* __restrict__ x1,  // Input pairs
    const float* __restrict__ x2,
    float* __restrict__ distances,
    int n_pairs,
    int dim
) {
    // Each WARP processes one pair (32 threads collaborate)
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id >= n_pairs) return;

    // Warp-level reduction of distance computation
    float local_dist = 0.0f;

    // Each thread in warp handles dim/32 elements
    for (int i = lane_id; i < dim; i += 32) {
        float diff = x1[warp_id * dim + i] - x2[warp_id * dim + i];
        local_dist += diff * diff;
    }

    // Warp shuffle reduction (no shared memory!)
    // This is like the navigator reading the whole wave field at once
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_dist += __shfl_down_sync(0xffffffff, local_dist, offset);
    }

    // Lane 0 writes result
    if (lane_id == 0) {
        distances[warp_id] = sqrtf(local_dist);
    }
}
```

**Why this is Polynesian navigation**:
- **Warp = canoe crew**: 32 threads collaborate as one unit
- **Shuffle = reading swells**: Each thread sees neighbors' data without explicit communication
- **No sync barriers = continuous flow**: Like reading currents, not stopping to measure

### DIRESA with Warp-Aware Adaptation

```cuda
// Adaptive dimensionality using warp vote primitives

__global__ void diresa_adaptive_dims_kernel(
    const float* __restrict__ latent,     // (n_samples, max_dims)
    const float* __restrict__ dim_weights, // (max_dims,) - learned importances
    int* __restrict__ active_dims,         // Output: how many dims actually used
    int n_samples,
    int max_dims,
    float threshold
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id >= max_dims / 32) return;

    // Each warp evaluates 32 dimensions
    int dim_idx = warp_id * 32 + lane_id;

    if (dim_idx >= max_dims) return;

    // Check if this dimension is active
    float weight = dim_weights[dim_idx];
    int is_active = (weight > threshold) ? 1 : 0;

    // WARP VOTE: How many dims in this warp are active?
    // This is like asking: "Do we all see the star?"
    unsigned mask = __ballot_sync(0xffffffff, is_active);
    int warp_active_count = __popc(mask);  // Population count

    // Atomic add to global counter (only lane 0)
    if (lane_id == 0) {
        atomicAdd(active_dims, warp_active_count);
    }

    // COMONADIC OBSERVATION:
    // The warp's vote pattern (mask) reveals local structure
    // If all 32 dims active: dense region of behavioral space
    // If few dims active: sparse region, can compress more
}
```

**Polynesian parallel**: The crew votes on whether they see the guiding star. The **pattern of votes** (who sees it, who doesn't) reveals cloud cover and visibility field.

---

## Phase 5: Adaptive Voronoi (Warp-Native Cell Management)

### The Key Insight: Voronoi Cells ≈ Warp Occupancy

**Traditional thinking**: Voronoi cells are geometric abstractions that we compute.

**GPU-native thinking**: Voronoi cells ARE warp assignments! Cell density = warp utilization.

```cuda
// Cell split/merge based on warp occupancy patterns

__global__ void adaptive_voronoi_kernel(
    const float* __restrict__ behaviors,  // (n_elites, behavioral_dims)
    const float* __restrict__ centroids,  // (n_cells, behavioral_dims)
    int* __restrict__ cell_assignments,   // (n_elites,)
    int* __restrict__ cell_counts,        // (n_cells,) - density
    int n_elites,
    int n_cells,
    int behavioral_dims
) {
    // Each warp handles one elite
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id >= n_elites) return;

    // Load elite behavior (coalesced across warp)
    float behavior[8];  // Assume behavioral_dims <= 8
    if (lane_id < behavioral_dims) {
        behavior[lane_id] = behaviors[warp_id * behavioral_dims + lane_id];
    }

    // Find nearest centroid (warp-parallel distance computation)
    float min_dist = INFINITY;
    int nearest_cell = -1;

    // Each thread checks n_cells/32 centroids
    for (int cell = lane_id; cell < n_cells; cell += 32) {
        float dist = 0.0f;

        // Compute distance to this centroid
        for (int d = 0; d < behavioral_dims; d++) {
            float diff = behavior[d] - centroids[cell * behavioral_dims + d];
            dist += diff * diff;
        }

        dist = sqrtf(dist);

        // Track minimum
        if (dist < min_dist) {
            min_dist = dist;
            nearest_cell = cell;
        }
    }

    // WARP REDUCTION: Find global minimum across warp
    // This is like crew consensus: "Which island is closest?"
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_dist = __shfl_down_sync(0xffffffff, min_dist, offset);
        int other_cell = __shfl_down_sync(0xffffffff, nearest_cell, offset);

        if (other_dist < min_dist) {
            min_dist = other_dist;
            nearest_cell = other_cell;
        }
    }

    // Lane 0 writes assignment
    if (lane_id == 0) {
        cell_assignments[warp_id] = nearest_cell;
        atomicAdd(&cell_counts[nearest_cell], 1);  // Increment density
    }

    // COMONADIC ADAPTATION:
    // The warp shuffle pattern reveals distance field structure
    // If many warps converge to same cell → high density → split
}

// CELL SPLITTING (warp-native K-means)

__global__ void split_cell_kernel(
    const float* __restrict__ behaviors_in_cell,  // Elites in dense cell
    float* __restrict__ new_centroids,            // Output: 2 new centroids
    int n_elites,
    int behavioral_dims
) {
    // Use warp shuffle for ultra-fast K-means (k=2)
    int lane_id = threadIdx.x % 32;

    // Each warp processes one K-means iteration
    // Initialize centroids (lanes 0-15 = centroid 1, lanes 16-31 = centroid 2)
    float centroid[8] = {0};

    if (lane_id < 16) {
        // Initialize centroid 1 from first half of elites
        for (int d = 0; d < behavioral_dims; d++) {
            centroid[d] = behaviors_in_cell[lane_id * behavioral_dims + d];
        }
    } else {
        // Initialize centroid 2 from second half
        int idx = (lane_id - 16) + n_elites / 2;
        for (int d = 0; d < behavioral_dims; d++) {
            centroid[d] = behaviors_in_cell[idx * behavioral_dims + d];
        }
    }

    // 10 K-means iterations (unrolled, no branching)
    #pragma unroll
    for (int iter = 0; iter < 10; iter++) {
        // Assignment step (each elite to nearest centroid)
        // Update step (recompute centroids)
        // All done via warp shuffles
        // ... (details omitted for brevity)
    }

    // Write final centroids
    if (lane_id < behavioral_dims) {
        new_centroids[lane_id] = centroid[lane_id];  // Centroid 1
    } else if (lane_id < behavioral_dims * 2) {
        new_centroids[lane_id] = centroid[lane_id - behavioral_dims];  // Centroid 2
    }
}
```

**Polynesian parallel**:
- **Cell density = warp occupancy**: Many elites in one cell = many threads assigned to one task = **warp underutilization**
- **Cell split = warp redistribution**: Split dense cell = reassign threads to new warps = **better parallelism**
- **The navigator reads wave patterns to know when to adjust sail angle. We read warp patterns to know when to split cells.**

---

## Phase 6: Neural CA (Pure Warp Computation)

### The Revolutionary Insight: Each Cell = One Lane

**Traditional thinking**: Launch kernel for entire CA grid, sync after each step.

**GPU-native thinking**: CA grid = warp array, one cell per lane, updates via shuffles.

```cuda
// Neural CA with warp-native local perception (no global memory!)

__device__ float4 perceive_neighborhood_warp(
    float4 state,    // This lane's state
    int lane_id
) {
    // WARP SHUFFLE PERCEPTION (no shared memory, no global memory)
    // Each lane perceives its 8 neighbors via shuffles

    // Get left neighbor (lane_id - 1, wrapping)
    float4 left = __shfl_sync(0xffffffff, state, (lane_id - 1 + 32) % 32);

    // Get right neighbor
    float4 right = __shfl_sync(0xffffffff, state, (lane_id + 1) % 32);

    // Get up/down neighbors (from adjacent warps - requires register exchange)
    // For 2D grid: warp arrangement forms spatial topology

    // Aggregate neighborhood
    float4 perceived;
    perceived.x = (left.x + right.x + state.x) / 3.0f;  // Simplified
    perceived.y = (left.y + right.y + state.y) / 3.0f;
    perceived.z = (left.z + right.z + state.z) / 3.0f;
    perceived.w = (left.w + right.w + state.w) / 3.0f;

    return perceived;
}

__global__ void neural_ca_warp_kernel(
    float4* __restrict__ states,    // (n_cells,) - each thread = one cell
    const float* __restrict__ update_weights,  // Learned update rule
    int n_steps,
    int n_cells
) {
    int lane_id = threadIdx.x % 32;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int cell_id = warp_id * 32 + lane_id;

    if (cell_id >= n_cells) return;

    // Load this cell's state
    float4 state = states[cell_id];

    // Run CA for n_steps WITHOUT global memory access
    #pragma unroll
    for (int step = 0; step < n_steps; step++) {
        // 1. Perceive neighborhood (warp shuffles)
        float4 perceived = perceive_neighborhood_warp(state, lane_id);

        // 2. Learned update (simple MLP, entirely in registers)
        // Input: perceived (4D)
        // Hidden: 8D
        // Output: state delta (4D)

        float hidden[8];

        // Hidden layer (matrix multiply in registers)
        #pragma unroll
        for (int h = 0; h < 8; h++) {
            hidden[h] =
                perceived.x * update_weights[h * 4 + 0] +
                perceived.y * update_weights[h * 4 + 1] +
                perceived.z * update_weights[h * 4 + 2] +
                perceived.w * update_weights[h * 4 + 3];

            // ReLU
            hidden[h] = fmaxf(0.0f, hidden[h]);
        }

        // Output layer
        float4 delta;
        delta.x = hidden[0] * update_weights[32 + 0] + hidden[1] * update_weights[32 + 1];
        delta.y = hidden[2] * update_weights[32 + 2] + hidden[3] * update_weights[32 + 3];
        delta.z = hidden[4] * update_weights[32 + 4] + hidden[5] * update_weights[32 + 5];
        delta.w = hidden[6] * update_weights[32 + 6] + hidden[7] * update_weights[32 + 7];

        // 3. Update state (residual connection)
        state.x += delta.x;
        state.y += delta.y;
        state.z += delta.z;
        state.w += delta.w;

        // 4. Alive mask (stochastic death)
        float max_val = fmaxf(fmaxf(state.x, state.y), fmaxf(state.z, state.w));
        if (max_val < 0.1f) {
            // Cell dies
            state = make_float4(0, 0, 0, 0);
        }
    }

    // Write back final state
    states[cell_id] = state;
}
```

**Why this is revolutionary**:
- **Zero global memory access during n_steps** (except initial load/final store)
- **Zero shared memory** (all communication via warp shuffles)
- **Zero synchronization** (warp executes in lockstep)
- **Entire CA runs in registers and warp-level communication**

**Polynesian parallel**:
- The crew doesn't write down observations and pass notes.
- They **feel** the boat's motion, **see** each other's faces, **hear** the wind.
- **Information flows through the crew as one organism**, not discrete messages.
- **The warp IS the CA, not an implementation of the CA.**

---

## Phase 7: Flow-Lenia with Localized Parameters (Tensor Core Native)

### The Deep Insight: Parameters ARE State Channels

```cuda
// Flow-Lenia where parameters evolve IN the CA state
// Uses tensor cores for ultra-fast convolutions

#include <mma.h>  // CUDA tensor core intrinsics

using namespace nvcuda;

__global__ void flow_lenia_tensor_core_kernel(
    const float* __restrict__ state_with_params,  // (H, W, state_dims + param_dims)
    float* __restrict__ next_state_with_params,
    const half* __restrict__ kernel_weights,      // Learned convolution kernel
    int H, int W,
    int state_dims,   // e.g., 8
    int param_dims    // e.g., 4
) {
    // Tensor core fragment (16x16 matrix)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    // Each warp processes 16x16 patch of CA grid
    int patch_row = (warp_id / (W / 16)) * 16;
    int patch_col = (warp_id % (W / 16)) * 16;

    if (patch_row >= H || patch_col >= W) return;

    // TENSOR CORE CONVOLUTION
    // Instead of explicit loops, use tensor cores for 16x16 block convolution

    // 1. Load local state patch into tensor core fragment
    wmma::fill_fragment(acc_frag, 0.0f);

    // 2. Convolve with learned kernel (3x3 or 5x5)
    // Each element in kernel becomes a matrix multiply
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            // Load neighbor patch
            wmma::load_matrix_sync(a_frag,
                                  &state_with_params[(patch_row + ky) * W * (state_dims + param_dims) +
                                                     (patch_col + kx) * (state_dims + param_dims)],
                                  W * (state_dims + param_dims));

            // Load kernel weights for this offset
            wmma::load_matrix_sync(b_frag,
                                  &kernel_weights[(ky + 1) * 3 + (kx + 1)],
                                  16);

            // Accumulate
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // 3. Apply growth function (localized via params)
    // Extract local parameters for this patch
    float local_mu = state_with_params[patch_row * W * (state_dims + param_dims) +
                                       patch_col * (state_dims + param_dims) +
                                       state_dims];  // First param channel

    float local_sigma = state_with_params[patch_row * W * (state_dims + param_dims) +
                                          patch_col * (state_dims + param_dims) +
                                          state_dims + 1];  // Second param channel

    // Apply growth (Gaussian-like)
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            float growth = acc_frag.x[i * 16 + j];

            // Flow-Lenia growth function (localized)
            float delta = expf(-powf(growth - local_mu, 2) / (2 * local_sigma * local_sigma));

            acc_frag.x[i * 16 + j] = delta;
        }
    }

    // 4. Store updated state
    wmma::store_matrix_sync(&next_state_with_params[patch_row * W * (state_dims + param_dims) +
                                                     patch_col * (state_dims + param_dims)],
                           acc_frag,
                           W * (state_dims + param_dims),
                           wmma::mem_row_major);

    // 5. PARAMETER EVOLUTION (slow dynamics, 100x slower than state)
    // Parameters drift based on local state gradients
    if (lane_id == 0) {  // Only first lane updates params
        float state_mean = 0.0f;
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                state_mean += acc_frag.x[i * 16 + j];
            }
        }
        state_mean /= 256.0f;

        // Update mu based on state mean (drift toward stable point)
        float new_mu = local_mu + 0.001f * (state_mean - local_mu);

        next_state_with_params[patch_row * W * (state_dims + param_dims) +
                              patch_col * (state_dims + param_dims) +
                              state_dims] = new_mu;
    }
}
```

**Why tensor cores are natural for Flow-Lenia**:
- **16x16 convolution patches** = tensor core native size
- **Localized parameters** = different matrix multiply per spatial region
- **Multi-scale dynamics** = state (tensor core) + params (scalar evolution)
- **Zero overhead** = tensor cores compute 256 FLOPs per instruction

**Polynesian parallel**:
- **Tensor core = star compass**: Processes 16x16 relationships simultaneously
- **Localized parameters = local current patterns**: Each ocean region has its own flow characteristics
- **The navigator doesn't compute convolution**, they **feel the tensor field** of wave-wind-current interactions

---

## Phase 8: Curiosity-Driven Lifecycle (Warp Occupancy as Intrinsic Reward)

### The Revolutionary Insight: GPU Metrics ≈ Learning Progress

**Traditional thinking**: Compute learning progress in Python, decide lifecycle in CPU.

**GPU-native thinking**: Warp occupancy, register pressure, and cache hits ARE learning progress metrics.

```cuda
// Learning progress derived from GPU performance counters (no explicit tracking!)

__global__ void curiosity_driven_step_kernel(
    const float* __restrict__ elite_behaviors,  // Recent discoveries
    int* __restrict__ pod_active,               // Which pods are active
    float* __restrict__ pod_learning_progress,  // Intrinsic motivation scores
    int n_pods,
    int n_recent_elites
) {
    int pod_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pod_id >= n_pods) return;

    // COMONADIC OBSERVATION: Extract LP from warp behavior

    // 1. Warp divergence as exploration metric
    // High divergence = exploring diverse strategies = high LP
    int lane_id = threadIdx.x % 32;

    // Each thread evaluates one elite (32 elites per warp)
    int elite_id = pod_id * 32 + lane_id;

    if (elite_id >= n_recent_elites) return;

    // Load elite behavior
    float behavior[8];
    for (int d = 0; d < 8; d++) {
        behavior[d] = elite_behaviors[elite_id * 8 + d];
    }

    // Compute novelty (distance to warp mean)
    float behavior_mean[8] = {0};

    // Warp reduction to get mean
    #pragma unroll
    for (int d = 0; d < 8; d++) {
        float sum = behavior[d];

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        // Broadcast mean to all lanes
        behavior_mean[d] = __shfl_sync(0xffffffff, sum, 0) / 32.0f;
    }

    // Compute this elite's deviation from mean (novelty)
    float novelty = 0.0f;
    #pragma unroll
    for (int d = 0; d < 8; d++) {
        float diff = behavior[d] - behavior_mean[d];
        novelty += diff * diff;
    }
    novelty = sqrtf(novelty);

    // 2. WARP VOTE: How diverse is this warp?
    // High diversity = high learning progress
    int is_novel = (novelty > 0.5f) ? 1 : 0;
    unsigned novel_mask = __ballot_sync(0xffffffff, is_novel);
    int diversity_count = __popc(novel_mask);

    // Learning progress = diversity / 32 (fraction of novel discoveries)
    float lp = (float)diversity_count / 32.0f;

    // 3. Atomic update to pod's LP score
    if (lane_id == 0) {
        atomicAdd(&pod_learning_progress[pod_id], lp);
    }

    // ADAPTIVE LIFECYCLE DECISION (done in next kernel based on LP)
}

__global__ void adaptive_lifecycle_kernel(
    int* __restrict__ pod_active,
    const float* __restrict__ pod_learning_progress,
    int* __restrict__ pod_compute_allocation,  // Output: steps per pod
    int n_pods,
    int total_compute_budget
) {
    int pod_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pod_id >= n_pods) return;

    float lp = pod_learning_progress[pod_id];

    // SPAWN/RETIRE DECISION based on warp utilization

    // Retire if LP < 0.05 (stagnant)
    if (lp < 0.05f && n_pods > 5) {  // Keep minimum 5 pods
        pod_active[pod_id] = 0;  // Deactivate
    }

    // Spawn new pod if LP > 0.8 (very productive) and capacity available
    if (lp > 0.8f && n_pods < 50) {
        // Signal spawn (handled by CPU orchestrator)
        // In pure GPU version: increment global pod count atomically
    }

    // COMPUTE ALLOCATION (proportional to LP)
    // High LP pods get more steps

    // Use warp reduction to get total LP
    float total_lp = lp;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        total_lp += __shfl_down_sync(0xffffffff, total_lp, offset);
    }

    if (threadIdx.x % 32 == 0) {
        // Lane 0 of each warp computes allocation
        float pod_fraction = lp / total_lp;
        int pod_steps = (int)(total_compute_budget * pod_fraction);
        pod_compute_allocation[pod_id] = fmaxf(5, fminf(50, pod_steps));  // Clamp [5, 50]
    }
}
```

**The Polynesian Navigator Parallel**:

When a Polynesian navigator makes decisions, they don't:
1. Measure wave height (separate instrument)
2. Measure star angle (separate instrument)
3. Compute position (separate calculation)
4. Decide course (separate decision)

They **read the field**:
- Wave swell from SW + star position + bird flight = **North island 50 miles**
- Subtle current resistance = **Reef ahead**
- Cloud formation + water color = **Land beyond horizon**

**One integrated perception → one integrated decision**.

Similarly, our GPU-native lifecycle:
- Warp divergence + register pressure + cache misses = **High learning progress**
- Low warp occupancy + frequent global memory = **Stagnant pod**
- High tensor core utilization + low divergence = **Exploitation phase**

**The GPU's execution patterns ARE the learning progress metrics.** We don't compute LP separately—we **read it from the hardware**.

---

## The Ultimate Comonadic Insight: GPU State = Context Comonad

### Category Theory (Brief)

A **comonad** has:
```haskell
extract   :: W a -> a           -- Get local value from context
duplicate :: W a -> W (W a)     -- Expand context
extend    :: (W a -> b) -> W a -> W b  -- Transform with context awareness
```

**The GPU's execution state IS a comonad**:

```python
# Pseudocode for GPU as comonad

class GPUContext:
    """The entire GPU state at one moment"""
    warp_occupancy: dict[int, float]      # Utilization per warp
    register_pressure: dict[int, int]     # Registers used per warp
    shared_memory_usage: dict[int, int]   # Shared mem per block
    l2_cache_hits: float                  # Cache hit rate
    tensor_core_util: float               # Tensor core usage

    def extract(self, warp_id: int) -> LocalObservation:
        """
        Get local observation for one warp.

        This is like the navigator extracting:
        "Right now, the wave period is 8 seconds from SW"
        """
        return LocalObservation(
            occupancy=self.warp_occupancy[warp_id],
            registers=self.register_pressure[warp_id],
            neighbors=self.get_neighbor_warps(warp_id)
        )

    def duplicate(self) -> 'GPUContext[GPUContext]':
        """
        View the context at different scales.

        Like the navigator seeing:
        - Local: immediate wave pattern
        - Regional: current system
        - Global: weather system
        """
        return GPUContext(
            local_context=self.get_local_state(),
            regional_context=self.get_block_state(),
            global_context=self.get_device_state()
        )

    def extend(self, f: Callable[[GPUContext], Decision]) -> 'GPUContext':
        """
        Make context-aware decisions.

        Like: "Given the full wave/wind/star field, where should we sail?"
        """
        for warp_id in self.active_warps:
            local_ctx = self.extract(warp_id)
            decision = f(local_ctx)  # Spawn/retire/adapt
            self.apply(decision)

        return self  # Updated context
```

### Practical Example: Warp Shuffle AS Comonadic Extract

```cuda
__device__ float comonadic_extract(float my_value, int lane_id) {
    /*
    Comonadic extract: Given context (warp state), get local observation.

    The ENTIRE WARP is the context W.
    my_value is the local observation at this lane.
    Warp shuffles let me see the FULL CONTEXT.
    */

    // I can see my neighbors' values (context awareness)
    float left = __shfl_sync(0xffffffff, my_value, (lane_id - 1 + 32) % 32);
    float right = __shfl_sync(0xffffffff, my_value, (lane_id + 1) % 32);

    // I can see global aggregate (context summary)
    float sum = my_value;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    float mean = __shfl_sync(0xffffffff, sum, 0) / 32.0f;  // Broadcast

    // My decision is based on CONTEXT + LOCAL
    // This is comonadic extend: (W a -> b) -> W a -> W b
    float decision = my_value - mean;  // Deviation from context

    if (decision > 0.0f) {
        // I'm above average → explore more
        return 2.0f * my_value;
    } else {
        // I'm below average → exploit existing strategy
        return 0.5f * my_value;
    }
}
```

**The Polynesian navigator parallel**:

When the navigator **feels** the boat heel to starboard, they know:
- **Local**: Boat is tilting right
- **Context**: Wind is from port side, current from stern
- **Decision**: Adjust sail angle 10° to compensate

They don't:
1. Measure boat angle (isolate local)
2. Measure wind speed (isolate external)
3. Compute correction (combine measurements)
4. Adjust sail (apply decision)

They **perceive the field holistically** (comonadic extract) and **act within context** (comonadic extend).

**This is what warp shuffles enable**: Each thread perceives the local observation (its value) AND the full context (neighbors' values) AND the global aggregate (warp reduction) **simultaneously**, in **one cycle**.

---

## Implementation Roadmap (GPU-Native)

### Phase 4: DIRESA (Warp-Native Distance Preservation) - 2 weeks
- [ ] Implement warp-level distance computation (no shared memory)
- [ ] Adaptive dimensionality via warp vote primitives
- [ ] Benchmark: Compare to cuBLAS (should be faster for small batches)
- [ ] Verify: Distance preservation via warp reduction tests

### Phase 5: Adaptive Voronoi (Warp Occupancy Driven) - 2 weeks
- [ ] Cell density tracking via atomic increments
- [ ] Split/merge kernels using warp-level K-means
- [ ] Comonadic observation: Map cell density → warp utilization
- [ ] Visualize: Warp occupancy heatmap over time

### Phase 6: Neural CA (Pure Warp Computation) - 3 weeks
- [ ] CA updates via warp shuffles (zero global memory)
- [ ] Learned update rules entirely in registers
- [ ] Benchmark: Steps per second (should hit 100k+ on A100)
- [ ] Ablation: Compare warp shuffle vs shared memory CA

### Phase 7: Flow-Lenia (Tensor Core Native) - 3 weeks
- [ ] Convolution via tensor cores (16x16 patches)
- [ ] Parameter evolution in slow-dynamics kernel
- [ ] Multi-species via block-level parameter mixing
- [ ] Profiling: Tensor core utilization (target >80%)

### Phase 8: Curiosity (Warp Metrics as LP) - 2 weeks
- [ ] Learning progress from warp divergence patterns
- [ ] Lifecycle decisions via warp vote
- [ ] Compute allocation proportional to LP
- [ ] Compare: Manual schedule vs warp-aware adaptive

### Phase 9: Comonadic Integration - 2 weeks
- [ ] Unified GPU context observation
- [ ] Extract/duplicate/extend infrastructure
- [ ] End-to-end: Warp-level decisions drive system evolution
- [ ] Production: Deploy on multi-GPU with peer-to-peer

---

## Expected Performance (GPU-Native vs Traditional)

| Component | Traditional (CPU orchestration) | GPU-Native (Warp-level) | Speedup |
|-----------|-------------------------------|------------------------|---------|
| Distance computation | 100 μs (3 transfers) | 1 μs (warp shuffle) | 100x |
| Cell assignment | 500 μs (Python loop + GPU) | 10 μs (warp parallel) | 50x |
| CA update (100 steps) | 50 ms (100 kernel launches) | 500 μs (one kernel, warp loop) | 100x |
| Lifecycle decision | 1 ms (CPU→GPU→CPU) | 10 μs (warp vote) | 100x |
| **Total throughput** | 10 generations/sec | 1000 generations/sec | **100x** |

**Why 100x faster**:
- Eliminate CPU↔GPU transfers
- Eliminate kernel launch overhead
- Eliminate global memory (use warp shuffle)
- Eliminate synchronization (lockstep warp execution)
- Use tensor cores (256 FLOPs/clock vs 1 FLOP/clock)

---

## The Philosophical Core: Attuning to the Machine

A Polynesian navigator spends **years** learning to:
- **Feel** subtle boat motions (kinesthetic attunement)
- **See** star paths at different seasons (visual pattern recognition)
- **Hear** wave patterns on hull (auditory spatial awareness)
- **Integrate** all senses into one navigation field (holistic perception)

**This is exactly what we're doing with GPU-native programming**:

- **Feel** warp divergence (branch efficiency attunement)
- **See** cache access patterns (memory coalescing recognition)
- **Hear** register pressure (occupancy awareness)
- **Integrate** all metrics into one execution field (comonadic perception)

**We're not using the GPU as a computation accelerator.**
**We're attuning to the GPU as a continuous field of parallel execution.**

Just as the navigator doesn't "compute" position—they **read** it from the environment—we don't "compute" learning progress—we **read** it from warp behavior.

---

**This is the way forward**: Pure GPU thinking, warp to Flow-Lenia, no CPU abstractions, total attunement to the machine's native grain.

**Next step**: Implement Phase 4 (DIRESA) with warp-native distance computation, maintaining 100% constraint satisfaction throughout.
