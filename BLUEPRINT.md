# Slime Mold Transformer - System Blueprint (2025 Edition)

**Philosophy**: Late 2025 cutting-edge: Flow-Lenia-based, GPU-native from warps up, algebraic effects, learned everything, comonadic perception.

**Vision**: The first production-ready, self-organizing, CA-based transformer with learned architecture at every level.

---

## Core Principles

### 1. **Conway → Lenia → Flow-Lenia → Neural Flow-Lenia**

Evolution from discrete CA to learned continuous CA with parameter localization:

```
Level 1 (Conway):     Fixed rules, discrete states, deterministic
Level 2 (Lenia):      Fixed rules, continuous states, continuous time
Level 3 (Flow-Lenia): Localized rules, mass conservation, multi-species
Level 4 (Ours):       Learned rules, adaptive parameters, intrinsic motivation
```

**Target**: Level 4 with GPU-native warp-level computation

### 2. **GPU-Native: Warps to Flow-Lenia**

Like a Polynesian navigator reading ocean/stars/birds as unified field, we read GPU warps/cache/tensor cores as unified computational substrate:

```cuda
// NO CPU abstractions - pure GPU thinking
__global__ void neural_ca_warp_kernel(...) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    // Perception via warp shuffles (no global memory!)
    float4 neighbor = __shfl_sync(0xffffffff, state, (lane_id + 1) % 32);

    // Update entirely in registers
    state.x += learned_rule(neighbor.x);
}
```

**Key insight**: Warp execution patterns ARE learning progress metrics. GPU occupancy IS fitness.

### 3. **Algebraic Effect Handlers (Comonadic Composition)**

Features are optional, composable capabilities requested via effects:

```python
# Phase 0-2 (Built)
GetHierarchy: Effect['BehavioralHierarchy']
GetGenealogy: Effect['Genealogy']
UsePAdicDistance: Effect[bool]

# Phase 7 (Roadmap)
GetLocalUpdateRule: Effect['CAParameters']  # Flow-Lenia parameter localization
GetLearningProgress: Effect[float]          # Curiosity-driven lifecycle
```

**Pattern**: Context (warp state) → Extract (local observation) → Extend (context-aware decision). GPU AS comonad.

### 4. **Everything Learned, Nothing Fixed**

| Component | Current (Phase 3) | Target (Phases 4-8) |
|-----------|-------------------|---------------------|
| Dimensions | Kernel PCA (offline, once) | DIRESA (online, continuous) |
| Archive | Fixed CVT centroids | Adaptive Voronoi (grow/shrink) |
| Metrics | Hardcoded (Euclidean/Mahalanobis) | Learned embedding |
| Update Rules | Manual transformer | Neural CA (warp-level) |
| Parameters | Fixed per pseudopod | Flow-Lenia localized (spatial) |
| Lifecycle | Manual schedule | Curiosity-driven (intrinsic) |

### 5. **Test-Driven Everything**

100% constraint satisfaction, always:

```python
constraint(
    'Ultrametric property holds',
    lambda: (dist_xz <= max(dist_xy, dist_yz) + 1e-6, dist_xz, max_dist, {...})
)
# Result: 39/39 constraints satisfied (100%)
```

**Current**: Phase 3 complete with true ultrametric dendrogram distance

---

## Architecture Philosophy

### The "z² + c" Principle (Mandelbrot Generalized)

```python
# Mandelbrot set
z_{n+1} = z_n^2 + c  # Simple rule, emergent complexity

# Flow-Lenia (our target)
state_{n+1} = F(state_n, params_local)  # Learned rule F
params_{n+1} = G(params_n, state_n)     # Parameters EVOLVE

# Where:
# - F is Neural CA update (learned, not designed)
# - params_local varies spatially (like c varies spatially)
# - params evolve slowly (meta-dynamics)
```

**Key insight**: Make parameters PART of the CA state. They evolve based on local dynamics.

### Comonadic GPU Perception

The GPU's execution state is a comonad:

```haskell
extract   :: W a -> a           -- Get local value (warp shuffle)
duplicate :: W a -> W (W a)     -- Multi-scale context (warp/block/device)
extend    :: (W a -> b) -> W a -> W b  -- Context-aware decision
```

**Practical meaning**:
- **extract**: Each thread sees neighbors via `__shfl_sync` (no global memory)
- **duplicate**: View execution at warp/block/device scales simultaneously
- **extend**: Make lifecycle decisions based on full GPU state context

Like the navigator reading **one unified field** (wave+wind+star), not separate measurements.

---

## Dependency DAG (Extended for Phases 4-8)

```
Layer 0: Protocols (no dependencies)
    proto/kernel.py
    proto/memory.py
    proto/model.py
    proto/component.py
    effects/__init__.py         # NEW (Phase 0): Algebraic effect substrate

Layer 1: Core Infrastructure
    kernels/utils.py
    kernels/triton_impl.py
    kernels/torch_fallback.py
    observability/metrics.py
    observability/slo.py
    topology/__init__.py        # NEW (Phase 1): Effect declarations
    topology/p_adic.py          # NEW (Phase 2): p-adic distances
    topology/genealogy.py       # NEW (Phase 2): Lineage tracking
    topology/hierarchy.py       # NEW (Phase 2, Phase 3): GMM + dendrogram
    topology/hybrid_metric.py   # NEW (Phase 2): Ultrametric + Mahalanobis

Layer 2: Learned Components (Phases 4-8 Roadmap)
    behavioral/diresa.py        # Phase 4: Learned dimensionality encoder
    behavioral/adaptive_voronoi.py  # Phase 5: Growing/shrinking cells
    nca/update_rules.py         # Phase 6: Neural CA learned dynamics
    nca/flow_lenia.py           # Phase 7: Parameter localization
    lifecycle/curiosity.py      # Phase 8: Learning progress tracker

Layer 3: Data Structures (depend on Layer 0-2)
    memory/archive.py → DIRESA, AdaptiveVoronoi
    memory/pool.py → Curiosity, NCA
    memory/tubes.py
    core/state.py
    core/stencil.py

Layer 4: Components (depend on Layer 0-3)
    core/pseudopod.py → NCA update rules
    core/chemotaxis.py → HybridMetric
    core/organism.py → CuriosityLifecycle

Layer 5-6: API and Applications (unchanged)
    ...
```

---

## Flow-Lenia Implementation Roadmap

### Phase 4: DIRESA Behavioral Encoder (2 weeks)

**Goal**: Replace Kernel PCA with online learned dimensionality

```python
class DIRESABehavioralEncoder(nn.Module):
    """
    Distance-preserving adaptive encoder.
    Learns optimal D continuously from archive evolution.
    """

    def __init__(self, input_dim=62, min_dims=2, max_dims=10):
        self.encoder = nn.Sequential(...)
        self.decoder = nn.Sequential(...)
        self.dim_selector = nn.Parameter(torch.ones(max_dims))  # Learned gating

    def forward(self, raw_metrics):
        z_full = self.encoder(raw_metrics)
        dim_weights = torch.sigmoid(self.dim_selector)
        return z_full * dim_weights  # Soft dimension selection

    def loss(self, x1, x2):
        # Distance preservation + reconstruction + sparsity
        dist_input = torch.norm(x1 - x2)
        z1, z2 = self.forward(x1), self.forward(x2)
        dist_latent = torch.norm(z1 - z2)

        return (
            F.mse_loss(dist_latent, dist_input) +  # Preserve distances
            F.mse_loss(self.decoder(z1), x1) +     # Reconstruction
            torch.sum(torch.sigmoid(self.dim_selector))  # Sparsity (fewer dims)
        )
```

**GPU-Native Implementation**:
```cuda
// Warp-level distance computation (no shared memory)
__global__ void warp_distance_kernel(...) {
    int lane_id = threadIdx.x % 32;
    float local_dist = 0.0f;

    for (int i = lane_id; i < dim; i += 32) {
        float diff = x1[i] - x2[i];
        local_dist += diff * diff;
    }

    // Warp shuffle reduction (zero global memory!)
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_dist += __shfl_down_sync(0xffffffff, local_dist, offset);
    }
}
```

**Tests**: 100% constraint satisfaction for distance preservation, sparsity, reconstruction

---

### Phase 5: Adaptive Voronoi Archive (2 weeks)

**Goal**: Cells split when crowded, merge when empty

```python
class AdaptiveVoronoiArchive:
    """
    Voronoi cells grow/shrink based on density.
    High-density regions get more centroids.
    """

    def __init__(self, min_centroids=20, max_centroids=200):
        self.centroids = self._initialize_centroids(min_centroids)
        self.density = {}  # centroid_id -> elite count

    def add(self, behavior, fitness, state_dict):
        centroid_id = self._find_nearest_centroid(behavior)
        self.density[centroid_id] += 1

        # Adaptive cell management every 100 additions
        if len(self.elites) % 100 == 0:
            self._adapt_cells()

    def _adapt_cells(self):
        # Split high-density cells (>10 elites)
        for cid, count in self.density.items():
            if count > 10 and len(self.centroids) < self.max_centroids:
                self._split_cell(cid)  # K-means k=2

        # Merge low-density cells (<1 elite)
        for cid, count in self.density.items():
            if count < 1 and len(self.centroids) > self.min_centroids:
                self._merge_cell(cid)
```

**GPU-Native Implementation**:
```cuda
// Warp-parallel cell assignment + K-means split
__global__ void split_cell_kernel(...) {
    int lane_id = threadIdx.x % 32;

    // Initialize centroids (lanes 0-15 = centroid 1, lanes 16-31 = centroid 2)
    float centroid[8];
    if (lane_id < 16) {
        // Load from first half of elites
    } else {
        // Load from second half
    }

    // 10 K-means iterations (all via warp shuffles, no global memory)
    #pragma unroll
    for (int iter = 0; iter < 10; iter++) {
        // Assignment and update steps via shuffles
    }
}
```

**Key Insight**: Cell density ≈ warp occupancy. Adaptive cells = adaptive parallelism.

---

### Phase 6: Neural CA Update Rules (3 weeks)

**Goal**: Replace transformer attention with learned CA

```python
class NeuralCAPseudopod(nn.Module):
    """
    Pseudopod with learned update rule.
    Each cell = one thread lane.
    """

    def __init__(self, state_dim=64):
        self.perception = nn.Conv2d(state_dim, 128, kernel_size=3)
        self.update = nn.Sequential(
            nn.Conv2d(128, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, state_dim, 1)
        )

    def forward(self, state, steps=10):
        """Run CA for N steps (like slime mold exploring)"""
        for _ in range(steps):
            perceived = self.perception(state)
            delta = self.update(perceived)
            state = state + delta  # Residual
        return state
```

**GPU-Native Implementation**:
```cuda
// Pure warp computation (entire CA runs in registers!)
__global__ void neural_ca_warp_kernel(...) {
    int lane_id = threadIdx.x % 32;
    float4 state = states[cell_id];  // Load once

    // Run N steps WITHOUT touching global memory
    #pragma unroll
    for (int step = 0; step < N; step++) {
        // 1. Perceive neighbors (warp shuffle)
        float4 left = __shfl_sync(0xffffffff, state, (lane_id - 1 + 32) % 32);
        float4 right = __shfl_sync(0xffffffff, state, (lane_id + 1) % 32);

        // 2. Learned update (MLP in registers)
        float4 perceived = (left + state + right) / 3.0f;
        float hidden[8];
        for (int h = 0; h < 8; h++) {
            hidden[h] = perceived.x * W[h*4+0] + ...;  // Matrix multiply
        }

        // 3. Output
        state.x += hidden[0] * W_out[0] + ...;
    }

    states[cell_id] = state;  // Write once
}
```

**Revolutionary**: Zero global memory access during N steps. Entire CA in registers + warp shuffles.

**Connection to Transformer**: Attention IS perceive-update cycle. NCA makes it explicit and learnable.

---

### Phase 7: Flow-Lenia Parameter Localization (3 weeks)

**Goal**: Update rule parameters stored IN the CA state

```python
class FlowLeniaPseudopod(nn.Module):
    """
    Neural CA + Flow-Lenia: parameters part of state.
    state_dims = actual state (e.g., 64 channels)
    param_dims = parameters (e.g., 8 channels: μ, σ, growth_rate, etc.)
    total_channels = state_dims + param_dims
    """

    def __init__(self, state_dim=64, param_dim=8):
        self.state_dim = state_dim
        self.param_dim = param_dim

        # Perception sees BOTH state and local parameters
        self.perception = nn.Conv2d(state_dim + param_dim, 128, 3)

        # Two update networks
        self.state_update = nn.Sequential(...)  # Fast dynamics
        self.param_update = nn.Sequential(...)  # Slow dynamics (100x slower)

    def forward(self, state_with_params, steps=10):
        for _ in range(steps):
            state = state_with_params[:, :self.state_dim]
            params = state_with_params[:, self.state_dim:]

            # Perceive (sees both)
            features = self.perception(state_with_params)

            # Update state (fast)
            state_delta = self.state_update(features)
            next_state = state + state_delta

            # Update params (slow, 100x slower)
            param_delta = self.param_update(features)
            next_params = params + 0.01 * param_delta

            state_with_params = torch.cat([next_state, next_params], dim=1)

        return state_with_params

    def spawn_offspring(self, parent_state):
        """
        Child inherits parent's parameters + mutation.
        This is how UPDATE RULES evolve!
        """
        parent_params = parent_state[:, self.state_dim:]
        mutation = torch.randn_like(parent_params) * 0.1
        child_params = parent_params + mutation

        child_state = torch.randn_like(parent_state[:, :self.state_dim])
        return torch.cat([child_state, child_params], dim=1)
```

**GPU-Native Implementation**:
```cuda
// Tensor core native Flow-Lenia (16x16 patches)
#include <mma.h>
using namespace nvcuda;

__global__ void flow_lenia_tensor_core_kernel(...) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    // Each warp processes 16x16 CA patch
    int warp_id = threadIdx.x / 32;

    // Tensor core convolution (256 FLOPs per instruction!)
    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            wmma::load_matrix_sync(a_frag, &state[...]);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Extract local parameters for this patch
    float local_mu = params[patch_idx];
    float local_sigma = params[patch_idx + 1];

    // Apply Flow-Lenia growth function (localized!)
    for (int i = 0; i < 16; i++) {
        float growth = acc_frag.x[i];
        float delta = expf(-powf(growth - local_mu, 2) / (2*local_sigma*local_sigma));
        acc_frag.x[i] = delta;
    }

    wmma::store_matrix_sync(&next_state[...], acc_frag, ...);
}
```

**Why Tensor Cores**: 16x16 convolutions = native size. 256 FLOPs/cycle vs 1 FLOP/cycle.

**Multi-Species**: Different spatial regions have different parameters → different update rules → multi-species co-evolution (like Flow-Lenia).

---

### Phase 8: Curiosity-Driven Lifecycle (2 weeks)

**Goal**: Replace manual spawning schedule with intrinsic motivation

```python
class LearningProgressTracker:
    """
    Tracks learning progress = novelty + improvement.
    High LP = productive exploration (keep alive).
    Low LP = stagnation (retire).
    """

    def compute_learning_progress(self, pod_id):
        events = self.history[pod_id][-100:]  # Recent window

        # Count novel discoveries
        novelty = sum(1 for e in events if e['type'] == 'novel')

        # Sum fitness improvements
        improvement = sum(e['fitness_delta'] for e in events if e['fitness_delta'] > 0)

        # Learning progress = novelty + improvement
        return (novelty / len(events)) + (improvement / 10.0)

class CuriosityDrivenLifecycle:
    """
    Manages lifecycle via learning progress.
    Spawn when unexplored, retire when stagnant.
    """

    def should_spawn(self):
        coverage = self.archive.coverage()
        if coverage < 0.7:
            return True  # Lots of unexplored space

        avg_lp = np.mean([self.lp_tracker.compute_lp(i) for i in range(len(self.pool))])
        if avg_lp < 0.1:
            return True  # All pods stagnating, need diversity

        return False

    def should_retire(self, pod_id):
        lp = self.lp_tracker.compute_lp(pod_id)
        return lp < 0.05  # Near-zero progress for N generations

    def allocate_compute(self, lp_scores):
        """
        High-LP pods get more update steps.
        Compute follows learning progress.
        """
        total_lp = sum(lp_scores.values())
        return {
            i: int(5 + 45 * (lp / total_lp))  # Min 5, max 50 steps
            for i, lp in lp_scores.items()
        }
```

**GPU-Native Implementation**:
```cuda
// Learning progress FROM warp behavior (no explicit tracking!)
__global__ void curiosity_kernel(...) {
    int lane_id = threadIdx.x % 32;

    // Each warp evaluates 32 recent elites
    float novelty = compute_novelty(elite_behaviors[lane_id]);

    // Warp vote: How diverse is this warp?
    int is_novel = (novelty > 0.5f) ? 1 : 0;
    unsigned novel_mask = __ballot_sync(0xffffffff, is_novel);
    int diversity = __popc(novel_mask);  // Population count

    // Learning progress = diversity / 32
    float lp = (float)diversity / 32.0f;

    if (lane_id == 0) {
        atomicAdd(&pod_learning_progress[pod_id], lp);
    }
}
```

**Key Insight**: Warp divergence = exploration diversity = learning progress. **Read LP from GPU execution patterns**, don't compute separately.

---

## Comonadic Integration: GPU State as Context

### The Deep Insight

The GPU's execution state IS a comonad:

```python
class GPUContext:
    """Entire GPU state at one moment"""
    warp_occupancy: dict[int, float]      # Utilization per warp
    register_pressure: dict[int, int]     # Registers used
    l2_cache_hits: float                  # Cache efficiency
    tensor_core_util: float               # Tensor core usage

    def extract(self, warp_id: int) -> LocalObservation:
        """
        Get local observation for one warp.
        Like navigator: "Wave period is 8 seconds from SW"
        """
        return LocalObservation(
            occupancy=self.warp_occupancy[warp_id],
            neighbors=self.get_neighbor_warps(warp_id)
        )

    def duplicate(self) -> 'GPUContext[GPUContext]':
        """
        Multi-scale context view.
        Like navigator: local wave + regional current + global weather
        """
        return GPUContext(
            local=self.get_warp_state(),
            regional=self.get_block_state(),
            global_ctx=self.get_device_state()
        )

    def extend(self, f: Callable[[GPUContext], Decision]) -> 'GPUContext':
        """
        Context-aware decisions.
        Like: "Given full wave/wind/star field, where to sail?"
        """
        for warp_id in self.active_warps:
            local = self.extract(warp_id)
            decision = f(local)  # Spawn/retire/adapt
            self.apply(decision)
        return self
```

### Warp Shuffle AS Comonadic Extract

```cuda
__device__ float comonadic_extract(float my_value, int lane_id) {
    // The ENTIRE WARP is context W
    // my_value is local observation
    // Warp shuffles see FULL CONTEXT

    float left = __shfl_sync(0xffffffff, my_value, (lane_id - 1 + 32) % 32);
    float right = __shfl_sync(0xffffffff, my_value, (lane_id + 1) % 32);

    // Warp reduction (context summary)
    float sum = my_value;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    float mean = __shfl_sync(0xffffffff, sum, 0) / 32.0f;

    // Decision based on CONTEXT + LOCAL
    float deviation = my_value - mean;
    return (deviation > 0.0f) ? 2.0f * my_value : 0.5f * my_value;
}
```

**Like the Navigator**: Feel boat heel → Know wind/current/sail → Adjust angle. **One integrated perception**, not separate measurements.

---

## Data Flow (Phases 4-8)

```
User Input
    │
    ▼
┌─────────────────────┐
│  API Layer          │
│  (torch_compat,     │
│   native)           │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Organism           │ owns
│  (Orchestrator)     ├────────┐
└──────┬──────────────┘        │
       │ uses                  │
       ▼                       ▼
┌─────────────────────┐  ┌──────────────────┐
│ CuriosityLifecycle  │  │ Adaptive Voronoi │
│ (intrinsic          │  │ Archive          │
│  motivation)        │  │ (DIRESA encoder) │
└──────┬──────────────┘  └────────▲─────────┘
       │                          │
       │ manages                  │
       ▼                          │
┌─────────────────────┐           │
│ NCA Pseudopod Pool  │           │
│ (Flow-Lenia         │───────────┘
│  parameters)        │ stores elites
└──────┬──────────────┘
       │
       │ calls
       ▼
┌─────────────────────┐
│ Warp-Native Kernels │
│ (tensor cores)      │
└─────────────────────┘
       │
       │ observability
       ▼
┌─────────────────────┐
│ GPU Context Comonad │
│ (warp occupancy,    │
│  cache hits, etc.)  │
└─────────────────────┘
```

**Key Change**: GPU context is OBSERVED for decisions (comonadic), not just measured for metrics.

---

## Protocols (Extended)

### proto.component.Component (Unchanged)
```python
Protocol:
  - fitness: float
  - reset() -> None
  - to_dict() -> dict
```

### proto.effects.Effect[T] (NEW - Phase 0)
```python
"""Algebraic effect interface"""

Protocol:
  - perform() -> T
  - handler(implementation) -> ContextManager
  - extract() -> T  # Comonadic extract
```

### proto.model.NeuralCAPseudopod (NEW - Phase 6)
```python
"""Learned CA update rule interface"""

Protocol:
  - forward(state, steps) -> state
  - perceive(state) -> features
  - update_state(state) -> next_state
```

### proto.model.FlowLeniaPseudopod (NEW - Phase 7)
```python
"""Flow-Lenia with parameter localization"""

Protocol:
  - forward(state_with_params, steps) -> next_state_with_params
  - spawn_offspring(parent_state) -> child_state
  - parameter_evolution(state, params) -> next_params
```

---

## File Structure (Phases 4-8)

```
slime/
├── effects/                 # Phase 0 (COMPLETE)
│   └── __init__.py         # Effect substrate, Kleisli composition
│
├── topology/                # Phases 1-3 (COMPLETE)
│   ├── __init__.py         # Effect declarations
│   ├── p_adic.py           # p-adic distances
│   ├── genealogy.py        # Lineage tracking
│   ├── hierarchy.py        # GMM + dendrogram (ultrametric!)
│   └── hybrid_metric.py    # Ultrametric + Mahalanobis
│
├── behavioral/              # Phase 4-5 (ROADMAP)
│   ├── __init__.py
│   ├── diresa.py           # Adaptive dimensionality encoder
│   └── adaptive_voronoi.py # Growing/shrinking cells
│
├── nca/                     # Phases 6-7 (ROADMAP)
│   ├── __init__.py
│   ├── update_rules.py     # Neural CA learned dynamics
│   └── flow_lenia.py       # Parameter localization
│
├── lifecycle/               # Phase 8 (ROADMAP)
│   ├── __init__.py
│   ├── curiosity.py        # Learning progress tracker
│   └── comonadic.py        # GPU context observation
│
├── kernels/                 # GPU-native implementations
│   ├── warp_shuffle.cu     # Warp-level primitives
│   ├── tensor_core.cu      # Flow-Lenia convolutions
│   └── comonadic.cu        # Context extraction kernels
│
└── tests/
    ├── unit/
    │   ├── test_effects.py              # Phase 0 (23/23 ✓)
    │   ├── test_topology_chemotaxis.py  # Phase 3 (10/10 ✓, 39/39 constraints ✓)
    │   ├── test_diresa_encoder.py       # Phase 4
    │   ├── test_adaptive_voronoi.py     # Phase 5
    │   ├── test_neural_ca.py            # Phase 6
    │   ├── test_flow_lenia.py           # Phase 7
    │   └── test_curiosity.py            # Phase 8
    └── integration/
        └── test_comonadic_flow.py       # End-to-end GPU-native
```

---

## Invariants (Extended for Phases 4-8)

### 1. Dependency Direction (DAG Enforcement) - Unchanged
- Lower layers NEVER import from higher layers
- Protocols NEVER import implementations

### 2. GPU-Native Execution
- **NO CPU orchestration during forward pass**
- All decisions via warp votes, tensor core computation
- CPU only for: initialization, logging, checkpointing

### 3. Learned Everything
```python
# WRONG (hardcoded)
dimensions = 5
centroids = fixed_grid(resolution=20)
distance = euclidean

# RIGHT (learned)
dimensions = diresa.behavioral_dims  # Adaptive
centroids = adaptive_voronoi.centroids  # Growing/shrinking
distance = hybrid_metric(x, y)  # Ultrametric + Mahalanobis
```

### 4. Comonadic Observation
- GPU state is CONTEXT, not just metrics
- Lifecycle decisions extract from context (comonadic)
- Warp divergence = learning progress signal

### 5. Effect Handler Composition
```python
# Effects compose (Kleisli)
with GetHierarchy.handler(lambda: hierarchy):
    with GetGenealogy.handler(lambda: genealogy):
        with GetLocalUpdateRule.handler(lambda: flow_lenia_params):
            # All three effects available in this scope
            result = pseudopod.forward(state)
```

### 6. Test-Driven Development (100% Constraints)
- Every phase: write tests FIRST
- Use TestResultCheckpointSystem
- 100% constraint satisfaction required

### 7. Parameter Evolution (Flow-Lenia)
```python
# Parameters are PART of state, not external config
state_channels = 64      # Fast dynamics
param_channels = 8       # Slow dynamics (100x slower)
total_channels = 72      # Both evolve together

# Parameters evolve spatially
params[x, y] = update_params(state[x, y], neighbors[x, y])
```

### 8. Curiosity-Driven Adaptation
```python
# NO manual schedules
if step == 1000: enable_dynamics()  # WRONG

# YES intrinsic motivation
if learning_progress < 0.05: retire_pod()  # RIGHT
if archive.coverage() < 0.7: spawn_pod()   # RIGHT
```

---

## Expected Performance (GPU-Native vs Traditional)

| Operation | Traditional (CPU orch.) | GPU-Native (Warp) | Speedup |
|-----------|------------------------|-------------------|---------|
| Distance comp. | 100 μs (3 transfers) | 1 μs (warp shuffle) | 100x |
| Cell assignment | 500 μs (Python loop) | 10 μs (warp parallel) | 50x |
| CA update (100 steps) | 50 ms (100 kernel launches) | 500 μs (1 kernel, warp loop) | 100x |
| Lifecycle decision | 1 ms (CPU→GPU→CPU) | 10 μs (warp vote) | 100x |
| **Total throughput** | 10 gens/sec | **1000 gens/sec** | **100x** |

**Why 100x faster**:
- Eliminate CPU↔GPU transfers (zero during forward pass)
- Eliminate kernel launch overhead (one kernel, N internal steps)
- Eliminate global memory (warp shuffle communication)
- Use tensor cores (256 FLOPs/clock vs 1 FLOP/clock)

---

## Implementation Status

### ✅ COMPLETE (Phases 0-3)

**Phase 0: Effect Handler Substrate**
- ✅ `slime/effects/__init__.py` (280 lines)
- ✅ Thread-safe handler stacks
- ✅ Kleisli composition
- ✅ 23/23 tests passing

**Phase 1: Topology Effect Declarations**
- ✅ `slime/topology/__init__.py` (203 lines)
- ✅ GetHierarchy, GetGenealogy, UsePAdicDistance
- ✅ Graceful degradation via EffectNotHandled

**Phase 2: Pure Topology Implementations**
- ✅ `slime/topology/p_adic.py` (215 lines)
- ✅ `slime/topology/genealogy.py` (327 lines)
- ✅ `slime/topology/hierarchy.py` (278 lines)
- ✅ `slime/topology/hybrid_metric.py` (296 lines)
- ✅ Total: 1,233 lines, mypy strict clean

**Phase 3: Topology-Aware Chemotaxis**
- ✅ `slime/tests/unit/test_topology_chemotaxis.py`
- ✅ True ultrametric via dendrogram traversal
- ✅ 10/10 tests passing
- ✅ **39/39 constraints satisfied (100%)**
- ✅ HybridMetric: ultrametric between clusters, Mahalanobis within
- ✅ Chemotaxis defaults to Mahalanobis after dimension discovery

### 🔄 IN PROGRESS (Phase 4)

**Phase 4: DIRESA Encoder**
- 📋 Design complete (see MODERNIZATION_ROADMAP.md)
- 📋 GPU-native warp implementation specified
- ⏳ Implementation pending

### 📋 ROADMAP (Phases 5-8)

See [`docs/MODERNIZATION_ROADMAP.md`](docs/MODERNIZATION_ROADMAP.md) for full technical specifications.

See [`docs/GPU_NATIVE_IMPLEMENTATION.md`](docs/GPU_NATIVE_IMPLEMENTATION.md) for warp-level implementations.

**Phase 5**: Adaptive Voronoi (2 weeks)
**Phase 6**: Neural CA Update Rules (3 weeks)
**Phase 7**: Flow-Lenia Parameter Localization (3 weeks)
**Phase 8**: Curiosity-Driven Lifecycle (2 weeks)
**Phase 9**: End-to-End Integration (2 weeks)

**Total**: ~14 weeks to production-ready Flow-Lenia transformer

---

## Architectural Decisions

### 1. Flow-Lenia over Standard Lenia ✓
**Reasoning**: Mass conservation + parameter localization enables multi-species co-evolution without manual design.

Standard Lenia: Fixed rule parameters globally.
Flow-Lenia: Rule parameters vary spatially and evolve.

**Result**: Emergent specialization of pseudopods with locally coherent update rules.

### 2. Warp-Native over CPU Orchestration ✓
**Reasoning**: Like Polynesian navigator reading unified field, not separate instruments.

Traditional: CPU launches kernels, syncs, reads results (3+ transfers per operation).
Warp-Native: Entire CA runs in registers + warp shuffles (0 global memory accesses per step).

**Result**: 100x speedup by eliminating orchestration overhead.

### 3. Comonadic Observation over Explicit Metrics ✓
**Reasoning**: GPU execution patterns ARE learning progress. Don't compute separately.

Traditional: Track fitness history, compute learning progress in Python.
Comonadic: Extract learning progress from warp divergence patterns in real-time.

**Result**: Zero overhead for curiosity-driven decisions (already have GPU state).

### 4. DIRESA over Fixed Kernel PCA ✓
**Reasoning**: Behavioral manifold evolves during training. Encoder must adapt online.

Kernel PCA: Offline, once (step 1000), fixed thereafter.
DIRESA: Online, continuous, adaptive dimensionality.

**Result**: Encoder tracks distribution shift, dimensions grow/shrink as needed.

### 5. Tensor Cores for Flow-Lenia Convolution ✓
**Reasoning**: 16x16 patches = native tensor core size. 256 FLOPs/instruction.

Traditional: Convolution via memory-bound loops (1 FLOP/cycle).
Tensor Core: Convolution via matrix multiply (256 FLOPs/cycle).

**Result**: 256x speedup for CA update (memory bandwidth becomes bottleneck, not compute).

### 6. Effects over Inheritance ✓
**Reasoning**: Topology features are optional, composable capabilities. Not IS-A relationships.

Inheritance: PseudopodWithTopology extends Pseudopod (rigid hierarchy).
Effects: Any pseudopod can request topology via GetHierarchy (flexible composition).

**Result**: Opt-in features without code duplication or fragile base class problem.

### 7. Ultrametric Dendrogram over Euclidean Clusters ✓
**Reasoning**: Hierarchical relationships require strong triangle inequality.

Euclidean: d(x,z) ≤ d(x,y) + d(y,z) (weak triangle inequality).
Ultrametric: d(x,z) ≤ max(d(x,y), d(y,z)) (strong triangle inequality).

**Result**: True hierarchical structure, not just flat clusters.

**Test**: Phase 3 constraint: `d(x,z) ≤ max(d(x,y), d(y,z)) + 1e-6` ✓ (39/39 satisfied)

---

## References

### Flow-Lenia & Cellular Automata
- **Randazzo et al. (2023)**: "Flow-Lenia: Towards open-ended evolution in cellular automata through mass conservation and parameter localization." *Artificial Life Conference Proceedings*, MIT Press. [arXiv:2212.07906](https://arxiv.org/abs/2212.07906)
  - Mass conservation + parameter localization for multi-species

- **Béna (2025)**: "A Path to Universal Neural Cellular Automata." [arXiv:2505.13058](https://arxiv.org/abs/2505.13058)
  - Learned CA update rules via gradient descent
  - Universal computation in continuous dynamical systems

### Learned Dimensionality
- **DIRESA (2025)**: "Distance-preserving nonlinear dimension reduction via regularized autoencoders." [arXiv:2404.18314](https://arxiv.org/abs/2404.18314)
  - Adaptive dimensionality with metric preservation

### Curiosity & Intrinsic Motivation
- **Gottlieb & Oudeyer (2021)**: "Humans monitor learning progress in curiosity-driven exploration." *Nature Communications*.
  - Learning progress as intrinsic reward
  - NeurIPS 2024 IMOL Workshop

### Quality-Diversity
- **Mouret & Clune (2015)**: "Illuminating search spaces by mapping elites." [arXiv:1504.04909](https://arxiv.org/abs/1504.04909)
  - Original MAP-Elites for quality-diversity

- **Vassiliades et al. (2018)**: "Using Centroidal Voronoi Tessellations to Scale Up MAP-Elites." *IEEE Trans. Evolutionary Computation*.
  - CVT-MAP-Elites for scalable behavioral spaces

### GPU Architecture
- **Dao et al. (2022)**: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS 2022*. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
  - IO-aware tiled attention (HBM ↔ SRAM optimization)

---

## Next Steps

**Immediate**: Begin Phase 4 (DIRESA encoder) with:
1. Warp-native distance computation tests
2. Adaptive dimensionality via warp vote
3. Integration with CVTArchive
4. 100% constraint satisfaction

**Vision**: Production-ready self-organizing CA transformer by Q2 2026.

**Philosophy**: Like a Polynesian navigator attuned to ocean/stars/birds, we attune to GPU warps/cache/tensor cores. **Not using GPU as accelerator. Attuning to GPU as computational substrate.**

---

**Blueprint Version**: 2.0 (2025 Edition - Flow-Lenia/GPU-Native/Comonadic)
**Status**: Phase 3 Complete (100%), Phase 4 Ready to Begin
**Last Updated**: 2025-10-12
