# Slime Mold Transformer - System Blueprint

## Architecture Philosophy

1. **Protocol-First**: All interfaces defined before implementations
2. **Dynamic Everything**: No static allocations, lifecycle-managed components
3. **Comonadic Extraction**: GPU computation is spatial (extract from context), not sequential (inject into context)
4. **CVT-MAP-Elites Core**: Archive-driven evolution using Centroidal Voronoi Tessellation (scales to 4-5 behavioral dimensions)
5. **IO-Aware Kernels**: FlashAttention-style tiled computation (HBM ↔ SRAM management)
6. **Low-Rank Archive**: Store weight factorizations (U, V) not full matrices (10-100x memory reduction)
7. **Validated Behavioral Space**: KMO test ensures dimensions correlate with hardware structure
8. **Deterministic Random**: Hash-based seeded random for reproducibility
9. **SRE Built-In**: Observability, SLOs, error budgets from day one
10. **GPU-Native**: 100% device execution, zero CPU synchronization
11. **DRY Principle**: Single source of truth for each concept

## Dependency DAG

```
Layer 0: Protocols (no dependencies)
    proto/kernel.py
    proto/memory.py
    proto/model.py
    proto/component.py

Layer 1: Implementations of protocols (depend only on Layer 0)
    kernels/utils.py
    kernels/triton_impl.py → proto.kernel
    kernels/torch_fallback.py → proto.kernel
    observability/metrics.py → (no proto, pure collection)
    observability/slo.py → (no proto, pure validation)

Layer 2: Data structures (depend on Layer 0-1)
    memory/archive.py → proto.component
    memory/pool.py → proto.component
    memory/tubes.py → proto.memory
    core/state.py → (no dependencies, plain dataclass)
    core/comonad.py → (comonad protocol, no dependencies)

Layer 3: Components (depend on Layer 0-2)
    core/pseudopod.py → proto.model, proto.kernel, kernels/*, observability/*
    core/chemotaxis.py → proto.model, memory/archive
    memory/pool.py → core/comonad (for spatial context extraction)

Layer 4: Orchestration (depend on Layer 0-3)
    core/organism.py → proto.model, core/pseudopod, core/chemotaxis, memory/*, observability/*

Layer 5: API (depend on Layer 0-4)
    api/torch_compat.py → core/organism
    api/native.py → core/organism

Layer 6: Applications (depend on Layer 0-5)
    training/fitness.py → proto.component
    training/trainer.py → core/organism, observability/*, memory/archive
    bench/profile.py → core/organism
    tools/export.py → core/organism
    tools/package.py → (standalone, no deps)
    config/loader.py → (reads Layer 5)

    tests/unit/test_protocol_component.py → proto.component
    tests/unit/test_protocol_kernel.py → proto.kernel
    tests/unit/test_protocol_memory.py → proto.memory
    tests/unit/test_protocol_model.py → proto.model
    tests/unit/test_triton_kernels.py → kernels/triton_impl
    tests/unit/test_torch_fallback.py → kernels/torch_fallback
    tests/unit/test_kernel_equivalence.py → kernels/*, proto.kernel
    tests/unit/test_archive_operations.py → memory/archive
    tests/unit/test_pool_lifecycle.py → memory/pool
    tests/unit/test_tubes_memory.py → memory/tubes
    tests/unit/test_pseudopod_component.py → core/pseudopod
    tests/unit/test_chemotaxis_selection.py → core/chemotaxis
    tests/unit/test_organism_orchestration.py → core/organism
    tests/unit/test_dag_enforcement.py → (all layers, import analysis)
    tests/unit/test_ownership_hierarchy.py → core/organism, core/pseudopod, memory/*
    tests/unit/test_timescale_separation.py → training/trainer, memory/archive
    tests/unit/test_gpu_memory_safety.py → kernels/*, memory/*
    tests/unit/test_behavioral_kmo.py → memory/archive, scipy.stats
    tests/unit/test_behavioral_bartlett.py → memory/archive, scipy.stats
    tests/unit/test_behavioral_kernel_pca.py → memory/archive, sklearn
    tests/unit/test_race_conditions.py → core/organism, memory/pool
    tests/unit/test_gradient_flow_edge_cases.py → core/pseudopod, kernels/*
    tests/unit/test_attention_numerical_limits.py → kernels/triton_impl
    tests/unit/test_archive_elite_replacement.py → memory/archive
    tests/unit/test_fitness_stability.py → training/fitness
    tests/unit/test_device_placement_determinism.py → memory/archive, core/organism
    tests/unit/test_bootstrap_interpolation.py → memory/archive
    tests/unit/test_multi_gpu_partitioning.py → memory/archive, core/organism
    tests/unit/test_memory_budget_enforcement.py → memory/pool
    tests/unit/test_lifecycle_birth_death.py → memory/pool, core/pseudopod
    tests/unit/test_metrics_collection.py → observability/metrics
    tests/unit/test_slo_validation.py → observability/slo
    tests/unit/test_torch_compat_api.py → api/torch_compat
    tests/unit/test_native_api.py → api/native
    tests/unit/test_config_validation.py → config/loader
    tests/integration/test_end_to_end_training.py → training/trainer, all layers
    tests/ablations/test_vs_baseline_transformer.py → training/trainer
    tests/ablations/test_with_without_archive.py → training/trainer, memory/archive
    tests/ablations/test_with_without_lifecycle.py → training/trainer, memory/pool
    tests/ablations/test_behavioral_vs_random_placement.py → core/organism
    tests/ablations/test_efficiency_in_fitness.py → training/fitness
```

## Data Flow

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
│  (Orchestrator)     ├──────┐
└──────┬──────────────┘      │
       │                     │
       │ uses                │
       ▼                     ▼
┌─────────────────┐   ┌─────────────┐
│ Pseudopod Pool  │   │  Archive    │
│ (Dynamic)       │   │ (MAP-Elites)│
└──────┬──────────┘   └──────▲──────┘
       │                     │
       │ delegates           │
       ▼                     │ stores
┌─────────────────┐          │
│ Pseudopod       │──────────┘
│ (Component)     │ adds self
└──────┬──────────┘
       │
       │ calls
       ▼
┌─────────────────┐
│ Kernels         │
│ (GPU compute)   │
└─────────────────┘
       │
       │ metrics collected by
       ▼
┌─────────────────┐
│ Observability   │
│ (side channel)  │
└─────────────────┘
```

No cycles. Archive doesn't call anything. Observability is passive collector.

## Protocols

### proto.component.Component
```python
"""Base component protocol for pool management"""

Protocol:
  - fitness: float (property)
  - reset() -> None
  - to_dict() -> dict
  - from_dict(data: dict) -> Component

Purpose:
  - Unified interface for ALL pooled components
  - Archive stores any Component
  - Pools manage any Component
```

### proto.memory.Memory
```python
"""Temporal memory interface (NOT for pools)"""

Protocol:
  - store(data: Tensor, weight: float) -> None
  - recall() -> Optional[Tensor]
  - clear() -> None

Implementations:
  - memory.tubes.TubeNetwork

Purpose:
  - ONLY for temporal memory with decay
  - NOT for component lifecycle (that's pool.py)
```

### proto.model.Pseudopod
```python
"""Sensory probe interface"""

Protocol:
  - forward(latent, stimulus) -> output
  - correlation: Tensor (property)
  - effective_rank() -> Tensor
  - coherence() -> Tensor

Implementations:
  - core.pseudopod.Pseudopod

Dependencies:
  - MUST use proto.kernel.Kernel for all compute
  - MUST implement proto.component.Component
```

### proto.model.Chemotaxis
```python
"""Behavioral space navigator"""

Protocol:
  - add_source(nutrient, location, concentration) -> None
  - sample(behavior, metabolic_rate, hunger) -> Optional[Tensor]
  - clear() -> None

Implementations:
  - core.chemotaxis.Chemotaxis

Dependencies:
  - Uses memory.archive for spatial indexing
  - NO direct component management
```

### proto.model.Organism
```python
"""Top-level orchestrator"""

Protocol:
  - forward(stimulus, state) -> (output, new_state)
  - reset_state() -> None
  - stats() -> dict

Implementations:
  - core.organism.Organism

Dependencies:
  - Owns: Pool[Pseudopod], Archive, Chemotaxis
  - Uses: Kernels via Pseudopods
  - Records: Observability metrics
```

## File Structure

```
./
├── BLUEPRINT.md            # System architecture (this file)
├── README.md               # User documentation with examples
├── setup.py                # Python package setup
├── pyproject.toml          # Modern Python project configuration
├── requirements.txt        # Python dependencies
├── .python-version         # Python version specification
└── strip_docstrings.py     # AST-based docstring removal tool

slime/
├── proto/
│   ├── __init__.py
│   ├── component.py        # Base component interface
│   ├── kernel.py           # Kernel compute interface
│   ├── memory.py           # Temporal memory interface (tubes only)
│   └── model.py            # Model component interfaces
│
├── kernels/
│   ├── __init__.py
│   ├── utils.py            # Validation utilities
│   ├── triton_impl.py      # Triton GPU kernels (attention, correlation, effective_rank)
│   └── torch_fallback.py   # PyTorch CPU/GPU fallback implementations
│
├── observability/
│   ├── __init__.py
│   ├── metrics.py          # Passive metrics collector (latency, throughput, memory)
│   ├── slo.py              # SLO definitions and validation
│   └── tracing.py          # Distributed tracing (spans, contexts)
│
├── memory/
│   ├── __init__.py
│   ├── archive.py          # MAP-Elites archive (stores Component protocol)
│   ├── pool.py             # Dynamic component pools (manages Component protocol)
│   └── tubes.py            # TubeNetwork temporal memory (implements Memory protocol)
│
├── core/
│   ├── __init__.py
│   ├── state.py            # FlowState dataclass (input, hidden, residual)
│   ├── comonad.py          # Comonadic spatial context (extract from neighborhood)
│   ├── pseudopod.py        # Pseudopod component (implements Component + Model.Pseudopod)
│   ├── chemotaxis.py       # Chemotaxis selection (implements Model.Chemotaxis)
│   └── organism.py         # Organism orchestrator (implements Model.Organism)
│
├── api/
│   ├── __init__.py
│   ├── torch_compat.py     # SlimeMoldEncoder (nn.Module interface)
│   └── native.py           # SlimeModel (native API)
│
├── training/
│   ├── __init__.py
│   ├── trainer.py          # Training loop orchestrator
│   ├── losses.py           # Loss functions (cross-entropy, contrastive)
│   ├── stability.py        # Gradient clipping, numerical stability checks
│   ├── fitness.py          # Fitness computation (gradient magnitude, task correlation)
│   └── lifecycle.py        # Component birth/death decisions (hard limits, loss gates)
│
├── config/
│   ├── __init__.py
│   ├── loader.py           # YAML configuration loader with validation
│   ├── model.yaml          # Model architecture configuration
│   ├── training.yaml       # Training hyperparameters
│   └── slo.yaml            # SLO thresholds and error budgets
│
├── bench/
│   ├── __init__.py
│   ├── datasets.py         # Dataset loaders (TinyStories, WikiText)
│   ├── transformer.py      # Baseline transformer for ablations
│   ├── profile.py          # Performance profiling (latency, memory, FLOPS)
│   └── toy_tasks.py        # Simple tasks (y=sin(x), XOR, parity) for validation
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_protocol_component.py
│   │   ├── test_protocol_kernel.py
│   │   ├── test_protocol_memory.py
│   │   ├── test_protocol_model.py
│   │   ├── test_triton_kernels.py
│   │   ├── test_torch_fallback.py
│   │   ├── test_kernel_equivalence.py
│   │   ├── test_kernel_utils.py
│   │   ├── test_archive_operations.py
│   │   ├── test_archive_elite_replacement.py
│   │   ├── test_archive_serialization.py
│   │   ├── test_pool_lifecycle.py
│   │   ├── test_pool_culling.py
│   │   ├── test_tubes_memory.py
│   │   ├── test_tubes_temporal_access.py
│   │   ├── test_state_dataclass.py
│   │   ├── test_pseudopod_component.py
│   │   ├── test_pseudopod_fitness.py
│   │   ├── test_chemotaxis_selection.py
│   │   ├── test_chemotaxis_behavioral_search.py
│   │   ├── test_organism_orchestration.py
│   │   ├── test_organism_forward_pass.py
│   │   ├── test_torch_compat_api.py
│   │   ├── test_native_api.py
│   │   ├── test_trainer_loop.py
│   │   ├── test_losses.py
│   │   ├── test_stability.py
│   │   ├── test_fitness_computation.py
│   │   ├── test_lifecycle_decisions.py
│   │   ├── test_config_validation.py
│   │   ├── test_config_yaml_parsing.py
│   │   ├── test_datasets_loaders.py
│   │   ├── test_baseline_transformer.py
│   │   ├── test_profiling_metrics.py
│   │   ├── test_metrics_collection.py
│   │   ├── test_slo_validation.py
│   │   ├── test_tracing_spans.py
│   │   ├── test_visualize_behavioral_space.py
│   │   ├── test_export_onnx.py
│   │   ├── test_export_torchscript.py
│   │   ├── test_package_windows.py
│   │   ├── test_dag_enforcement.py
│   │   ├── test_ownership_hierarchy.py
│   │   ├── test_timescale_separation.py
│   │   ├── test_gpu_memory_safety.py
│   │   ├── test_behavioral_kmo.py
│   │   ├── test_behavioral_bartlett.py
│   │   ├── test_behavioral_kernel_pca.py
│   │   ├── test_race_conditions.py
│   │   ├── test_gradient_flow_edge_cases.py
│   │   ├── test_attention_numerical_limits.py
│   │   ├── test_fitness_stability.py
│   │   ├── test_device_placement_determinism.py
│   │   ├── test_bootstrap_interpolation.py
│   │   ├── test_multi_gpu_partitioning.py
│   │   ├── test_memory_budget_enforcement.py
│   │   └── test_lifecycle_birth_death.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_end_to_end_training.py
│   │   ├── test_training_stability.py
│   │   ├── test_gradient_flow.py
│   │   └── test_behavioral_space_coverage.py
│   ├── ablations/
│   │   ├── __init__.py
│   │   ├── test_vs_baseline_transformer.py
│   │   ├── test_with_without_archive.py
│   │   ├── test_with_without_lifecycle.py
│   │   ├── test_static_vs_dynamic.py
│   │   ├── test_behavioral_vs_random_placement.py
│   │   ├── test_efficiency_in_fitness.py
│   │   └── test_fitness_metrics_comparison.py
│   └── slo/
│       ├── __init__.py
│       ├── test_latency_slo.py
│       ├── test_throughput_slo.py
│       └── test_memory_slo.py
│
└── tools/
    ├── __init__.py
    ├── visualize.py        # Archive behavioral space visualization
    ├── export.py           # ONNX/TorchScript export
    └── package.py          # Windows .exe packaging
```

## Invariants

### 1. Dependency Direction (DAG Enforcement)
- Lower layers NEVER import from higher layers
- Protocols NEVER import implementations
- Components NEVER import API layer
- **Violation = compilation error**

### 2. Ownership Hierarchy
```
Organism owns:
  - Pool[Pseudopod]
  - Archive
  - Chemotaxis

Pool owns:
  - List[Component]

Archive owns:
  - Dict[cell, Elite] where Elite.genome = dict (NO object refs)

NO CYCLES
```

### 3. Protocol Compliance
- Every component MUST declare which protocols it implements
- Archive stores via `Component.to_dict()` (immutable serialization)
- Pools manage via `Component.fitness` (no type checking at runtime)

### 4. GPU Memory Safety
- Kernels check allocation before launch
- Organism enforces memory budget
- Pool culling triggered by OOM

### 5. Observability Injection
- Metrics collector passed to Organism.__init__()
- All forward passes record to metrics
- NO global state for metrics

### 6. Timescale Separation
```
Fast (every step):
  - Weight updates via backprop
  - Fitness tracking
  - Metrics collection
  - Loss monitoring

Medium (every 100 steps):
  - Fitness assessment
  - Archive elite updates
  - Pool spawn decisions
  - Loss gate check

Slow (every 1000 steps):
  - Pool culling
  - Memory budget enforcement
  - Behavioral space analysis
  - Hard limit enforcement (max pool size, max archive)
```

### 7. Archive Bootstrapping Policy 
- Archive provides INITIALIZATION only
- Bootstrapped components trained with rest of network
- NO frozen parameters injected mid-training
- Prevents mode collapse and gradient conflicts

### 8. Comonadic Spatial Context ✓
**Reasoning (GPU Architecture):** GPU computation is spatial (SIMD, tiles, neighborhoods), not sequential. Extract values from context, don't inject context into values.

**Comonad laws:**
```python
extract :: W a → a                    # Get focal value from context
extend :: (W a → b) → W a → W b      # Apply context-aware function
duplicate :: W a → W (W a)            # Nested contexts
```

**SpatialContext implementation:**
```python
@dataclass
class SpatialContext(Generic[T]):
    focus: T                           # Focal component
    neighborhood: List[T]              # All other components
    distance_fn: Callable[[T, T], float]  # Behavioral distance

    def extract(self) -> T:
        return self.focus

    def get_k_nearest(self, k: int) -> List[T]:
        # Extract k nearest neighbors by behavioral distance
```

**Comonadic extractors (context → value):**
```python
# Fitness IS relative to competition
extract_relative_fitness(ctx: W Component) → float
# Z-score: (my_fitness - mean_neighbor) / std_neighbor

# Diversity IS relative to neighborhood
extract_behavioral_divergence(ctx: W Component) → Tensor[5]
# my_behavior - mean(neighbor_behaviors)

# Learning rate IS relative to neighborhood activity
extract_gradient_magnitude_rank(ctx: W Component) → float
# Percentile ∈ [0,1] among k neighbors

# Synchronization IS relative to what neighbors attend to
extract_attention_coherence(ctx: W Component) → float
# Cosine similarity with neighbor attention patterns
```

**Why comonads:**
- Monads (sequential): a → M a (inject into context) - good for I/O, state
- Comonads (spatial): W a → a (extract from context) - good for convolution, attention, GPU

**FlashAttention is comonadic:**
```python
q_tile = load_context()  # W Q (full query context)
for k_tile, v_tile:
    output += extract(q_tile, k_tile, v_tile)  # Extract from tiles
```

**Pool provides spatial context:**
```python
ctx = pool.extract_component_context(component)  # W Component
relative_fitness = extract_relative_fitness(ctx)  # Extract from competition
```

**Pattern:** All contextual computations go through comonad extract. No component computes in isolation.

### 9. Fitness Correlation with Task ✓
Fitness MUST correlate with loss reduction. Options:
- Gradient magnitude (components affecting loss)
- Attention alignment with targets
- Information bottleneck metrics (mutual information)
- **Relative fitness** (gradient magnitude vs neighborhood via comonad)

NOT attention entropy alone (doesn't correlate with task)

### 9. CVT-MAP-Elites Architecture ✓
**Reasoning (Scalability):** Fixed grid scales as resolution^dims. CVT scales linearly with num_centroids.

**Fixed grid problem:**
```python
# 3D behavioral space, resolution 20
grid_cells = 20^3 = 8000 cells
# 4D space: 20^4 = 160,000 cells (exponential explosion)
# 5D space: 20^5 = 3,200,000 cells (infeasible)
```

**CVT solution:**
```python
# 4D or 5D behavioral space
num_centroids = 1000  # Linear scaling, user-defined
# Voronoi partitioning: each component maps to nearest centroid
# No exponential explosion, handles continuous space naturally
```

**Behavioral dimensions (4-5 dims):**
```python
behavior = (
    avg_attention_span,         # Mean(attention_weights * position_distance)
    activation_sparsity,         # Fraction of activations near zero
    gradient_flow_magnitude,     # L2 norm of gradients through component
    memory_access_locality,      # Coherence of attention patterns
    computational_intensity,     # FLOPs per forward pass
)
```

Each dimension correlates with hardware structure: memory locality, compute efficiency, task relevance.

### 10. Low-Rank Archive Storage ✓
**Reasoning (Memory Efficiency):** Inspired by Ha et al. (2016) hypernetworks memory optimization.

**Naive storage:**
```python
# Store full weight matrices
elite.weights = [W_q, W_k, W_v, W_o]  # Each D×D matrix
memory_per_elite = 4 * D^2 * 4 bytes
# For D=512: 4MB per elite, 1000 elites = 4GB
```

**Low-rank factorization:**
```python
# Store U (D×k) and V (k×D) where k << D
elite.weights_compressed = [(U_q, V_q), (U_k, V_k), (U_v, V_v), (U_o, V_o)]
memory_per_elite = 4 * 2 * D * k * 4 bytes
# For D=512, k=64: 0.5MB per elite, 1000 elites = 500MB (8x reduction)
```

Reconstruction: W = U @ V. Gradient updates apply to U and V during training.

### 11. Lifecycle Safety Guardrails ✓
```python
# Hard limits (never exceed)
MAX_POOL_SIZE = 64
MAX_ARCHIVE_CENTROIDS = 1000  # CVT centroids, not grid cells
MAX_LOSS_RATIO = 10.0  # vs moving average

# Loss gates (halt lifecycle if loss diverging)
if current_loss > 10 * loss_ema:
    freeze_lifecycle()  # Stop births/deaths, train normally

# Phased training (gradually enable dynamics)
if step < 1000:
    phase = "warmup"  # Static pool, no lifecycle
elif step < 5000:
    phase = "gentle"  # Allow births, no deaths
else:
    phase = "full"    # All dynamics enabled
```

## Implementation Checklist

### MUST FIX NOW
- [ ] **core/chemotaxis.py** - Behavioral space navigation
- [ ] **core/organism.py** - Fix to inject metrics, kernel, use chemotaxis
- [ ] **observability/metrics.py** - MetricsCollector (injectable)
- [ ] **observability/slo.py** - SLO definitions
- [ ] **observability/tracing.py** - Tracing spans

### MUST ADD
- [ ] **memory/archive.py** - CVT-MAP-Elites with Voronoi partitioning and low-rank storage
- [ ] **kernels/triton_impl.py** - FlashAttention-style tiled attention (HBM ↔ SRAM)
- [ ] **core/pseudopod.py** - 5D behavioral metric computation
- [ ] **training/stability.py** - Simulated annealing temperature schedule
- [ ] **training/fitness.py** - Gradient-based fitness with efficiency signals
- [ ] **training/losses.py** - Multi-objective loss functions
- [ ] **training/trainer.py** - Training loop with annealing-driven lifecycle
- [ ] **training/lifecycle.py** - Annealing-based birth/death decisions, loss gates

### MUST TEST
- [ ] **tests/integration/test_training_stability.py** - Convergence with lifecycle
- [ ] **tests/integration/test_gradient_flow.py** - Gradients through dynamic pools
- [ ] **tests/ablations/test_static_vs_dynamic.py** - Prove dynamic helps
- [ ] **tests/ablations/test_with_without_archive.py** - Prove archive helps
- [ ] **bench/toy_tasks.py** - Tasks (y=sin(x), XOR, parity)

### Phase 1 Completion
- [ ] kernels/triton_impl.py
- [ ] tests/unit/test_archive.py
- [ ] tests/unit/test_pool.py
- [ ] tests/unit/test_pseudopod.py
- [ ] tests/unit/test_tubes.py
- [ ] tests/unit/test_chemotaxis.py
- [ ] tests/unit/test_kernels.py
- [ ] tests/unit/test_metrics.py

### Phase 2
- [ ] bench/toy_tasks.py
- [ ] tests/ablations/test_fitness_metrics.py (which fitness works?)
- [ ] tests/integration/test_behavioral_space.py (coverage, gradients)
- [ ] tools/visualize.py (behavioral space plots)

### Phase 3
- [ ] bench/transformer.py (vs baseline on real tasks)
- [ ] tests/integration/test_optimization_landscape.py
- [ ] config/* (complete YAML schemas)

### Phase 4
- [ ] tools/export.py (ONNX, TorchScript)
- [ ] tools/package.py (Windows .exe)
- [ ] setup.py / pyproject.toml
- [ ] README.md with examples

## Architectural Decisions

### 1. IO-Aware Tiled Attention (FlashAttention) ✓
**Reasoning (Dao et al., 2022):** Standard attention is memory-bound, not compute-bound. Tile to maximize SRAM usage.

**Problem:** Attention loads Q, K, V from HBM repeatedly.
```python
# Naive: O(M²) memory accesses to HBM
for i in range(M):
    for j in range(M):
        S[i,j] = Q[i] @ K[j]  # Load K[j] from HBM M times
```

**FlashAttention solution:** Tile computation to fit in SRAM.
```python
BLOCK_M = 128  # Query tile size (fits in SRAM)
BLOCK_N = 128  # Key/value tile size (fits in SRAM)

for block_m in range(0, M, BLOCK_M):
    # Load Q tile to SRAM once
    Q_tile = Q[block_m:block_m+BLOCK_M]  # HBM → SRAM

    for block_n in range(0, M, BLOCK_N):
        # Load K, V tiles to SRAM
        K_tile = K[block_n:block_n+BLOCK_N]  # HBM → SRAM
        V_tile = V[block_n:block_n+BLOCK_N]  # HBM → SRAM

        # Compute in SRAM (fast)
        S_tile = Q_tile @ K_tile.T
        P_tile = softmax(S_tile)
        O_tile += P_tile @ V_tile

        # Write back to HBM once
        O[block_m:block_m+BLOCK_M] = O_tile  # SRAM → HBM
```

**IO complexity:**
```
Naive: O(M² * D) HBM accesses
Tiled: O(M² * D / SRAM_size) HBM accesses
Speedup: ~3x on GPT-2 (Dao et al., 2022)
```

**Implementation:** kernels/triton_impl.py uses tiling with BLOCK_M=128, BLOCK_N=128, BLOCK_D=64.

### 2. Kernel injection: Constructor Injection ✓
**Reasoning (Bitter Lesson):** Let user provide compute capability. Scale with available hardware, not our assumptions.
```python
Pseudopod.__init__(head_dim, kernel: Kernel, device)
```

### 3. Multi-GPU: Hash-based partitioning ✓
**Reasoning (Bitter Lesson):** Hash function scales arbitrarily. No hand-coded spatial assumptions.
```python
device_id = hash(behavior_coords) % num_gpus
```

### 4. Determinism: Sort keys on iteration ✓
**Reasoning (Architecture):** Spatial structure over temporal accidents. Reproducible science.
```python
for key in sorted(archive.keys()):
```

### 5. Memory limits: Soft limit with graceful degradation ✓
**Reasoning (SRE + Bitter Lesson):** Adapt to constraints, don't crash. Trade quality for capacity automatically.
```python
if memory > budget: pool.cull_worst(fraction=0.2)
```

### 6. Metrics injection: Dependency injection ✓
**Reasoning (SRE + Testing):** Explicit dependencies. No globals. Testable.
```python
Organism.__init__(metrics_collector: Optional[MetricsCollector])
```

### 7. Fitness metric: Gradient magnitude ✓
**Reasoning (Training Stability):** Fitness must correlate with task performance, not internal diversity metrics.
```python
fitness = grad_norm * attention_to_targets  # Task-relevant
```

### 8. Archive bootstrapping: Initialization only ✓
**Reasoning (Gradient Flow):** Don't inject frozen weights mid-training. Bootstrap init, then train together.
```python
if new_component_needed:
    component = archive.bootstrap_component(...)  # Init from elite
    component.requires_grad_(True)  # Train with network
```

### 9. Timescale separation: 1x / 100x / 1000x ✓
**Reasoning (Stability):** Separate fast (weights) from medium (fitness) from slow (lifecycle).
```python
if step % 1000 == 0: pool.cull()
elif step % 100 == 0: archive.add_if_elite()
fitness_ema.update()  # Every step
```

### 10. KMO-validated behavioral dimensions (4-5 dims) ✓
**Reasoning (MAP-Elites Research):** Behavioral characterization is CRITICAL. Wrong dimensions = useless diversity. CVT allows 4-5 dimensions without exponential grid explosion.

**Validation requirement:** KMO (Kaiser-Meyer-Olkin) test ensures behavioral dimensions are factorable and capture meaningful variance.

```python
# Before training: validate behavioral space
from scipy.stats import KMO
behavioral_samples = sample_components(n=100)
kmo_statistic = KMO(behavioral_samples)

if kmo_statistic < 0.5:
    raise ValueError("Behavioral dimensions are not factorable - diversity will be noise")
elif kmo_statistic < 0.6:
    warnings.warn("Marginal behavioral space quality")
# KMO > 0.6: Acceptable, KMO > 0.8: Good, KMO > 0.9: Excellent
```

**5-dimensional behavioral space (CVT-compatible):**
```python
behavior = (
    avg_attention_span,         # Mean(attention_weights * position_distance)
    activation_sparsity,         # Fraction of activations < threshold
    gradient_flow_magnitude,     # L2 norm of gradients through component
    memory_access_locality,      # Variance of attention positions
    computational_intensity,     # FLOPs per forward pass
)
```

Each dimension correlates with measurable hardware metrics:
- Attention span → Memory bandwidth usage
- Sparsity → Compute utilization
- Gradient flow → Task relevance
- Memory locality → Cache hit rate
- Compute intensity → GPU occupancy

**CVT partitioning for device placement:**
```python
# Compute 1000 Voronoi centroids in 5D behavioral space
centroids = compute_cvt_centroids(num_centroids=1000, dims=5)
# Each component maps to nearest centroid
centroid_id = find_nearest_centroid(component.behavior(), centroids)
device_id = hash(centroid_id) % num_gpus
```

**Failure mode prevention:** KMO < 0.6 indicates dimensions don't capture structured variance. Replace dimensions before training.

**Test:** Does KMO-validated behavioral space beat random dimensions? If no, MAP-Elites adds overhead without benefit.

### 11. Deterministic hash-based random ✓
**Reasoning (Reproducibility):** Non-deterministic random breaks reproducibility. Hash-based seeded random is cheap (~100ns) vs gradient computation (ms).

**Deterministic random primitive:**
```python
def _deterministic_random(seed: int, step: int, context: str) -> float:
    hash_input = f"{seed}:{step}:{context}".encode('utf-8')
    hash_digest = hashlib.sha256(hash_input).digest()
    random_bytes = int.from_bytes(hash_digest[:8], byteorder='big')
    return random_bytes / (2**64 - 1)  # [0, 1]
```

**Applications:**
```python
# Birth decisions
random_val = _deterministic_random(seed, step, f"birth:{behavior_hash}")
if random_val < birth_probability:
    spawn_component()

# Death decisions
random_val = _deterministic_random(seed, step, f"cull:{component_id}")
if random_val < death_probability:
    cull_component()

# Centroid initialization
rng = np.random.RandomState(seed)
centroids = rng.randn(num_centroids, behavioral_dims)

# Component mutation
torch.manual_seed(seed + generation)
noise = torch.randn_like(weights) * mutation_std
```

**Benefits:**
- Same seed → identical training trajectory → identical final model
- Debuggable: replay exact sequence of births/deaths
- Scientific: ablation tests require identical random seeds
- Cost: SHA256 hash ~100ns, negligible vs ~10ms gradient computation

**Pattern:** NO np.random.random() or random.random(). All random decisions via hash(seed, step, context).

### 12. Fitness must include efficiency signals ✓
**Reasoning (Hardware Awareness):** Task accuracy alone won't discover hardware-optimal patterns. Fitness must reward efficiency.

```python
fitness = (
    task_performance * 0.7 +           # Does it help the task?
    compute_efficiency * 0.2 +          # Is it fast?
    gradient_magnitude * 0.1            # Is it relevant?
)
```

**Without efficiency in fitness:**
- Slow components survive if they help task accuracy
- No evolutionary pressure for hardware-friendly patterns
- You get diversity, but not useful diversity

**With efficiency in fitness:**
- Cross-GPU communication → slower training → lower fitness → culled
- Poor cache behavior → slower training → lower fitness → culled
- Excessive memory → exceeds budget → culled

MAP-Elites doesn't "know" about GPUs, but fitness does. Hardware-optimal patterns emerge from selection pressure, not explicit programming.

**Test:** Train with and without efficiency in fitness. Does it discover faster configurations?

### 12. Quality-diversity maintains architectural variety ✓
**Reasoning (Avoid Mode Collapse):** Standard transformers: all heads learn similar features. MAP-Elites: forced diversity.

**Standard transformer failure mode:**
```python
# All 8 attention heads learn to do the same thing
head_similarity = cosine_similarity(head_0, head_1)  # 0.95
```

**MAP-Elites pressure:**
```python
archive[(0.1, 0.2)] = local_syntax_checker     # Short distance, sparse
archive[(0.8, 0.9)] = global_coherence_tracker # Long distance, dense
archive[(0.5, 0.3)] = mid_range_dependency     # Medium distance, sparse
```

Each cell MUST be occupied by behaviorally-distinct component. No collapse to single strategy.

**Benefit 1: Graceful degradation under device loss**
```python
# Lost a GPU? Components redistribute deterministically
device = hash(behavior) % 3  # was 4 GPUs, now 3
# No retraining needed - hash mapping is pure function
```

**Benefit 2: Interpretability**
```python
# Ask system: "show me all short-range sparse components"
short_range_components = [
    elite for coords, elite in archive.items()
    if coords[0] < 0.3 and coords[1] < 0.4
]
```

**Test:** Does MAP-Elites maintain higher behavioral diversity than standard training?

### 13. Ablation tests determine architectural value ✓
**Reasoning (Scientific Method):** Don't assume architectural choices work. Test them.

**Required comparisons:**
1. **Slime Mold vs Baseline Transformer** (same parameters, same compute)
   - If slower or worse accuracy: architecture is self-indulgent
   - If faster or better accuracy: investigate why

2. **With Archive vs Without Archive** (dynamic pool only)
   - Does archive-guided bootstrapping improve convergence?
   - Or is it overhead with no benefit?

3. **With Lifecycle vs Static Pool** (fixed number of components)
   - Does birth/death improve over fixed architecture?
   - Or does training instability hurt more than variety helps?

4. **Behavioral Device Placement vs Random Placement**
   - Does hash(behavior) % num_gpus beat random device assignment?
   - Test requires: multi-GPU setup, measure cross-GPU communication

5. **Efficiency in Fitness vs Accuracy Only**
   - Does including compute_efficiency in fitness discover faster configurations?
   - Measure: throughput (samples/sec), memory usage

**Acceptance criteria:**
- Must beat baseline transformer on at least one dimension (speed OR accuracy)
- If worse on all dimensions: architecture is a failure, simplify
- If better on some dimensions: document tradeoffs, make configurable

**Phase 2 implementation:** Create tests/ablations/ with automated comparisons.

## Computational Cost Analysis

### Cost Structure

**Training costs (per step):**
```
Forward pass: O(B * M * D^2)          B=batch, M=sequence, D=model_dim
Backward pass: O(B * M * D^2)         Standard backprop
Fitness computation: O(P * D)         P=num_pseudopods, per-component gradients
Archive update (1/100 steps): O(P)    Insert/replace elite
Pool lifecycle (1/1000 steps): O(P)   Birth/death decisions
```

**Memory costs:**
```
Pseudopod weights: P * D^2 * 4 bytes (fp32)
Archive storage: C * D^2 * 4 bytes     C=num_cells (<1000 typical)
Gradients: P * D^2 * 4 bytes
Activations: B * M * D * 4 bytes
```

**Total overhead vs baseline transformer:**
```
Baseline: Forward + Backward
Slime: Forward + Backward + Fitness(~1%) + Archive(0.01%) + Lifecycle(0.001%)
Net overhead: ~1-2% per training step
```

### Comparison to DARTS (Modern NAS)

**DARTS (Liu et al., 2018) baseline:**
- Differentiable architecture search with continuous relaxation
- Uses weight sharing across candidate operations
- Search cost: 1-4 GPU days on CIFAR-10/ImageNet
- 1000x faster than early NAS methods (NASNet: 2000 GPU days)

**Key difference:**
```
DARTS: Find single best architecture → train it from scratch
Slime: Maintain diverse components → continuously adapt during training
```

**Slime approach:**
```
1. No separate search phase
   - Components evolve during task training
   - CVT-MAP-Elites maintains 1000 diverse solutions
   - Quality-diversity, not single-objective optimization

2. Continuous adaptation
   - DARTS architecture is fixed after search
   - Slime components birth/death based on fitness
   - Adapts to distribution shift during training

3. Estimated cost
   - Base training: 100% (same as DARTS final training)
   - Fitness computation: +5-10%
   - CVT archive ops: +1-3%
   - Lifecycle decisions: +1-2%
   - Total: 107-115% of baseline training time
```

**Honest comparison:**
```
DARTS: 4 GPU days search + N GPU days training = (4 + N) total
Slime: 1.15 × N GPU days (search happens during training)

If N > 30 days: Slime is faster
If N < 30 days: DARTS is faster

Advantage: Slime maintains architectural diversity throughout training.
Disadvantage: Slime has 15% overhead vs fixed architecture.
```

**Hypothesis (requires empirical validation):** For long training runs (100+ GPU days), amortized search cost favors Slime. For short runs (<30 days), DARTS is more efficient.

### Comparison to Hypernetworks (Ha et al., 2016)

**Hypernetwork approach (Ha et al., ICLR 2017):**
- Small network generates weights for larger network
- Achieves parameter efficiency: fewer learnable params than standard networks
- Memory-efficient formulation: O(Nz × hidden_units) not O(Nz × all_params)
- Weight sharing across layers via generation scheme

**Key insight from Ha et al.:** Low-rank weight generation can be MORE efficient than storing full matrices.

**Slime borrows this insight for archive storage:**
```python
# Archive stores low-rank factorizations (U, V) inspired by hypernetworks
elite.weights = [(U_q, V_q), (U_k, V_k), (U_v, V_v), (U_o, V_o)]
# Reconstruction: W = U @ V
# Memory: O(D × k) instead of O(D²)
```

**Key differences:**
```
Hypernetworks: Generate weights dynamically at runtime
Slime: Store compressed weights statically in archive

Hypernetworks: Single generator network for all components
Slime: Each component has independent weights (no generator bottleneck)

Hypernetworks: Meta-learning (learn to generate good weights)
Slime: Quality-diversity (maintain diverse proven weights)
```

**Computational tradeoffs:**
```
Hypernetwork:
  + Fast adaptation to new tasks (generate new weights)
  + Parameter efficient (small generator)
  - Generation cost at runtime
  - Gradient flow through generator adds complexity

Slime:
  + No generation cost (weights are direct parameters)
  + No meta-learning instability
  - Archive memory cost (mitigated by low-rank storage)
  - Slower adaptation to new tasks (need to add/train components)
```

**Complementary approaches:** Hypernetworks excel at few-shot adaptation. Slime excels at maintaining diverse specialists for single-task training.

### Simulated Annealing for Component Lifecycle

**Insight:** Quality-diversity needs exploration-exploitation balance. Simulated annealing provides principled temperature schedule.

**Application 1: Birth decisions (exploration temperature)**
```python
# Early training: high temperature = accept diverse components
# Late training: low temperature = only accept high-fitness components

def birth_probability(fitness, step, max_steps):
    temperature = initial_temp * (1 - step / max_steps)  # Linear cooling
    if fitness > threshold:
        return 1.0  # Always accept good components
    else:
        # Accept suboptimal components with probability exp(-ΔE/T)
        delta_e = threshold - fitness
        return exp(-delta_e / temperature)

# Early: temperature=1.0, accept fitness=0.5 with prob=exp(-0.5)=0.61
# Late: temperature=0.1, accept fitness=0.5 with prob=exp(-5.0)=0.007
```

**Application 2: CVT centroid refinement**
```python
# Use simulated annealing to optimize Voronoi centroid positions
# Start with random centroids, gradually move to minimize quantization error

def refine_centroids(behavioral_samples, num_centroids):
    centroids = initialize_random(num_centroids)
    temperature = 1.0

    for iteration in range(max_iterations):
        # Propose centroid move
        new_centroids = perturb(centroids, std=temperature)
        # Accept if improves coverage or with annealing probability
        if coverage(new_centroids) > coverage(centroids):
            centroids = new_centroids
        elif random() < exp(-delta_coverage / temperature):
            centroids = new_centroids

        temperature *= cooling_rate  # Geometric cooling

    return centroids
```

**Application 3: Archive mutation strength**
```python
# When bootstrapping from archive, mutation strength follows annealing
# Early: large mutations for exploration
# Late: small mutations for exploitation

def bootstrap_from_archive(elite, step, max_steps):
    weights = elite.reconstruct()  # U @ V
    temperature = initial_temp * (1 - step / max_steps)
    noise = torch.randn_like(weights) * temperature
    return weights + noise
```

**Pattern:** Annealing naturally transitions from exploration → exploitation without manual phase boundaries. Smoother than hard-coded warmup/gentle/full phases.

**Fundamental difference:**
```
DARTS: Search for single best architecture (optimization)
Hypernetworks: Learn to generate diverse weights (meta-learning)
Slime: Maintain diverse components (quality-diversity)
Simulated Annealing: Principled exploration-exploitation schedule
```

**Slime uniqueness:**
1. **No search vs exploitation tradeoff**: Archive maintains both
   - Exploit: Use current best components for task
   - Explore: Archive keeps diverse alternatives alive
   - Switch cost: zero (deterministic bootstrap from archive)

2. **Emergent specialization**: Components discover niches automatically
   - No pre-specified roles (unlike multi-head attention with fixed heads)
   - No manual architecture engineering
   - Behavioral space captures relevant variance

3. **Hardware co-optimization**: Fitness includes efficiency signals
   - NAS: Architecture search is task-accuracy only
   - Hypernetworks: No mechanism for hardware awareness
   - Slime: Fast components have higher fitness → survive

**Test this claim:**
```python
# Required test: Train all three approaches on same task, same budget
baseline_transformer = train(model, task, epochs=100)
nas_architecture = nas_search(search_space, task, budget=10000_gpu_hours)
hypernetwork = train(hypernetwork_model, task, epochs=100)
slime_organism = train(slime_model, task, epochs=100)

# Compare:
# 1. Final task accuracy
# 2. Training throughput (samples/sec)
# 3. Total GPU-hours to convergence
# 4. Diversity of learned components (behavioral variance)
# 5. Graceful degradation under component removal
```

Hypothesis: Slime matches or exceeds task accuracy with 50-100x less total compute than NAS, and 1.3-1.5x better throughput than hypernetworks.

**If hypothesis fails:** Architecture is self-indulgent. Simplify or abandon.

## References

### Core Algorithms

- Mouret, J.-B. & Clune, J. (2015). "Illuminating search spaces by mapping elites." *arXiv:1504.04909*
  - Original MAP-Elites algorithm for quality-diversity optimization
  - Foundation for archive-based behavioral diversity

- Vassiliades, V., Chatzilygeroudis, K., & Mouret, J.-B. (2018). "Using Centroidal Voronoi Tessellations to Scale Up the Multidimensional Archive of Phenotypic Elites Algorithm." *IEEE Transactions on Evolutionary Computation*, 22(4), 623-630.
  - CVT-MAP-Elites for scalable behavioral space partitioning
  - Solves exponential grid explosion with fixed-resolution grids

- Pugh, J. K., Soros, L. B., & Stanley, K. O. (2016). "Quality Diversity: A New Frontier for Evolutionary Computation." *Frontiers in Robotics and AI*, 3:40.
  - Survey of quality-diversity algorithms and applications
  - Distinguishes QD from pure optimization and novelty search

### Neural Architecture

- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS 2022*. *arXiv:2205.14135*
  - IO-aware tiled attention implementation
  - 3x speedup on GPT-2, 15% speedup on BERT via HBM ↔ SRAM tiling

- Liu, H., Simonyan, K., & Yang, Y. (2019). "DARTS: Differentiable Architecture Search." *ICLR 2019*. *arXiv:1806.09055*
  - Modern NAS baseline: 1-4 GPU days (vs 2000 for early NAS)
  - Continuous relaxation with weight sharing

- Ha, D., Dai, A., & Le, Q. V. (2017). "HyperNetworks." *ICLR 2017*. *arXiv:1609.09106*
  - Small network generates weights for larger network
  - Memory-efficient formulation: O(Nz × hidden_units) not O(Nz × all_params)
  - Low-rank weight generation inspiration for archive storage

### Statistical Validation

- Kaiser, H. F. (1970). "A second generation little jiffy." *Psychometrika*, 35(4), 401-415.
  - Kaiser-Meyer-Olkin (KMO) test for factor analysis adequacy
  - Used to validate behavioral dimensions are factorable

- Bartlett, M. S. (1950). "Tests of significance in factor analysis." *British Journal of Statistical Psychology*, 3(2), 77-85.
  - Bartlett's test of sphericity for correlation matrices
  - Tests null hypothesis that behavioral dimensions are uncorrelated

### Optimization Theory

- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). "Optimization by simulated annealing." *Science*, 220(4598), 671-680.
  - Simulated annealing for combinatorial optimization
  - Temperature schedule for exploration-exploitation balance
