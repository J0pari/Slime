# Slime Mold Transformer - System Blueprint

## Architecture Philosophy

1. **Protocol-First**: All interfaces defined before implementations
2. **Dynamic Everything**: No static allocations, lifecycle-managed components
3. **MAP-Elites Core**: Archive-driven evolution and bootstrapping
4. **SRE Built-In**: Observability, SLOs, error budgets from day one
5. **GPU-Native**: 100% device execution, zero CPU synchronization
6. **DRY Principle**: Single source of truth for each concept

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

Layer 3: Components (depend on Layer 0-2)
    core/pseudopod.py → proto.model, proto.kernel, kernels/*, observability/*
    core/chemotaxis.py → proto.model, memory/archive

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

### 8. Fitness Correlation with Task
Fitness MUST correlate with loss reduction. Options:
- Gradient magnitude (components affecting loss)
- Attention alignment with targets
- Information bottleneck metrics (mutual information)

NOT attention entropy alone (doesn't correlate with task)

### 9. Lifecycle Safety Guardrails 
```python
# Hard limits (never exceed)
MAX_POOL_SIZE = 64
MAX_ARCHIVE_SIZE = 1000
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
- [ ] **training/stability.py** - Phased training (warmup/gentle/full)
- [ ] **training/fitness.py** - Gradient-based fitness computation
- [ ] **training/losses.py** - Multi-objective loss functions
- [ ] **training/trainer.py** - Training loop with lifecycle timescales
- [ ] **training/lifecycle.py** - Hard limits, loss gates, safety checks

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

### 1. Kernel injection: Constructor Injection ✓
**Reasoning (Bitter Lesson):** Let user provide compute capability. Scale with available hardware, not our assumptions.
```python
Pseudopod.__init__(head_dim, kernel: Kernel, device)
```

### 2. Multi-GPU: Hash-based partitioning ✓
**Reasoning (Bitter Lesson):** Hash function scales arbitrarily. No hand-coded spatial assumptions.
```python
device_id = hash(behavior_coords) % num_gpus
```

### 3. Determinism: Sort keys on iteration ✓
**Reasoning (Architecture):** Spatial structure over temporal accidents. Reproducible science.
```python
for key in sorted(archive.keys()):
```

### 4. Memory limits: Soft limit with graceful degradation ✓
**Reasoning (SRE + Bitter Lesson):** Adapt to constraints, don't crash. Trade quality for capacity automatically.
```python
if memory > budget: pool.cull_worst(fraction=0.2)
```

### 5. Metrics injection: Dependency injection ✓
**Reasoning (SRE + Testing):** Explicit dependencies. No globals. Testable.
```python
Organism.__init__(metrics_collector: Optional[MetricsCollector])
```

### 6. Fitness metric: Gradient magnitude ✓
**Reasoning (Training Stability):** Fitness must correlate with task performance, not internal diversity metrics.
```python
fitness = grad_norm * attention_to_targets  # Task-relevant
```

### 7. Archive bootstrapping: Initialization only ✓
**Reasoning (Gradient Flow):** Don't inject frozen weights mid-training. Bootstrap init, then train together.
```python
if new_component_needed:
    component = archive.bootstrap_component(...)  # Init from elite
    component.requires_grad_(True)  # Train with network
```

### 8. Timescale separation: 1x / 100x / 1000x ✓
**Reasoning (Stability):** Separate fast (weights) from medium (fitness) from slow (lifecycle).
```python
if step % 1000 == 0: pool.cull()
elif step % 100 == 0: archive.add_if_elite()
fitness_ema.update()  # Every step
```

### 9. Behavioral dimensions must correlate with hardware costs ✓
**Reasoning (Emergent Hardware Optimization):** MAP-Elites can discover hardware-optimal patterns IF behavioral space captures hardware-relevant structure.

**Critical requirement:** Behavioral dimensions MUST correlate with actual compute patterns. If dimensions are arbitrary, diversity is meaningless.

**Good behavioral dimensions:**
```python
behavior = (
    attention_distance,      # Correlates with memory access patterns
    activation_sparsity,     # Correlates with compute efficiency
    weight_magnitude_std,    # Correlates with numerical precision needs
)
```

**Bad behavioral dimensions:**
```python
behavior = (
    random_noise,            # No correlation with anything
    component_id_hash,       # Arbitrary identifier
    creation_timestamp,      # Temporal accident, not structure
)
```

**Hash partitioning for device placement:**
```python
device_id = hash(behavior_coords) % num_gpus
```

This creates deterministic device placement. IF behavioral space captures hardware structure:
- Short attention distance → GPU 0 (local memory access)
- Long attention distance → GPU 3 (global memory access)
- Components naturally partition by actual compute patterns

**Failure mode:** If behavioral dimensions don't correlate with hardware, this is just random device assignment with extra steps.

**Test:** Does archive-guided device placement beat random placement? If no, behavioral space is wrong.

### 10. Fitness must include efficiency signals ✓
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

### 11. Quality-diversity maintains architectural variety ✓
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

### 12. Ablation tests determine architectural value ✓
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

### Why Better Than NAS

**Neural Architecture Search problems:**
1. **Outer loop cost**: Train candidate → evaluate → pick best → repeat
   - Each candidate requires full training run
   - Total cost: N_candidates * training_time
   - Typical: 1000 candidates * 100 GPU-days = 100,000 GPU-days

2. **Discrete search space**: Must evaluate entire architectures
   - Can't mix-and-match partial solutions
   - Every structural change requires full re-evaluation

3. **Transferability**: Architectures found on small datasets/tasks may not transfer

**Slime advantages:**
```
1. Single training run with continuous adaptation
   - No outer loop: components evolve during task training
   - Cost: 1.02x baseline (2% overhead)

2. Continuous search space via MAP-Elites
   - Archive maintains diverse solutions simultaneously
   - Interpolation between elites for new components
   - Amortized exploration: pay once, sample forever

3. Task-adaptive: Components specialize to actual data distribution
   - Fitness computed on real task gradients
   - No transferability gap
```

**Cost comparison:**
```
NAS: 100,000 GPU-days to find architecture, then train it
Slime: 1.02x GPU-days to train and discover architecture simultaneously
Speedup: ~98,000x cheaper
```

### Why Better Than Hypernetworks

**Hypernetwork problems:**
1. **Meta-overfitting**: Hypernetwork learns to generate weights, but:
   - Adds meta-parameters (hypernetwork weights)
   - Meta-parameters must generalize across weight space
   - Tends to collapse to average solution

2. **Computational overhead**:
   - Generate weights → forward pass → backward through generator
   - Gradient flow: task_loss → primary_weights → hypernetwork_params
   - 2x memory (store both hypernetwork and generated weights)

3. **Expressiveness limit**:
   - Hypernetwork output dimension = primary network parameters
   - For D=512 model: output ~260k values per component
   - Bottleneck: compress weight space into hypernetwork capacity

**Slime advantages:**
```
1. No meta-parameters: Components are primary parameters
   - Direct gradient flow from task to component weights
   - No compression bottleneck
   - Each component learns independently

2. Computational efficiency:
   - No weight generation step
   - Standard backprop through components
   - Memory: 1x (just component weights + gradients)

3. Unbounded expressiveness:
   - Archive stores actual weight matrices, not generators
   - No reconstruction loss
   - Each elite is fully optimized for its niche
```

**Cost comparison:**
```
Hypernetwork: 2x memory, 1.5x compute (generation + backprop through generator)
Slime: 1.02x compute (fitness overhead), 1.01x memory (archive is tiny)
Speedup: ~1.5x faster than hypernetworks
```

### Why Better Than Both

**Fundamental difference:**
```
NAS: Search for single best architecture (optimization)
Hypernetworks: Learn to generate diverse weights (meta-learning)
Slime: Maintain diverse components (quality-diversity)
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
