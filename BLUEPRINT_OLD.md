# Slime Mold Transformer - System Blueprint

## Architecture Philosophy

**Foundation**: Conway → Lenia → Flow-Lenia → Neural Flow-Lenia evolution path. Our Slime Mold Transformer is a **Neural Flow-Lenia organism** where Pseudopods are learned CA update rules with spatially-localized parameters, mass-conservative dynamics, and intrinsic curiosity-driven lifecycle.

**Core Principles**

1. **Protocol-First**: All interfaces defined before implementations (algebraic effect handlers for optional capabilities)
2. **Dynamic Everything**: No static allocations, lifecycle-managed components (curiosity-driven birth/death via learning progress)
3. **Flow-Lenia Neural CA Substrate**: Pseudopod updates are learned continuous CA rules with mass conservation, parameter localization, warp-level GPU execution
   - Conway (Level 1): Fixed rules, discrete states → Lenia (Level 2): Fixed rules, continuous states → Flow-Lenia (Level 3): Localized parameters, mass conservation → **Ours (Level 4)**: Learned rules, adaptive parameters, intrinsic motivation
4. **Adaptive Voronoi MAP-Elites Core**: Archive-driven evolution with cells that grow/shrink based on density, DIRESA learned embeddings (adaptive 2-10D)
5. **Warp-Native GPU Kernels**: Like Polynesian navigator reading ocean/stars/birds as unified field, read warps/cache/tensor-cores as unified substrate
   - FlashAttention-style tiling (HBM ↔ SRAM), warp shuffles for zero-global-memory neighbor access, tensor cores for 256 FLOPs/cycle convolutions
6. **Content-Addressable Low-Rank Archive**: SVD factorization + content-addressed delta compression (80-160x memory reduction)
7. **Validated Behavioral Space**: Distance-preserving embeddings ensure dimensions correlate with hardware structure (ultrametric topology via topology/{p_adic,genealogy,hierarchy,hybrid_metric})
8. **DIRESA Learned Embeddings**: Adaptive dimensionality via distance-preserving nonlinear autoencoders, learns online, dimension count adapts via warp vote (2-10D)
9. **Deterministic Random**: Hash-based seeded random for reproducibility
10. **SRE Built-In**: Observability, SLOs, error budgets from day one (100% constraint satisfaction always)
11. **Local-Hierarchical Duality**: Neural CA reads from spatial neighbors (local perception) while Archive/Genealogy broadcasts to descendants (hierarchical memory). GPU grid operations extract neighborhood context, phylogenetic tree propagates information through lineages. Complementary structures for different timescales
12. **DRY Principle**: Single source of truth for each concept

## Dependency DAG

**Layer 0: Protocols** (no dependencies)
- proto/kernel.py
- proto/memory.py
- proto/model.py
- proto/component.py

**Layer 1: Implementations** (depend only on Layer 0)
- kernels/utils.py, kernels/triton_impl.py, kernels/torch_fallback.py → proto.kernel
- observability/metrics.py, observability/slo.py (passive collection/validation)

**Layer 2: Data structures** (depend on Layer 0-1)
- memory/archive.py, memory/pool.py → proto.component
- memory/tubes.py → proto.memory
- core/state.py (FlowState dataclass, no dependencies)
- core/stencil.py (GPU-parallel spatial ops, no dependencies)

**Layer 3: Components** (depend on Layer 0-2)
- core/pseudopod.py → proto.model, proto.kernel, kernels/*, observability/*
- core/chemotaxis.py → proto.model, memory/archive
- memory/pool.py → core/stencil (batched spatial computation)

**Layer 4: Orchestration** (depend on Layer 0-3)
- core/organism.py → proto.model, core/pseudopod, core/chemotaxis, memory/*, observability/*

**Layer 5: API** (depend on Layer 0-4)
- api/torch_compat.py, api/native.py → core/organism

**Layer 6: Applications** (depend on Layer 0-5)
- training/trainer.py, training/fitness.py, training/lifecycle.py
- bench/profile.py, tools/export.py, tools/package.py
- config/loader.py
- tests/* (unit, integration, ablations, slo)


## Data Flow

**User Input** → **API Layer** (torch_compat, native)
    ↓
**Organism** (orchestrator, owns Pool + Archive + Chemotaxis)
    ↓ uses                    ↓ owns
**Pseudopod Pool** ←→ **Archive** (MAP-Elites storage)
    ↓ delegates
**Pseudopod** (Component) → adds self to Archive
    ↓ calls
**Kernels** (GPU compute, warp-level execution)
    ↓ metrics collected by
**Observability** (passive side channel, no callbacks)

No cycles. Archive doesn't call anything. Observability is passive collector.

**Computation as ensemble over trajectories:**
```python
pseudopods = self.pseudopod_pool.get_at(behavior, max_count=self._max_pseudopods)
for pod in pseudopods:
    outputs.append(pod(pod_input, stim_input))
merged = torch.stack(outputs).mean(0)  # Weighted sum over computational trajectories
```

Each forward pass computes the ensemble average over all active pseudopods at a behavioral location. The archive maintains the history of successful trajectories through configuration space, weighted by fitness. Selection collapses the ensemble to high-fitness trajectories that persist to the archive.

## Protocols

### proto.component.Component
**Purpose**: Unified interface for ALL pooled components

**Interface**:
- fitness: float (property) - Component quality metric
- reset() → None - Reset internal state
- to_dict() → dict - Immutable serialization for Archive storage
- from_dict(data: dict) → Component - Reconstruction from serialized form

**Usage**: Archive stores any Component via to_dict(), Pools manage any Component via fitness property

### proto.memory.Memory
**Purpose**: Temporal memory interface with decay (NOT for component lifecycle - that's pool.py)

**Interface**:
- store(data: Tensor, weight: float) → None - Store with decay weight
- recall() → Optional[Tensor] - Retrieve with temporal blending
- clear() → None - Reset memory state

**Implementations**: memory.tubes.TubeNetwork (flowing memory with exponential decay)

### proto.model.Pseudopod
**Purpose**: Sensory probe with Neural CA update rule (Flow-Lenia substrate)

**Interface**:
- forward(latent, stimulus) → output - Learned CA update with Flow-Lenia dynamics
- correlation: Tensor (property) - Mass conservation metric (∑ output = ∑ input)
- effective_rank() → Tensor - Parameter localization metric (spatial variation of update rule)
- coherence() → Tensor - Learning progress metric for curiosity-driven lifecycle

**Neural CA Substrate**:
- forward() implements learned continuous CA rule with mass conservation
- Parameter localization: Spatial variation of update rule parameters (not global)
- Learned via gradient descent on downstream task loss
- Warp-level GPU execution via proto.kernel.Kernel

**Configuration space paths**: Each forward pass traces a trajectory through parameter space (CA weights, attention weights, normalization scales). The CA update rule defines local dynamics. Training modifies the landscape these trajectories traverse. Each pseudopod explores a different region of this configuration manifold.

**Dependencies**: MUST use proto.kernel.Kernel for all compute, MUST implement proto.component.Component

### proto.model.Chemotaxis
**Purpose**: Behavioral space navigator with curiosity-driven search (Adaptive Voronoi MAP-Elites)

**Interface**:
- add_source(nutrient, location, concentration) → None - Add elite to archive (grow Voronoi cell)
- sample(behavior, metabolic_rate, hunger) → Optional[Tensor] - Sample genome from archive
- clear() → None - Reset archive state

**Curiosity-Driven Lifecycle**:
- hunger = learning_progress_deficit (intrinsic motivation via coherence() metric)
- High coherence() (learning fast) → low hunger → survive
- Low coherence() (plateaued) → high hunger → sample new genome from archive
- Natural selection via intrinsic curiosity, not external reward

**Archive as trajectory history**: The archive stores successful parameter configurations that reached behavioral locations. When sampling from archive, the system retrieves trajectories that previously contributed non-negligible fitness at that location. Bootstrapping initializes new pseudopods from these historical trajectories, weighted by their fitness contributions.

**Dependencies**: Uses memory.archive for spatial indexing (Adaptive Voronoi cells), NO direct component management

### proto.model.Organism
**Purpose**: Top-level orchestrator with context-aware GPU execution

**Interface**:
- forward(stimulus, state) → (output, new_state) - Collective Pseudopod updates
- reset_state() → None - Reset organism state
- stats() → dict - GPU occupancy, learning progress, archive coverage

**Context-Aware GPU Execution**:
- GPU state is the execution context (not external orchestration)
- extract(warp_id) → LocalObservation (warp occupancy, neighbor state, cache hits)
- Context-aware decisions: spawn/retire Pseudopods based on whole computational field
- Whole field (warps/cache/tensor-cores) informs local decisions

**Dependencies**: Owns Pool[Pseudopod] + Archive + Chemotaxis, Uses Kernels via Pseudopods, Records Observability metrics

## File Structure

**Repository Layout**:
- BLUEPRINT.md (system architecture), README.md (user documentation with examples)
- setup.py, pyproject.toml, requirements.txt, .python-version
- strip_docstrings.py (AST-based docstring removal tool)

**slime/ package structure**:
- **proto/** - Protocol definitions (component, kernel, memory, model interfaces)
- **kernels/** - GPU compute implementations (triton_impl, torch_fallback, utils)
- **observability/** - Passive metrics collection (metrics, slo, tracing)
- **memory/** - Data structures (archive MAP-Elites storage, pool lifecycle, tubes temporal memory)
- **core/** - Components (state FlowState dataclass, stencil GPU-parallel ops, pseudopod Neural CA, chemotaxis navigator, organism orchestrator)
- **api/** - Public interfaces (torch_compat nn.Module, native SlimeModel)
- **training/** - Training loop (trainer, losses, stability, fitness computation, lifecycle decisions)
- **config/** - Configuration (loader with validation, YAML files for model/training/slo)
- **bench/** - Benchmarking (datasets loaders, baseline transformer, profiling, toy tasks)
- **tests/** - Test suites (unit/ protocol + implementation tests, integration/ end-to-end, ablations/ comparative studies, slo/ performance validation)
- **tools/** - Utilities (visualize behavioral space, export ONNX/TorchScript, package .exe)


## Invariants

### 1. Dependency Direction (DAG Enforcement)
- Lower layers NEVER import from higher layers
- Protocols NEVER import implementations
- Components NEVER import API layer
- **Violation = compilation error**

### 2. Ownership Hierarchy

**Organism owns:**
- Pool[Pseudopod]
- Archive
- Chemotaxis

**Pool owns:**
- List[Component]

**Archive owns:**
- Dict[cell, Elite] where Elite.genome = dict (NO object refs)

**NO CYCLES**

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

**Fast (every step):**
- Weight updates via backprop
- Fitness tracking
- Metrics collection
- Loss monitoring

**Medium (every 100 steps):**
- Fitness assessment
- Archive elite updates
- Pool spawn decisions
- Loss gate check

**Slow (every 1000 steps):**
- Pool culling
- Memory budget enforcement
- Behavioral space analysis
- Hard limit enforcement (max pool size, max archive)

### 7. Ultrametric Topology

**Strong triangle inequality**: d(x, z) ≤ max(d(x, y), d(y, z)) for all x, y, z

**Implementation**: True dendrogram traversal via linkage matrix merge height

**Topology Types**:
- **p-Adic**: Distance based on common prefix length (hierarchical codes)
- **Genealogy**: Distance based on common ancestor recency
- **Hierarchy**: Distance via dendrogram merge height (ultrametric guarantee)

**Chemotaxis Integration**: HybridMetric (ultrametric between clusters, Mahalanobis within), DIRESA preserves ultrametric in learned embeddings

### 8. Archive Bootstrapping Policy 
- Archive provides INITIALIZATION only
- Bootstrapped components trained with rest of network
- NO frozen parameters injected mid-training
- Prevents mode collapse and gradient conflicts

### 8. GPU-Parallel Spatial Stencil Computation
**Reasoning (GPU Architecture):** GPU computation is spatial (SIMD, tiles, stencil convolution), not sequential. Batched operations on entire populations, not individual components.

**JAX vmap pattern (push loops to primitives):**
- BAD (sequential): Loop over pool computing per-component z-scores → O(N) sequential
- GOOD (parallel): vmap_relative_fitness(fitnesses, neighbor_mask) → O(1) GPU call

**SpatialStencil**: JAX vmap-inspired batched computation of contextual metrics (pairwise distances, k-nearest neighbors, vectorized metrics) - 100x-1000x speedup vs sequential

**Pattern:** Stencil kernel applied to every component position (SIMD), matches GPU architecture perfectly.

### 9. Fitness Correlation with Task
Fitness MUST correlate with loss reduction. Options:
- Gradient magnitude (components affecting loss)
- CA mass conservation correlation with targets
- Information bottleneck metrics (mutual information)
- **Relative fitness** (gradient magnitude z-score vs k-nearest neighbors)

NOT activation entropy alone (doesn't correlate with task)

### 9. CVT-MAP-Elites Architecture
**Reasoning (Scalability):** Fixed grid scales as resolution^dims. CVT scales linearly with num_centroids.

**Fixed grid problem:** Exponential explosion (3D: 8k cells, 4D: 160k, 5D: 3.2M)

**CVT solution:** Linear scaling (1000 centroids for any dimensionality)

**Behavioral dimensions:** DIRESA learns 2-10 nonlinear dimensions from 10-20 raw metrics online. Trustworthiness/Continuity validation ensures distance preservation.

### 10. Content-Addressable Low-Rank Archive Storage
**Reasoning (Memory Efficiency):** SVD low-rank factorization with content-addressable delta compression.

**Storage strategy:**
1. SVD low-rank factorization: D×D → (D×k + k×D), 8x compression
2. Content-addressable hashing: Deduplicate identical elites
3. Delta compression: Store diffs vs parent in same centroid, 10-20x additional compression

**Result:** 80-160x memory reduction (D=512, k=64: 4MB → 25-50KB per elite)

**Key insight:** Elites in same centroid have similar behaviors → similar weights → tiny deltas

### 11. Lifecycle Safety Guardrails
**Hard limits:** MAX_POOL_SIZE=64, MAX_ARCHIVE_CENTROIDS=1000, MAX_LOSS_RATIO=10.0
**Loss gates:** Freeze lifecycle if loss > 10× moving average
**Training:** DIRESA learns embeddings online, annealing schedule for exploration→exploitation, curiosity-driven lifecycle

**Cellular lattice as discrete spacetime:** The CA operates on a discrete spatial lattice where each cell undergoes local update rules. Mass conservation couples neighboring cells. Each forward pass applies the update rule across all lattice positions simultaneously (SIMD), computing one timestep of the discrete dynamics. Training gradient descent modifies the update rule parameters, changing which computational trajectories are accessible from given initial conditions.

## Premortem Analysis

**Purpose**: Identify failure modes before they occur. Each item describes a plausible way the system could fail, why it might happen, and how to detect it early.

### Training Instability from Archive Updates

**Failure mode**: Archive updates cause gradient variance spikes → training loss divergence

**Why it happens**: 
- New pseudopods bootstrapped from archive have different loss landscape positions
- Sudden parameter distribution shifts confuse optimizer momentum
- Lifecycle events (birth/death) create discontinuities in gradient flow

**Early detection**:
- Monitor loss variance: `np.std(losses[-100:]) > 3.0 × np.std(losses[-1000:-100])`
- Track gradient norm spikes: `grad_norm > 10.0 × grad_norm_ema`
- Watch lifecycle event correlation: High birth rate within 50 steps before loss spike

**Mitigation**:
- Loss gates: Freeze lifecycle when loss > 10× EMA (already implemented)
- Warmup period: 100 steps no lifecycle, 500 steps reduced frequency (implemented)
- Gradient clipping: Clip to 1.0 during gentle phase, 0.5 during warmup
- Archive bootstrapping only for initialization: New pseudopods train from scratch with gradient flow

### DIRESA Embeddings Don't Converge

**Failure mode**: Behavioral embeddings fail to preserve distances → archive loses diversity signal

**Why it happens**:
- Insufficient training data: DIRESA needs ~1000 samples to learn manifold structure
- Metric instability: Raw behavioral metrics fluctuate wildly during early training
- Dimensionality mismatch: Intrinsic dimensionality > learned dimensions (e.g., 8D manifold compressed to 3D)
- Adversarial dynamics: Training modifies behavioral distribution faster than DIRESA adapts

**Early detection**:
- Trustworthiness < 0.70 (should be ≥ 0.85)
- Continuity < 0.70 (should be ≥ 0.85)
- Reconstruction error > 0.8 (should be ≤ 0.5)
- Embedding dimensions stuck at max (10D) or min (2D) for > 5000 steps

**Mitigation**:
- Delayed DIRESA activation: Use Euclidean distance for first 2000 steps, accumulate behavioral samples
- Metric stabilization: Use EMA-smoothed behavioral metrics for embedding training
- Adaptive dimension bounds: Allow 2-10D, monitor reconstruction error to detect under-compression
- Separate learning rate: DIRESA lr = 0.1 × main lr (slower adaptation)
- Fallback policy: If validation fails for > 1000 steps, revert to PCA embeddings

### Pool Collapse to Single Strategy

**Failure mode**: All pseudopods converge to identical behavior → diversity loss → archive coverage stalls

**Why it happens**:
- Fitness pressure dominates diversity pressure (lifecycle culls too aggressively)
- Archive sampling bias: High-fitness centroids sampled repeatedly, low-fitness never explored
- Gradient alignment: All pseudopods receive similar gradients → parameters converge
- Behavioral metrics too coarse: Can't distinguish actually-different strategies

**Early detection**:
- Pseudopod coherence std < 0.05 (should be > 0.1)
- Archive coverage plateaus: `len(archive.centroid_refs) / num_centroids` unchanged for 5000 steps
- Behavioral metric correlation: `np.corrcoef(behavioral_vectors)` off-diagonal > 0.9
- Pool size shrinks: `len(pool._components) < 0.5 × pool.max_size`

**Mitigation**:
- Diversity bonus in fitness: `fitness = 0.7 × task_fitness + 0.3 × diversity_bonus`
- Archive sampling with exploration: ε-greedy (ε=0.2) samples random centroid instead of fitness-weighted
- Birth threshold jitter: Add noise to birth_threshold (±0.1) to prevent deterministic culling
- Behavioral metric expansion: Add more raw metrics (target: 80-100 dimensions before DIRESA)
- Forced exploration: Every 1000 steps, spawn 1 pseudopod from random archive centroid

### Memory Budget Violation from Archive Growth

**Failure mode**: Archive grows unbounded → OOM crash → training interruption

**Why it happens**:
- Every elite addition creates new storage (low-rank + deltas)
- Delta chains grow without GC → orphaned objects accumulate
- Content-addressable hash table doesn't shrink → fragmentation
- DIRESA autoencoder weights grow with each dimension increase

**Early detection**:
- Memory usage growth rate: `(mem_now - mem_1000_steps_ago) / mem_now > 0.1` (10% growth per 1000 steps)
- Archive size: `len(archive._content_store) > 2 × num_centroids` (deduplication failing)
- Delta chain depth: `max(chain_lengths) > 50` (GC not running)
- GPU memory allocation failures: `torch.cuda.OutOfMemoryError` during archive operations

**Mitigation**:
- Hard archive limit: MAX_ARCHIVE_CENTROIDS=1000 (implemented), MAX_ELITES_PER_CENTROID=10
- Aggressive GC: Mark-and-sweep every 100 additions (implemented), reference counting
- Delta chain pruning: Collapse chains > 20 deltas into new base weights
- Memory monitoring: Track archive memory usage, trigger early culling at 80% budget
- Graceful degradation: Remove oldest elites in least-visited centroids when memory tight

### GPU Kernel Launch Failures

**Failure mode**: Triton kernel launches fail due to resource exhaustion → fallback to slow PyTorch → throughput collapse

**Why it happens**:
- Too many pseudopods active simultaneously → register pressure → kernel launch fails
- Tile sizes too large for SRAM → shared memory exhaustion
- Tensor shapes misaligned with warp size (32) → poor occupancy
- CUDA context switching overhead from multi-GPU hash partitioning

**Early detection**:
- Kernel fallback rate: `num_torch_calls / (num_triton_calls + num_torch_calls) > 0.1`
- Occupancy metrics: `torch.cuda.get_device_properties().multi_processor_count × active_warps < 0.5 × theoretical_max`
- Launch latency spikes: `kernel_launch_time > 2.0 × kernel_launch_ema`
- Memory allocation retries: Monitor `torch.cuda.memory_stats()['num_alloc_retries']`

**Mitigation**:
- Adaptive max_pseudopods: Scale with GPU memory (implemented), reduce when kernel launches fail
- Tile size autotuning: Start with BLOCK_M=128, reduce to 64 if SRAM pressure detected
- Batched kernel launches: Group pseudopod forward passes into single kernel with different offsets
- Hash partition validation: Monitor cross-device communication, fall back to single-GPU if overhead > 20%
- Graceful degradation: If Triton fails, log warning and use PyTorch fallback (already implemented)

### Loss Gates Over-Trigger (Lifecycle Frozen Too Long)

**Failure mode**: Loss gates freeze lifecycle for thousands of steps → stale pool → performance plateau

**Why it happens**:
- Loss EMA miscalibrated: Too low threshold triggers on normal variance
- Noisy tasks (e.g., RL) have inherently high loss variance
- Lifecycle freeze prevents adaptation → loss stays high → gates stay active (vicious cycle)

**Early detection**:
- Lifecycle frozen fraction: `frozen_steps / total_steps > 0.5`
- Loss EMA calibration: `loss_ema < 0.1 × np.mean(losses)` (EMA too optimistic)
- Pool staleness: No births/deaths for > 2000 steps while loss > threshold

**Mitigation**:
- Adaptive loss threshold: `threshold = max(10.0 × loss_ema, 2.0 × np.std(losses[-1000:]))`
- Cooldown period: After gate triggers, wait 500 steps before re-enabling (prevent oscillation)
- Gate timeout: If frozen for > 5000 steps, force re-enable lifecycle and recalibrate EMA
- Task-specific thresholds: Accept threshold multiplier as config parameter (default=10.0)

## Resource Budget

**Purpose**: Explicit computational and memory constraints for each component. Ensures system scales to available hardware without silent degradation.

### GPU Memory Budget (per-device)

**Total Available**: Detect via `torch.cuda.get_device_properties(device).total_memory`

**Allocation Strategy**:
- **Model weights (40%)**: Pseudopod parameters (CA weights, attention, norms)
- **Activations (30%)**: Forward pass intermediate tensors, gradients
- **Archive storage (15%)**: Low-rank weight storage, delta chains
- **DIRESA embeddings (5%)**: Autoencoder weights, behavioral metric buffers
- **Optimizer state (10%)**: Adam momentum/variance, gradient buffers

**Example (RTX 3090: 24GB)**:
- Model weights: 9.6 GB → ~80M parameters at fp32 (20M at fp16 mixed precision)
- Activations: 7.2 GB → Supports batch_size=32, max_pseudopods=16
- Archive: 3.6 GB → 1000 centroids × 10 elites × 360KB per elite (low-rank)
- DIRESA: 1.2 GB → 65 raw dims → 10 learned dims, batch=1024
- Optimizer: 2.4 GB → Adam state for 80M params

**Adaptive Limits**:
- `max_pseudopods = max(4, min(16, int(gpu_memory_gb * 0.3 / memory_per_pod_gb)))` (implemented in organism.py:40-45)
- `max_archive_centroids = min(1000, int(gpu_memory_gb * 0.15 / memory_per_centroid_gb))`
- `diresa_batch_size = min(1024, int(gpu_memory_gb * 0.05 / memory_per_sample_gb))`

**Safety Margins**:
- Reserve 10% headroom for fragmentation and CUDA overhead
- Monitor actual usage: `torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() < 0.9`
- Trigger GC when > 85% allocated: `torch.cuda.empty_cache()`

### Compute Budget (per-step)

**Target Latency**: 100ms per training step (10 steps/sec)

**Breakdown**:
- **Forward pass (50ms)**: All active pseudopods, ensemble averaging
  - Per-pseudopod: 3ms (Triton kernels), 12ms (PyTorch fallback)
  - Max pseudopods: 16 × 3ms = 48ms (parallel), 16 × 12ms = 192ms (fallback)
- **Backward pass (30ms)**: Gradient computation, optimizer step
- **Lifecycle decisions (5ms)**: Fitness evaluation, birth/death decisions (only every 100 steps)
- **Archive operations (10ms)**: Elite addition, sampling (only when needed)
- **Metrics collection (5ms)**: Behavioral metrics, observability

**Amortized Costs** (averaged over 100 steps):
- Archive elite addition: 200ms / 100 = 2ms per step
- Pool culling: 500ms / 1000 = 0.5ms per step
- DIRESA embedding update: 100ms / 100 = 1ms per step
- Voronoi adaptation: 1000ms / 1000 = 1ms per step

**Total**: 50 + 30 + 5 + 10 + 5 + 4.5 = 104.5ms → Within budget

**Scaling**:
- **More pseudopods**: Latency increases linearly (forward pass bottleneck)
- **Larger archive**: Logarithmic cost increase (hash table lookups)
- **Higher dimensions**: Quadratic cost in DIRESA (pairwise distances)

**Performance Requirements**:
- Triton kernel occupancy: ≥ 50% (warp utilization)
- Fallback overhead: < 10% of steps use PyTorch fallback
- Multi-GPU efficiency: ≥ 80% scaling (2 GPUs → 1.6× throughput)

### Training Budget (wall-clock time)

**Target Tasks**:
- **MNIST (TINY config)**: 10 minutes (5000 steps × 100ms, 1 GPU)
- **CIFAR-10 (SMALL config)**: 4 hours (80k steps × 150ms, 1 GPU)
- **ImageNet (MEDIUM config)**: 3 days (1M steps × 250ms, 4 GPUs)

**Baseline Comparisons** (untested hypotheses):
- **Standard Transformer (MNIST)**: ~5 minutes typical
- **DARTS (CIFAR-10)**: ~1.5 GPU-days typical
- **Hypernetworks (MNIST)**: ~8 minutes typical

**Expected Cost Breakdown (CIFAR-10)** (untested):
- **Forward/backward (70%)**: 2.8 hours pure gradient updates
- **Lifecycle overhead (20%)**: 48 minutes births/deaths/archive
- **Logging/checkpointing (10%)**: 24 minutes I/O

**Amortization Hypothesis** (requires empirical validation): 
- Short runs (< 10k steps): Lifecycle overhead may dominate
- Long runs (> 100k steps): Architecture discovery may amortize overhead
- Very long runs (> 1M steps): Adaptation may enable performance gains

### Storage Budget (disk)

**Checkpoints**:
- **Full checkpoint**: 500 MB (model weights + optimizer state + archive)
- **Archive-only**: 50 MB (compressed elites + delta chains)
- **Checkpoint frequency**: Every 1000 steps → 50 MB × (total_steps / 1000)

**Example (CIFAR-10: 80k steps)**:
- Full checkpoints: 500 MB × 5 (every 16k steps) = 2.5 GB
- Archive snapshots: 50 MB × 80 = 4 GB
- Logs/metrics: 500 MB (tensorboard + observability)
- **Total**: ~7 GB per training run

**Cleanup Policy**:
- Keep last 3 full checkpoints (1.5 GB)
- Keep archive snapshots for analysis (4 GB)
- Delete intermediate checkpoints after training completes

### Network Budget (multi-GPU)

**Hash-based Partitioning Overhead**:
- **Assumption**: `device_id = hash(behavior_coords) % num_gpus`
- **Worst case**: Every forward pass requires cross-device communication
- **Bandwidth**: PCIe 4.0: 32 GB/s, NVLink: 600 GB/s

**Communication Volume (per step)**:
- **Pseudopod migration**: 10 MB per pseudopod (weights + state)
- **Archive sync**: 50 MB per elite addition (low-rank weights + deltas)
- **Gradient sync**: 100 MB (all-reduce for optimizer step)

**Total (4 GPUs, 16 pseudopods)**:
- Pseudopod comm: 0 MB (assuming good hash distribution, no migration)
- Archive sync: 50 MB / 100 steps = 0.5 MB/step
- Gradient sync: 100 MB/step
- **Total**: ~100 MB/step → 1 ms latency on NVLink, 3 ms on PCIe

**Expected Scaling Efficiency** (untested):
- 2 GPUs: Target 1.8× throughput (90% efficiency)
- 4 GPUs: Target 3.2× throughput (80% efficiency, communication overhead)
- 8 GPUs: Target 5.6× throughput (70% efficiency, hash imbalance)

**Mitigation**:
- Use NVLink when available (200× faster than PCIe)
- Batch archive updates to reduce sync frequency
- Async gradient sync (overlap communication with computation)

## Decision Tree

**Purpose**: Guide implementation decisions with clear criteria. Each node asks a question, branches on measurable conditions, and leads to concrete actions.

### When to Add a New Component to Pool?

```
Q: Is pool below min_size?
├─ YES → Spawn immediately (safety: pool must have min_size components)
└─ NO → Q: Is current step < warmup_steps (100)?
    ├─ YES → Don't spawn (let existing components stabilize)
    └─ NO → Q: Is current step < gentle_phase_end (500)?
        ├─ YES → Q: Is step % 50 == 0? (reduced frequency)
        │   ├─ YES → Continue to fitness check
        │   └─ NO → Don't spawn
        └─ NO → Continue to fitness check
        
Fitness Check:
Q: Is there a component with fitness < death_threshold (0.1)?
├─ YES → Q: Is archive coverage < 0.5?
│   ├─ YES → Sample from random archive centroid (exploration)
│   └─ NO → Sample from fitness-weighted archive (exploitation)
└─ NO → Q: Is pool below max_size AND average fitness > birth_threshold (0.8)?
    ├─ YES → Spawn from archive (high-performing pool, room to grow)
    └─ NO → Don't spawn
```

### When to Cull Components from Pool?

```
Q: Is pool above max_size (hard limit)?
├─ YES → Cull immediately (fraction=0.3, remove worst performers)
└─ NO → Q: Is step % cull_interval (1000) == 0?
    ├─ YES → Q: Is loss > 10.0 × loss_ema? (loss gate)
    │   ├─ YES → Skip culling (training unstable, freeze lifecycle)
    │   └─ NO → Q: Is pool size > 1.5 × min_size?
    │       ├─ YES → Cull bottom 20% by fitness
    │       └─ NO → Don't cull (too close to min_size)
    └─ NO → Don't cull (not time yet)
```

### When to Update Archive with New Elite?

```
Q: Has component survived for > 100 steps?
├─ NO → Don't archive (too young, fitness unstable)
└─ YES → Q: Is fitness > current_elite_fitness at this centroid?
    ├─ YES → Replace elite (found better solution)
    └─ NO → Q: Is centroid empty?
        ├─ YES → Add as first elite (expand coverage)
        └─ NO → Q: Is centroid below capacity (10 elites per centroid)?
            ├─ YES → Add as additional elite (maintain diversity within cell)
            └─ NO → Q: Is fitness > worst_elite_fitness in this centroid?
                ├─ YES → Replace worst elite
                └─ NO → Don't archive (not competitive)
```

### When to Activate DIRESA Embeddings?

```
Q: Is step < 2000?
├─ YES → Use Euclidean distance on raw metrics (accumulate samples)
└─ NO → Q: Are there ≥ 1000 behavioral samples collected?
    ├─ NO → Continue using Euclidean (insufficient training data)
    └─ YES → Q: Train DIRESA for 500 steps, then validate
        Q: Is Trustworthiness ≥ 0.70 AND Continuity ≥ 0.70?
        ├─ YES → Activate DIRESA embeddings (sufficient quality)
        └─ NO → Q: Have we retried < 3 times?
            ├─ YES → Retrain with 2× learning rate
            └─ NO → Fallback to PCA (DIRESA failing to converge)
```

### When to Trigger Loss Gate (Freeze Lifecycle)?

```
Q: Is loss > 10.0 × loss_ema?
├─ YES → Q: Has loss gate been active for < 5000 steps?
│   ├─ YES → Freeze lifecycle (loss unstable, protect training)
│   └─ NO → Q: Is loss still decreasing (gradient of loss_ema < 0)?
│       ├─ YES → Continue freeze (making progress despite high loss)
│       └─ NO → Force unfreeze and recalibrate EMA (stuck, need adaptation)
└─ NO → Q: Is loss gate currently active?
    ├─ YES → Q: Has loss been stable for > 500 steps?
    │   ├─ YES → Unfreeze lifecycle (loss recovered)
    │   └─ NO → Continue freeze (wait for stabilization)
    └─ NO → Normal lifecycle operation
```

### How to Choose Tile Sizes for Triton Kernels?

```
Q: What is SRAM size per SM?
├─ >= 128 KB (A100) → Try BLOCK_M=128, BLOCK_N=128, BLOCK_D=64
├─ >= 64 KB (RTX 3090) → Try BLOCK_M=64, BLOCK_N=64, BLOCK_D=64
└─ < 64 KB (older GPUs) → Try BLOCK_M=32, BLOCK_N=32, BLOCK_D=32

Q: Did kernel launch succeed?
├─ YES → Q: Is occupancy > 50%?
│   ├─ YES → Keep current tile sizes
│   └─ NO → Reduce tile sizes by 2× (improve occupancy)
└─ NO → Q: Was error "out of shared memory"?
    ├─ YES → Reduce tile sizes by 2×
    └─ NO → Q: Was error "out of registers"?
        ├─ YES → Reduce BLOCK_D (fewer registers per thread)
        └─ NO → Fallback to PyTorch (unknown error)
```

### When to Use Archive Sampling vs Random Initialization?

```
Q: Is archive coverage > 0.1? (at least 10% of centroids filled)
├─ NO → Random initialization (archive too sparse)
└─ YES → Q: Is this an exploration step? (ε=0.2 probability)
    ├─ YES → Sample from random archive centroid (force diversity)
    └─ NO → Q: What is the behavioral location for this spawn?
        Q: Does archive have elite at this centroid?
        ├─ YES → Sample from this centroid (behavioral locality)
        └─ NO → Q: Does archive have elites in k=5 nearest centroids?
            ├─ YES → Sample from nearest centroid (approximate locality)
            └─ NO → Random initialization (no relevant history)
```

### When to Increase vs Decrease max_pseudopods?

```
Q: Is GPU memory utilization > 90%?
├─ YES → Decrease max_pseudopods by 25% (approaching OOM)
└─ NO → Q: Is GPU memory utilization < 60%?
    ├─ YES → Q: Is kernel occupancy > 70%?
    │   ├─ YES → Increase max_pseudopods by 25% (underutilized)
    │   └─ NO → Keep current (occupancy bottleneck, not memory)
    └─ NO → Q: Did we encounter kernel launch failures?
        ├─ YES → Decrease max_pseudopods by 50% (register/SRAM pressure)
        └─ NO → Keep current (healthy utilization)
```

### How to Handle DIRESA Embedding Dimension Count?

```
Q: Is reconstruction error > 0.8?
├─ YES → Increase dimensions by 1 (under-compressing, losing information)
└─ NO → Q: Is reconstruction error < 0.3?
    ├─ YES → Q: Are dimensions > 2?
    │   ├─ YES → Decrease dimensions by 1 (over-parameterized)
    │   └─ NO → Keep at 2D (minimum dimensionality)
    └─ NO → Q: Has dimension count been stable for > 2000 steps?
        ├─ YES → Keep current (converged to good dimensionality)
        └─ NO → Q: Is Trustworthiness < 0.85?
            ├─ YES → Increase dimensions (neighborhoods not preserved)
            └─ NO → Keep current (quality acceptable)
```

## Architectural Decisions

### 1. IO-Aware Tiled Attention (FlashAttention)
**Reasoning (Dao et al., 2022):** Standard attention is memory-bound, not compute-bound. Tile to maximize SRAM usage.

**Problem:** Attention loads Q, K, V from HBM repeatedly.

**FlashAttention solution:** Tile computation to fit in SRAM.

**IO complexity:**
- Naive: O(M² × D) HBM accesses
- Tiled: O(M² × D / SRAM_size) HBM accesses
- Speedup: ~3x on GPT-2 (Dao et al., 2022)

**Implementation:** kernels/triton_impl.py uses tiling with BLOCK_M=128, BLOCK_N=128, BLOCK_D=64.

### 2. Kernel injection: Constructor Injection
**Reasoning (Bitter Lesson):** Let user provide compute capability. Scale with available hardware, not our assumptions.

**Pattern**: Pseudopod constructor accepts Kernel interface, allowing user to provide compute capability (Triton GPU, PyTorch CPU fallback, custom implementations). Kernel is injected at construction time, not hardcoded.

### 3. Multi-GPU: Hash-based partitioning
**Reasoning (Bitter Lesson):** Hash function scales arbitrarily. No hand-coded spatial assumptions.

**Pattern**: Device assignment via hash function: device_id = hash(behavior_coords) modulo num_gpus. Scales to arbitrary GPU counts without manual partitioning logic.

### 4. Determinism: Sort keys on iteration
**Reasoning (Architecture):** Spatial structure over temporal accidents. Reproducible science.

**Pattern**: Archive iteration uses sorted keys to ensure deterministic order. Prevents non-deterministic behavior from hash table iteration order.

### 5. Memory limits: Soft limit with graceful degradation
**Reasoning (SRE + Bitter Lesson):** Adapt to constraints, don't crash. Trade quality for capacity automatically.

**Pattern**: When memory exceeds budget, pool culls worst-performing components (fraction=0.2). System degrades quality gracefully rather than crashing on OOM.

### 6. Metrics injection: Dependency injection
**Reasoning (SRE + Testing):** Explicit dependencies. No globals. Testable.

**Pattern**: Organism constructor accepts optional MetricsCollector parameter. Metrics are injected as explicit dependencies, not accessed via global state.

### 7. Fitness metric: Gradient magnitude
**Reasoning (Training Stability):** Fitness must correlate with task performance, not internal diversity metrics.

**Formula**: fitness = gradient_norm × correlation_with_targets. Combines gradient magnitude (task impact) with CA mass conservation metric (relevance).

### 8. Archive bootstrapping: Initialization only
**Reasoning (Gradient Flow):** Don't inject frozen weights mid-training. Bootstrap init, then train together.

**Pattern**: When new component needed, archive provides initialization (bootstrap_component), then component is trained with full gradient flow (requires_grad=True). No frozen weights injected.

### 9. Timescale separation: 1x / 100x / 1000x
**Reasoning (Stability):** Separate fast (weights) from medium (fitness) from slow (lifecycle).

**Schedule**: Every step (1x) - update fitness EMA. Every 100 steps - check archive elite conditions. Every 1000 steps - pool culling. Prevents lifecycle churn from interfering with gradient updates.

### 10. DIRESA Behavioral Dimension Learning
**Reasoning:** Behavioral characterization is CRITICAL. Wrong dimensions = useless diversity. Hardcoded dimensions are arbitrary. DIRESA learns distance-preserving nonlinear embeddings online with adaptive dimensionality.

**DIRESA Architecture:** Autoencoder with learned gating for adaptive dimensions (2-10D), distance preservation loss, online training

**Raw metrics (10-20 metrics):** CA_mass_conservation, activation_sparsity, gradient_flow_magnitude, memory_access_locality, computational_intensity, CA_parameter_localization, weight_magnitude, gradient_variance, activation_magnitude, CA_neighborhood_coherence, etc.

**Learned embeddings:** Nonlinear projections preserving pairwise distances via autoencoder with distance loss

**Validation:** Trustworthiness ≥ 0.85, Continuity ≥ 0.85, Procrustes distance ≤ 0.15, reconstruction error ≤ 0.5

### 10a. DIRESA Adaptive Dimensionality
**Question: How many dimensions should behavioral embeddings use?**

**WRONG (arbitrary fixed):** hardcode 5 dimensions

**RIGHT (adaptive learned):** DIRESA autoencoder with learned gating determines dimensionality online

**Mechanism**: DIRESA encoder has gating layer that learns which dimensions to activate. Dimension count adapts via warp vote mechanism (2-10D range). System learns optimal dimensionality based on task, not predetermined.

**Validation**: Trustworthiness ≥ 0.85 (neighbors preserved in low-D), Continuity ≥ 0.85 (neighborhoods preserved), Procrustes distance ≤ 0.15 (shape similarity), reconstruction error ≤ 0.5 ensure learned embeddings are factorable and distance-preserving.

### 10b. Content-Addressable Storage: Delta Protocol Specification
**Question 3: What operations does delta compression support?**

**Delta format (structured operations, NOT raw byte diffs):**

**Operations**: Delta is list of structured weight-level operations:
- **sparse_add**: Add values at specified 2D indices (for sparse updates with >95% sparsity)
- **low_rank**: Low-rank update W += dU @ dV where dU is D×r, dV is r×D, r << k (for dense medium-sparsity updates)
- **dense**: Full replacement (for small tensors like biases)
- **scale_add**: Scalar multiplication plus sparse add (for small perturbations)

**Application**: apply_delta reconstructs weights by applying operations sequentially to base weights. Each operation modifies specific weight matrix.

**Compression strategy**: Choose operation based on sparsity and size. Sparsity >95% → sparse_add. Small tensors → dense. Otherwise → low_rank SVD with rank r=8.

### 10c. Content-Addressable Storage: Garbage Collection Policy
**Question 4: When are unreferenced objects deleted?**

**GC Policy: Reference Counting + Periodic Mark-and-Sweep**


**GC guarantees:**
1. **No premature deletion:** Reference counting prevents deletion while object is reachable
2. **No memory leaks:** Mark-and-sweep catches orphaned delta chains
3. **Bounded overhead:** GC runs every 100 add() calls, amortized O(1) per operation
4. **Deterministic:** Same sequence of operations → same GC decisions (given same seed)

### 11. Deterministic hash-based random
**Reasoning (Reproducibility):** Non-deterministic random breaks reproducibility. Hash-based seeded random is cheap (~100ns) vs gradient computation (ms).

**Deterministic random primitive:**
**Implementation**: Hash-based seeded random: `hash(seed, step, context) → [0,1]` for all stochastic decisions

**Benefits**: Reproducible training, debuggable birth/death sequences, ablation-ready

**Pattern**: NO unseeded random. All decisions via `_deterministic_random(seed, step, context)`

### 12. Fitness must include efficiency signals
**Reasoning (Hardware Awareness):** Task accuracy alone won't discover hardware-optimal patterns. Fitness must reward efficiency.

**Fitness formula**: `task_performance (70%) + compute_efficiency (20%) + gradient_magnitude (10%)`

**Result**: Hardware-optimal patterns emerge from selection pressure (fast components survive, slow components culled)

### 12. Quality-diversity maintains architectural variety
**Reasoning (Avoid Mode Collapse):** Standard transformers: all heads learn similar features. MAP-Elites: forced diversity.

**Standard transformer**: All heads learn similar features (head_similarity > 0.9)
**MAP-Elites**: Each archive cell requires behaviorally-distinct component (forced diversity, no mode collapse)

**Benefit 1**: Graceful degradation under device loss (hash-based redistribution, no retraining)
**Benefit 2**: Interpretability (query archive by behavioral coordinates)

### 13. Ablation testing methodology

Tests should make specific predictions that could be disproven. Design tests where failure is possible and informative.

**Approach:**
1. State measurable predictions with numeric thresholds
2. Design counter-examples where the prediction should fail
3. Probe live system state (Archive coverage, Pseudopod coherence, Voronoi densities)
4. Isolate component-level causation

**Avoid:**
- Black-box model comparison
- Recomputing metrics the system already tracks
- Descriptive statistics without falsification criteria
- Static end-of-training snapshots

**Test cases:**

**Curiosity-driven lifecycle maintains population diversity**
- Prediction: Pseudopod coherence std > 0.1 throughout training
- Counter-test: Static pool should collapse to std < 0.05
- Probe: `[pod.coherence().item() for pod in organism.pseudopod_pool._components]` every 100 steps
- If static pool maintains diversity, remove lifecycle

**DIRESA discovers intrinsic dimensionality**
- Prediction: Archive.behavioral_dims < 0.5 * Archive.num_raw_metrics
- Counter-test: Pure Gaussian noise should fail to compress (dims ≈ num_raw_metrics)
- Probe: `archive.behavioral_dims` vs `archive.num_raw_metrics` at discovery
- If DIRESA compresses noise, embedding is broken

**Adaptive Voronoi prevents density variance explosion**
- Prediction: `np.var(list(archive._cell_densities.values())) < 5.0` after 1000 additions
- Counter-test: Static Voronoi should exceed variance > 20.0
- Probe: `archive._cell_densities` histogram every 100 additions
- If static has same variance, remove adaptive logic

**Coherence-based state blending improves sample efficiency**
- Prediction: Adaptive blend reaches 90% accuracy in < 0.8x epochs vs fixed 0.5/0.5
- Counter-test: Fixed blend on same task
- Probe: Training curves for adaptive vs `body = 0.5*fresh + 0.5*state`
- If fixed is as fast, revert to simpler implementation

**Archive bootstrapping accelerates convergence**
- Prediction: Archive-bootstrapped Pseudopods reach fitness > 0.5 in < 200 steps
- Counter-test: Random init should take > 500 steps
- Probe: `pod.fitness` trajectory for archive-sampled vs factory-spawned
- If random init converges as fast, disable bootstrapping

**Mass conservation enables substrate stability**
- Prediction: CA mass conservation > 0.95 throughout training
- Counter-test: Unconstrained CA should diverge (< 0.7)
- Probe: `pod._ca_metrics['CA_mass_conservation']` every forward pass
- If unconstrained is stable, simplify

**Archive coverage correlates with generalization**
- Prediction: Test accuracy increases with coverage (0.2 to 0.8 coverage = +5% test acc)
- Counter-test: Limit coverage to 0.3 should hurt test performance
- Probe: `len(archive.centroid_refs) / archive.num_centroids` vs test accuracy
- If uncorrelated, archive is overhead

**Implementation:**
- Probe live Organism state: `organism.archive`, `organism.pseudopod_pool._components`, CA metrics
- Track trajectories: coherence over time, coverage evolution, density histograms
- Mechanistic interventions: disable components, inject controlled noise
- Numeric thresholds for pass/fail

**Criteria:**
- Failed prediction means component is broken or unnecessary
- Survived counter-test means provisionally supported
- No measurable improvement means remove the component
- Test without failure mode is invalid

## Computational Cost Analysis

### Cost Structure

**Training costs (per step):**

**Memory costs:**

**Total overhead vs baseline transformer:**

### Comparison to DARTS (Modern NAS)

**DARTS (Liu et al., 2018) baseline:**
- Differentiable architecture search with continuous relaxation
- Uses weight sharing across candidate operations
- Search cost: 1-4 GPU days on CIFAR-10/ImageNet
- 1000x faster than early NAS methods (NASNet: 2000 GPU days)

**Key difference:**

**Slime approach:**

**Honest comparison:**

**Hypothesis (requires empirical validation):** For long training runs (100+ GPU days), amortized search cost favors Slime. For short runs (<30 days), DARTS is more efficient.

### Comparison to Hypernetworks (Ha et al., 2016)

**Hypernetwork approach (Ha et al., ICLR 2017):**
- Small network generates weights for larger network
- Achieves parameter efficiency: fewer learnable params than standard networks
- Memory-efficient formulation: O(Nz × hidden_units) not O(Nz × all_params)
- Weight sharing across layers via generation scheme

**Key insight from Ha et al.:** Low-rank weight generation can be MORE efficient than storing full matrices.

**Slime borrows this insight for archive storage:**

**Key differences:**

**Computational tradeoffs:**

**Complementary approaches:** Hypernetworks excel at few-shot adaptation. Slime excels at maintaining diverse specialists for single-task training.

### Simulated Annealing for Component Lifecycle

**Insight:** Quality-diversity needs exploration-exploitation balance. Simulated annealing provides principled temperature schedule.

**Applications**:
- Birth decisions: Temperature schedule for accepting diverse vs high-fitness components
- CVT centroid refinement: Annealing to minimize quantization error
- Archive mutation strength: Large mutations (early) → small mutations (late)

**Pattern:** Annealing naturally transitions from exploration → exploitation without manual phase boundaries.

**Fundamental difference:**

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

**Untested Hypothesis**: Slime may match or exceed task accuracy with significantly less compute than NAS methods, while maintaining competitive throughput with hypernetworks.

**Empirical validation required**: If comparative testing shows no advantage over baselines, simplify or abandon architecture.

**Decision Tree Check**: See "When to Add a New Component to Pool?" for lifecycle spawn logic, "When to Activate DIRESA Embeddings?" for embedding training schedule.

## System Components

**Complete Architecture**: Algebraic effect handlers, Ultrametric topology, DIRESA learned embeddings (adaptive 2-10D), Adaptive Voronoi MAP-Elites, Neural CA Pseudopods (Flow-Lenia substrate), Curiosity-driven lifecycle (learning progress), Context-aware GPU execution

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

### Flow-Lenia & Cellular Automata (2023-2025)

- Randazzo, E., Mordvintsev, A., Niklasson, E., Levin, M., & Greydanus, S. (2023). "Flow-Lenia: Towards open-ended evolution in cellular automata through mass conservation and parameter localization." *Artificial Life Conference Proceedings*, MIT Press. [arXiv:2212.07906](https://arxiv.org/abs/2212.07906)
  - Mass-conservative continuous CA with spatially-localized parameters
  - Multi-species dynamics without global rules
  - **Foundation substrate for our Pseudopod update rules**

- Béna, G. (2025). "A Path to Universal Neural Cellular Automata." [arXiv:2505.13058](https://arxiv.org/abs/2505.13058)
  - Learned CA update rules via gradient descent on downstream tasks
  - Continuous state spaces, differentiable dynamics
  - **Target: Pseudopods as learned Neural CA**

### Learned Embeddings & Dimension Reduction (2025)

- Zhang, Y., et al. (2025). "DIRESA: Distance-preserving nonlinear dimension reduction via regularized autoencoders." [arXiv:2404.18314](https://arxiv.org/abs/2404.18314)
  - Adaptive dimensionality via learned gating (2-10 dimensions)
  - Distance-preserving autoencoder with explicit pairwise distance loss
  - **Foundation for our behavioral embedding learning**

### Curiosity & Intrinsic Motivation (2021-2024)

- Gottlieb, J., & Oudeyer, P.-Y. (2021). "Humans monitor learning progress in curiosity-driven exploration." *Nature Communications*, 12:5972.
  - Learning progress (derivative of prediction error) drives exploration
  - Intrinsic motivation via competence signals, not external rewards
  - **Foundation: coherence() metric → curiosity-driven Pseudopod lifecycle**

- NeurIPS 2024 Workshop: Intrinsic Motivation and Open-Ended Learning (IMOL)
  - State-of-the-art intrinsic motivation research
  - Connections to developmental psychology, meta-learning, open-endedness
  - **Informs our curiosity-driven selection pressure**


### Dimensionality Reduction Validation

- Venna, J., Peltonen, J., Nybo, K., Aidos, H., & Kaski, S. (2010). "Information retrieval perspective to nonlinear dimensionality reduction for data visualization." *Journal of Machine Learning Research*, 11, 451-490.
  - Trustworthiness and Continuity metrics for embedding quality
  - Measures preservation of local neighborhoods in dimension reduction

- Gower, J. C., & Dijksterhuis, G. B. (2004). *Procrustes Problems*. Oxford University Press.
  - Procrustes distance for comparing geometric configurations
  - Used to validate shape-preservation in DIRESA embeddings

### Optimization Theory

- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). "Optimization by simulated annealing." *Science*, 220(4598), 671-680.
  - Simulated annealing for combinatorial optimization
  - Temperature schedule for exploration-exploitation balance

## Wave 2: Bio-Inspired Enhancements (Effect Handler Gated)

**Philosophy**: Optional capabilities via algebraic effect handlers. Wave 1 (current) must work first. Wave 2 adds sophistication component-by-component with zero overhead when disabled.

### Effect Handler Architecture

**Pattern**: Each Wave 2 feature is an algebraic effect that components can request:

```python
# Component requests capability
@effect_handler('conformational_switching')
def forward(self, x):
    if has_effect('conformational_switching'):
        return self.conformational_ode(x)
    else:
        return self.standard_forward(x)
```

**Benefits**:
- Zero overhead when disabled (no runtime checks, compile-time optimization)
- Gradual migration (enable per-component, not system-wide)
- A/B testing (compare with/without each effect)
- Composable (combine multiple effects freely)

### 2.1: Conformational Switching (Neural ODE Bifurcations)

**Biological Inspiration**: Proteins switch between conformational states via energy barriers.

**Effect**: `conformational_switching`
**Protocol**: `proto/effects/conformational.py`
**Implementation**: `core/conformational_ode.py`

### 2.2: Collective Memory (Modern Hopfield Networks)

**Biological Inspiration**: Neural ensembles reach consensus via attractor dynamics.

**Effect**: `collective_memory`
**Protocol**: `proto/effects/collective.py`
**Implementation**: `memory/collective_memory.py`

### 2.3: Mitotic Division (Asexual Reproduction)

**Biological Inspiration**: Successful cells divide with small variation.

**Effect**: `mitotic_division`
**Protocol**: `proto/effects/reproduction.py`
**Implementation**: `lifecycle/mitotic_division.py`

### 2.4: Meiotic Recombination (Sexual Reproduction)

**Biological Inspiration**: Crossover creates diversity under stress.

**Effect**: `meiotic_recombination`
**Protocol**: `proto/effects/reproduction.py`
**Implementation**: `lifecycle/meiotic_recombination.py`

### 2.5: Self-Modification (Hypernetwork + Learned Optimizer)

**Biological Inspiration**: Cells modify their own gene expression.

**Effect**: `self_modification`
**Protocol**: `proto/effects/meta_learning.py`
**Implementation**: `meta/hypernetwork.py`, `meta/learned_optimizer.py`

### 2.6: Adaptive Reproduction Strategy

**Integration**: Pool decides reproduction mode based on environmental conditions.

**Effect**: `adaptive_reproduction`
**Implementation**: `lifecycle/adaptive_strategy.py`

## Wave 2 Activation Strategy

**Gradual Enablement**:
1. Get Wave 1 baseline (current system working, ablations run)
2. Enable one effect at a time, measure delta
3. If improvement: keep enabled
4. If regression: disable and debug
5. Combine effects that show synergy

**Testing Requirements**:
- Each effect must have A/B test comparing enabled vs disabled
- Must measure: accuracy, throughput, memory, training stability
- Must document when each effect helps vs hurts

**Acceptance Criteria**:
- Effect improves at least one metric without regressing others >10%
- Effect composes cleanly with other effects (no interactions)
- Effect has zero overhead when disabled (compile-time optimization)

**File Structure for Wave 2**:
```
slime/
├── proto/
│   └── effects/          # Wave 2 effect protocols
│       ├── conformational.py
│       ├── collective.py
│       ├── reproduction.py
│       └── meta_learning.py
├── core/
│   └── conformational_ode.py
├── memory/
│   └── collective_memory.py
├── lifecycle/
│   ├── mitotic_division.py
│   ├── meiotic_recombination.py
│   └── adaptive_strategy.py
└── meta/
    ├── hypernetwork.py
    └── learned_optimizer.py
```

