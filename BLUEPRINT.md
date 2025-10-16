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

---

## Part I: Abstract Architecture

### Dependency DAG

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

### Data Flow

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

### Protocols

#### proto.component.Component
**Purpose**: Unified interface for ALL pooled components

**Interface**:
- fitness: float (property) - Component quality metric
- reset() → None - Reset internal state
- to_dict() → dict - Immutable serialization for Archive storage
- from_dict(data: dict) → Component - Reconstruction from serialized form

**Usage**: Archive stores any Component via to_dict(), Pools manage any Component via fitness property

#### proto.memory.Memory
**Purpose**: Temporal memory interface with decay (NOT for component lifecycle - that's pool.py)

**Interface**:
- store(data: Tensor, weight: float) → None - Store with decay weight
- recall() → Optional[Tensor] - Retrieve with temporal blending
- clear() → None - Reset memory state

**Implementations**: memory.tubes.TubeNetwork (flowing memory with exponential decay)

#### proto.model.Pseudopod
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

#### proto.model.Chemotaxis
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

#### proto.model.Organism
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

### File Structure

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

### Invariants (Structural Rules)

**1. Dependency Direction** (DAG Enforcement)
- Lower layers NEVER import from higher layers
- Protocols NEVER import implementations
- Components NEVER import API layer
- **Violation = compilation error**

**2. Ownership Hierarchy**
- Organism owns: Pool[Pseudopod], Archive, Chemotaxis
- Pool owns: List[Component]
- Archive owns: Dict[cell, Elite] where Elite.genome = dict (NO object refs)
- **NO CYCLES**

**3. Protocol Compliance**
- Every component MUST declare which protocols it implements
- Archive stores via `Component.to_dict()` (immutable serialization)
- Pools manage via `Component.fitness` (no type checking at runtime)

**4. Observability Injection**
- Metrics collector passed to Organism.__init__()
- All forward passes record to metrics
- NO global state for metrics

**5. Determinism**
- Archive iteration uses sorted keys (deterministic order)
- All stochastic decisions via `_deterministic_random(seed, step, context)`
- Hash-based seeded random for reproducibility

**6. Ultrametric Topology**
- **Strong triangle inequality**: d(x, z) ≤ max(d(x, y), d(y, z)) for all x, y, z
- **Implementation**: True dendrogram traversal via linkage matrix merge height
- **Topology Types**: p-Adic (hierarchical codes), Genealogy (common ancestor), Hierarchy (dendrogram merge)
- **Chemotaxis Integration**: HybridMetric (ultrametric between clusters, Mahalanobis within), DIRESA preserves ultrametric in learned embeddings

---

## Part II: Behavioral & Lifecycle Architecture

### Archive & Quality-Diversity

**CVT-MAP-Elites with Content-Addressable Storage**

**Core Properties**:
- Adaptive Voronoi tessellation (centroids grow/shrink based on density)
- DIRESA learned behavioral embeddings (2-10D adaptive, distance-preserving)
- Low-rank storage: SVD factorization (8x compression) + delta compression (10-20x)
- Content-addressable: SHA256 deduplication prevents duplicate storage

Fixed grid approaches scale as resolution^dims, causing exponential explosion (3D: 8k cells, 4D: 160k, 5D: 3.2M). CVT scales linearly with num_centroids (1000 centroids for any dimensionality).

**Storage Strategy**:
1. SVD low-rank factorization: D×D → (D×k + k×D), 8x compression
2. Content-addressable hashing: Deduplicate identical elites
3. Delta compression: Store diffs vs parent in same centroid, 10-20x additional compression
4. **Result**: 80-160x memory reduction (D=512, k=64: 4MB → 25-50KB per elite)

Key insight: Elites in same centroid have similar behaviors → similar weights → tiny deltas.

**Delta Protocol Operations**:
- **sparse_add**: Add values at specified 2D indices (for sparse updates with >95% sparsity)
- **low_rank**: Low-rank update W += dU @ dV where dU is D×r, dV is r×D, r << k (for dense medium-sparsity updates)
- **dense**: Full replacement (for small tensors like biases)
- **scale_add**: Scalar multiplication plus sparse add (for small perturbations)

Application: apply_delta reconstructs weights by applying operations sequentially to base weights. Compression strategy chooses operation based on sparsity and size: Sparsity >95% → sparse_add. Small tensors → dense. Otherwise → low_rank SVD with rank r=8.

**Garbage Collection**: Reference counting + periodic mark-and-sweep every 100 add() calls
- No premature deletion (reference counting)
- No memory leaks (mark-and-sweep catches orphaned chains)
- Bounded overhead (amortized O(1))
- Deterministic (same sequence → same GC decisions)

**Bootstrapping Policy**:
- Archive provides INITIALIZATION only
- Bootstrapped components trained with rest of network (requires_grad=True)
- NO frozen parameters injected mid-training
- Prevents mode collapse and gradient conflicts

### DIRESA Behavioral Embeddings

**Purpose**: Learn distance-preserving nonlinear embeddings from raw behavioral metrics

**Architecture**: Autoencoder with learned gating for adaptive dimensions (2-10D), distance preservation loss, online training

**Raw Behavioral Metrics** (dimensionality discovered via covariance rank):
- CA pattern statistics: CA_mass_conservation, CA_neighborhood_coherence, CA_parameter_localization, CA_flow_divergence, CA_reaction_diffusion_balance
- Weight gradient norms: gradient_flow_magnitude, gradient_variance, gradient_spatial_locality
- Activation statistics: activation_sparsity, activation_magnitude, activation_entropy, activation_kurtosis
- Compute metrics: memory_access_locality, computational_intensity, cache_hit_rate, warp_divergence
- Weight statistics: weight_magnitude, weight_sparsity, weight_spectral_norm, weight_condition_number
- Correlation structure: effective_rank, mutual_information_with_loss, pairwise_correlation_entropy
- Hardware alignment: tensor_core_utilization, memory_bandwidth_efficiency, FLOP_efficiency

**Dimensionality Discovery**: Track covariance matrix rank via eigenvalue spectrum. If top-k eigenvalues explain >95% variance, raw dimensionality is k. DIRESA further compresses to 2-10D learned embeddings.

**Learned Embeddings**: Nonlinear projections preserving pairwise distances via autoencoder with distance loss

**Adaptive Dimensionality**:
- DIRESA encoder has gating layer that learns which dimensions to activate
- Dimension count adapts via warp vote mechanism (2-10D range)
- System learns optimal dimensionality based on task, not predetermined

**Validation Thresholds**:
- Trustworthiness ≥ 0.85 (neighbors preserved in low-D)
- Continuity ≥ 0.85 (neighborhoods preserved)
- Procrustes distance ≤ 0.15 (shape similarity)
- Reconstruction error ≤ 0.5 (information preservation)

**Training Schedule**:
- Delayed activation: Use Euclidean distance for first 2000 steps
- Accumulate ≥1000 behavioral samples before training
- Train DIRESA for 500 steps, then validate
- Fallback to PCA if validation fails after 3 retries

### Lifecycle & Curiosity-Driven Selection

**Fitness Metric**: `fitness = effective_rank() × coherence()`
- effective_rank(): Parameter localization (spatial variation of CA rule)
- coherence(): Learning progress (how fast component is improving)

**Curiosity-Driven Dynamics**:
- High fitness → survive and get archived
- Low fitness → replaced by sampling from archive history
- Learning progress drives which trajectories persist (intrinsic motivation)

**Timescale Separation**:

**Fast (every step)**:
- Weight updates via backprop
- Fitness tracking (EMA)
- Metrics collection
- Loss monitoring

**Medium (every 100 steps)**:
- Fitness assessment
- Archive elite updates
- Pool spawn decisions
- Loss gate check

**Slow (every 1000 steps)**:
- Pool culling
- Memory budget enforcement
- Behavioral space analysis
- Hard limit enforcement

**Loss Gates (Training Stability)**:
- Freeze lifecycle when loss > threshold × loss_ema
- Adaptive threshold: `max(10.0 × loss_ema, 2.0 × std(losses[-1000:]))`
- Cooldown period: 500 steps after gate triggers
- Timeout: Force re-enable after 5000 frozen steps

**Warmup & Gentle Phases**:
- Warmup (0-100 steps): No lifecycle, let trajectories stabilize
- Gentle (100-500 steps): Reduced culling frequency, gradient clipping
- Normal (500+ steps): Full lifecycle operation

**GPU-Parallel Spatial Stencil**: GPU computation is spatial (SIMD, tiles, stencil convolution), not sequential. JAX vmap-inspired batched computation of contextual metrics (pairwise distances, k-nearest neighbors). BAD (sequential): Loop over pool computing per-component z-scores → O(N) sequential. GOOD (parallel): `vmap_relative_fitness(fitnesses, neighbor_mask)` → O(1) GPU call. Stencil kernel applied to every component position (SIMD), matches GPU architecture perfectly. 100x-1000x speedup vs sequential.

**Cellular Lattice Dynamics**: The CA operates on a discrete spatial lattice where each cell undergoes local update rules. Mass conservation couples neighboring cells. Each forward pass applies the update rule across all lattice positions simultaneously (SIMD), computing one timestep of the discrete dynamics. Training gradient descent modifies the update rule parameters, changing which computational trajectories are accessible from given initial conditions.

---

## Part III: Implementation Decisions

### GPU Kernel Architecture

**IO-Aware Tiled Attention** (FlashAttention pattern)
- Tile computation to fit in SRAM (HBM ↔ SRAM tiling)
- Naive attention: O(M² × D) HBM accesses (load Q, K, V repeatedly for each output element)
- Tiled attention: O(M² × D / SRAM_size) HBM accesses (load tiles once, compute in SRAM)
- Implementation: BLOCK_M=128, BLOCK_N=128, BLOCK_D=64 (adaptive to GPU SRAM)

**Kernel Injection**: Constructor injection pattern
- Pseudopod accepts Kernel interface at construction
- Allows Triton GPU, PyTorch CPU fallback, custom implementations
- Scale with available hardware, not hardcoded assumptions
- Bitter Lesson: Let user provide compute capability, don't hardcode our assumptions about what hardware is available

**Multi-GPU Hash Partitioning**:
- Device assignment: `device_id = hash(behavior_coords) % num_gpus`
- Scales to arbitrary GPU counts without manual partitioning
- Hash function ensures load balancing

**Graceful Degradation**:
- Forward uses Triton (speed), backward uses PyTorch einsum (gradient flow)
- Fallback to PyTorch if Triton kernel fails
- Adaptive tile sizes based on SRAM availability

### Fitness & Selection

**Fitness Components**:
- Task performance (70%): Gradient magnitude (components affecting loss)
- Compute efficiency (20%): FLOPs, memory bandwidth, tensor core utilization
- CA conservation quality (10%): Mass conservation correlation with targets

**NOT activation entropy alone** (doesn't correlate with task performance)

**Why this formula**: Task accuracy alone won't discover hardware-optimal patterns. Fitness must reward efficiency so fast components survive and slow components get culled. CA mass conservation ensures substrate stability.

**Relative Fitness**: Gradient magnitude z-score vs k-nearest neighbors (contextual fitness)
- Compute absolute fitness: gradient_magnitude × CA_mass_conservation
- Find k=5 nearest neighbors in behavioral space
- Compute z-score: (fitness - mean(neighbor_fitness)) / std(neighbor_fitness)
- Contextual: same absolute fitness scores differently in crowded vs sparse regions

**Simulated Annealing**: Temperature schedule for exploration→exploitation
- Birth decisions: Accept diverse vs high-fitness components
- CVT centroid refinement: Minimize quantization error
- Archive mutation strength: Large mutations (early) → small mutations (late)

**Quality-Diversity Benefits**:
- Graceful degradation under device loss: Hash-based redistribution to remaining GPUs, no retraining required
- Interpretability via behavioral coordinates: Query archive by specific behavioral properties, understand what strategies exist
- No mode collapse: Each archive cell requires behaviorally-distinct component (forced diversity vs standard transformers where all heads learn similar features with head_similarity > 0.9)

### Memory & Resource Management

**Soft Limits with Graceful Degradation**:
- When memory exceeds budget, pool culls worst performers (fraction=0.2)
- System degrades quality gracefully rather than crashing on OOM
- Adaptive max_pseudopods based on GPU memory availability
- Bitter Lesson: Adapt to constraints, don't crash. Trade quality for capacity automatically.

**GPU Memory Safety**:
- Kernels check allocation before launch
- Organism enforces memory budget
- Pool culling triggered by OOM
- Reserve 10% headroom for fragmentation

**Why this approach**: Hard memory limits cause abrupt OOM crashes. Soft limits with graceful degradation maintain system stability by trading diversity (pool size) for continued operation. System automatically finds the right quality/capacity tradeoff for available hardware.

---

## Part IV: Testing & Validation

### Ablation Testing Methodology

**Approach**: Tests make specific predictions that could be disproven. Design tests where failure is possible and informative.

**Pattern**:
1. State measurable predictions with numeric thresholds
2. Design counter-examples where prediction should fail
3. Probe live system state (Archive coverage, Pseudopod coherence, Voronoi densities)
4. Isolate component-level causation

**Avoid**:
- Black-box model comparison
- Recomputing metrics system already tracks
- Descriptive statistics without falsification criteria
- Static end-of-training snapshots

**Key Test Cases**:

**Curiosity-driven lifecycle maintains population diversity**
- Prediction: Pseudopod coherence std > 0.1 throughout training
- Counter-test: Static pool should collapse to std < 0.05
- Probe: `[pod.coherence().item() for pod in organism.pseudopod_pool._components]` every 100 steps
- If static pool maintains diversity, remove lifecycle

**DIRESA discovers intrinsic dimensionality**
- Prediction: Archive.behavioral_dims < 0.5 * Archive.num_raw_metrics
- Counter-test: Pure Gaussian noise should fail to compress
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

**Implementation**:
- Probe live Organism state: `organism.archive`, `organism.pseudopod_pool._components`, CA metrics
- Track trajectories: coherence over time, coverage evolution, density histograms
- Mechanistic interventions: disable components, inject controlled noise
- Numeric thresholds for pass/fail

**Criteria**:
- Failed prediction means component is broken or unnecessary
- Survived counter-test means provisionally supported
- No measurable improvement means remove the component
- Test without failure mode is invalid

---

## Part V: Operational Guidance

### Premortem Analysis

**Purpose**: Identify failure modes before they occur.

**1. Training Instability from Archive Updates**
- Failure: Archive updates cause gradient variance spikes → loss divergence
- Why: New pseudopods bootstrapped from archive have different loss landscape positions. Sudden parameter distribution shifts confuse optimizer momentum. Lifecycle events (birth/death) create discontinuities in gradient flow.
- Detection: `np.std(losses[-100:]) > 3.0 × np.std(losses[-1000:-100])`, gradient norm spikes > 10.0 × EMA, high birth rate within 50 steps before loss spike
- Mitigation: Loss gates (freeze lifecycle), warmup period, gradient clipping

**2. DIRESA Embeddings Don't Converge**
- Failure: Behavioral embeddings fail to preserve distances → diversity loss
- Why: Insufficient training data (needs ~1000 samples). Raw metrics fluctuate wildly during early training. Intrinsic dimensionality > learned dimensions (e.g., 8D manifold compressed to 3D). Training modifies behavioral distribution faster than DIRESA adapts.
- Detection: Trustworthiness < 0.70, Continuity < 0.70, reconstruction error > 0.8, dimensions stuck at min/max for > 5000 steps
- Mitigation: Delayed activation (2000 steps), EMA-smoothed metrics, adaptive dimension bounds (2-10D), fallback to PCA

**3. Pool Collapse to Single Strategy**
- Failure: All pseudopods converge to identical behavior → coverage stalls
- Why: Fitness pressure dominates diversity pressure (culls too aggressively). Archive sampling bias (high-fitness centroids sampled repeatedly). Gradient alignment (all pseudopods receive similar gradients). Behavioral metrics too coarse.
- Detection: Coherence std < 0.05, archive coverage plateaus for 5000 steps, behavioral correlation > 0.9, pool size < 0.5 × max_size
- Mitigation: Diversity bonus in fitness (30%), ε-greedy sampling (ε=0.2), birth threshold jitter (±0.1), forced exploration every 1000 steps

**4. Memory Budget Violation from Archive Growth**
- Failure: Archive grows unbounded → OOM crash
- Why: Every elite addition creates storage (low-rank + deltas). Delta chains grow without GC. Content-addressable hash table doesn't shrink (fragmentation). DIRESA autoencoder grows with dimension increases.
- Detection: Memory growth > 10% per 1000 steps, archive size > 2 × num_centroids, delta chains > 50, torch.cuda.OutOfMemoryError
- Mitigation: Hard limits (1000 centroids, 10 elites/centroid), aggressive GC every 100 additions, delta chain pruning (collapse > 20), remove oldest elites when tight

**5. GPU Kernel Launch Failures**
- Failure: Triton launches fail → fallback to slow PyTorch → throughput collapse
- Why: Too many pseudopods active (register pressure). Tile sizes too large for SRAM. Tensor shapes misaligned with warp size (32). CUDA context switching overhead from multi-GPU hash partitioning.
- Detection: Fallback rate > 10%, occupancy < 50%, launch latency spikes > 2.0 × EMA, memory allocation retries
- Mitigation: Adaptive max_pseudopods (scale with GPU memory), tile size autotuning (start 128, reduce to 64), batched kernel launches, hash partition validation

**6. Loss Gates Over-Trigger**
- Failure: Lifecycle frozen too long → stale pool → performance plateau
- Why: Loss EMA miscalibrated (too low threshold). Noisy tasks (RL) have high loss variance. Lifecycle freeze prevents adaptation → loss stays high (vicious cycle).
- Detection: Frozen fraction > 0.5, no births/deaths for > 2000 steps, loss EMA < 0.1 × mean(losses)
- Mitigation: Adaptive threshold max(10.0 × EMA, 2.0 × std(losses)), cooldown period 500 steps, timeout 5000 steps, task-specific threshold multiplier

### Hyperparameter Tuning Protocol

**Loss gate threshold**:
- Measurement: Compute loss std over last 1000 training steps
- Procedure: Try threshold values 5.0, 10.0, 20.0. Track gate trigger frequency.
- Hypothesis: If gate triggers >30% of steps, threshold too low. If <1%, might miss real instability.
- Test: Does changing threshold affect final accuracy? If not, threshold doesn't matter.

**Culling fraction**:
- Measurement: Track coherence std before and after culling
- Procedure: Try fractions 0.1, 0.2, 0.3. Measure coherence std after 1000 steps.
- Hypothesis: Coherence std drop >50% suggests over-culling.
- Test: Does lower culling improve final coverage? Does higher culling improve final accuracy?

**Warmup steps**:
- Measurement: Plot gradient norm over first 1000 steps
- Procedure: Visual inspection for when variance stabilizes
- Hypothesis: Lifecycle before stabilization causes training instability
- Test: Does reducing warmup cause loss spikes? Does increasing warmup delay convergence?

**Max pseudopods**:
- Measurement: GPU memory % and kernel occupancy during forward pass
- Procedure: Increase until memory >90% or occupancy <50%
- Hypothesis: Memory limit or compute limit, whichever hits first
- Test: Does increasing beyond limit cause OOM or slower throughput?

**Archive centroids**:
- Measurement: Cell density variance (max_density / mean_density)
- Procedure: Try 10^d, 10^(d+1), 10^(d-1) where d=DIRESA dims
- Hypothesis: High variance means unbalanced cells. High empty % means too many centroids.
- Test: Does centroid count affect final coverage? Does it affect memory usage?

### Reproducibility Tests

**Test: Same seed, same GPU**
- Run: `python train.py --seed=42` twice
- Check: `np.corrcoef(losses_run1, losses_run2)[0,1]`
- Hypothesis: Deterministic RNG + sorted iteration → identical results
- If different: Search for `random()` without seed, unsorted dict iteration, CUDA atomics

**Test: Same seed, different GPU**
- Run: `python train.py --seed=42` on two GPU models
- Check: `scipy.stats.kendalltau(fitness_gpu1, fitness_gpu2)`
- Hypothesis: FP rounding differs but rank order preserved
- If rank order differs: Logic branches on hardware properties

**Test: Different seeds**
- Run: `for seed in range(10): python train.py --seed=$seed`
- Check: `np.std([final_acc_seed0, final_acc_seed1, ...])`
- Hypothesis: Low std means system not sensitive to initialization
- If high std: System explores different archive regions, check if some seeds consistently fail

### Resource Budget

**GPU Memory (per-device)**:
- Model weights (40%): Pseudopod parameters
- Activations (30%): Forward pass tensors, gradients
- Archive storage (15%): Low-rank weights, delta chains
- DIRESA embeddings (5%): Autoencoder, behavioral buffers
- Optimizer state (10%): Adam momentum/variance

**Example (RTX 3090: 24GB)**:
- Model weights: 9.6 GB → ~80M params fp32 (20M fp16)
- Activations: 7.2 GB → batch_size=32, max_pseudopods=16
- Archive: 3.6 GB → 1000 centroids × 10 elites × 360KB
- DIRESA: 1.2 GB → adaptive dims (2-10D learned), batch=1024
- Optimizer: 2.4 GB → Adam state

**Adaptive Limits**:
- `max_pseudopods = max(4, min(16, int(gpu_memory_gb * 0.3 / memory_per_pod_gb)))`
- `max_archive_centroids = min(1000, int(gpu_memory_gb * 0.15 / memory_per_centroid_gb))`

**Compute Budget (per-step, target 100ms)**:
- Forward pass (50ms): Per-pseudopod 3ms Triton, 12ms PyTorch
- Backward pass (30ms): Gradients, optimizer
- Lifecycle (5ms amortized): Birth/death every 100 steps
- Archive ops (10ms amortized): Elite addition, sampling
- Metrics (5ms): Behavioral metrics, observability

**Training Budget** (untested hypotheses):
- MNIST (TINY): 10 minutes (5000 steps × 100ms, 1 GPU)
- CIFAR-10 (SMALL): 4 hours (80k steps × 150ms, 1 GPU)
- ImageNet (MEDIUM): 3 days (1M steps × 250ms, 4 GPUs)

**Network Budget (multi-GPU, untested)**:
- Target 2 GPUs: 1.8× throughput (90% efficiency)
- Target 4 GPUs: 3.2× throughput (80% efficiency)
- Target 8 GPUs: 5.6× throughput (70% efficiency)

### Decision Tree

**Purpose**: Guide implementation decisions with clear criteria. Each node asks a question, branches on measurable conditions, and leads to concrete actions.

**When to Add Component to Pool?**
```
Q: pool < min_size? → YES: Spawn immediately
Q: step < 100? → YES: Don't spawn (warmup)
Q: step < 500? → YES: Spawn only if step % 50 == 0 (gentle)
Q: component with fitness < 0.1?
  → YES: Q: coverage < 0.5?
      → YES: Random centroid (exploration)
      → NO: Fitness-weighted (exploitation)
  → NO: Q: pool < max_size AND avg_fitness > 0.8?
      → YES: Spawn from archive
      → NO: Don't spawn
```

**When to Cull from Pool?**
```
Q: pool > max_size? → YES: Cull 30% immediately
Q: step % 1000 == 0?
  → YES: Q: loss > 10.0 × loss_ema? (loss gate)
      → YES: Skip culling (unstable)
      → NO: Q: pool > 1.5 × min_size?
          → YES: Cull bottom 20%
          → NO: Don't cull
```

**When to Update Archive?**
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

**When to Activate DIRESA?**
```
Q: step < 2000? → YES: Use Euclidean (accumulate samples)
Q: samples ≥ 1000? → NO: Continue Euclidean
Q: Train 500 steps → Trustworthiness ≥ 0.70 AND Continuity ≥ 0.70?
  → YES: Activate DIRESA
  → NO: Retry < 3 times? → YES: Retrain 2× lr
                         → NO: Fallback to PCA
```

**When to Trigger Loss Gate?**
```
Q: loss > 10.0 × loss_ema?
  → YES: Q: gate active < 5000 steps?
      → YES: Freeze lifecycle
      → NO: Q: loss decreasing?
          → YES: Continue freeze
          → NO: Force unfreeze, recalibrate
  → NO: Q: gate currently active?
      → YES: Q: loss stable > 500 steps?
          → YES: Unfreeze
          → NO: Continue freeze
```

**Triton Tile Size Selection?**
```
Q: SRAM ≥ 128 KB? → Try BLOCK=128
Q: SRAM ≥ 64 KB? → Try BLOCK=64
Q: SRAM < 64 KB? → Try BLOCK=32

Q: Launch succeeded?
  → YES: Q: Occupancy > 50%? → YES: Keep
                             → NO: Reduce 2×
  → NO: Q: Out of SRAM? → Reduce 2×
        Q: Out of registers? → Reduce BLOCK_D
        → Otherwise: Fallback PyTorch
```

**Archive Sampling vs Random Init?**
```
Q: coverage > 0.1? → NO: Random init (sparse archive)
Q: ε-greedy exploration (ε=0.2)? → YES: Random centroid
Q: Elite at this centroid? → YES: Sample from centroid
Q: Elite in k=5 nearest? → YES: Sample from nearest
→ Otherwise: Random init
```

**Adjust max_pseudopods?**
```
Q: GPU memory > 90%? → Decrease 25%
Q: GPU memory < 60%?
  → Q: Occupancy > 70%? → Increase 25%
                        → Keep (occupancy bottleneck)
Q: Kernel launch failures? → Decrease 50%
```

**DIRESA Dimension Count?**
```
Q: reconstruction_error > 0.8? → Increase dims +1
Q: reconstruction_error < 0.3? → Q: dims > 2? → Decrease -1
                                              → Keep at 2D
Q: dims stable > 2000 steps? → Keep
Q: Trustworthiness < 0.85? → Increase dims
```

---

## Part VI: Related Work & Comparisons

### Computational Cost Analysis

**Hypothesis (requires empirical validation)**: For long training runs (100+ GPU days), amortized search cost favors Slime. For short runs (<30 days), DARTS more efficient.

**Untested Hypothesis**: Slime may match or exceed task accuracy with significantly less compute than NAS methods, while maintaining competitive throughput with hypernetworks.

**Empirical validation required**: If comparative testing shows no advantage over baselines, simplify or abandon architecture.

### Comparison to DARTS (Modern NAS)

**DARTS baseline (Liu et al., 2018)**:
- Differentiable architecture search with continuous relaxation
- Weight sharing across candidate operations
- Search cost: 1-4 GPU days on CIFAR-10/ImageNet
- 1000x faster than early NAS methods (NASNet: 2000 GPU days)

**Key difference**: DARTS searches discrete architecture space. Slime continuously adapts component population.

### Comparison to Hypernetworks (Ha et al., 2016)

**Hypernetwork approach (Ha et al., ICLR 2017)**:
- Small network generates weights for larger network
- Parameter efficiency: fewer learnable params than standard networks
- Memory-efficient: O(Nz × hidden_units) not O(Nz × all_params)

**Key insight**: Low-rank weight generation can be MORE efficient than storing full matrices.

**Slime borrows this for archive storage**: Low-rank SVD + delta compression

**Complementary approaches**: Hypernetworks excel at few-shot adaptation. Slime excels at maintaining diverse specialists for single-task training.

### Simulated Annealing Integration

**Insight**: Quality-diversity needs exploration-exploitation balance. Simulated annealing provides principled temperature schedule.

**Applications**:
- Birth decisions: Temperature schedule for accepting diverse vs high-fitness components
- CVT centroid refinement: Annealing to minimize quantization error
- Archive mutation strength: Large mutations (early) → small mutations (late)

### Slime Uniqueness

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

---

## References

- Illuminating search spaces by mapping elites. Mouret & Clune (2015) arXiv:1504.04909
- Using Centroidal Voronoi Tessellations to Scale Up the Multidimensional Archive of Phenotypic Elites Algorithm. Vassiliades et al. (2018) IEEE Trans. Evolutionary Computation 22(4):623-630
- Quality Diversity: A New Frontier for Evolutionary Computation. Pugh et al. (2016) Frontiers in Robotics and AI 3:40
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. Dao et al. (2022) arXiv:2205.14135
- DARTS: Differentiable Architecture Search. Liu et al. (2019) arXiv:1806.09055
- HyperNetworks. Ha et al. (2017) arXiv:1609.09106
- Flow-Lenia: Towards open-ended evolution in cellular automata through mass conservation and parameter localization. Randazzo et al. (2023) arXiv:2212.07906
- A Path to Universal Neural Cellular Automata. Béna et al. (2025) arXiv:2505.13058
- DIRESA: Distance-preserving nonlinear dimension reduction via regularized autoencoders. De Paepe & De Cruz (2025) arXiv:2404.18314
- Humans monitor learning progress in curiosity-driven exploration. Gottlieb & Oudeyer (2021) Nature Communications 12:5972
- Information retrieval perspective to nonlinear dimensionality reduction for data visualization. Venna et al. (2010) JMLR 11:451-490
- Procrustes Problems. Gower & Dijksterhuis (2004) Oxford University Press
- Optimization by simulated annealing. Kirkpatrick et al. (1983) Science 220(4598):671-680

---

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
- Zero overhead when disabled (compile-time optimization)
- Gradual migration (enable per-component)
- A/B testing (compare with/without each effect)
- Composable (combine multiple effects freely)

### Wave 2 Features

**2.1: Conformational Switching** (Neural ODE Bifurcations)
- Biological: Proteins switch between conformational states via energy barriers
- Effect: `conformational_switching`

**2.2: Collective Memory** (Modern Hopfield Networks)
- Biological: Neural ensembles reach consensus via attractor dynamics
- Effect: `collective_memory`

**2.3: Mitotic Division** (Asexual Reproduction)
- Biological: Successful cells divide with small variation
- Effect: `mitotic_division`

**2.4: Meiotic Recombination** (Sexual Reproduction)
- Biological: Crossover creates diversity under stress
- Effect: `meiotic_recombination`

**2.5: Self-Modification** (Hypernetwork + Learned Optimizer)
- Biological: Cells modify their own gene expression
- Effect: `self_modification`

**2.6: Adaptive Reproduction Strategy**
- Integration: Pool decides reproduction mode based on environment
- Effect: `adaptive_reproduction`

### Wave 2 Activation Strategy

**Gradual Enablement**:
1. Get Wave 1 baseline (current system working)
2. Enable one effect at a time, measure delta
3. If improvement: keep enabled
4. If regression: disable and debug
5. Combine effects that show synergy

**Testing Requirements**:
- Each effect must have A/B test (enabled vs disabled)
- Measure: accuracy, throughput, memory, training stability
- Document when each effect helps vs hurts

**Acceptance Criteria**:
- Effect improves ≥1 metric without regressing others >10%
- Effect composes cleanly with other effects (no interactions)
- Effect has zero overhead when disabled

**File Structure for Wave 2**:
```
slime/
├── proto/effects/          # Wave 2 effect protocols
│   ├── conformational.py
│   ├── collective.py
│   ├── reproduction.py
│   └── meta_learning.py
├── core/conformational_ode.py
├── memory/collective_memory.py
├── lifecycle/
│   ├── mitotic_division.py
│   ├── meiotic_recombination.py
│   └── adaptive_strategy.py
└── meta/
    ├── hypernetwork.py
    └── learned_optimizer.py
```
