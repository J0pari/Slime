# Slime Mold Transformer - System Blueprint

## Architecture Philosophy

**Foundation**: Conway → Lenia → Flow-Lenia → Neural Flow-Lenia evolution path. Our Slime Mold Transformer is a **Neural Flow-Lenia organism** where Pseudopods are learned CA update rules with spatially-localized parameters, mass-conservative dynamics, and intrinsic curiosity-driven lifecycle.

### Core Principles (2025 Substrate)

1. **Protocol-First**: All interfaces defined before implementations (algebraic effect handlers for optional capabilities)
2. **Dynamic Everything**: No static allocations, lifecycle-managed components (curiosity-driven birth/death via learning progress)
3. **Flow-Lenia Neural CA Substrate**: Pseudopod updates are learned continuous CA rules with mass conservation, parameter localization, warp-level GPU execution
   - Conway (Level 1): Fixed rules, discrete states → Lenia (Level 2): Fixed rules, continuous states → Flow-Lenia (Level 3): Localized parameters, mass conservation → **Ours (Level 4)**: Learned rules, adaptive parameters, intrinsic motivation
4. **Adaptive Voronoi MAP-Elites Core**: Archive-driven evolution with cells that grow/shrink based on density, DIRESA learned embeddings (adaptive 2-10D)
5. **Warp-Native GPU Kernels**: Like Polynesian navigator reading ocean/stars/birds as unified field, read warps/cache/tensor-cores as unified substrate
   - FlashAttention-style tiling (HBM ↔ SRAM), warp shuffles for zero-global-memory neighbor access, tensor cores for 256 FLOPs/cycle convolutions
6. **Content-Addressable Low-Rank Archive**: SVD factorization + content-addressed delta compression (80-160x memory reduction)
7. **Validated Behavioral Space**: KMO test ensures dimensions correlate with hardware structure (ultrametric topology via p-adic/genealogy/hierarchy)
8. **DIRESA Learned Embeddings**: Adaptive dimensionality via distance-preserving nonlinear autoencoders, learns online, dimension count adapts via warp vote (2-10D)
9. **Deterministic Random**: Hash-based seeded random for reproducibility
10. **SRE Built-In**: Observability, SLOs, error budgets from day one (100% constraint satisfaction always)
11. **GPU-Native Comonadic Perception**: GPU execution state AS comonad (extract local observation, extend with context-aware decisions)
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

**Dependencies**: Uses memory.archive for spatial indexing (Adaptive Voronoi cells), NO direct component management

### proto.model.Organism
**Purpose**: Top-level orchestrator with comonadic GPU perception

**Interface**:
- forward(stimulus, state) → (output, new_state) - Collective Pseudopod updates
- reset_state() → None - Reset organism state
- stats() → dict - GPU occupancy, learning progress, archive coverage

**Comonadic GPU Perception**:
- GPU execution state AS comonad (not external orchestration)
- extract(warp_id) → LocalObservation (warp occupancy, neighbor state, cache hits)
- extend(decision_fn) → Apply context-aware decisions (spawn/retire Pseudopods based on whole field)
- Whole computational field (warps/cache/tensor-cores) informs local decisions

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
- Attention alignment with targets
- Information bottleneck metrics (mutual information)
- **Relative fitness** (gradient magnitude z-score vs k-nearest neighbors)

NOT attention entropy alone (doesn't correlate with task)

### 9. CVT-MAP-Elites Architecture
**Reasoning (Scalability):** Fixed grid scales as resolution^dims. CVT scales linearly with num_centroids.

**Fixed grid problem:** Exponential explosion (3D: 8k cells, 4D: 160k, 5D: 3.2M)

**CVT solution:** Linear scaling (1000 centroids for any dimensionality)

**Behavioral dimensions:** DIRESA learns 2-10 nonlinear dimensions from 10-20 raw metrics online. KMO validation ensures factorability.

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

**Learned embeddings:** Nonlinear projections preserving pairwise distances better than PCA/t-SNE/UMAP

**Validation (2025 metrics):** Trustworthiness ≥ 0.85, Continuity ≥ 0.85, Procrustes distance ≤ 0.15, reconstruction error ≤ 0.5

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

### 13. Ablation tests determine architectural value
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

**Testing approach:** tests/ablations/ contains automated comparisons.

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

Hypothesis: Slime matches or exceeds task accuracy with 50-100x less total compute than NAS, and 1.3-1.5x better throughput than hypernetworks.

**If hypothesis fails:** Architecture is self-indulgent. Simplify or abandon.

## System Components

**Complete Architecture**: Algebraic effect handlers, Ultrametric topology, DIRESA learned embeddings (adaptive 2-10D), Adaptive Voronoi MAP-Elites, Neural CA Pseudopods (Flow-Lenia substrate), Curiosity-driven lifecycle (learning progress), Comonadic GPU orchestration

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
  - Preserves pairwise distances better than PCA/t-SNE/UMAP
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
