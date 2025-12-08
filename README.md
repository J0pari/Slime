# Slime: GPU-Native Evolutionary Neural CA

**Core idea**: Evolutionary system combining gradient-based learning with quality-diversity archiving. Multi-head neural cellular automata (8 heads, 512-channel state) evolve via fitness selection while learning classification tasks through autodiff tape backpropagation. MAP-Elites archive maintains diverse elite strategies in 3-axis behavioral space (hardware efficiency, task accuracy, generalization gap). DIRESA autoencoders compress genomes (1024→128D, 8x) enabling latent-space mutations and archive storage. All computation device-side via CUDA Dynamic Parallelism.

**Intended behavior**: Archive accumulates elites across behavioral niches - organisms specializing in different trade-offs (high-accuracy/low-efficiency vs balanced vs exploration-focused). Voronoi tessellation partitions concatenated 3D space (hw_coords + task_coords + gen_coords) with adaptive cells tracking density fluctuations. Power-law fitness (task^α × gen^β × rank^γ × efficiency^δ) drives task-grounded evolution where α,β,γ,δ exponents genome-evolved, enabling species that evolve their own selection criteria. Hybrid lifecycle: gradient learning every CHECKPOINT_INTERVAL=20 generations, evolutionary selection every generation.

**Implementation**: Multi-head neural CAs with FP16 WMMA tensor core matrix operations (perception/interaction/value layers). MAP-Elites archive with adaptive Voronoi cells tessellating concatenated 3-axis behavioral space: hw_coords (hardware efficiency features → DIRESA embedding) + task_coords (classification performance) + gen_coords (train/test generalization). Power-law fitness: task^α × (1-gen_gap)^β × rank^γ × efficiency^δ where α,β,γ,δ exponents genome-derived via genome_to_param with contextual modulation. Genomes (1024 floats) compressed to 128D DIRESA latent space (8x ratio) stored in archive, enabling direct latent-space mutations without decompress/recompress cycles. Dynamic parallelism (CDP) with hybrid synchronization: warp-level CooperativeSync for DIRESA encode/decode ops, device-level cudaDeviceSynchronize for recursive kernel launches.

## Requirements

- NVIDIA GPU: sm_86+ (Ampere/Hopper) with Tensor Cores, 6GB+ VRAM
- CUDA Toolkit: 12.0+
- Windows: MSVC 2022 Build Tools (nvcc host compiler)

## Build

```bash
cd build
compile.bat  # Windows
# or
make  # Linux
```

## Quick Start

```bash
cd build
slime.exe  # Run evolution (Windows)
```

Output: Console logs every 5 generations with pool/archive statistics, telemetry every 100 generations.

## Scripts

```bash
scripts/commit.sh              # Commit with web-flow author, push to origin
scripts/dev.py verify-kernels  # Check compiled PTX for expected kernels
scripts/audit_connectivity.py  # Verify kernel launch connectivity
```

## Architecture

### Neural Cellular Automaton with Flow-Lenia Morphology

Each CA operates on a 64×64×512 grid with 8 parallel heads. Multi-head output produces spatially-varying affinity map U^t. Reintegration tracking conserves mass via geometric overlap integrals:

```
Concentration A^t [64×64×512]
  ↓
Perception: 512×256 matrix (FP16 WMMA)
  ↓
Interaction: 512×256 matrix (FP16 WMMA)
  ↓
Value: 256×512 matrix (FP16 WMMA)
  ↓
Multi-head output [8 heads × 64×64×16]
  ↓
Warp-reduce affinity: U^t = Σ_{heads,dims} CA_output  (WarpReduce<32>::sum across heads per spatial cell)
  ↓
Flow computation: F^t = (1-α^t)∇U^t - α^t∇A_Σ^t  where α^t = [(A_Σ^t/β_A)^n]_0^1
                  Gradients via Sobel (Stencils::gradients_at)
  ↓
Reintegration tracking: A^(t+dt)(x) = Σ_{x'} A^t(x') · I(x'→x)
                        where I(x'→x) is overlap integral at displaced position x' + dt·F^t(x')
```

**Mass conservation**: Σ(A^(t+dt)) = Σ(A^t) emerges from reintegration geometry - each cell's mass redistributes to overlapping neighbors, overlap integrals sum to 1 per source. Gaussian overlap integral I(x'→x) computed via FlowLeniaOps::gaussian_overlap_integral with spatially-varying spread s (genome-derived). Atomic adds (atomicAdd) prevent race conditions during concurrent reintegration writes (reintegration_redistribute_kernel).

**Parameter localization**: Multi-head CA weights define spatially-varying affinity landscape U^t. Each head's perception/interaction/value weights create different local preferences. Flow field F^t weights affinity gradient ∇U^t vs diffusion -∇A_Σ^t based on local mass concentration via α^t = [(A_Σ^t/β_A)^n]_0^1 where β_A (mass threshold) and n (flow steepness) genome-derived via genome_to_param.

**Tensor core mapping** (pseudopod_tensor.cu): Perception (512×256), interaction (512×256), value (256×512) matrices use nvcuda::wmma::load_matrix_sync/mma_sync/store_matrix_sync with wmma::fragment<matrix_a/matrix_b/accumulator, 16, 16, 16, half/float>. 16×16×16 tile size aligns to Ampere/Hopper WMMA fragment shape. Accumulation in FP32, weights in FP16 for 2× memory bandwidth vs FP32.

**Throughput**: Tensor cores: ~13 TFLOPS FP16 on RTX 3060 (sm_86, 28 SMs) vs ~0.4 TFLOPS scalar FP32 (30× speedup). CA forward pass (8 heads, 64×64 grid, 512 channels): ~2ms with tensor cores, ~60ms with scalar ops.

### Fitness = task^α × (1-gen_gap)^β × rank^γ × efficiency^δ

**Power-law composition** via compute_fitness_from_diresa_kernel (organism.cu:661-696):
- task^α: Classification accuracy, exponent α ∈ [0.3, 2.0] from FITNESS_TASK_EXPONENT_MIN/MAX
- (1-gen_gap)^β: Train/test consistency (1 - |train_acc - test_acc|), β ∈ [0.3, 2.0] from FITNESS_GEN_EXPONENT_MIN/MAX
- rank^γ: DIRESA latent variance → effective_rank = sqrt(variance) × latent_dim (manifold spread in 128D space), γ ∈ [0.3, 2.0] from FITNESS_RANK_EXPONENT_MIN/MAX
- efficiency^δ: Weighted sum of hardware_features (IPC, bandwidth, occupancy, cache hits), δ ∈ [0.1, 1.5] from FITNESS_EFFICIENCY_EXPONENT_MIN/MAX

**Exponent derivation**: Each α,β,γ,δ computed via genome_to_param with 7-dimensional context: (metabolic, stress, morphogen, hash_entropy, novelty_gradient, behavioral_drift, task_accuracy). Enables adaptive fitness landscapes - organisms evolve their own selection criteria.

**Computation schedule** (organism.cu:507, triggered in component_evolution_kernel): Fitness computed every generation when pool->entries[0] updated. Exponents re-derived from current genome state each evaluation, allowing fitness function itself to evolve.

### MAP-Elites Archive (archive.cu, organism.cu:759-820)

**3-axis behavioral space** (GPUElite struct fields):
- hw_coords: HARDWARE_FEATURES_DIM=15 (warp occupancy, memory throughput, IPC, cache hits, divergence) → DIRESA embedding → genome-derived hw_dim
- task_coords: per-class classification accuracies → genome-derived task_dim
- gen_coords: train/test generalization metrics → genome-derived gen_dim

**Storage**: latent_genome (128D DIRESA compressed) + fitness/coherence/effective_rank scalars + parent_ids genealogy + hardware_features raw + task_performance + generation timestamp. Archive size MAX_ARCHIVE_SIZE with wraparound indexing.

**Voronoi tessellation** (VoronoiCell struct): Cells track hw_centroid + task_centroid + gen_centroid, radius (power-law from density fluctuation), density/density_prev for gradient, best_elite_idx, quality_threshold. Distance via compute_three_axis_distance_sq concatenating all 3 subspaces.

**Genome compression**: Delta encoding (delta.cu) stores only changed indices/values from parent. DIRESA latent (128D) stored in archive enables mutations without full decompression. Combined: sparse deltas reference DIRESA latent parents.

### Dynamic Parallelism (CUDA Dynamic Parallelism / CDP)

**Zero CPU synchronization**: Parent kernels launch child kernels device-side via `<<<...>>>` syntax within __device__ code. All evolution computation GPU-resident - CPU only triggers top-level kernel, polls completion.

**Launch hierarchy** (organism.cu):
- component_evolution_kernel (top-level) spawns:
  - compute_fitness_from_diresa_kernel (fitness evaluation)
  - selection_kernel (MAP-Elites behavioral selection)
  - archive_update_kernel (Voronoi cell update)
  - behavioral_update_kernel (3-axis coords update)
  - lifecycle progression kernels (hunger, coherence, gradient fitness)
- hybrid_lifecycle_kernel spawns:
  - autodiff tape allocation/deallocation
  - gradient descent steps
  - DIRESA compression/decompression

**Synchronization**: cudaDeviceSynchronize() enforces global dependencies (e.g., fitness before selection). CooperativeSync::sync_warp() for warp-level deps (DIRESA warp reductions). No CPU round-trips during generation.

**Compilation**: `-rdc=true` (relocatable device code), `-lcudadevrt` (device runtime library), `-arch=sm_86` (Ampere minimum for CDP stack depth).

### Automatic Differentiation (Tape-based Reverse-mode AD)

**Tape recording** (autodiff_tape.cu): Forward pass records operations (matrix multiplies, activations) with operand indices, shapes, operation type. Each thread maintains tape_index counter incremented after recording. Overflow checks before every write - tape full triggers reallocation or gradient checkpoint.

**Backward pass**: Reverse-order tape traversal computes ∂loss/∂weights via chain rule. Matrix multiply gradients: ∂loss/∂A = ∂loss/∂C · B^T, ∂loss/∂B = A^T · ∂loss/∂C. Activation gradients: GELU derivative, ReLU mask. Accumulated in FP32 gradient buffers per weight matrix (perception_grads, interaction_grads, value_grads).

**CA integration** (autodiff_integration.cu): Perception/interaction/value matrix operations in multi_head_ca_kernel record to tape during forward pass. Tape indices threaded through kernel parameters. Gradient descent (hybrid_lifecycle.cu) applies accumulated gradients every CHECKPOINT_INTERVAL=20 generations with learning rate from genome_to_param (LR_MIN=1e-5, LR_MAX=1e-2 range).

**Memory management**: Tape preallocated to TAPE_MAX_SIZE ops. Checkpointing: recompute forward from last checkpoint if tape overflows, trading compute for memory. Device-side tape allocation/free via cudaMalloc/cudaFree in CDP kernels.

## Training (Hybrid Gradient-Evolutionary Metalearning)

**Dual timescales**: Gradient learning every CHECKPOINT_INTERVAL=20 generations (hybrid_lifecycle.cu), evolutionary selection every generation (selection_kernel).

**Gradient fitness** (gradient_fitness.cu): Organisms evaluated on learning speed - how fast CA weights improve classification accuracy via autodiff backprop. CAs with higher gradient magnitudes (∂loss/∂weights norm) receive fitness bonus, selecting for "learnability". Coherence bonus (loss improvement consistency across steps) rewards exploration vs exploitation balance.

**Lifecycle progression** (lifecycle_stages.cu): Hunger state (energy depletion) tracked per organism. Hunger increases each generation without fitness improvement, decreases when fitness exceeds threshold. Hunger ≥ hunger_threshold triggers archive resampling (replace stagnant organism with archive elite). Coherence (gradient_fitness.cu) measures training trajectory smoothness - organisms with erratic loss curves penalized.

**Dataset support**: 12 datasets registered (dataset_loader.cu) - MNIST, Fashion-MNIST, CIFAR-10, PathMNIST, ESC-50, Speech-Commands, UrbanSound8K, SVHN, EMNIST, QuickDraw, UCI-HAR, SST-2. Vision datasets (2D spatial), audio (spectral via mel-spectrogram), timeseries (1D temporal), text (embedding). Default: MNIST for rapid prototyping.

**Loss components** (losses.cu):
- Cross-entropy: classification accuracy (∂CE/∂logits backpropped through CA)
- Gradient magnitude: ||∂loss/∂weights||_2 (metalearning signal - select CAs that learn fast)
- Coherence bonus: Σ_t max(0, (loss_t - loss_{t+1})/loss_t) / T (reward consistent improvement)

## Configuration

Key constants in `config/config.cu`:

```cuda
// CA architecture
GRID_SIZE = 64              // Spatial grid dimensions (64×64 cells)
CHANNELS = 512              // CA state channels per cell
NUM_HEADS = 8               // Multi-head attention heads
HIDDEN_DIM = 256            // Perception/interaction hidden dim
HEAD_DIM = 16               // Output dim per head (CHANNELS / NUM_HEADS = 512 / 8 = 64, compressed to 16)

// Genome and compression
GENOME_SIZE = 1024                  // Full genome float count
GENOME_LATENT_DIM_MAX = 128         // DIRESA compressed latent dimension (8× compression)

// Evolution
MAX_COMPONENTS = 256                // Pool size (organisms per generation)
MAX_ARCHIVE_SIZE = 10000            // MAP-Elites archive capacity
CHECKPOINT_INTERVAL = 20            // Gradient learning frequency (generations)

// Behavioral space
BEHAVIORAL_DIM_MIN = 2              // Minimum behavioral coords dimension
BEHAVIORAL_DIM_MAX = 10             // Maximum (genome-adaptive)
HARDWARE_FEATURES_DIM = 15          // Hardware efficiency features

// Fitness exponents (genome-derived ranges)
FITNESS_TASK_EXPONENT_MIN/MAX = 0.3, 2.0        // task^α exponent
FITNESS_GEN_EXPONENT_MIN/MAX = 0.3, 2.0         // (1-gen_gap)^β exponent
FITNESS_RANK_EXPONENT_MIN/MAX = 0.3, 2.0        // effective_rank^γ exponent
FITNESS_EFFICIENCY_EXPONENT_MIN/MAX = 0.1, 1.5  // efficiency^δ exponent
```

## File Structure

```
slime/
├── config/              # System constants (GENOME_SIZE=1024, GENOME_LATENT_DIM_MAX=128, fitness exponent ranges)
├── core/                # Multi-head CA kernels (pseudopod.cu, pseudopod_tensor.cu), organism lifecycle, behavioral navigation
├── kernels/             # Tensor core utilities, warp operations
├── memory/              # MAP-Elites archive (archive.cu), component pools (pool.cu), temporal memory tubes
├── learning/            # Autodiff tape (autodiff_tape.cu), DIRESA autoencoder (diresa.cu)
├── training/            # Gradient fitness (gradient_fitness.cu), classification (classification.cu), hybrid lifecycle (hybrid_lifecycle.cu)
├── lifecycle/           # Genealogy tracking, archive elite sampling
├── metrics/             # Hardware feature extraction (15D: IPC, bandwidth, occupancy, cache hits, divergence)
├── data/                # 12 dataset loaders (MNIST, Fashion-MNIST, CIFAR-10, SVHN, etc.)
├── compression/         # Delta compression (delta.cu) for sparse genome storage
├── tests/               # Unit tests (autodiff, CA integration, lifecycle)
├── runtime.cu           # Host-side orchestration, device memory allocation
└── extract_state.cu     # JSON telemetry logging
```

See `docs/ARCHITECTURE.md` for complete interface contracts.

## Testing

**Unit tests** (tests/ directory): Autodiff tape recording/backprop, DIRESA compression/decompression, MAP-Elites archive insertion, lifecycle progression, gradient fitness computation.

```bash
cd build
compile.bat tests

# Autodiff validation
test_autodiff_tape.exe
# Tests: tape recording during matrix ops, backward pass gradient correctness (∂loss/∂A = ∂loss/∂C · B^T),
# overflow handling, tape reallocation, checkpointing

# End-to-end classification
test_mnist_ca_integration.exe
# Tests: CA forward pass with tensor cores, loss computation (cross-entropy + gradient magnitude),
# autodiff backward through CA, weight updates, accuracy improvement over training steps

# DIRESA compression
test_diresa_embedding.exe
# Tests: 1024→128D encoding, 128→1024D decoding, reconstruction error < threshold,
# distance preservation (trustworthiness/continuity metrics), triplet loss convergence

# Lifecycle mechanics
test_lifecycle_stages.exe
# Tests: hunger state updates, coherence tracking, archive resampling triggers,
# fitness threshold evaluation, genealogy parent_ids correctness
```

**Integration validation**: Compare tensor core vs scalar CA outputs (should match within FP16 epsilon). Verify MAP-Elites archive never stores duplicate elites in same Voronoi cell. Check DIRESA latent mutations produce valid genomes after decode.

## Performance

**Reference hardware (sm_86 Ampere, 28 SMs, 6GB VRAM)**:
- Tensor core throughput: ~13 TFLOPS FP16 (wmma::mma_sync), ~0.4 TFLOPS FP32 scalar
- CA forward pass (8 heads, 64×64 grid, 512 channels): ~2ms with tensor cores, ~60ms scalar
- DIRESA compression (1024→128D): ~0.5ms encode, ~0.5ms decode (warp reductions + matrix ops)
- Fitness computation (compute_fitness_from_diresa_kernel): ~0.1ms per organism (latent variance calculation)
- MAP-Elites archive insertion (Voronoi cell search + elite storage): ~5ms per generation
- Gradient descent step (autodiff backward + weight update): ~15ms per checkpoint (every 20 gen)
- Full generation (selection + mutation + fitness + archive): ~10-15ms average

**Memory footprint** (6GB VRAM):
- CA state (64×64×512 FP16): ~4MB per organism × 256 pool = ~1GB
- Archive (128D latent + metadata): ~0.5KB per elite × 10000 = ~5MB
- Autodiff tape (TAPE_MAX_SIZE ops): ~100MB preallocated
- DIRESA autoencoder weights: ~50MB (encoder + decoder)
- Perception/interaction/value weights (FP16): ~3MB per organism × 256 = ~768MB
- Hardware feature buffers, temporal tubes, chemical fields: ~500MB
- **Total**: ~2.5GB active, ~3.5GB peak during gradient checkpoints

**Bottlenecks**:
- DIRESA autoencoder training: 500-step triplet loss minimization every 2000 generation steps (distance preservation learning)
- Voronoi tessellation density updates: O(archive_size × behavioral_dim) distance computations per elite insertion
- Memory compaction: Stream compaction (thrust::remove_if) when temporal tubes exceed capacity
- Autodiff tape overflow: Checkpoint/recomputation when tape exceeds TAPE_MAX_SIZE during long CA rollouts

## References

- Illuminating search spaces by mapping elites. Mouret & Clune (2015) arXiv:1504.04909
- Flow-Lenia: Towards open-ended evolution in cellular automata through mass conservation and parameter localization. Randazzo et al. (2023) arXiv:2212.07906
- A Path to Universal Neural Cellular Automata. Béna et al. (2025) arXiv:2505.13058
- DIRESA, a distance-preserving nonlinear dimension reduction technique based on regularized autoencoders. Geert De Paepe, Lesley De Cruz (2025) arXiv:2404.18314
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. (2022) NEURIPS2022_67d57c32

## License

MIT
