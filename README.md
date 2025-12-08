# Slime: GPU-Native Evolutionary Neural CA

**Core idea**: Combine gradient-based learning with evolutionary diversity maintenance. Neural cellular automata learn via backpropagation on external task datasets while a MAP-Elites archive prevents mode collapse. Tensor cores accelerate both CA computation and task evaluation.

**Intended behavior**: Traditional neural networks collapse to single solutions. This system maintains a population of diverse CA update rules, each specialized for different task performance niches (accuracy vs generalization vs efficiency trade-offs). The archive preserves successful strategies across three behavioral axes: hardware efficiency, task performance, and generalization capacity.

**Implementation**: Multi-head neural CAs with FP16 tensor core acceleration. MAP-Elites archive with adaptive Voronoi cells in concatenated 3D behavioral space (hardware + task + generalization). Power-law fitness composition: task_accuracy^α × generalization^β × effective_rank^γ × hardware_efficiency^δ where exponents are genome-evolved. All computation on GPU via dynamic parallelism.

## Requirements

```bash
# NVIDIA GPU with Tensor Cores (sm_86+)
# RTX 3060 or newer
# CUDA 12.0+
nvcc --version  # Verify installation
```

## Build

```bash
cd build
compile.bat  # Windows
# or
make  # Linux
```

## Quick Start

```bash
./slime.exe  # Run evolution with default parameters
```

Output: JSON snapshots every 100 generations containing archive state, fitness trajectories, and behavioral embeddings.

## Python Tooling (developer helpers)

Run common developer checks via a single entry point:

```bash
# Verify expected kernels were compiled (auto-detects PTX dir and exe)
python tools/dev.py verify-kernels

# Audit MNIST→CA training path against compiled artifacts (run build first or add --build)
python tools/dev.py audit-connectivity --build
```

- verify-kernels:
  - Scans PTX and preprocessed sources (*.ii, *.ptx, *.cudafe1.cpp) for expected kernels, including MNIST gradient path components.
  - PTX directory auto-detection: prefers build/logs/ptx (from compile.bat), falls back to logs/ptx.
  - Executable default: build/slime.exe (if present). Override with: `python tools/dev.py verify-kernels --exe <path>`

- audit-connectivity (real audit):
  - Verifies required connections both in source and in compiled artifacts (preprocessed launches under logs/ptx). Fails if nothing is compiled unless `--allow-source-only` is provided.
  - Add `--build` to invoke build/compile.bat automatically before auditing.
  - Flags: `--ptx-dir <dir>`, `--exe <path>`, `--build`, `--allow-source-only`.
  - Required kernel launch sites checked:
    - MNIST→CA injection: mnist_to_ca_grid_kernel or inject_mnist_to_ca_kernel
    - Features→Logits: classification_head_kernel
    - Logits→Loss: cross_entropy_loss_kernel
    - Backward: ad_backward_kernel
    - Optimizer: adam_update_kernel or adam_update_fp16_kernel

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

**Mass conservation**: Σ(A^(t+dt)) = Σ(A^t) emerges from reintegration geometry - each cell's mass redistributes to overlapping neighbors, overlap integrals sum to 1 per source. No rescaling needed.

**Parameter localization**: Multi-head CA weights define spatially-varying affinity landscape U^t. Each head's perception/interaction/value weights create different local preferences. Flow field F^t weights affinity gradient ∇U^t vs diffusion -∇A_Σ^t based on local mass concentration via α^t.

**Tensor core mapping**: Matrix dimensions align to 16×16 WMMA fragments
**Throughput**: 256 TFLOPS/s vs 5 TFLOPS/s for scalar FP32 (50× speedup)

### Fitness = effective_rank × coherence

**effective_rank** (parameter diversity):
- Correlation matrix of CA weights
- SVD via Jacobi sweeps (warp_ca.cu)
- Shannon entropy of singular values
- exp(entropy) = effective dimensionality
- High rank = diverse parameter usage

**coherence** (learning progress):
- Genome predicts chemical field evolution
- prediction_errors[t] = ||predicted - actual||²
- coherence = rate of error decrease
- High coherence = organism learning fast
- Computed via temporal correlation kernel

**Schedule**: Prediction errors tracked every generation. Full fitness (SVD + coherence) computed every 100 generations.

### MAP-Elites Archive

**Behavioral space**: Hardware-geometric features → DIRESA embedding
- 15D input: warp occupancy, memory throughput, IPC, cache hits, divergence, etc.
- 2-10D output: Distance-preserving nonlinear projection
- Adaptive dimensionality discovered via validation
- Trustworthiness ≥ 0.70, Continuity ≥ 0.70 required for activation

**DIRESA training schedule**:
1. Steps 0-2000: Use Euclidean distance, accumulate samples
2. Step 2000: Train encoder/decoder for 500 steps if ≥1000 samples
3. Validate distance preservation
4. Activate if validation passes, else retry 3× with 2× learning rate
5. Fallback to Euclidean if all attempts fail

**Voronoi tessellation**: Adaptive cells partition behavioral space
- Distance: ||elite.behavioral_coords - cell.centroid||
- Cells reshape when DIRESA changes dimensionality
- Density-based quality thresholds

**Compression**: SVD low-rank (8×) + delta from parent (10-20×) = 80-160× total

### Curiosity-Driven Lifecycle

**Hunger = 1.0 - coherence**:
- High coherence → low hunger → survival
- Low coherence → high hunger → culling
- Archive sampling replaces culled organisms

**Warmup phases** (prevent early chaos):
- 0-100 steps: No lifecycle, stabilize trajectories
- 100-500 steps: Reduced culling frequency
- 500+ steps: Full lifecycle operation

**Loss gates**: Freeze lifecycle when loss > 10× EMA (prevent catastrophic forgetting)

### Gradient + Evolution Dual Training

**Gradient fitness**: Magnitude of weight updates during MNIST classification
- Classification head: Spatial pooling → FC layer → logits
- Cross-entropy loss → backprop through autodiff tape
- Gradient magnitude = learning potential

**Evolutionary fitness**: effective_rank × coherence
- Parameter diversity × learning progress
- GPU-native via SVD and temporal correlation

**Hybrid**: 70% gradient, 30% evolution weights

### Dynamic Parallelism

**Zero CPU synchronization**: Parent kernels launch child kernels on GPU
- 41 device-side kernel launches in organism.cu
- component_evolution_kernel spawns: selection, archive, lifecycle kernels
- Compilation flags: `-rdc=true -lcudadevrt -arch=sm_86`

### Automatic Differentiation

**Tape-based backprop**:
- Record operations during CA forward pass
- Thread tape indices: perception → interaction → value
- Backward pass computes ∂loss/∂weights
- Overflow checks before every tape operation

**Integration**: autodiff_integration.cu threads indices through CA kernels

## Training

The system metalearns on MNIST: CAs that learn to classify digits faster have higher gradient fitness.

```bash
./slime.exe  # MNIST metalearning enabled by default
```

Loss components:
- Cross-entropy (classification accuracy)
- Gradient magnitude (learning speed)
- Coherence bonus (exploration reward)

## Configuration

Key constants in `config/config.cu`:

```cuda
GRID_SIZE = 64
CHANNELS = 512
NUM_HEADS = 8
HIDDEN_DIM = 256
GENOME_SIZE = 1024
MAX_COMPONENTS = 256
MAX_ARCHIVE_SIZE = 10000
BEHAVIORAL_DIM = 10  // Adaptive 2-10
HARDWARE_FEATURES_DIM = 15
```

## File Structure

```
slime/
├── config/              # System constants
├── core/                # CA kernels, organism lifecycle, behavioral navigation
├── kernels/             # SVD, tensor cores, utilities
├── memory/              # Archive, pools, temporal memory
├── learning/            # Autodiff, DIRESA embedding
├── training/            # Gradient fitness, classification, lifecycle
├── lifecycle/           # Genealogy, archive sampling
├── metrics/             # Hardware feature extraction
├── data/                # MNIST loader
├── compression/         # Delta compression
├── tests/               # Unit tests
├── runtime.cu           # Host-side orchestration
└── extract_state.cu     # JSON logging
```

See `docs/ARCHITECTURE.md` for complete interface contracts.

## Testing

```bash
cd build
compile.bat tests
test_autodiff_tape.exe          # Gradient validation
test_mnist_ca_integration.exe   # End-to-end classification
```

## Performance

**RTX 3060 (sm_86, 28 SMs)**:
- Tensor core throughput: ~13 TFLOPS FP16
- CA forward (8 heads): ~2ms
- Fitness computation (every 100 gen): ~50ms
- Archive insertion: ~5ms
- Generation time: ~10ms

**Bottlenecks**:
- SVD (Jacobi sweeps): O(n³) iterations
- DIRESA training: 500-step triplet loss every 2000 steps
- Memory compaction: Stream compaction when tubes full

## References

- Illuminating search spaces by mapping elites. Mouret & Clune (2015) arXiv:1504.04909
- Flow-Lenia: Towards open-ended evolution in cellular automata through mass conservation and parameter localization. Randazzo et al. (2023) arXiv:2212.07906
- A Path to Universal Neural Cellular Automata. Béna et al. (2025) arXiv:2505.13058
- DIRESA, a distance-preserving nonlinear dimension reduction technique based on regularized autoencoders. Geert De Paepe, Lesley De Cruz (2025) arXiv:2404.18314
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. (2022) NEURIPS2022_67d57c32

## License

MIT
