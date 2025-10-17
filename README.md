# Slime Mold Transformer

**Core idea**: A neural network learns multiple cellular automaton update rules simultaneously. Each rule explores a different computational path. Successful paths survive and get archived. Failed paths are replaced by sampling from archive history.

**Intended behavior**: Traditional neural networks collapse to a single solution. This system would maintain a population of diverse solutions, each specialized for different computational patterns. The archive should prevent mode collapse while selection ensures quality.

**Implementation**: Multi-head Neural CAs with Flow-Lenia dynamics (mass-conserving learned update rules). CVT-MAP-Elites archive with distance-preserving embeddings. Curiosity-driven lifecycle where learning progress determines survival.

## Installation

```bash
# Requires NVIDIA GPU with CUDA 12.0+
# RTX 3060 minimum (Tensor Cores required)
nvcc --version  # Verify CUDA installation

# Build GPU-native system
make clean
make all
```

## Quick Start

```bash
./slime --seed=42  # Run GPU-native slime mold transformer
```

## Architecture

### Foundation: Neural Cellular Automaton

A pseudopod is a learned update rule for a cellular automaton. Standard neural networks apply the same computation everywhere. CAs let computation vary spatially - different cells can follow different update rules.

Each pseudopod traces a trajectory through parameter space:

```
Input (latent + stimulus)
  ↓
Multi-head CA projections (Perception/Interaction/Value)
  ↓
CA neighborhood convolution (3×3 learned kernels per head)
  ↓
Flow-Lenia growth function (bell curve modulation)
  ↓
CA activation pattern (like attention but for CA)
  ↓
Mass-conserving value propagation (∑ output = ∑ input)
  ↓
Spatially-modulated output projection
```

**Mass conservation** (∑ output = ∑ input): Physical constraint for stability, enforced via warp reductions.

**Parameter localization**: CA rule parameters vary by position (Flow-Lenia dynamics).

**Multi-head**: 8 parallel CA update rules execute concurrently.

**Ensemble computation**: Multiple pseudopods active simultaneously, archive stores successful configurations.

### GPU Acceleration

**Tensor Core mapping**: 8 multi-head CA rules operate on 16×16 tiles matching WMMA fragment size. CA convolutions become matrix multiplies.

**Memory hierarchy**: Operations tiled to fit in 128KB SRAM per SM. Warp shuffles keep data in registers.

**Fused kernels**: fitness = effective_rank × coherence computed in single kernel. No intermediate memory writes.

### Archive: Trajectory Memory

**MAP-Elites CVT**: Adaptive Voronoi tessellation with cells that grow/shrink based on density. Content-addressable storage via SHA256 deduplication.

**Behavioral space**: DIRESA learns 2-10D embeddings from raw metrics. Distance-preserving nonlinear projection.

**Storage optimization**: 
- SVD low-rank factorization (8x compression)
- Delta compression vs parent (10-20x additional)
- Content-addressable hashing prevents duplicates
- Achieved: 80-160x total compression

**GPU operations**: Voronoi updates via batched matrix ops on Tensor Cores. Archive sampling uses warp-level parallel search.

### Selection: Curiosity-Driven Survival

**Fitness formula = effective_rank() × coherence()**:
- Weights: 70% gradient magnitude, 20% compute efficiency, 10% mass conservation
- GPU-native computation via Jacobi SVD and temporal correlation

**Curiosity-driven lifecycle**: 
- hunger = 1.0 - coherence (learning progress deficit)
- High coherence → low hunger → survival
- Low coherence → high hunger → replacement from archive

**Stability mechanisms**: 
- Warmup: 100 steps no lifecycle
- Gentle: 100-500 steps reduced culling
- Loss gates: Freeze lifecycle when loss > 10× EMA

## Configuration

`slime/config/model.yaml`:

```yaml
architecture:
  num_heads: 8
  head_dim: 64
  grid_size: 128
  
behavioral_space:
  min_dims: 2
  max_dims: 10  # DIRESA adaptive
  
fitness_weights:
  gradient_magnitude: 0.7
  compute_efficiency: 0.2
  conservation_quality: 0.1
```

## Training

```bash
./slime --mode=train --dataset=mnist  # Train on MNIST
./slime --mode=train --dataset=cifar10 --gpus=2  # Multi-GPU training
```

Loss components:
- Task loss (reconstruction/classification)
- Rank regularization (prevent collapse)
- Coherence bonus (reward learning progress)
- Coverage loss (explore behavioral space)

## Testing

```bash
make test  # Run GPU kernel tests
make bench  # Performance benchmarks
```

## File Structure

```
slime/
├── proto/              # Protocol headers (kernel, memory, model, component)
├── kernels/            # GPU kernels (warp_ca.cu, utils.cu, triton_impl.cu)
├── core/               # Components (pseudopod.cu, organism.cu, chemotaxis.cu)
├── memory/             # Data structures (archive.cu, pool.cu, tubes.cu)
├── training/           # Training loop (trainer.cu, fitness.cu, lifecycle.cu)
├── api/                # Public interface (gpu_native.cu)
├── config/             # YAML configurations
└── tests/              # GPU kernel tests
```

## Development Focus

- Architecture specification: BLUEPRINT.md
- Build plan: buildplan.md

## References

- Illuminating search spaces by mapping elites. Mouret & Clune (2015) arXiv:1504.04909
- Flow-Lenia: Towards open-ended evolution in cellular automata through mass conservation and parameter localization. Randazzo et al. (2023) arXiv:2212.07906
- A Path to Universal Neural Cellular Automata. Béna et al. (2025) arXiv:2505.13058
- DIRESA, a distance-preserving nonlinear dimension reduction technique based on regularized autoencoders. Geert De Paepe, Lesley De Cruz (2025) arXiv:2404.18314
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. (2022) NEURIPS2022_67d57c32

## License

MIT
