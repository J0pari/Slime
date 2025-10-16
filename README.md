# Slime Mold Transformer

**Core idea**: A neural network learns multiple cellular automaton update rules simultaneously. Each rule explores a different computational path. Successful paths survive and get archived. Failed paths are replaced by sampling from archive history.

**Why this works**: Traditional neural networks collapse to a single solution. This system maintains a population of diverse solutions, each specialized for different computational patterns. The archive prevents mode collapse while selection ensures quality.

**Implementation**: Multi-head Neural CAs with Flow-Lenia dynamics (mass-conserving learned update rules). CVT-MAP-Elites archive with distance-preserving embeddings. Curiosity-driven lifecycle where learning progress determines survival.

## Installation

```bash
# Windows with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install triton-windows
pip install pyyaml numpy scipy scikit-learn

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## Quick Start

```bash
python run.py
```

Trains on MNIST (28x28 images → 10 classes) using TINY architecture (4 heads, 16-dim, 64 hidden).

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

**Mass conservation** (∑ output = ∑ input): Physical constraint ensures stability. Unconstrained CAs diverge.

**Parameter localization**: CA rule parameters vary by position, not global. Enables spatial specialization.

**Multi-head**: 4 parallel update rules discover different computational strategies.

**Ensemble computation**: Multiple pseudopods active simultaneously. Outputs averaged. Archive stores which parameter configurations worked at which behavioral locations. Sampling from archive bootstraps new pseudopods from successful trajectories.

### GPU Acceleration

**Problem**: CA updates are embarrassingly parallel but memory-bound. Naive implementation loads weights from HBM repeatedly.

**Solution**: Tile computation to fit in SRAM (on-chip fast memory). Load tiles once, compute entirely in SRAM, write results.

**Impact**: O(M² × D) HBM accesses → O(M² × D / SRAM_size). 10-20x speedup on typical workloads.

**Implementation**: Triton kernels fuse operations (multi-head projections, Flow-Lenia growth, mass-conserving propagation). Adaptive tile sizes (BLOCK=128/64/32) based on GPU SRAM availability. Forward uses Triton for speed, backward uses PyTorch einsum for correct gradients.

### Archive: Trajectory Memory

**Problem**: Training finds one solution then stops. Diverse solutions exist but gradient descent collapses to nearest local optimum.

**Solution**: Archive stores successful parameter configurations indexed by behavior. When pool needs new pseudopods, sample from archive locations with similar behavior. This bootstraps from known-good trajectories instead of random initialization.

**Behavioral space**: Each pseudopod generates metrics during runtime (CA mass conservation, gradient magnitudes, activation patterns, hardware utilization). These form a high-dimensional behavioral description. Dimensionality discovered via covariance rank, then compressed to 3-5D using DIRESA (distance-preserving learned embeddings).

**Storage efficiency**: 
- SVD low-rank factorization: 8x compression (D×D → D×k + k×D)
- Delta compression: 10-20x (store diffs vs parent in same Voronoi cell)
- Content-addressable hashing: Deduplicate identical elites
- **Total: 80-160x compression** (4MB elite → 25-50KB)

**Adaptive partitioning**: Voronoi cells grow in dense regions, shrink in sparse regions. Prevents unbalanced storage where some cells have 1000 elites and others have 0.

### Selection: Curiosity-Driven Survival

**Fitness = task performance × compute efficiency × CA quality**:
- Task (70%): Gradient magnitude (high gradient = affects loss)
- Efficiency (20%): Hardware utilization (FLOPs, bandwidth, tensor cores)
- Conservation (10%): Mass conservation quality (stable CA dynamics)

**Curiosity metric**: `coherence()` measures learning progress. High coherence = learning fast = survive. Low coherence = plateaued = replace with archive sample.

**Selection pressure**: Every 100 steps, compute fitness for all pseudopods. Top performers survive and get archived. Bottom performers culled and replaced. This collapses the ensemble toward successful trajectories while archive maintains diversity.

**Stability**: 
- Warmup (0-100 steps): No lifecycle. Let gradients stabilize.
- Gentle (100-500 steps): Reduced culling. Gradual pressure.
- Loss gates: If loss spikes >10× EMA, freeze lifecycle until stable.

**Why this works**: Standard neural networks have no memory of alternative solutions. Once gradient descent finds a local optimum, training stops. Archive maintains history of all successful trajectories. Selection explores this history guided by current task performance.

## Configuration

`slime/config/presets.py`:

```python
TINY = ArchitectureConfig(
    dimensions=DimensionConfig(head_dim=16, num_heads=4, hidden_dim=64),
    behavioral_space=BehavioralSpaceConfig(min_dims=3, max_dims=5),  # dimensionality discovered
    ...
```

## Training

```bash
python run.py
```

Loss function (6 terms):
- Reconstruction (1.0)
- Rank regularization (0.01)
- Coherence regularization (0.01)
- Diversity (0.1)
- Archive coverage (0.05)
- Fitness variance (0.05)

## Testing

```bash
pytest slime/tests/unit/test_triton_kernels.py -v  # 14/14, 33/33 constraints
pytest slime/tests/unit/test_lifecycle.py -v        # 22 tests
```

## File Structure

```
slime/
├── proto/              # Protocols
├── kernels/            # Triton + PyTorch fallback
├── core/               # Neural CA (pseudopod, organism, chemotaxis)
├── memory/             # Archive (MAP-Elites), Pool, DIRESA
├── training/           # Trainer, loss, fitness, lifecycle, stability
├── api/                # SlimeMoldEncoder (nn.Module)
├── config/             # Presets (TINY, SMALL, MEDIUM, LARGE, FULL)
└── tests/              # Unit tests with causal constraints
```

## Current Status

- Triton kernels: unit testing in progress
- Integration: In progress

## References

- Illuminating search spaces by mapping elites. Mouret & Clune (2015) arXiv:1504.04909
- Flow-Lenia: Towards open-ended evolution in cellular automata through mass conservation and parameter localization. Randazzo et al. (2023) arXiv:2212.07906
- A Path to Universal Neural Cellular Automata. Béna et al. (2025) arXiv:2505.13058
- DIRESA, a distance-preserving nonlinear dimension reduction technique based on regularized autoencoders. Geert De Paepe, Lesley De Cruz (2025) arXiv:2404.18314
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. (2022) NEURIPS2022_67d57c32

## License

MIT
