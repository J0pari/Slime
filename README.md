# Slime Mold Transformer

A neural architecture that learns cellular automaton update rules for adaptive computation. Each forward pass computes an ensemble average over multiple active components (pseudopods), where each pseudopod explores a different trajectory through parameter space. The archive maintains a history of successful computational trajectories, weighted by fitness. Selection naturally collapses the ensemble toward high-fitness paths that persist.

Components (pseudopods) are multi-head Neural CAs implementing Flow-Lenia dynamics. Quality-diversity search via CVT-MAP-Elites archive with DIRESA learned embeddings discovers diverse behavioral strategies.

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

### Neural Cellular Automaton

Each pseudopod is a learned CA update rule that traces a trajectory through configuration space (CA weights, attention weights, normalization scales):

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

**Key properties:**
- Multi-head: 4 parallel CA update rules
- Parameter localization: CA rule parameters vary spatially
- Mass conservation: Biological inspiration from physical constraints
- GPU-accelerated: Triton kernels with Flash Attention-style online softmax

**Computational ensemble:** Each forward pass activates multiple pseudopods at a behavioral location. Their outputs are averaged to compute the ensemble result. The archive stores parameter configurations that reached each behavioral region, weighted by their fitness contributions. This creates a form of memory that guides future exploration.

### Triton Kernels

GPU kernels implement:
- Fused multi-head CA projections
- Flow-Lenia growth function (Gaussian bell curve per head)
- CA activation with temperature-modulated softmax
- Mass-conserving value propagation
- Correlation and effective rank (behavioral metrics)

Autograd support: Forward uses Triton (speed), backward uses PyTorch einsum (gradient flow).

### Quality-Diversity Archive

**Trajectory history:** The archive stores successful parameter configurations that reached different behavioral locations. When the pool needs new pseudopods, it samples from archive trajectories that previously succeeded at similar behaviors, weighted by their fitness.

CVT-MAP-Elites with:
- 50 centroids (TINY) - adaptive Voronoi partitioning
- DIRESA learned embeddings (3-5D adaptive behavioral space)
- Low-rank storage (16x compression) + delta compression (5-10x)
- Content-addressable: SHA256 deduplication

Behavioral metrics (65 raw dimensions → 3-5D learned space) from pseudopod runtime:
- CA pattern statistics
- Weight gradient norms
- Activation statistics
- Compute metrics
- Correlation structure

### Lifecycle Management

**Trajectory collapse through selection:** The pseudopod pool maintains multiple computational paths simultaneously. At each step:
1. Active pseudopods compute their fitness (effective_rank × coherence)
2. High-fitness trajectories survive and get archived
3. Low-fitness trajectories are replaced by sampling from archive history
4. This collapses the ensemble toward paths that work, while maintaining diversity

**Curiosity-driven selection:**
- fitness = effective_rank() × coherence()
- High fitness → survive and archive, low fitness → resample from archive
- Learning progress (coherence metric) drives which trajectories persist

**Stability guardrails:**
- Warmup: 100 steps (no lifecycle, let trajectories stabilize)
- Gentle: 500 steps (reduced culling frequency)
- Loss gates: Freeze lifecycle if loss > 10× EMA (protect during training instability)

## Configuration

`slime/config/presets.py`:

```python
TINY = ArchitectureConfig(
    dimensions=Dimension Config(head_dim=16, num_heads=4, hidden_dim=64),
    behavioral_space=BehavioralSpaceConfig(num_raw_metrics=65, min_dims=3, max_dims=5),
    ...
)
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
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (2022) NEURIPS2022_67d57c32

## License

MIT
