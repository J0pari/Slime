# Slime Mold Transformer

A neural architecture that learns cellular automaton update rules for adaptive computation. Components (pseudopods) are multi-head Neural CAs implementing Flow-Lenia dynamics. Quality-diversity search via CVT-MAP-Elites archive with DIRESA learned embeddings.

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

Each pseudopod is a learned CA update rule:

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

### Triton Kernels

GPU kernels implement:
- Fused multi-head CA projections
- Flow-Lenia growth function (Gaussian bell curve per head)
- CA activation with temperature-modulated softmax
- Mass-conserving value propagation
- Correlation and effective rank (behavioral metrics)

Autograd support: Forward uses Triton (speed), backward uses PyTorch einsum (gradient flow).

### Quality-Diversity Archive

CVT-MAP-Elites with:
- 50 centroids (TINY) - adaptive Voronoi partitioning
- DIRESA learned embeddings (3-5D adaptive behavioral space)
- Low-rank storage (16x compression) + delta compression (5-10x)
- Content-addressable: SHA256 deduplication

Behavioral metrics (62 dimensions) from pseudopod runtime:
- CA pattern statistics
- Weight gradient norms
- Activation statistics
- Compute metrics
- Correlation structure

### Lifecycle Management

**Curiosity-driven selection:**
- fitness = effective_rank() × coherence()
- High fitness → survive, low fitness → sample from archive

**Stability:**
- Warmup: 100 steps (no lifecycle)
- Gentle: 500 steps (reduced frequency)
- Loss gates: Freeze if loss > 10× EMA

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

- Triton kernels: 14/14 tests, 33/33 constraints (100%)
- Integration: In progress
- Compression: 80-160× memory reduction validated

## References

- MAP-Elites: Mouret & Clune (2015) arXiv:1504.04909
- Flow-Lenia: Randazzo et al. (2023) arXiv:2212.07906
- Neural CA: Béna (2025) arXiv:2505.13058
- DIRESA: Zhang et al. (2025) arXiv:2404.18314
- Flash Attention: Dao et al. (2022) NeurIPS

## License

MIT
