# Slime Mold Transformer

**Core idea**: A neural network learns multiple cellular automaton update rules simultaneously. Each rule explores a different computational path. Successful paths survive and get archived. Failed paths are replaced by sampling from archive history.

**Intended behavior**: Traditional neural networks collapse to a single solution. This system would maintain a population of diverse solutions, each specialized for different computational patterns. The archive should prevent mode collapse while selection ensures quality.

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
python run.py  # Currently demonstrates basic CA forward pass
```

## Architecture

### Foundation: Neural Cellular Automaton

A pseudopod is a learned update rule for a cellular automaton. Standard neural networks apply the same computation everywhere. CAs let computation vary spatially - different cells can follow different update rules.

Each pseudopod would trace a trajectory through parameter space:

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

**Mass conservation** (∑ output = ∑ input): Physical constraint for stability.

**Parameter localization**: CA rule parameters should vary by position.

**Multi-head**: Multiple parallel update rules planned.

**Ensemble computation**: Multiple pseudopods would be active simultaneously, with archive storing successful configurations.

### GPU Acceleration

**Target**: Tile computation to fit in SRAM. Triton kernels to fuse operations.

**Expected impact**: O(M² × D) HBM accesses → O(M² × D / SRAM_size).

**Planned implementation**: Adaptive tile sizes based on GPU SRAM availability.

### Archive: Trajectory Memory

**Challenge**: Training typically finds one solution then stops.

**Approach**: Archive to store successful parameter configurations indexed by behavior.

**Behavioral space**: Each pseudopod would generate runtime metrics. DIRESA embeddings planned for dimensionality reduction.

**Planned storage optimization**: 
- SVD low-rank factorization
- Delta compression
- Content-addressable hashing
- Target: 80-160x compression

**Adaptive partitioning**: Voronoi cells planned to adapt based on density.

### Selection: Curiosity-Driven Survival

**Target fitness formula = effective_rank() × coherence()**:
- Planned weights: 70% gradient magnitude, 20% efficiency, 10% conservation

**Curiosity metric**: `coherence()` to measure learning progress.

**Selection mechanism**: Periodic fitness evaluation planned.

**Stability mechanisms planned**: 
- Warmup phase
- Gentle introduction
- Loss gates

**Hypothesis**: Archive could maintain history of alternative solutions.

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
python run.py  # Basic forward pass demo
```

Planned loss components:
- Reconstruction
- Rank regularization
- Coherence regularization
- Diversity
- Archive coverage
- Fitness variance

## Testing

```bash
pytest slime/tests/unit/  # Unit tests for individual components
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

## Development Focus

- Primary reference: BLUEPRINT.md
- Construction plan: HEALING_PLAN.md

## References

- Illuminating search spaces by mapping elites. Mouret & Clune (2015) arXiv:1504.04909
- Flow-Lenia: Towards open-ended evolution in cellular automata through mass conservation and parameter localization. Randazzo et al. (2023) arXiv:2212.07906
- A Path to Universal Neural Cellular Automata. Béna et al. (2025) arXiv:2505.13058
- DIRESA, a distance-preserving nonlinear dimension reduction technique based on regularized autoencoders. Geert De Paepe, Lesley De Cruz (2025) arXiv:2404.18314
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. (2022) NEURIPS2022_67d57c32

## License

MIT
