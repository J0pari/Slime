# Slime Mold Transformer

A neural network where components are learned cellular automata with curiosity-driven lifecycle. Pseudopods extend into behavioral space, discovering niches through gradient-based selection. Archive maintains diversity via Adaptive Voronoi MAP-Elites. Memory efficiency through low-rank factorization and delta compression (80-160x reduction).

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

```python
from slime.api.torch_compat import SlimeMoldEncoder
import torch

model = SlimeMoldEncoder(
    sensory_dim=512,
    latent_dim=256,
    head_dim=64,
    num_pseudopods=8,
    device='cuda'
)

x = torch.randn(2, 128, 512, device='cuda')  # (batch, seq, dim)
output = model(x)  # (2, 128, 256)
```

## What It Is

Slime mold extends pseudopods searching for food. Successful pseudopods persist, unsuccessful ones retract. The organism remembers successful patterns.

Our system: **Pseudopods are learned cellular automata** (Conway → Lenia → Flow-Lenia → Neural Flow-Lenia). Each Pseudopod is a Neural CA update rule that learns via gradient descent on the task loss.

**Flow-Lenia substrate:**
- Mass conservation: ∑ output = ∑ input
- Parameter localization: CA rule parameters vary spatially (not global)
- Warp-level GPU execution: Neighbors accessed via shuffles, tensor cores for convolution

**Curiosity-driven lifecycle:**
- coherence() metric tracks learning progress (Δ prediction error)
- High coherence (learning fast) → low hunger → survive
- Low coherence (plateaued) → high hunger → sample new genome from archive
- Natural selection via intrinsic motivation, not external reward

**Adaptive Voronoi MAP-Elites:**
- Archive cells grow/shrink based on density (not fixed grid)
- DIRESA learns behavioral embeddings online (adaptive 2-10D)
- Dimension count adapts via warp vote mechanism
- Distance-preserving nonlinear autoencoder, not PCA

## Configuration

Edit `slime/config/model.yaml`:

```yaml
model:
  sensory_dim: 512
  latent_dim: 256
  head_dim: 64
  num_pseudopods: 8

archive:
  num_centroids: 1000                    # Adaptive Voronoi partitions
  diresa_dims: [2, 10]                   # Adaptive dimensionality range (learned)
  low_rank_k: 64                         # Factorization rank for weight compression
  delta_rank: 8                          # Delta compression rank (smaller)
  kmo_threshold: 0.6                     # KMO validation threshold
  reconstruction_error_threshold: 0.5    # Max reconstruction error
  gc_interval: 100                       # Garbage collection frequency
  seed: 42                               # Deterministic centroid init

lifecycle:
  curiosity_driven: true   # Use coherence() for hunger
  max_pool_size: 64
  min_pool_size: 4
  max_loss_ratio: 10.0     # Freeze lifecycle if loss > 10x EMA
  seed: 42                 # Deterministic birth/death

pool:
  min_size: 4
  max_size: 64
  cull_fraction: 0.2
```

Load with:

```python
from slime.config.loader import load_config
config = load_config('slime/config/model.yaml')
```

## Training

```python
from slime.training.trainer import Trainer
from slime.bench.datasets import load_tinystories

dataset = load_tinystories(split='train', max_samples=10000)
trainer = Trainer(model, config)

trainer.train(
    dataset=dataset,
    epochs=10,
    batch_size=8,
    learning_rate=1e-4
)
```

### Timescales

- **Fast (every step):** Weight updates, gradient computation, fitness tracking
- **Medium (every 100 steps):** Archive updates, component birth decisions
- **Slow (every 1000 steps):** Component culling, memory budget enforcement

### Safety

If loss exceeds 10x moving average, lifecycle freezes automatically. Training continues with existing components only.

## Architecture

**Elite Storage:**
1. SVD low-rank compression: W = U @ V (D×D → D×k + k×D, 8x compression)
2. Content addressing: SHA256 hash → automatic deduplication
3. Delta compression: Store diffs between consecutive elites (10-20x additional)
4. Automatic re-basing: When delta chain >70% of full size, store new blob

Combined: **80-160x memory reduction**

**Behavioral Embeddings:**

DIRESA autoencoder learns 2-10 dimensions online. Raw metrics:
- CA_mass_conservation (∑ output = ∑ input)
- CA_parameter_localization (spatial variation of rule parameters)
- CA_neighborhood_coherence (local vs global patterns)
- activation_sparsity
- gradient_flow_magnitude
- memory_access_locality
- computational_intensity
- weight_magnitude
- gradient_variance
- activation_magnitude

DIRESA learns nonlinear embeddings preserving pairwise distances. Dimension count adapts based on task. Validated via KMO ≥ 0.6, Bartlett's p < 0.05, reconstruction error ≤ 0.5.

## Testing

```bash
# Unit tests
pytest slime/tests/unit/ -v

# Integration tests
pytest slime/tests/integration/ -v

# Ablations (vs baseline transformer)
pytest slime/tests/ablations/ -v
```

### Critical Tests

```bash
# Verify Triton kernels
pytest slime/tests/unit/test_triton_kernels.py -v

# Verify DAG dependencies
pytest slime/tests/unit/test_dag_enforcement.py -v

# Verify ultrametric topology
pytest slime/tests/unit/test_topology_chemotaxis.py -v
```

## Profiling

```python
from slime.bench.profile import profile_model

results = profile_model(
    model=model,
    input_shape=(8, 512, 512),  # (batch, seq, dim)
    num_iterations=100,
    device='cuda'
)

print(f"Forward: {results['forward_mean_ms']:.2f}ms ± {results['forward_std_ms']:.2f}ms")
print(f"Backward: {results['backward_mean_ms']:.2f}ms ± {results['backward_std_ms']:.2f}ms")
print(f"Memory: {results['peak_memory_mb']:.1f}MB")
```

## Observability

```python
from slime.observability.metrics import MetricsCollector

metrics = MetricsCollector()
model = SlimeMoldEncoder(..., metrics_collector=metrics)

# After training
print(f"Pool size: {metrics.get('pool_size')}")
print(f"Archive coverage: {metrics.get('archive_coverage')}")
print(f"Component births: {metrics.get('component_births')}")
print(f"Component deaths: {metrics.get('component_deaths')}")
```

## Exporting

```python
from slime.tools.export import export_onnx, export_torchscript

# ONNX (for C++/other languages)
export_onnx(model, output_path='model.onnx', input_shape=(1, 128, 512))

# TorchScript (deployment without Python)
export_torchscript(model, output_path='model.pt')
```

## Troubleshooting

**"CUDA out of memory"**
- Reduce `batch_size`
- Reduce `max_size` in pool config
- Enable memory budget enforcement

**"Loss diverging during training"**
- Reduce `learning_rate`
- Verify fitness uses gradients: `pytest slime/tests/unit/test_fitness_computation.py`

**"Components not diversifying"**
- Check DIRESA embeddings: KMO > 0.6, reconstruction error < 0.5
- Verify fitness includes efficiency signal
- Increase `num_centroids` (more Voronoi cells)

**"Triton kernel errors"**
- Verify CUDA version: `nvidia-smi`
- Reinstall: `pip install --force-reinstall triton-windows`
- Fallback: `model = SlimeMoldEncoder(..., use_triton=False)`

## File Layout

```
slime/
├── proto/          # Protocols (Component, Kernel, Memory, Model)
├── kernels/        # GPU implementations (Triton + PyTorch fallback)
├── memory/         # Archive (MAP-Elites), Pool (lifecycle), Tubes (temporal)
├── core/           # Pseudopod (Neural CA), Chemotaxis, Organism, Stencil
├── topology/       # Ultrametric (p-adic, genealogy, hierarchy)
├── api/            # torch_compat (nn.Module), native (SlimeModel)
├── training/       # Trainer, losses, fitness, lifecycle, stability
├── config/         # YAML loaders
├── bench/          # Datasets, profiling, baseline transformer
├── tests/          # unit/, integration/, ablations/, slo/
├── tools/          # Visualization, export
└── observability/  # Metrics, logging
```

Dependencies follow strict DAG: proto → kernels → memory → core → api → training

## Fitness Computation

```python
fitness = (
    gradient_magnitude * 0.7 +       # Task relevance
    CA_mass_conservation * 0.2 +     # CA substrate quality
    compute_efficiency * 0.1         # Hardware utilization
)
```

Relative fitness (z-score vs k-nearest neighbors) computed in parallel via GPU stencil operations. Components with low relative fitness culled every 1000 steps.

## Multi-GPU

Device placement via centroid hash:

```python
centroid_id = find_nearest_centroid(component.behavior(), centroids)
device_id = hash(centroid_id) % num_gpus
```

Similar behaviors → same centroid → same GPU. Deterministic placement, no coordination overhead. GPU failure → automatic redistribution via same hash function.

## Limitations

- Requires CUDA-capable GPU (tested on RTX 3060+)
- Windows: Use `triton-windows` (bundled TinyCC compiler)
- Archive memory: 1000 centroids × D² × 4 bytes, but low-rank (k=64) reduces by 8x
- Lifecycle adds overhead vs static transformer
- DIRESA embeddings must be factorable (validated via KMO test)

## References

- **MAP-Elites:** Mouret & Clune (2015). "Illuminating search spaces by mapping elites" arXiv:1504.04909
- **CVT-MAP-Elites:** Vassiliades et al. (2018). IEEE Trans. Evolutionary Computation 22(4), 623-630
- **Flow-Lenia:** Randazzo et al. (2023). "Flow-Lenia: Towards open-ended evolution in cellular automata" arXiv:2212.07906
- **Neural CA:** Béna (2025). "A Path to Universal Neural Cellular Automata" arXiv:2505.13058
- **DIRESA:** Zhang et al. (2025). "Distance-preserving nonlinear dimension reduction" arXiv:2404.18314
- **Curiosity:** Gottlieb & Oudeyer (2021). "Humans monitor learning progress in curiosity-driven exploration" Nature Communications 12:5972
- **HyperNetworks:** Ha et al. (2017). ICLR 2017. arXiv:1609.09106

## License

MIT
