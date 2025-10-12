# Slime Mold Transformer

Neural network with dynamic component lifecycle. Components compete for survival based on gradient contribution. Archive maintains behavioral diversity via CVT-MAP-Elites (Centroidal Voronoi Tessellation). Uses FlashAttention-style tiled kernels and low-rank weight storage.

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

## Configuration

Edit `slime/config/model.yaml`:

```yaml
model:
  sensory_dim: 512
  latent_dim: 256
  head_dim: 64
  num_pseudopods: 8
  behavioral_dims: 2

archive:
  num_centroids: 1000  # CVT partitions, not grid cells
  behavioral_dims: 5   # Can use 4-5 dims without exponential explosion
  low_rank_k: 64       # Factorization rank for weight compression

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

- **Fast (every step):** Weight updates, gradient computation
- **Medium (every 100 steps):** Archive updates, component birth decisions
- **Slow (every 1000 steps):** Component culling, memory budget enforcement

### Safety Guardrails

Training uses simulated annealing for smooth exploration-exploitation transition:

```python
temperature = initial_temp * (1 - step / max_steps)  # Linear cooling

# Early training (high temp): accept diverse low-fitness components
# Late training (low temp): only accept high-fitness components
birth_prob = exp(-fitness_deficit / temperature)
```

If loss exceeds 10x moving average, lifecycle freezes automatically.

## Architecture Analogy

Think of a slime mold foraging for food. It extends pseudopods (components) in different directions. Successful pseudopods (high fitness) persist. Unsuccessful ones retract (culling). The organism remembers successful patterns (archive using low-rank compressed weights) and reuses them when exploring new areas.

**CVT-MAP-Elites:** Archive uses Voronoi partitioning of behavioral space (not fixed grid). Scales to 4-5 behavioral dimensions:
- Attention span (memory locality)
- Activation sparsity (compute efficiency)
- Gradient flow magnitude (task relevance)
- Memory access locality (cache friendliness)
- Computational intensity (GPU occupancy)

Key difference from standard transformers: attention heads don't have fixed roles. Components discover specializations through gradient-based selection pressure.

## Testing

```bash
# Unit tests (protocol compliance, kernel correctness)
pytest slime/tests/unit/ -v

# Integration tests (end-to-end training)
pytest slime/tests/integration/ -v

# Ablations (compare vs baseline transformer)
pytest slime/tests/ablations/ -v
```

### Critical Tests

```bash
# Verify Triton kernels work on your GPU
pytest slime/tests/unit/test_triton_kernels.py -v

# Verify DAG dependencies are not violated
pytest slime/tests/unit/test_dag_enforcement.py -v

# Verify behavioral space is factorable (KMO test)
pytest slime/tests/unit/test_behavioral_kmo.py -v
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
print(f"FLOPS: {results['estimated_flops'] / 1e9:.2f}G")
```

## Observability

```python
from slime.observability.metrics import MetricsCollector

metrics = MetricsCollector()
model = SlimeMoldEncoder(..., metrics_collector=metrics)

# After training
print(f"Pool size over time: {metrics.get('pool_size')}")
print(f"Archive coverage: {metrics.get('archive_coverage')}")
print(f"Component births: {metrics.get('component_births')}")
print(f"Component deaths: {metrics.get('component_deaths')}")
```

## Exporting

```python
from slime.tools.export import export_onnx, export_torchscript

# ONNX (for inference in C++/other languages)
export_onnx(model, output_path='model.onnx', input_shape=(1, 128, 512))

# TorchScript (for deployment without Python)
export_torchscript(model, output_path='model.pt')
```

## Troubleshooting

**"CUDA out of memory"**
- Reduce `batch_size` in training
- Reduce `max_size` in pool config
- Enable memory budget enforcement: `pool.enforce_memory_budget = True`

**"Loss diverging during training"**
- Check lifecycle is in warmup phase first 1000 steps
- Verify fitness computation uses actual gradients: `pytest slime/tests/unit/test_fitness_computation.py`
- Reduce `learning_rate`

**"Components not diversifying"**
- Check behavioral dimensions correlate with compute patterns: `pytest slime/tests/unit/test_behavioral_kmo.py`
- Verify KMO statistic > 0.6 (measures factorability of behavioral dimensions)
- Verify efficiency is included in fitness: see `training/fitness.py`
- Increase `num_centroids` in archive config (CVT partitions)

**"Triton kernel errors"**
- Verify CUDA version matches: `nvidia-smi`
- Reinstall triton-windows: `pip install --force-reinstall triton-windows`
- Fall back to PyTorch: `model = SlimeMoldEncoder(..., use_triton=False)`

## Computational Cost

**Estimated overhead vs baseline transformer: 7-15% per training step**

- Forward/backward: Standard O(B * M * D²)
- FlashAttention tiling: 2-3x speedup (Dao et al., 2022)
- Fitness computation: O(P * D) where P = num_pseudopods
- Behavioral metrics: O(P * M * D) for attention span, sparsity, etc.
- CVT archive update (1/100 steps): O(P * num_centroids) nearest centroid search
- Lifecycle decisions (1/1000 steps): O(P) with simulated annealing

**Comparison to DARTS:**
- DARTS: 4 GPU days search + N days training
- Slime: 1.15 × N days (search during training)
- Break-even: N > 30 days

Slime favors long training runs where amortized search cost is negligible.

## File Layout

```
slime/
├── proto/          # Interfaces (Component, Kernel, Memory, Model)
├── kernels/        # GPU implementations (Triton + PyTorch fallback)
├── memory/         # Archive (MAP-Elites), Pool (lifecycle), Tubes (temporal)
├── core/           # Pseudopod, Chemotaxis, Organism
├── api/            # torch_compat (nn.Module), native (SlimeModel)
├── training/       # Trainer, losses, fitness, lifecycle, stability
├── config/         # YAML loaders and schemas
├── bench/          # Datasets, profiling, baseline transformer
├── tests/          # unit/, integration/, ablations/, slo/
└── tools/          # Visualization, export, packaging
```

Dependencies follow strict DAG: proto → kernels → memory → core → api → training

## Behavioral Space

**CVT-MAP-Elites uses Voronoi partitioning**, not fixed grid. Scales to 4-5 dimensions without exponential explosion.

**5D behavioral space:**
1. **Attention span:** Mean(attention_weights × position_distance) - correlates with memory bandwidth
2. **Activation sparsity:** Fraction of activations near zero - correlates with compute efficiency
3. **Gradient flow magnitude:** L2 norm of gradients - correlates with task relevance
4. **Memory access locality:** Variance of attention positions - correlates with cache hit rate
5. **Computational intensity:** FLOPs per forward pass - correlates with GPU occupancy

**Validation:** KMO (Kaiser-Meyer-Olkin) test ensures dimensions are factorable. KMO < 0.6 = dimensions are noise, not structure.

**Example:** Component near centroid (0.1, 0.8, 0.5, 0.2, 0.3) specializes in: local attention, sparse activations, medium gradient flow, coherent memory access, low compute intensity.

When spawning new component, sample from archive near desired behavioral centroid. Low-rank factorized weights (U, V) provide warm initialization.

## Fitness Computation

```python
fitness = (
    task_performance * 0.7 +      # Does it reduce loss?
    compute_efficiency * 0.2 +     # Is it fast?
    gradient_magnitude * 0.1       # Is it relevant?
)
```

Components with low fitness get culled every 1000 steps. Archive stores high-fitness components for reuse.

## Multi-GPU

Device placement via CVT centroid hash:

```python
centroid_id = find_nearest_centroid(component.behavior(), centroids)
device_id = hash(centroid_id) % num_gpus
```

Components with similar behavior map to same centroid, land on same GPU. Deterministic placement means no coordination overhead. If GPU fails, components redistribute automatically via same hash function. Centroid-based hashing naturally clusters by compute patterns.

## Limitations

- Requires CUDA-capable GPU (tested on RTX 3060+)
- Windows: Use `triton-windows` package (bundled TinyCC compiler)
- CVT centroids scale linearly (not exponentially), but 1000 centroids × D² × 4 bytes per elite
- Low-rank storage (k=64) reduces archive memory by 8x vs full matrices
- Component lifecycle adds 7-15% training overhead vs static transformer
- Behavioral dimensions must correlate with hardware metrics (validated via KMO test)

## References

- **MAP-Elites:** Mouret, J.-B. & Clune, J. "Illuminating search spaces by mapping elites" (2015). arXiv:1504.04909
- **CVT-MAP-Elites:** Vassiliades, V., Chatzilygeroudis, K., & Mouret, J.-B. "Using Centroidal Voronoi Tessellations to Scale Up the Multidimensional Archive of Phenotypic Elites Algorithm" (2018). IEEE Trans. Evolutionary Computation, 22(4), 623-630.
- **FlashAttention:** Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022). NeurIPS 2022. arXiv:2205.14135
- **DARTS:** Liu, H., Simonyan, K., & Yang, Y. "DARTS: Differentiable Architecture Search" (2019). ICLR 2019. arXiv:1806.09055
- **HyperNetworks:** Ha, D., Dai, A., & Le, Q. V. "HyperNetworks" (2017). ICLR 2017. arXiv:1609.09106
- **Simulated Annealing:** Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. "Optimization by simulated annealing" (1983). Science, 220(4598), 671-680.

## License

MIT
