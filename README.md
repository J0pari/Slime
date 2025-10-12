# Slime Mold Transformer

Neural network with dynamic component lifecycle. Components compete for survival based on gradient contribution. Archive maintains behavioral diversity via MAP-Elites.

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
  grid_size: 10
  behavioral_bounds: [[0.0, 1.0], [0.0, 1.0]]

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

Training has three phases:

1. **Warmup (steps 0-1000):** Static pool, no lifecycle
2. **Gentle (steps 1000-5000):** Allow births, no deaths
3. **Full (steps 5000+):** All dynamics enabled

If loss exceeds 10x moving average, lifecycle freezes automatically.

## Architecture Analogy

Think of a slime mold foraging for food. It extends pseudopods (components) in different directions. Successful pseudopods (high fitness) persist. Unsuccessful ones retract (culling). The organism remembers successful patterns (archive) and reuses them when exploring new areas.

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
- Verify efficiency is included in fitness: see `training/fitness.py`
- Increase `grid_size` in archive config

**"Triton kernel errors"**
- Verify CUDA version matches: `nvidia-smi`
- Reinstall triton-windows: `pip install --force-reinstall triton-windows`
- Fall back to PyTorch: `model = SlimeMoldEncoder(..., use_triton=False)`

## Computational Cost

Overhead vs baseline transformer: ~1-2% per training step

- Forward/backward: Standard O(B * M * D²)
- Fitness computation: O(P * D) where P = num_pseudopods
- Archive update (1/100 steps): O(P)
- Lifecycle decisions (1/1000 steps): O(P)

Typical P=8-32, so overhead is negligible.

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

Archive grid cells correspond to behavioral coordinates. Example with 2D behavioral space:

- **Dimension 0:** Attention distance (0.0 = local, 1.0 = global)
- **Dimension 1:** Activation sparsity (0.0 = dense, 1.0 = sparse)

Component at (0.1, 0.2) specializes in short-range dense patterns. Component at (0.9, 0.1) specializes in long-range dense patterns. Archive maintains best component for each cell.

When spawning new component, sample from archive cells near desired behavior. This provides warm initialization based on proven patterns.

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

Device placement via behavioral hash:

```python
device_id = hash(behavior_coords) % num_gpus
```

Components with similar behavior land on same GPU. Deterministic placement means no coordination overhead. If GPU fails, components redistribute automatically via same hash function.

## Limitations

- Requires CUDA-capable GPU (tested on RTX 3060+)
- Windows: Use `triton-windows` package (bundled TinyCC compiler)
- Archive memory scales with grid_size^behavioral_dims (keep dims ≤ 3)
- Component lifecycle adds ~1000 lines of complexity vs standard transformer

## References

- MAP-Elites: Mouret & Clune, "Illuminating the Search Space" (2015)
- Flash Attention: Dao et al., "FlashAttention" (2022)
- Triton: Tillet et al., "Triton: GPU Programming for Neural Networks" (2019)

## License

MIT
