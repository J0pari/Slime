# Slime Mold Transformer - System Blueprint

## Architecture Philosophy

1. **Protocol-First**: All interfaces defined before implementations
2. **Dynamic Everything**: No static allocations, lifecycle-managed components
3. **MAP-Elites Core**: Archive-driven evolution and bootstrapping
4. **SRE Built-In**: Observability, SLOs, error budgets from day one
5. **GPU-Native**: 100% device execution, zero CPU synchronization
6. **DRY Principle**: Single source of truth for each concept

## Complete File Structure

```
slime/
├── __init__.py                    # Public API exports
│
├── proto/                         # Protocol definitions (interfaces)
│   ├── __init__.py
│   ├── kernel.py                  # Kernel protocol (attention, correlation, rank)
│   ├── memory.py                  # Memory protocol (store, recall)
│   └── model.py                   # Model protocols (Organism, Pseudopod, Chemotaxis)
│
├── kernels/                       # GPU kernel implementations
│   ├── __init__.py
│   ├── utils.py                   # ✓ Validation, grid config, safety
│   ├── triton_impl.py             # Triton kernels (implements proto.kernel.Kernel)
│   │   - fused_attention_kernel
│   │   - correlation_kernel
│   │   - effective_rank_kernel
│   └── torch_fallback.py          # CPU/PyTorch fallback (same interface)
│
├── memory/                        # Memory and lifecycle systems
│   ├── __init__.py
│   ├── archive.py                 # ✓ MAP-Elites behavioral archive
│   ├── pool.py                    # ✓ Dynamic component pools with apoptosis
│   └── tubes.py                   # Tube network (temporal memory with decay)
│
├── core/                          # Core components
│   ├── __init__.py
│   ├── state.py                   # ✓ FlowState dataclass
│   ├── pseudopod.py               # ✓ Pseudopod implementation
│   ├── organism.py                # ✓ Main Organism (Plasmodium)
│   └── chemotaxis.py              # FoodSource (behavioral space navigation)
│
├── observability/                 # SRE infrastructure
│   ├── __init__.py
│   ├── metrics.py                 # MetricsCollector (latency, throughput, memory)
│   ├── slo.py                     # SLO definitions and error budgets
│   └── tracing.py                 # Distributed tracing (span context)
│
├── api/                           # External interfaces
│   ├── __init__.py
│   ├── torch_compat.py            # ✓ SlimeMoldEncoder (nn.TransformerEncoder API)
│   └── native.py                  # ✓ SlimeModel (direct API)
│
├── training/                      # Training infrastructure
│   ├── __init__.py
│   ├── trainer.py                 # Training loop with observability
│   ├── losses.py                  # Multi-objective losses
│   └── optimizer.py               # Optimizer factory with metabolic rate
│
├── config/                        # Configuration schemas
│   ├── __init__.py
│   ├── loader.py                  # YAML config loader with validation
│   ├── model.yaml                 # Model architecture defaults
│   ├── training.yaml              # Training hyperparameters
│   ├── slo.yaml                   # SLO definitions
│   └── deployment.yaml            # Deployment settings
│
├── bench/                         # Benchmarking
│   ├── __init__.py
│   ├── transformer.py             # Benchmark vs standard Transformer
│   ├── datasets.py                # Data loaders (CIFAR10, ImageNet, WikiText)
│   └── profile.py                 # GPU profiling utilities
│
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_archive.py
│   │   ├── test_pool.py
│   │   ├── test_pseudopod.py
│   │   ├── test_organism.py
│   │   └── test_kernels.py
│   ├── integration/
│   │   ├── test_api.py
│   │   ├── test_training.py
│   │   └── test_end_to_end.py
│   └── slo/
│       └── test_slo_compliance.py
│
└── tools/                         # Utilities
    ├── __init__.py
    ├── visualize.py               # Real-time CUDA-GL visualization
    ├── export.py                  # Model export (ONNX, TorchScript)
    └── package.py                 # Windows .exe builder

# Root level files
├── setup.py                       # Package installation
├── requirements.txt               # Python dependencies
├── README.md                      # User documentation
├── BLUEPRINT.md                   # This file
├── .gitignore                     # ✓
└── pyproject.toml                 # Modern Python packaging
```

## Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                         proto/*                             │
│          (All interfaces, no dependencies)                  │
└─────────────────────────────────────────────────────────────┘
                           ▲
                           │ implements
          ┌────────────────┼────────────────┐
          │                │                │
┌─────────▼─────────┐ ┌───▼────────┐ ┌────▼─────────┐
│   kernels/*       │ │  memory/*  │ │ observability│
│  (GPU primitives) │ │ (Archive,  │ │  (Metrics,   │
│                   │ │  Pools)    │ │   SLOs)      │
└─────────┬─────────┘ └────┬───────┘ └──────┬───────┘
          │                │                │
          └────────────────┼────────────────┘
                           │ uses
                    ┌──────▼────────┐
                    │    core/*     │
                    │  (Pseudopod,  │
                    │   Organism)   │
                    └──────┬────────┘
                           │ wraps
                    ┌──────▼────────┐
                    │    api/*      │
                    │ (Public APIs) │
                    └──────┬────────┘
                           │ used by
          ┌────────────────┼────────────────┐
          │                │                │
    ┌─────▼──────┐  ┌─────▼─────┐   ┌─────▼─────┐
    │ training/* │  │  bench/*  │   │  tools/*  │
    └────────────┘  └───────────┘   └───────────┘
```

## Interface Contracts

### proto.kernel.Kernel
```python
Protocol:
  - attention(q, k, v, temperature) -> output
  - correlation(key, value) -> matrix
  - effective_rank(matrix) -> scalar

Implementations:
  - kernels.triton_impl.TritonKernel (GPU)
  - kernels.torch_fallback.TorchKernel (CPU)

Contract:
  - All tensors must be contiguous
  - Device must match across inputs
  - Returns same device/dtype as input
```

### proto.memory.Memory
```python
Protocol:
  - store(data, weight) -> None
  - recall() -> Optional[Tensor]
  - clear() -> None
  - __len__() -> int

Implementations:
  - memory.tubes.TubeNetwork (temporal decay)

Contract:
  - Thread-safe storage
  - No strong references to external objects
  - Automatic capacity management
```

### proto.model.Organism
```python
Protocol:
  - forward(stimulus, state) -> (output, new_state)
  - reset_state() -> None

Implementations:
  - core.organism.Organism

Contract:
  - Stateless forward pass (state passed explicitly)
  - No side effects except metrics collection
  - GPU memory managed internally
```

## Data Flow

```
User Input
    │
    ▼
┌───────────────────────┐
│  api.SlimeMoldEncoder │
│  or api.SlimeModel    │
└──────────┬────────────┘
           │
           ▼
    ┌─────────────┐
    │  Organism   │◄────────┐
    └──────┬──────┘         │
           │                │
    ┌──────▼──────────┐     │
    │  Pseudopod Pool │     │
    │  (Dynamic)      │     │
    └──────┬──────────┘     │
           │                │
    ┌──────▼──────────┐     │
    │  Kernels        │     │
    │  (GPU)          │     │
    └──────┬──────────┘     │
           │                │
    ┌──────▼──────────┐     │
    │  Archive        │─────┘
    │  (MAP-Elites)   │  bootstraps
    └─────────────────┘
           │
           ▼
    ┌─────────────┐
    │ Observability│
    │ (Metrics)    │
    └─────────────┘
```

## Critical Invariants

### 1. No Circular References
- Archive stores **immutable snapshots** (dicts)
- Components **read** from archive, never write to themselves
- Pools **weakly reference** consumers

### 2. GPU Memory Safety
- All allocations checked before execution
- OOM triggers graceful degradation (pool culling)
- Maximum memory budget enforced at organism level

### 3. Lifecycle Management
- Every component has `fitness` property
- Automatic culling below `death_threshold`
- No component lives forever (bounded lifetime)

### 4. Observability
- Every forward pass records metrics
- SLO violations logged immediately
- Distributed tracing spans all operations

### 5. Determinism
- Given same seed, same results
- Archive iteration order stable
- Pool spawning deterministic

## Implementation Order

### Phase 1: Foundation (CURRENT)
- [x] Proto definitions
- [x] Memory (archive, pools)
- [x] Core (pseudopod, organism, state)
- [x] API (torch_compat, native)
- [ ] Kernels (triton_impl)
- [ ] Memory (tubes)
- [ ] Core (chemotaxis)

### Phase 2: Training
- [ ] Observability (metrics, slo, tracing)
- [ ] Training (trainer, losses, optimizer)
- [ ] Config (loader, YAML schemas)

### Phase 3: Testing
- [ ] Unit tests (all components)
- [ ] Integration tests
- [ ] SLO compliance tests
- [ ] Benchmarks vs Transformer

### Phase 4: Packaging
- [ ] setup.py / pyproject.toml
- [ ] Tools (visualize, export, package)
- [ ] Documentation (README, examples)
- [ ] Windows .exe distribution

## Key Dependencies

### Python Packages
```
torch >= 2.0.0
triton >= 2.1.0
numpy >= 1.24.0
pyyaml >= 6.0
pytest >= 7.4.0
prometheus-client >= 0.17.0  # Metrics
opentelemetry-api >= 1.20.0  # Tracing
```

### System Requirements
```
CUDA >= 12.0
NVIDIA GPU with compute capability >= 7.5
Windows 10/11 (for .exe packaging)
Python >= 3.10
```

## Testing Strategy

### Unit Tests
- Each component tested in isolation
- Mock dependencies via protocols
- Property-based testing (hypothesis)

### Integration Tests
- Full forward/backward passes
- Archive/pool interaction
- Multi-GPU scenarios

### SLO Tests
- Latency percentiles (p95, p99)
- Memory usage bounds
- Throughput targets

### Benchmarks
- vs nn.Transformer (speed, memory, accuracy)
- Scaling curves (batch size, seq length)
- GPU utilization metrics

## Success Metrics

### Functional
- Drop-in replacement for nn.TransformerEncoder ✓
- Training converges on standard benchmarks
- Archive coverage > 50% after training

### Performance
- p95 latency < 100ms (batch=32, seq=512)
- GPU memory < 5GB (RTX 3060)
- Throughput > 1000 samples/sec

### Quality
- All tests pass
- SLO compliance > 99%
- No memory leaks (long-running tests)

## Open Questions

1. **Spatial indexing for large pools**: Current O(n) scan, need O(log n) kd-tree?
2. **Multi-GPU partitioning**: Partition by behavioral space or random?
3. **Kernel fusion opportunities**: Can we fuse attention+correlation+rank?
4. **Archive persistence**: Store to disk? How to handle large archives?
5. **Visualization**: Real-time or post-hoc? 2D projection of behavior space?
