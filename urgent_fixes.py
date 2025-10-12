# URGENT FIXES REQUIRED

## 1. Fix Component Protocol (BREAKING CHANGE)

The protocol needs to support parameterized reconstruction.

```python
# slime/proto/component.py
class Component(Protocol):
    """Component lifecycle protocol"""
    
    @property
    def fitness(self) -> float: ...
    
    def reset(self) -> None: ...
    
    def to_dict(self) -> dict: ...
    
    # REMOVED: from_dict (can't be in protocol due to varying signatures)
```

**Rationale**: Different components need different dependencies (kernel, device, etc).
Archive must handle reconstruction contextually, not via protocol.

## 2. Fix Archive Bootstrapping

```python
# slime/memory/archive.py - REPLACE bootstrap_component method

def bootstrap_component(
    self,
    component_factory: Callable[[dict], Component],  # Factory takes genome
    behavior: Tuple[float, ...],
    search_radius: float = 0.2,
) -> Optional[Component]:
    """Bootstrap new component from archive elites.
    
    Args:
        component_factory: Factory(genome_dict) -> Component
        behavior: Target behavioral location
        search_radius: Neighborhood radius to search
        
    Returns:
        Initialized component or None if no elites found
    """
    nearby = self.sample_near(behavior, search_radius)
    
    if not nearby:
        return None
    
    # Use best nearby elite
    best = max(nearby, key=lambda e: e.fitness)
    
    # Factory handles reconstruction with dependencies
    component = component_factory(best.genome)
    
    # Track with weak reference
    self._live_components.add(component)
    
    logger.debug(
        f"Bootstrapped component from elite at generation {best.generation}, "
        f"fitness={best.fitness:.4f}"
    )
    
    return component
```

## 3. Fix DynamicPool Factory

```python
# slime/memory/pool.py - REPLACE __init__

def __init__(
    self,
    component_factory: Callable[[], Component],  # No-arg factory
    bootstrap_factory: Optional[Callable[[dict], Component]] = None,  # With-genome factory
    config: PoolConfig,
    archive: Optional['BehavioralArchive'] = None,
):
    """Initialize dynamic pool.
    
    Args:
        component_factory: Factory() -> Component (for fresh spawns)
        bootstrap_factory: Factory(genome) -> Component (for archive spawns)
        config: Pool configuration
        archive: Optional archive for bootstrapping
    """
    self.factory = component_factory
    self.bootstrap_factory = bootstrap_factory or component_factory
    self.config = config
    self.archive = archive
    # ... rest unchanged
```

```python
# slime/memory/pool.py - UPDATE _spawn_component

def _spawn_component(
    self,
    behavior_location: Optional[Tuple[float, ...]] = None,
) -> Component:
    """Spawn new component, optionally bootstrapping from archive"""
    
    if self.archive is not None and behavior_location is not None:
        # Try bootstrapping from archive using bootstrap_factory
        component = self.archive.bootstrap_component(
            self.bootstrap_factory,
            behavior_location,
        )
        if component is not None:
            logger.debug("Bootstrapped component from archive")
            return component
    
    # Create from scratch
    return self.factory()
```

## 4. Fix Organism to Inject Kernel

```python
# slime/core/organism.py - UPDATE __init__

from slime.kernels.torch_fallback import TorchKernel  # Add import
from slime.core.chemotaxis import Chemotaxis  # Add import
from slime.observability.metrics import MetricsCollector  # Add import

def __init__(
    self,
    sensory_dim: int,
    latent_dim: int,
    head_dim: int,
    device: torch.device = None,
    kernel: Optional[Kernel] = None,  # NEW
    pool_config: Optional[PoolConfig] = None,
    metrics_collector: Optional[MetricsCollector] = None,  # NEW
):
    super().__init__()
    
    self.sensory_dim = sensory_dim
    self.latent_dim = latent_dim
    self.head_dim = head_dim
    self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Kernel (injected or default)
    self.kernel = kernel or TorchKernel(self.device)
    
    # Metrics (injected or null)
    self.metrics = metrics_collector
    
    # ... encode, decode, predict_rank, predict_coherence (unchanged)
    
    # MAP-Elites archive
    self.archive = BehavioralArchive(
        dimensions=['rank', 'coherence'],
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        resolution=50,
        device=self.device,
    )
    
    # Chemotaxis (NEW)
    self.chemotaxis = Chemotaxis(self.archive, self.device)
    
    # Pool config
    if pool_config is None:
        pool_config = PoolConfig(
            min_size=4,
            max_size=32,
            birth_threshold=0.8,
            death_threshold=0.1,
            cull_interval=100,
        )
    
    # Dynamic pseudopod pool with BOTH factories
    self.pseudopod_pool = DynamicPool(
        component_factory=lambda: Pseudopod(head_dim, self.kernel, self.device),
        bootstrap_factory=lambda genome: Pseudopod.from_dict(
            genome, self.kernel, self.device
        ),
        config=pool_config,
        archive=self.archive,
    )
    
    self._generation = 0
```

## 5. Fix Organism.forward to Use Metrics

```python
# slime/core/organism.py - UPDATE forward method

def forward(
    self,
    stimulus: torch.Tensor,
    state: Optional[FlowState] = None,
) -> Tuple[torch.Tensor, FlowState]:
    """Single forward pass through organism."""
    
    # Start metrics tracking
    if self.metrics:
        self.metrics.start_step()
    
    batch_size = stimulus.shape[0]
    
    # ... existing forward logic (encode, predict, extend, merge)
    
    # End metrics tracking
    if self.metrics:
        self.metrics.end_step(
            batch_size=batch_size,
            pool_size=self.pseudopod_pool.size(),
            archive_size=self.archive.size(),
            archive_coverage=self.archive.coverage(),
            loss=None,  # Loss computed externally during training
        )
    
    # ... return output, new_state
```

## 6. Update API Layers

```python
# slime/api/torch_compat.py - UPDATE __init__

from slime.kernels.torch_fallback import TorchKernel  # Add import

def __init__(
    self,
    d_model: int,
    nhead: int = 8,
    # ... other args
    pool_config: Optional[PoolConfig] = None,
    kernel: Optional[Kernel] = None,  # NEW
):
    # ... setup
    
    # Create kernel if not provided
    if kernel is None:
        kernel = TorchKernel(self.device)
    
    # Internal organism with kernel
    self.organism = Organism(
        sensory_dim=d_model,
        latent_dim=d_model,
        head_dim=head_dim,
        device=self.device,
        kernel=kernel,  # PASS IT
        pool_config=pool_config,
    )
```

```python
# slime/api/native.py - Same changes as above
```

## TESTING CHECKLIST

After these fixes:

1. **Import Test**: `python -c "from slime import SlimeMoldEncoder"`
2. **Instantiation Test**: 
   ```python
   from slime import SlimeMoldEncoder
   model = SlimeMoldEncoder(d_model=128, nhead=4)
   print("✓ Model created")
   ```
3. **Forward Test**:
   ```python
   import torch
   x = torch.randn(2, 10, 128)  # batch=2, seq=10, d=128
   y = model(x)
   print(f"✓ Forward pass: {y.shape}")
   ```
4. **Archive Bootstrap Test**: Create pseudopod, archive it, bootstrap from archive

## PRIORITY ORDER

1. Fix Component protocol (remove from_dict)
2. Fix Archive.bootstrap_component (take factory)
3. Fix Pool factories (separate fresh vs bootstrap)
4. Fix Organism (inject kernel, metrics, chemotaxis)
5. Fix APIs (pass kernel)
6. Run tests

**DO THESE IN ORDER. Each depends on the previous.**
