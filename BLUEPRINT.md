# Slime Mold Transformer - System Blueprint

## Architecture Philosophy

1. **Protocol-First**: All interfaces defined before implementations
2. **Dynamic Everything**: No static allocations, lifecycle-managed components
3. **MAP-Elites Core**: Archive-driven evolution and bootstrapping
4. **SRE Built-In**: Observability, SLOs, error budgets from day one
5. **GPU-Native**: 100% device execution, zero CPU synchronization
6. **DRY Principle**: Single source of truth for each concept

## Dependency DAG (Strict Hierarchy)

```
Layer 0: Protocols (no dependencies)
    proto/kernel.py
    proto/memory.py
    proto/model.py
    proto/component.py  # NEW: Component lifecycle interface

Layer 1: Implementations of protocols (depend only on Layer 0)
    kernels/utils.py
    kernels/triton_impl.py → proto.kernel
    kernels/torch_fallback.py → proto.kernel
    observability/metrics.py → (no proto, pure collection)
    observability/slo.py → (no proto, pure validation)

Layer 2: Data structures (depend on Layer 0-1)
    memory/archive.py → proto.component
    memory/pool.py → proto.component
    memory/tubes.py → proto.memory
    core/state.py → (no dependencies, plain dataclass)

Layer 3: Components (depend on Layer 0-2)
    core/pseudopod.py → proto.model, proto.kernel, kernels/*, observability/*
    core/chemotaxis.py → proto.model, memory/archive

Layer 4: Orchestration (depend on Layer 0-3)
    core/organism.py → proto.model, core/pseudopod, core/chemotaxis, memory/*, observability/*

Layer 5: API (depend on Layer 0-4)
    api/torch_compat.py → core/organism
    api/native.py → core/organism

Layer 6: Applications (depend on Layer 0-5)
    training/*
    bench/*
    tools/*
    config/* → (reads Layer 5)
```

## Corrected Data Flow

```
User Input
    │
    ▼
┌─────────────────────┐
│  API Layer          │
│  (torch_compat,     │
│   native)           │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Organism           │ owns
│  (Orchestrator)     ├──────┐
└──────┬──────────────┘      │
       │                     │
       │ uses                │
       ▼                     ▼
┌─────────────────┐   ┌─────────────┐
│ Pseudopod Pool  │   │  Archive    │
│ (Dynamic)       │   │ (MAP-Elites)│
└──────┬──────────┘   └──────▲──────┘
       │                     │
       │ delegates           │
       ▼                     │ stores
┌─────────────────┐          │
│ Pseudopod       │──────────┘
│ (Component)     │ adds self
└──────┬──────────┘
       │
       │ calls
       ▼
┌─────────────────┐
│ Kernels         │
│ (GPU compute)   │
└─────────────────┘
       │
       │ metrics collected by
       ▼
┌─────────────────┐
│ Observability   │
│ (side channel)  │
└─────────────────┘
```

**Key Fix**: No cycles. Archive doesn't call anything. Observability is passive collector.

## Protocol Corrections

### proto.component.Component
```python
"""NEW: Base component protocol for pool management"""

Protocol:
  - fitness: float (property)
  - reset() -> None
  - to_dict() -> dict
  - from_dict(data: dict) -> Component

Purpose:
  - Unified interface for ALL pooled components
  - Archive stores any Component
  - Pools manage any Component
```

### proto.memory.Memory
```python
"""Temporal memory interface (NOT for pools)"""

Protocol:
  - store(data: Tensor, weight: float) -> None
  - recall() -> Optional[Tensor]
  - clear() -> None

Implementations:
  - memory.tubes.TubeNetwork (exponential decay)

Purpose:
  - ONLY for temporal memory with decay
  - NOT for component lifecycle (that's pool.py)
```

### proto.model.Pseudopod
```python
"""Sensory probe interface"""

Protocol:
  - forward(latent, stimulus) -> output
  - correlation: Tensor (property)
  - effective_rank() -> Tensor
  - coherence() -> Tensor

Implementations:
  - core.pseudopod.Pseudopod

Dependencies:
  - MUST use proto.kernel.Kernel for all compute
  - MUST implement proto.component.Component
```

### proto.model.Chemotaxis
```python
"""Behavioral space navigator"""

Protocol:
  - add_source(nutrient, location, concentration) -> None
  - sample(behavior, metabolic_rate, hunger) -> Optional[Tensor]
  - clear() -> None

Implementations:
  - core.chemotaxis.Chemotaxis

Dependencies:
  - Uses memory.archive for spatial indexing
  - NO direct component management
```

### proto.model.Organism
```python
"""Top-level orchestrator"""

Protocol:
  - forward(stimulus, state) -> (output, new_state)
  - reset_state() -> None
  - stats() -> dict

Implementations:
  - core.organism.Organism

Dependencies:
  - Owns: Pool[Pseudopod], Archive, Chemotaxis
  - Uses: Kernels via Pseudopods
  - Records: Observability metrics
```

## File Structure (Corrected)

```
slime/
├── proto/
│   ├── __init__.py
│   ├── component.py        # NEW: Base component interface
│   ├── kernel.py           # ✓ Kernel compute interface
│   ├── memory.py           # ✓ Temporal memory interface (tubes only)
│   └── model.py            # ✓ Model component interfaces
│
├── kernels/
│   ├── __init__.py
│   ├── utils.py            # ✓ Validation utilities
│   ├── triton_impl.py      # Triton kernels
│   └── torch_fallback.py   # PyTorch fallback
│
├── observability/
│   ├── __init__.py
│   ├── metrics.py          # Passive metrics collector
│   ├── slo.py              # SLO definitions
│   └── tracing.py          # Distributed tracing
│
├── memory/
│   ├── __init__.py
│   ├── archive.py          # ✓ MAP-Elites (stores Component protocol)
│   ├── pool.py             # ✓ Dynamic pools (manages Component protocol)
│   └── tubes.py            # TubeNetwork (implements Memory protocol)
│
├── core/
│   ├── __init__.py
│   ├── state.py            # ✓ FlowState dataclass
│   ├── pseudopod.py        # ✓ Pseudopod (implements Component + Model.Pseudopod)
│   ├── chemotaxis.py       # Chemotaxis (implements Model.Chemotaxis)
│   └── organism.py         # ✓ Organism (implements Model.Organism)
│
├── api/
│   ├── __init__.py
│   ├── torch_compat.py     # ✓ SlimeMoldEncoder
│   └── native.py           # ✓ SlimeModel
│
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   ├── losses.py
│   └── optimizer.py
│
├── config/
│   ├── __init__.py
│   ├── loader.py           # ✓ (partial)
│   ├── model.yaml          # ✓ (partial)
│   ├── training.yaml
│   └── slo.yaml
│
├── bench/
│   ├── __init__.py
│   ├── datasets.py         # ✓ (partial)
│   ├── transformer.py
│   └── profile.py
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   └── slo/
│
└── tools/
    ├── __init__.py
    ├── visualize.py
    ├── export.py
    └── package.py
```

## Critical Invariants (Refined)

### 1. Dependency Direction (DAG Enforcement)
- Lower layers NEVER import from higher layers
- Protocols NEVER import implementations
- Components NEVER import API layer
- **Violation = compilation error**

### 2. Ownership Hierarchy
```
Organism owns:
  - Pool[Pseudopod]
  - Archive
  - Chemotaxis

Pool owns:
  - List[Component]

Archive owns:
  - Dict[cell, Elite] where Elite.genome = dict (NO object refs)

NO CYCLES
```

### 3. Protocol Compliance
- Every component MUST declare which protocols it implements
- Archive stores via `Component.to_dict()` (immutable serialization)
- Pools manage via `Component.fitness` (no type checking at runtime)

### 4. GPU Memory Safety
- Kernels check allocation before launch
- Organism enforces total memory budget
- Pool culling triggered by OOM

### 5. Observability Injection
- Metrics collector passed to Organism.__init__()
- All forward passes record to metrics
- NO global state for metrics

## Implementation Checklist

### MUST FIX NOW:
- [ ] Add proto/component.py (Component protocol)
- [ ] Fix memory/archive.py to use Component protocol
- [ ] Fix memory/pool.py to use Component protocol
- [ ] Fix core/pseudopod.py to inject Kernel via __init__
- [ ] Add core/chemotaxis.py
- [ ] Add memory/tubes.py
- [ ] Fix observability/* to be injected, not global

### Phase 1 Completion:
- [ ] kernels/triton_impl.py
- [ ] kernels/torch_fallback.py
- [ ] observability/* (all 3 files)

### Phase 2:
- [ ] training/* (all 3 files)
- [ ] config/* (complete YAML schemas)

### Phase 3:
- [ ] tests/unit/* (all components)
- [ ] tests/integration/*
- [ ] bench/transformer.py

### Phase 4:
- [ ] tools/* (all 3 files)
- [ ] setup.py
- [ ] README.md

## Open Questions (Refined)

1. **Kernel injection**: Constructor injection or factory pattern?
2. **Multi-GPU**: Partition archive by behavioral space hash?
3. **Determinism**: Archive dict ordering via OrderedDict or sort keys?
4. **Memory limits**: Hard limit or soft limit with culling?
5. **Metrics injection**: Pass collector to each component or singleton?
