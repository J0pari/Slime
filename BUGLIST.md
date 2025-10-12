# BUGLIST: Implementation gaps vs BLUEPRINT.md

This document tracks where the implementation doesn't match the blueprint architecture.

## Curiosity-Driven Lifecycle (CRITICAL)

**Blueprint**: hunger = learning_progress_deficit (intrinsic motivation)
**Current Implementation**: hunger = manual schedule
**Impact**: Missing self-organizing component pool, no emergent specialization

**What needs to change**:
- coherence() metric → learning progress (Δ prediction error)
- High coherence → low hunger → survive
- Low coherence → high hunger → sample archive
- Remove manual fitness function

## Adaptive Voronoi Archive (HIGH)

**Blueprint**: Archive cells grow/shrink based on density
**Current Implementation**: Fixed CVT centroids
**Impact**: Poor coverage in sparse regions, inflexible partitioning

**What needs to change**:
- Cell density monitoring (elites per cell)
- Subdivision trigger (density > threshold → split)
- Merge trigger (density < threshold → merge neighbors)
- Lloyd's relaxation for centroid adjustment

## Comonadic GPU Orchestration (MEDIUM)

**Blueprint**: GPU execution state AS comonad (extract/extend for context-aware decisions)
**Current Implementation**: Manual Organism orchestration
**Impact**: No GPU-aware resource allocation, missing 2x hardware utilization improvement

**What needs to change**:
- GPUContext: Warp occupancy, cache hits, tensor core utilization
- extract(warp_id) → LocalObservation
- extend(decision_fn) → Apply context-aware spawn/retire decisions
- Polynesian navigator metaphor: whole field informs local decisions

## Warp-Level GPU Kernels (MEDIUM)

**Blueprint**: Zero-global-memory CA updates via warp shuffles
**Current Implementation**: Standard PyTorch/Triton kernels with global memory access
**Impact**: Missing 100x GPU utilization, no tensor core usage for convolutions

**What needs to change**:
- Warp shuffles for neighbor access (no global memory)
- Tensor cores for 16x16 matrix multiply (256 FLOPs/instruction)
- Entire CA update in registers
- CUDA kernel: `__shfl_sync()` for neighbor communication

## Algebraic Effect Handlers (LOW - Already implemented)

**Status**: Effect handlers exist for GetHierarchy, GetGenealogy, UsePAdicDistance
**Note**: Need to add GetLocalUpdateRule, GetLearningProgress for future features

## Ultrametric Topology (COMPLETE)

**Status**: True dendrogram traversal implemented, 100% constraint satisfaction
**File**: slime/topology/hierarchy.py
**Tests**: slime/tests/unit/test_topology_chemotaxis.py

---

**Priority**: CRITICAL > HIGH > MEDIUM > LOW
**Principle**: Never weaken constraints - implement correct solutions
