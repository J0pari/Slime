# BUGLIST: Implementation gaps vs BLUEPRINT.md

This document tracks where the implementation doesn't match the blueprint architecture.

## Neural CA Pseudopod Updates (CRITICAL)

**Blueprint**: Pseudopod.forward() is learned CA update with Flow-Lenia dynamics
**Current Implementation**: Pseudopod.forward() uses transformer attention
**Impact**: Missing mass conservation, parameter localization, warp-level GPU execution

**What needs to change**:
- Replace attention mechanism with learned CA convolution
- Add mass conservation constraint: ∑ output = ∑ input
- Implement parameter localization (spatial variation of CA rule parameters)
- Warp-level CUDA kernel (neighbors via shuffles, tensor cores for convolution)

## Curiosity-Driven Lifecycle (CRITICAL)

**Blueprint**: hunger = learning_progress_deficit (intrinsic motivation)
**Current Implementation**: hunger = manual schedule
**Impact**: Missing self-organizing component pool, no emergent specialization

**What needs to change**:
- coherence() metric → learning progress (Δ prediction error)
- High coherence → low hunger → survive
- Low coherence → high hunger → sample archive
- Remove manual fitness function

## DIRESA Learned Embeddings (CRITICAL)

**Blueprint**: DIRESA autoencoder learns behavioral embeddings online (adaptive 2-10D)
**Current Implementation**: Offline Kernel PCA discovers dimensions once
**Impact**: No online adaptation, fixed dimensionality, no distance preservation learning

**What needs to change**:
- Implement DIRESABehavioralEncoder (autoencoder + learned gating)
- Warp-native distance computation (shuffle reductions)
- Loss: Reconstruction + distance preservation + KL regularization
- Online training alongside main model

n## Validation Metrics (MEDIUM)

**Blueprint**: Trustworthiness ≥ 0.85, Continuity ≥ 0.85, Procrustes distance ≤ 0.15
**Current Implementation**: KMO ≥ 0.6, Bartlett's p < 0.05
**Impact**: Using 1970s factor analysis metrics instead of modern distance-preservation metrics

**Files affected**:
- slime/memory/archive.py (lines 338-339, 382-386, 417-429, 599-615)
- slime/core/organism.py (line 51)
- run.py (line 147)
- slime/tests/unit/test_archive.py (all CVTArchive instantiations)

**What needs to change**:
- Replace calculate_kmo() with compute_trustworthiness()
- Replace bartlett() with compute_continuity()
- Add compute_procrustes_distance()
- Update all kmo_threshold parameters to trustworthiness_threshold
- Remove factor_analyzer dependency, add scikit-learn.manifold
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
