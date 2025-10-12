# Topology Module Integration Plan

## Objective
Integrate `slime/topology/` module into the existing architecture with **Kleisli rigor** and **DAG clarity**, ensuring:
- No random edits or silos
- Natural emergence from existing structures
- Proper dependency ordering
- Composable, category-theoretic design

---

## Current Architecture Analysis

### Dependency DAG (Layer-by-Layer)
```
Layer 0: Protocols (proto/)
    ↓
Layer 1: Kernels + Observability (kernels/, observability/)
    ↓
Layer 2: Data Structures (memory/, core/state, core/stencil)
    ├── CVTArchive (memory/archive.py)
    ├── DynamicPool (memory/pool.py)
    ├── TubeNetwork (memory/tubes.py)
    └── SpatialStencil (core/stencil.py)
    ↓
Layer 3: Components (core/)
    ├── Pseudopod (core/pseudopod.py)
    └── Chemotaxis (core/chemotaxis.py)
    ↓
Layer 4: Orchestration (core/)
    └── Organism (core/organism.py)
    ↓
Layer 5: API (api/)
    ├── SlimeMoldEncoder (api/torch_compat.py)
    └── SlimeModel (api/native.py)
    ↓
Layer 6: Training + Tools (training/, tools/, bench/, config/)
```

### Natural Integration Points (Where Topology Already Exists)

1. **CVTArchive (Layer 2)** - Content-addressable storage with SHA256 hashes
   - Already has `object_store: Dict[str, bytes]` with 256-ary tree structure
   - Hash deduplication via `_hash_object()` and `_write_object()`
   - **Topology emerges**: Common prefix length → p-adic distance

2. **DynamicPool (Layer 2)** - Pseudopod spawning and culling
   - Already has `_spawn_component()` creating parent-child relationships
   - No genealogy tracking currently
   - **Topology emerges**: Spawning tree → ultrametric distance

3. **Chemotaxis (Layer 3)** - Behavioral navigation
   - Already uses distance metrics (Euclidean/Mahalanobis/Manhattan/Cosine)
   - No hierarchical clustering currently
   - **Topology emerges**: GMM clusters → hybrid metric (ultrametric between, Mahalanobis within)

4. **Archive Dimension Discovery (Layer 2)** - Kernel PCA filtration
   - Already has `discover_dimensions()` filtering 62D → ~12D → 3-5D
   - No multi-resolution awareness currently
   - **Topology emerges**: Filtration levels → p-adic valuation

---

## Proposed Module Structure

### `slime/topology/` (New Layer 2.5 - Between Data Structures and Components)

```
slime/topology/
├── __init__.py              # Exports: p_adic_distance, Genealogy, BehavioralHierarchy, HybridMetric
├── p_adic.py                # p-Adic distance and valuations
├── genealogy.py             # Pseudopod phylogenetic tracking
├── hierarchy.py             # GMM-based behavioral clustering
└── hybrid_metric.py         # Composable distance metric interface
```

### Dependencies (Kleisli-Style)
```
topology/ depends on:
    - memory/archive.py (for SHA256 hashes, centroid data)
    - memory/pool.py (for pseudopod references)
    - numpy, scipy (for GMM, dendrogram)
    - NO dependency on core/ or higher layers

core/chemotaxis.py can optionally depend on:
    - topology/hybrid_metric.py (for distance computation)

memory/archive.py can optionally depend on:
    - topology/p_adic.py (for hash-based deduplication)

memory/pool.py can optionally depend on:
    - topology/genealogy.py (for lineage tracking)
```

**Kleisli Property**: Each module is a functor `Maybe T → Maybe T'` - can fail gracefully if topology features not needed.

---

## Implementation Plan (Step-by-Step)

### Phase 1: Pure Functions (No State Mutation)

#### Step 1.1: Create `slime/topology/p_adic.py`
```python
"""p-Adic distance metrics for hierarchical structures."""
import numpy as np
from typing import Tuple

def p_adic_distance(x: str, y: str, p: int = 256) -> float:
    """
    Distance = p^(-n) where n = common prefix length.

    Args:
        x, y: Strings (hex hashes or any prefix-comparable data)
        p: Base (256 for SHA256 first byte, 2 for binary)

    Returns:
        Distance in [0, 1], where 0 = identical, 1 = no common prefix
    """
    n = 0
    for a, b in zip(x, y):
        if a == b:
            n += 1
        else:
            break
    return p ** (-n) if n > 0 else 1.0

def p_adic_valuation(x: int, p: int = 2) -> int:
    """
    Valuation v_p(x) = max{k : p^k divides x}.

    Used for metric refinement: higher valuation = coarser resolution.
    """
    if x == 0:
        return float('inf')
    v = 0
    while x % p == 0:
        x //= p
        v += 1
    return v

def ultrametric_from_tree(lineage_a: Tuple[int, ...], lineage_b: Tuple[int, ...]) -> int:
    """
    Ultrametric distance on tree = depth to MRCA.

    Args:
        lineage_a, lineage_b: Paths from root (e.g., (0, 3, 7) = node 7 with ancestors 3, 0)

    Returns:
        Distance = sum of depths - 2*depth(MRCA)
    """
    mrca_depth = 0
    for i, (a, b) in enumerate(zip(lineage_a, lineage_b)):
        if a == b:
            mrca_depth = i + 1
        else:
            break
    return len(lineage_a) + len(lineage_b) - 2 * mrca_depth
```

**Justification**: Pure functions with no dependencies. Testable, composable, no side effects.

#### Step 1.2: Create `slime/topology/genealogy.py`
```python
"""Pseudopod phylogenetic tree tracking."""
from typing import Dict, Tuple, Optional
import weakref

class Genealogy:
    """
    Tracks pseudopod spawning lineages.

    Maintains parent-child relationships and computes ultrametric distances.
    """

    def __init__(self):
        self._lineages: Dict[int, Tuple[int, ...]] = {}  # pod_id → (ancestor_path)
        self._children: Dict[int, list] = {}  # parent_id → [child_ids]
        self._next_id = 0

    def register_genesis(self, pod_id: int) -> None:
        """Register a genesis pseudopod (no parent)."""
        self._lineages[pod_id] = (pod_id,)
        self._children[pod_id] = []

    def register_spawn(self, parent_id: int, child_id: int) -> None:
        """Register child spawned from parent."""
        if parent_id not in self._lineages:
            raise ValueError(f"Parent {parent_id} not in genealogy")

        parent_lineage = self._lineages[parent_id]
        self._lineages[child_id] = parent_lineage + (child_id,)
        self._children[child_id] = []

        if parent_id in self._children:
            self._children[parent_id].append(child_id)
        else:
            self._children[parent_id] = [child_id]

    def get_lineage(self, pod_id: int) -> Tuple[int, ...]:
        """Get full lineage path from root."""
        return self._lineages.get(pod_id, ())

    def ultrametric_distance(self, pod_a: int, pod_b: int) -> int:
        """Compute ultrametric distance via MRCA depth."""
        from slime.topology.p_adic import ultrametric_from_tree

        lineage_a = self.get_lineage(pod_a)
        lineage_b = self.get_lineage(pod_b)

        if not lineage_a or not lineage_b:
            return float('inf')  # One or both not registered

        return ultrametric_from_tree(lineage_a, lineage_b)

    def diversity_score(self) -> float:
        """
        Phylogenetic diversity = average pairwise ultrametric distance.

        Encourages exploration across lineages, not just behavioral space.
        """
        if len(self._lineages) < 2:
            return 0.0

        total_dist = 0
        count = 0
        ids = list(self._lineages.keys())

        for i, id_a in enumerate(ids):
            for id_b in ids[i+1:]:
                total_dist += self.ultrametric_distance(id_a, id_b)
                count += 1

        return total_dist / count if count > 0 else 0.0
```

**Justification**: Pure data structure. No dependency on DynamicPool yet. Can be tested standalone.

#### Step 1.3: Create `slime/topology/hierarchy.py`
```python
"""GMM-based hierarchical behavioral space clustering."""
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram
from typing import Optional

class BehavioralHierarchy:
    """
    Learns hierarchical structure from behavioral space via GMM + agglomerative clustering.

    Provides:
    - Cluster assignments for behavioral coordinates
    - Dendrogram for inter-cluster ultrametric distances
    - Covariance matrices for intra-cluster Mahalanobis distances
    """

    def __init__(self, n_clusters: int = 10, linkage_method: str = 'ward', random_state: int = 42):
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.random_state = random_state

        self.gmm: Optional[GaussianMixture] = None
        self.linkage_matrix: Optional[np.ndarray] = None
        self._cluster_means: Optional[np.ndarray] = None

    def fit(self, behavioral_coords: np.ndarray) -> None:
        """
        Fit GMM to behavioral coordinates and build dendrogram.

        Args:
            behavioral_coords: (N, D) array of behavioral space coordinates
        """
        # Fit GMM
        self.gmm = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type='full',
            random_state=self.random_state
        )
        self.gmm.fit(behavioral_coords)

        # Build dendrogram from cluster means
        self._cluster_means = self.gmm.means_
        self.linkage_matrix = linkage(self._cluster_means, method=self.linkage_method)

    def predict_cluster(self, x: np.ndarray) -> int:
        """Predict cluster assignment for behavioral coordinate."""
        if self.gmm is None:
            raise RuntimeError("Must call fit() before predict_cluster()")
        return self.gmm.predict(x.reshape(1, -1))[0]

    def get_covariance(self, cluster_id: int) -> np.ndarray:
        """Get covariance matrix for cluster (for Mahalanobis distance)."""
        if self.gmm is None:
            raise RuntimeError("Must call fit() before get_covariance()")
        return self.gmm.covariances_[cluster_id]

    def cluster_distance(self, cluster_a: int, cluster_b: int) -> float:
        """
        Ultrametric distance between clusters via dendrogram.

        Returns merge height from linkage matrix.
        """
        if self.linkage_matrix is None:
            raise RuntimeError("Must call fit() before cluster_distance()")

        if cluster_a == cluster_b:
            return 0.0

        # Find merge height in dendrogram (simplified - full implementation needs dendrogram traversal)
        # For now, use Euclidean distance between cluster means as proxy
        dist = np.linalg.norm(self._cluster_means[cluster_a] - self._cluster_means[cluster_b])
        return dist
```

**Justification**: Uses sklearn (already in dependencies). Pure transformer - takes data, returns structure. No mutation of archive or pool.

#### Step 1.4: Create `slime/topology/hybrid_metric.py`
```python
"""Composable distance metrics with hierarchical awareness."""
import numpy as np
from typing import Optional, Literal
from slime.topology.hierarchy import BehavioralHierarchy

DistanceMode = Literal['euclidean', 'manhattan', 'mahalanobis', 'cosine', 'hybrid']

class HybridMetric:
    """
    Composable distance metric that combines:
    - Ultrametric (between clusters)
    - Mahalanobis (within clusters)

    Falls back to Euclidean if no hierarchy provided.
    """

    def __init__(
        self,
        hierarchy: Optional[BehavioralHierarchy] = None,
        inter_cluster_mode: Literal['ultrametric', 'euclidean'] = 'ultrametric',
        intra_cluster_mode: Literal['mahalanobis', 'euclidean'] = 'mahalanobis',
        p: int = 2  # Base for p-adic distance
    ):
        self.hierarchy = hierarchy
        self.inter_cluster_mode = inter_cluster_mode
        self.intra_cluster_mode = intra_cluster_mode
        self.p = p

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute hybrid distance between two behavioral coordinates.

        Args:
            x, y: (D,) behavioral coordinates

        Returns:
            Distance respecting hierarchical structure if available
        """
        if self.hierarchy is None:
            # Fallback to Euclidean
            return np.linalg.norm(x - y)

        cluster_x = self.hierarchy.predict_cluster(x)
        cluster_y = self.hierarchy.predict_cluster(y)

        if cluster_x == cluster_y:
            # Same cluster: use intra-cluster metric
            if self.intra_cluster_mode == 'mahalanobis':
                cov = self.hierarchy.get_covariance(cluster_x)
                diff = x - y
                return np.sqrt(diff @ np.linalg.inv(cov) @ diff)
            else:
                return np.linalg.norm(x - y)
        else:
            # Different clusters: use inter-cluster metric
            if self.inter_cluster_mode == 'ultrametric':
                return self.hierarchy.cluster_distance(cluster_x, cluster_y)
            else:
                return np.linalg.norm(x - y)

    def batch_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorized distance computation for batches."""
        if self.hierarchy is None:
            return np.linalg.norm(x - y, axis=1)

        # TODO: Vectorize cluster prediction and distance computation
        return np.array([self(xi, yi) for xi, yi in zip(x, y)])
```

**Justification**: Composable functor. Takes optional hierarchy, returns distance function. Pure, testable, no side effects.

---

### Phase 2: Integration Points (Dependency Injection)

#### Step 2.1: Modify `memory/pool.py` to Track Genealogy (OPTIONAL)
```python
# In DynamicPool.__init__():
from slime.topology.genealogy import Genealogy

def __init__(self, ..., track_genealogy: bool = False):
    # ... existing code ...
    self.genealogy = Genealogy() if track_genealogy else None

    # Register initial pseudopods as genesis
    if self.genealogy:
        for i, comp in enumerate(self._components):
            self.genealogy.register_genesis(id(comp))

# In DynamicPool._spawn_component():
def _spawn_component(self, behavior_location, parent_id: Optional[int] = None):
    component = # ... existing spawn logic ...

    if self.genealogy and parent_id is not None:
        self.genealogy.register_spawn(parent_id, id(component))
    elif self.genealogy:
        self.genealogy.register_genesis(id(component))

    return component

# Add diversity metric:
def phylogenetic_diversity(self) -> float:
    if self.genealogy:
        return self.genealogy.diversity_score()
    return 0.0
```

**Justification**: Backward-compatible (default `track_genealogy=False`). No breaking changes. Kleisli: `Maybe Genealogy → Maybe float`.

#### Step 2.2: Modify `core/chemotaxis.py` to Support Hybrid Metrics (OPTIONAL)
```python
# In Chemotaxis.__init__():
from slime.topology.hybrid_metric import HybridMetric, DistanceMode

def __init__(self, archive, device, distance_metric: Union[DistanceMode, HybridMetric] = None, hierarchy: Optional[BehavioralHierarchy] = None):
    # ... existing code ...

    if isinstance(distance_metric, HybridMetric):
        self._metric_fn = distance_metric
    elif distance_metric == 'hybrid' and hierarchy is not None:
        self._metric_fn = HybridMetric(hierarchy=hierarchy)
    else:
        # Existing logic for euclidean/manhattan/mahalanobis/cosine
        self._metric_fn = None  # Use existing _compute_distance()

# In Chemotaxis._compute_distance():
def _compute_distance(self, pos1, pos2):
    if self._metric_fn is not None:
        return self._metric_fn(pos1, pos2)

    # Existing metric logic
    if self.distance_metric == 'euclidean':
        ...
```

**Justification**: Backward-compatible. Existing code unchanged. New `hybrid` mode is opt-in.

#### Step 2.3: Modify `memory/archive.py` to Use p-Adic Deduplication (OPTIONAL)
```python
# In CVTArchive:
from slime.topology.p_adic import p_adic_distance

def find_similar_elites(self, elite_sha: str, threshold: float = 1e-4) -> List[str]:
    """
    Find elites with similar hashes using p-adic distance.

    Useful for near-duplicate detection and delta compression optimization.
    """
    similar = []
    for existing_sha in self.object_store.keys():
        if p_adic_distance(elite_sha, existing_sha, p=256) < threshold:
            similar.append(existing_sha)
    return similar
```

**Justification**: New method, no mutation of existing logic. Can be used for future optimization.

#### Step 2.4: Add Hierarchy Learning to `core/organism.py` (OPTIONAL)
```python
# In Organism.forward():
def forward(self, stimulus, state):
    # ... existing code ...

    # After dimension discovery, optionally learn hierarchy
    if self.archive._discovered and not hasattr(self.archive, '_hierarchy'):
        from slime.topology.hierarchy import BehavioralHierarchy

        # Get all behavioral coordinates from archive
        if self.archive.size() > 20:  # Need enough data
            coords = np.array([self.archive.centroids[i] for i in range(len(self.archive.centroids))])

            hierarchy = BehavioralHierarchy(n_clusters=min(10, len(coords) // 2))
            hierarchy.fit(coords)

            self.archive._hierarchy = hierarchy

            # Update chemotaxis to use hybrid metric
            from slime.topology.hybrid_metric import HybridMetric
            hybrid_metric = HybridMetric(hierarchy=hierarchy)
            self.chemotaxis._metric_fn = hybrid_metric

            logger.info(f'Learned behavioral hierarchy with {hierarchy.n_clusters} clusters')
```

**Justification**: Triggered automatically after dimension discovery + sufficient data. Non-breaking. Falls back gracefully if disabled.

---

### Phase 3: Update `run.py` to Enable Topology Features

```python
# In create_full_system():
def create_full_system(config, device, enable_topology: bool = False):
    # ... existing Layer 1-3 code ...

    if enable_topology:
        logger.info("\n[Layer 2.5: Topology (EXPERIMENTAL)]")

        # Enable genealogy tracking in pool
        pool_config.track_genealogy = True
        logger.info("  ✓ Genealogy tracking enabled")

        # Hierarchy will be learned automatically after dimension discovery
        logger.info("  ✓ Hierarchical clustering will be enabled after dimension discovery")

# In main():
def main():
    # ... existing code ...

    # Add CLI flag for topology
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-topology', action='store_true', help='Enable experimental p-adic topology features')
    args = parser.parse_args()

    model = create_full_system(config, device, enable_topology=args.enable_topology)
```

**Justification**: Opt-in via CLI flag. No changes to default behavior. Kleisli: `Bool → Maybe Topology → Model`.

---

## DAG Verification

### New Dependency Graph
```
Layer 0: Protocols
    ↓
Layer 1: Kernels + Observability
    ↓
Layer 2: Data Structures (memory/)
    ↓
Layer 2.5: Topology (NEW)  ← depends ONLY on Layer 2 + numpy/scipy
    ├── p_adic.py (pure functions)
    ├── genealogy.py (pure data structure)
    ├── hierarchy.py (sklearn wrapper)
    └── hybrid_metric.py (composable functor)
    ↓
Layer 3: Components (core/) ← can optionally use Layer 2.5
    ↓
Layer 4: Orchestration (core/organism.py) ← triggers hierarchy learning
    ↓
Layer 5-6: API, Training, Tools
```

**No cycles**: Topology depends only on memory/, not core/. Core/ can optionally import topology/.

---

## Testing Plan

### Unit Tests
```python
# tests/unit/test_topology_p_adic.py
def test_p_adic_distance_identical():
    assert p_adic_distance("abc", "abc") == 256**(-3)

def test_p_adic_distance_no_common_prefix():
    assert p_adic_distance("abc", "def") == 1.0

def test_ultrametric_triangle_inequality():
    # Strong triangle: d(x,z) <= max(d(x,y), d(y,z))
    ...

# tests/unit/test_topology_genealogy.py
def test_spawn_lineage():
    gen = Genealogy()
    gen.register_genesis(0)
    gen.register_spawn(0, 1)
    gen.register_spawn(0, 2)
    assert gen.ultrametric_distance(1, 2) == 2

# tests/unit/test_topology_hierarchy.py
def test_gmm_clustering():
    # Synthetic 2-cluster data
    ...

# tests/unit/test_topology_hybrid_metric.py
def test_hybrid_within_cluster():
    # Should use Mahalanobis
    ...

def test_hybrid_between_clusters():
    # Should use ultrametric
    ...
```

### Integration Test
```python
# tests/integration/test_topology_full.py
def test_topology_enabled():
    """Test full training run with --enable-topology."""
    # Should track genealogy, learn hierarchy, use hybrid metric
    ...
```

---

## Summary

### Kleisli Properties
1. **Composability**: Each module is a functor `A → Maybe B`
2. **Optional**: All topology features are opt-in via flags
3. **Pure**: No global state, all functions return values
4. **Testable**: Each module tested standalone before integration

### DAG Properties
1. **Acyclic**: topology/ depends only on memory/, not core/
2. **Layered**: Fits cleanly between Layer 2 (data structures) and Layer 3 (components)
3. **Backward-compatible**: Default behavior unchanged

### Integration Strategy
1. **Phase 1**: Write pure functions (p_adic.py, genealogy.py, hierarchy.py, hybrid_metric.py)
2. **Phase 2**: Add optional integration points (pool.py, chemotaxis.py, archive.py, organism.py)
3. **Phase 3**: Update run.py with --enable-topology flag

### Justification
- **Not forced**: All 3 structures (hashes, spawning, GMM) already exist
- **Not siloed**: Integrated via dependency injection
- **Not random**: Every edit justified by DAG position and Kleisli type
- **Elegant**: Optional, composable, testable

This plan ensures topology/ emerges naturally from the architecture without breaking existing code.
