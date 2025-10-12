# Distance Metrics in Behavioral Space

## Why Centroid IDs Don't Work

**Problem:** Centroid IDs are arbitrary labels assigned by K-Means, NOT spatial coordinates.

```python
# WRONG - treats IDs as if they're on a number line
distance = abs(centroid_id_1 - centroid_id_2)
```

**Why it fails:**
- Centroid 5 could be geometrically closer to Centroid 47 than to Centroid 6
- K-Means assigns IDs randomly during initialization
- No correlation between ID difference and spatial proximity
- Creates nonsensical chemotaxis gradients (components chase ID numbers, not actual nutrients)

## Available Metrics

### 1. Euclidean (L2) - Default
```python
distance_metric='euclidean'
```

**Formula:** `d = sqrt(Σ(p1_i - p2_i)²)`

**Use when:**
- Behavioral space is continuous and isotropic
- All dimensions equally important
- Natural for Kernel PCA embeddings (which optimize L2 reconstruction)

**Pros:** Natural, fast, works well for most cases
**Cons:** Sensitive to curse of dimensionality, assumes spherical distributions

---

### 2. Manhattan (L1 / Taxicab)
```python
distance_metric='manhattan'
```

**Formula:** `d = Σ|p1_i - p2_i|`

**Use when:**
- Behavioral space has grid-like structure
- Need robustness to outliers
- Computational efficiency critical (no sqrt)

**Pros:** More robust than Euclidean, faster
**Cons:** Anisotropic (diagonal moves cost more than axis-aligned), less natural for Kernel PCA

---

### 3. Mahalanobis (Covariance-Weighted) - **RECOMMENDED**
```python
distance_metric='mahalanobis'
chemotaxis.update_covariance()  # Must call after dimension discovery
```

**Formula:** `d = sqrt((p1 - p2)ᵀ Σ⁻¹ (p1 - p2))` where Σ is covariance matrix

**Use when:**
- Behavioral dimensions are correlated (common after Kernel PCA)
- Dimensions have different variances
- Want to respect natural structure of behavioral space

**Pros:**
- Accounts for correlations between dimensions
- Automatically weights dimensions by variance
- **Most geometrically appropriate for Kernel PCA spaces**

**Cons:**
- Requires covariance estimation (needs >2 centroids)
- Expensive to compute (matrix inversion)
- Can be unstable if covariance matrix is ill-conditioned

**Why it's best:** Kernel PCA discovers correlated behavioral dimensions. Mahalanobis respects that correlation structure, while Euclidean naively treats all dimensions as independent.

---

### 4. Cosine (Angular)
```python
distance_metric='cosine'
```

**Formula:** `d = 1 - (p1·p2) / (||p1|| ||p2||)`

**Use when:**
- Only direction matters, not magnitude
- Behavioral vectors are naturally normalized

**Pros:** Magnitude-invariant, good for normalized spaces
**Cons:** Loses magnitude information (fitness=10 vs fitness=1000 look identical), not a true metric

---

## What About Ultrametrics?

**Ultrametrics** satisfy the **strong triangle inequality:**
```
d(x, z) ≤ max(d(x, y), d(y, z))
```

**Use when:**
- Behavioral space has hierarchical structure (taxonomy, phylogenetic trees)
- Can organize behaviors in a tree (e.g., locomotion → bipedal → running → sprinting)

**Why NOT used in Slime:**
- CVT-MAP-Elites assumes **flat Voronoi tessellation**, not hierarchy
- Behavioral space discovered by Kernel PCA is continuous, not tree-structured
- Would require complete architectural rewrite (archive as tree instead of flat grid)

**When to consider:** If you're working with hierarchical behaviors (skill trees, evolutionary taxonomy), use a tree-based archive with ultrametric distances. For general quality-diversity optimization, stick with flat metrics.

---

## Recommendation

**For most use cases:** Start with **Euclidean** (default)

**For correlated behavioral spaces:** Use **Mahalanobis** (best theoretical justification)
```python
chemotaxis = Chemotaxis(archive, distance_metric='mahalanobis')
chemotaxis.update_covariance()  # Call after dimension discovery
```

**For computational efficiency:** Use **Manhattan**

**For normalized behaviors:** Use **Cosine**

---

## Example: Switching Metrics

```python
from slime.core.chemotaxis import Chemotaxis

# Try different metrics to see which works best
for metric in ['euclidean', 'manhattan', 'mahalanobis', 'cosine']:
    chemotaxis = Chemotaxis(archive, distance_metric=metric)

    if metric == 'mahalanobis':
        chemotaxis.update_covariance()  # Estimate from centroids

    # Test convergence speed, archive coverage, etc.
    results = run_experiment(chemotaxis)
    print(f"{metric}: coverage={results['coverage']:.2f}, diversity={results['diversity']:.2f}")
```

The metric that maximizes archive coverage and behavioral diversity is usually the best choice.
