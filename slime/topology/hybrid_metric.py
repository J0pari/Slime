"""
Hybrid distance metrics combining ultrametric and Mahalanobis distances.

Provides composable distance functions that respect hierarchical structure:
- **Between clusters**: Ultrametric distance (discrete jumps)
- **Within clusters**: Mahalanobis distance (continuous gradients)

This creates a multi-scale metric:
- Coarse scale: Cluster-level topology (exploration)
- Fine scale: Within-cluster geometry (exploitation)

Applications:
- Chemotaxis navigation (hybrid gradients)
- Quality-diversity search (balance exploration/exploitation)
- Multi-resolution fitness landscapes

References:
- De Domenico et al., "Structural reducibility of multilayer networks" (2013)
- Carlsson, "Topology and data" (2009) - Persistent homology
"""

from typing import Optional, Literal, Callable
import numpy as np
from slime.topology.hierarchy import BehavioralHierarchy
import logging

logger = logging.getLogger(__name__)

DistanceMode = Literal['euclidean', 'manhattan', 'mahalanobis', 'cosine', 'hybrid']


class HybridMetric:
    """
    Composable distance metric with hierarchical awareness.

    Combines:
    - Ultrametric (between clusters) - respects discrete topology
    - Mahalanobis (within clusters) - respects continuous geometry

    Falls back gracefully if no hierarchy provided (pure Euclidean).
    """

    def __init__(
        self,
        hierarchy: Optional[BehavioralHierarchy] = None,
        inter_cluster_mode: Literal['ultrametric', 'euclidean'] = 'ultrametric',
        intra_cluster_mode: Literal['mahalanobis', 'euclidean'] = 'mahalanobis',
        p: int = 2  # Base for p-adic distance (unused currently)
    ) -> None:
        """
        Initialize hybrid metric.

        Args:
            hierarchy: Behavioral clustering (None = fallback to Euclidean)
            inter_cluster_mode: Distance between clusters
                - 'ultrametric': Dendrogram-based (discrete)
                - 'euclidean': Mean-based (continuous)
            intra_cluster_mode: Distance within cluster
                - 'mahalanobis': Covariance-weighted (respects correlations)
                - 'euclidean': Standard L2 (isotropic)
            p: Base for p-adic distance (future extension)

        Example:
            >>> from slime.topology.hierarchy import BehavioralHierarchy
            >>> hierarchy = BehavioralHierarchy(n_clusters=10)
            >>> hierarchy.fit(coords)
            >>> metric = HybridMetric(hierarchy=hierarchy)
            >>> dist = metric(coord_a, coord_b)
        """
        self.hierarchy = hierarchy
        self.inter_cluster_mode = inter_cluster_mode
        self.intra_cluster_mode = intra_cluster_mode
        self.p = p

        if hierarchy is not None:
            logger.info(
                f"HybridMetric initialized with {hierarchy.n_clusters} clusters "
                f"(inter={inter_cluster_mode}, intra={intra_cluster_mode})"
            )
        else:
            logger.info("HybridMetric initialized without hierarchy (fallback to Euclidean)")

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute hybrid distance between two behavioral coordinates.

        Args:
            x: First coordinate (D,)
            y: Second coordinate (D,)

        Returns:
            Distance (float)

        Example:
            >>> metric = HybridMetric(hierarchy)
            >>> dist = metric(np.array([0.5, 0.3]), np.array([0.6, 0.4]))
        """
        if self.hierarchy is None:
            # Fallback: no hierarchy available
            return self._euclidean(x, y)

        cluster_x = self.hierarchy.predict_cluster(x)
        cluster_y = self.hierarchy.predict_cluster(y)

        if cluster_x == cluster_y:
            # Same cluster: use intra-cluster metric
            if self.intra_cluster_mode == 'mahalanobis':
                return self._mahalanobis(x, y, cluster_x)
            else:
                return self._euclidean(x, y)
        else:
            # Different clusters: use inter-cluster metric
            if self.inter_cluster_mode == 'ultrametric':
                return self.hierarchy.cluster_distance(cluster_x, cluster_y)
            else:
                return self._euclidean(x, y)

    def _euclidean(self, x: np.ndarray, y: np.ndarray) -> float:
        """Euclidean (L2) distance."""
        return float(np.linalg.norm(x - y))

    def _mahalanobis(self, x: np.ndarray, y: np.ndarray, cluster_id: int) -> float:
        """
        Mahalanobis distance using cluster covariance.

        d = sqrt((x - y)^T Σ^(-1) (x - y))

        Args:
            x, y: Points in same cluster
            cluster_id: Cluster ID for covariance matrix

        Returns:
            Mahalanobis distance
        """
        if self.hierarchy is None:
            return self._euclidean(x, y)

        try:
            cov = self.hierarchy.get_covariance(cluster_id)
            diff = x - y

            # Compute (x - y)^T Σ^(-1) (x - y)
            inv_cov = np.linalg.inv(cov)
            mahal_sq = diff @ inv_cov @ diff

            return float(np.sqrt(np.abs(mahal_sq)))  # abs for numerical stability

        except np.linalg.LinAlgError:
            # Singular covariance - fallback to Euclidean
            logger.warning(f"Singular covariance for cluster {cluster_id}, using Euclidean")
            return self._euclidean(x, y)

    def batch_distance(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Vectorized distance computation for batches.

        Args:
            X: (N, D) array of points
            Y: (N, D) array of points

        Returns:
            (N,) array of distances

        Example:
            >>> X = np.random.randn(100, 3)
            >>> Y = np.random.randn(100, 3)
            >>> dists = metric.batch_distance(X, Y)
            >>> dists.shape
            (100,)
        """
        if X.shape != Y.shape:
            raise ValueError(f"Shape mismatch: X={X.shape}, Y={Y.shape}")

        # Fallback to row-wise computation (TODO: optimize)
        return np.array([self(x, y) for x, y in zip(X, Y)])

    def gradient(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epsilon: float = 1e-5
    ) -> np.ndarray:
        """
        Compute gradient of distance w.r.t. x via finite differences.

        ∇_x d(x, y) ≈ (d(x + ε, y) - d(x, y)) / ε

        Args:
            x: Point to compute gradient at
            y: Reference point
            epsilon: Finite difference step

        Returns:
            (D,) gradient vector

        Use Case:
            Gradient descent in behavioral space for chemotaxis:
            x_new = x - lr * gradient(x, nutrient_location)

        Example:
            >>> grad = metric.gradient(x, y)
            >>> # Move x toward y
            >>> x_new = x - 0.1 * grad
        """
        grad = np.zeros_like(x)

        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += epsilon

            grad[i] = (self(x_plus, y) - self(x, y)) / epsilon

        return grad

    def is_hierarchical(self) -> bool:
        """
        Check if metric uses hierarchical structure.

        Returns:
            True if hierarchy provided, False if pure Euclidean fallback
        """
        return self.hierarchy is not None

    def get_cluster(self, x: np.ndarray) -> Optional[int]:
        """
        Get cluster assignment for point.

        Args:
            x: Behavioral coordinate

        Returns:
            Cluster ID or None if no hierarchy

        Example:
            >>> cluster = metric.get_cluster(coord)
            >>> if cluster is not None:
            ...     print(f"Point in cluster {cluster}")
        """
        if self.hierarchy is None:
            return None

        return self.hierarchy.predict_cluster(x)


# ============================================================================
# Factory Functions
# ============================================================================

def create_metric(
    mode: DistanceMode,
    hierarchy: Optional[BehavioralHierarchy] = None
) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Factory function for creating distance metrics.

    Args:
        mode: Type of metric to create
        hierarchy: Optional hierarchical structure for hybrid metrics

    Returns:
        Distance function: (x, y) -> float

    Example:
        >>> metric = create_metric('euclidean')
        >>> dist = metric(x, y)

        >>> metric = create_metric('hybrid', hierarchy=hierarchy)
        >>> dist = metric(x, y)  # Uses hierarchical structure
    """
    if mode == 'euclidean':
        return lambda x, y: float(np.linalg.norm(x - y))

    elif mode == 'manhattan':
        return lambda x, y: float(np.sum(np.abs(x - y)))

    elif mode == 'mahalanobis':
        # Global Mahalanobis requires covariance matrix
        raise NotImplementedError(
            "Global Mahalanobis requires covariance matrix. "
            "Use HybridMetric with hierarchy instead."
        )

    elif mode == 'cosine':
        def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
            dot = np.dot(x, y)
            norm_x = np.linalg.norm(x)
            norm_y = np.linalg.norm(y)
            if norm_x < 1e-10 or norm_y < 1e-10:
                return 1.0  # Maximum distance for zero vectors
            return float(1.0 - (dot / (norm_x * norm_y)))
        return cosine_distance

    elif mode == 'hybrid':
        if hierarchy is None:
            raise ValueError("Hybrid metric requires hierarchy")
        metric = HybridMetric(hierarchy=hierarchy)
        return lambda x, y: metric(x, y)

    else:
        raise ValueError(f"Unknown metric mode: {mode}")


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'HybridMetric',
    'DistanceMode',
    'create_metric',
]
