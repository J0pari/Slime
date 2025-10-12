"""
GMM-based hierarchical behavioral space clustering.

Learns hierarchical structure from behavioral coordinates via:
1. Gaussian Mixture Model (GMM) - soft clustering
2. Agglomerative clustering - dendrogram for inter-cluster distances

Provides:
- Cluster assignments for behavioral coordinates
- Covariance matrices for intra-cluster Mahalanobis distances
- Dendrogram for inter-cluster ultrametric distances

Applications:
- Hybrid distance metrics (ultrametric between clusters, Mahalanobis within)
- Multi-scale fitness landscapes (coarse/fine search)
- Behavioral space partitioning for chemotaxis

References:
- Bishop, "Pattern Recognition and Machine Learning" (2006) - GMM
- Murtagh & Contreras, "Algorithms for hierarchical clustering" (2012)
"""

from typing import Optional, Literal
import numpy as np
from sklearn.mixture import GaussianMixture  # type: ignore[import-untyped]
from scipy.cluster.hierarchy import linkage  # type: ignore[import-untyped]
import logging

logger = logging.getLogger(__name__)

LinkageMethod = Literal['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']


class BehavioralHierarchy:
    """
    Learns hierarchical structure from behavioral space via GMM + dendrogram.

    Workflow:
        1. Fit GMM to behavioral coordinates (soft clustering)
        2. Build dendrogram from cluster means (hierarchical structure)
        3. Provide cluster assignments and covariance matrices

    Thread-safe: All state immutable after fit().
    """

    def __init__(
        self,
        n_clusters: int = 10,
        linkage_method: LinkageMethod = 'ward',
        random_state: int = 42,
        covariance_type: Literal['full', 'tied', 'diag', 'spherical'] = 'full'
    ) -> None:
        """
        Initialize hierarchical clustering parameters.

        Args:
            n_clusters: Number of GMM components
            linkage_method: Method for dendrogram (ward recommended)
            random_state: Random seed for reproducibility
            covariance_type: GMM covariance structure
                - 'full': Each cluster has own covariance (most flexible)
                - 'tied': All clusters share covariance
                - 'diag': Diagonal covariances (independent dimensions)
                - 'spherical': Isotropic covariances (circular clusters)

        Example:
            >>> hierarchy = BehavioralHierarchy(n_clusters=5, linkage_method='ward')
            >>> hierarchy.fit(behavioral_coords)
            >>> cluster_id = hierarchy.predict_cluster(new_coord)
        """
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.random_state = random_state
        self.covariance_type = covariance_type

        self.gmm: Optional[GaussianMixture] = None
        self.linkage_matrix: Optional[np.ndarray] = None
        self._cluster_means: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, behavioral_coords: np.ndarray) -> None:
        """
        Fit GMM to behavioral coordinates and build dendrogram.

        Args:
            behavioral_coords: (N, D) array of behavioral space coordinates

        Raises:
            ValueError: If n_clusters > N (not enough data)

        Example:
            >>> coords = np.random.randn(100, 3)  # 100 points in 3D
            >>> hierarchy = BehavioralHierarchy(n_clusters=5)
            >>> hierarchy.fit(coords)
        """
        if behavioral_coords.shape[0] < self.n_clusters:
            raise ValueError(
                f"Cannot fit {self.n_clusters} clusters to {behavioral_coords.shape[0]} points. "
                f"Reduce n_clusters or provide more data."
            )

        # Fit GMM
        logger.info(
            f"Fitting GMM with {self.n_clusters} clusters to "
            f"{behavioral_coords.shape[0]} points in {behavioral_coords.shape[1]}D space"
        )

        self.gmm = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=100,
            n_init=3
        )
        self.gmm.fit(behavioral_coords)

        # Build dendrogram from cluster means
        self._cluster_means = self.gmm.means_
        self.linkage_matrix = linkage(
            self._cluster_means,
            method=self.linkage_method
        )

        self._fitted = True
        logger.info(f"Hierarchical clustering complete (converged: {self.gmm.converged_})")

    def predict_cluster(self, x: np.ndarray) -> int:
        """
        Predict cluster assignment for behavioral coordinate.

        Args:
            x: (D,) behavioral coordinate

        Returns:
            Cluster ID in [0, n_clusters)

        Raises:
            RuntimeError: fit() not called yet

        Example:
            >>> hierarchy.fit(coords)
            >>> cluster_id = hierarchy.predict_cluster(np.array([0.5, 0.3, 0.2]))
            >>> 0 <= cluster_id < hierarchy.n_clusters
            True
        """
        if not self._fitted or self.gmm is None:
            raise RuntimeError("Must call fit() before predict_cluster()")

        return int(self.gmm.predict(x.reshape(1, -1))[0])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """
        Predict soft cluster assignments (probabilities).

        Args:
            x: (D,) behavioral coordinate

        Returns:
            (n_clusters,) array of probabilities

        Example:
            >>> probs = hierarchy.predict_proba(coord)
            >>> probs.sum()
            1.0  # Probabilities sum to 1
        """
        if not self._fitted or self.gmm is None:
            raise RuntimeError("Must call fit() before predict_proba()")

        return self.gmm.predict_proba(x.reshape(1, -1))[0]

    def get_covariance(self, cluster_id: int) -> np.ndarray:
        """
        Get covariance matrix for cluster (for Mahalanobis distance).

        Args:
            cluster_id: Cluster ID in [0, n_clusters)

        Returns:
            (D, D) covariance matrix

        Raises:
            RuntimeError: fit() not called yet
            IndexError: Invalid cluster_id

        Example:
            >>> cov = hierarchy.get_covariance(0)
            >>> cov.shape
            (3, 3)  # 3D behavioral space
        """
        if not self._fitted or self.gmm is None:
            raise RuntimeError("Must call fit() before get_covariance()")

        if cluster_id < 0 or cluster_id >= self.n_clusters:
            raise IndexError(
                f"Cluster ID {cluster_id} out of range [0, {self.n_clusters})"
            )

        return self.gmm.covariances_[cluster_id]

    def cluster_distance(self, cluster_a: int, cluster_b: int) -> float:
        """
        Compute ultrametric distance between clusters via dendrogram traversal.

        Uses linkage matrix to find the height at which clusters merge.
        This guarantees the ultrametric property: d(x,z) <= max(d(x,y), d(y,z))

        Args:
            cluster_a: First cluster ID
            cluster_b: Second cluster ID

        Returns:
            Ultrametric distance (height of merge in dendrogram)

        Example:
            >>> hierarchy.fit(coords)
            >>> dist = hierarchy.cluster_distance(0, 1)
            >>> dist >= 0
            True
        """
        if not self._fitted or self._cluster_means is None:
            raise RuntimeError("Must call fit() before cluster_distance()")

        if cluster_a == cluster_b:
            return 0.0

        if cluster_a < 0 or cluster_a >= self.n_clusters:
            raise IndexError(f"Cluster A ID {cluster_a} out of range")
        if cluster_b < 0 or cluster_b >= self.n_clusters:
            raise IndexError(f"Cluster B ID {cluster_b} out of range")

        # Use dendrogram to find merge height (true ultrametric)
        # Linkage matrix format: [idx1, idx2, distance, sample_count]
        # First n_clusters rows are original clusters (indexed 0..n_clusters-1)
        # Each subsequent row represents a merge creating new cluster n_clusters+i

        # Find which merged cluster contains cluster_a and cluster_b
        # Traverse dendrogram from leaves to root
        def find_merge_height(a: int, b: int, Z: np.ndarray) -> float:
            """Find height at which clusters a and b merge."""
            n = len(Z) + 1  # Number of original clusters

            # Track which merged cluster each original cluster belongs to
            # cluster_to_merged[i] = set of original clusters in merged cluster i
            cluster_to_merged: dict[int, set[int]] = {}
            for i in range(n):
                cluster_to_merged[i] = {i}

            # Process merges from bottom to top
            for i, row in enumerate(Z):
                idx1, idx2, height, count = row
                idx1, idx2 = int(idx1), int(idx2)

                # Merged cluster ID
                merged_id = n + i

                # Combine sets from children
                cluster_to_merged[merged_id] = (
                    cluster_to_merged[idx1] | cluster_to_merged[idx2]
                )

                # Check if this merge joins a and b
                if a in cluster_to_merged[merged_id] and b in cluster_to_merged[merged_id]:
                    # But they weren't together before this merge
                    if not (a in cluster_to_merged[idx1] and b in cluster_to_merged[idx1]):
                        if not (a in cluster_to_merged[idx2] and b in cluster_to_merged[idx2]):
                            return float(height)

            # Should never reach here if dendrogram is valid
            return float('inf')

        return find_merge_height(cluster_a, cluster_b, self.linkage_matrix)

    def log_likelihood(self, x: np.ndarray) -> float:
        """
        Compute log-likelihood of point under GMM.

        Args:
            x: (D,) behavioral coordinate

        Returns:
            Log-likelihood (higher = better fit)

        Example:
            >>> ll = hierarchy.log_likelihood(coord)
            >>> # Points far from cluster centers have low likelihood
        """
        if not self._fitted or self.gmm is None:
            raise RuntimeError("Must call fit() before log_likelihood()")

        return float(self.gmm.score(x.reshape(1, -1)))

    def stats(self) -> dict[str, float | int | bool]:
        """
        Get clustering statistics.

        Returns:
            Dict with keys:
            - n_clusters: Number of clusters
            - fitted: Whether fit() called
            - converged: Whether GMM converged
            - n_iter: Number of EM iterations
            - lower_bound: GMM log-likelihood lower bound
        """
        if not self._fitted or self.gmm is None:
            return {
                'n_clusters': self.n_clusters,
                'fitted': False,
                'converged': False,
                'n_iter': 0,
                'lower_bound': float('-inf'),
            }

        return {
            'n_clusters': self.n_clusters,
            'fitted': True,
            'converged': self.gmm.converged_,
            'n_iter': self.gmm.n_iter_,
            'lower_bound': self.gmm.lower_bound_,
        }


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'BehavioralHierarchy',
    'LinkageMethod',
]
