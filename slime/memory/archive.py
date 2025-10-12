import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Callable, Union
from dataclasses import dataclass
import weakref
import logging
import numpy as np
from scipy.spatial import distance

logger = logging.getLogger(__name__)


@dataclass
class Elite:
    """Elite individual in archive with low-rank compressed weights."""
    behavior: Tuple[float, ...]
    fitness: float
    weights_u: Dict[str, torch.Tensor]  # Low-rank U factorization
    weights_v: Dict[str, torch.Tensor]  # Low-rank V factorization
    generation: int
    metadata: dict

    def __post_init__(self):
        self.behavior = tuple(self.behavior)

    def reconstruct_weights(self) -> dict:
        """Reconstruct full weights from low-rank factorization: W = U @ V."""
        weights = {}
        for key in self.weights_u.keys():
            weights[key] = torch.matmul(self.weights_u[key], self.weights_v[key])
        return weights

    def memory_usage(self) -> int:
        """Total memory usage in bytes for compressed weights."""
        mem = 0
        for key in self.weights_u.keys():
            mem += self.weights_u[key].numel() * self.weights_u[key].element_size()
            mem += self.weights_v[key].numel() * self.weights_v[key].element_size()
        return mem


class CVTArchive:
    """
    CVT-MAP-Elites archive for quality-diversity optimization.

    Uses Centroidal Voronoi Tessellation to partition behavioral space.
    Scales linearly with num_centroids (vs exponentially with grid resolution).
    """

    def __init__(
        self,
        behavioral_dims: int,
        num_centroids: int = 1000,
        low_rank_k: int = 64,
        device: torch.device = None,
        kmo_threshold: float = 0.6,
        seed: int = 42
    ):
        self.behavioral_dims = behavioral_dims
        self.num_centroids = num_centroids
        self.low_rank_k = low_rank_k
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kmo_threshold = kmo_threshold
        self.seed = seed

        self.centroids: Optional[np.ndarray] = None
        self._archive: Dict[int, Elite] = {}
        self._generation = 0
        self._total_additions = 0
        self._total_replacements = 0
        self._behavioral_samples: List[Tuple[float, ...]] = []

    def initialize_centroids(self, initial_samples: Optional[np.ndarray] = None):
        """
        Initialize CVT centroids.

        Args:
            initial_samples: Behavioral samples for K-means clustering (optional)
                           If not provided, uses random initialization with seed
        """
        if initial_samples is not None and len(initial_samples) >= self.num_centroids:
            from scipy.cluster.vq import kmeans
            self.centroids, _ = kmeans(initial_samples.astype(np.float32), self.num_centroids)
            logger.info(f'Initialized {self.num_centroids} CVT centroids from {len(initial_samples)} samples')
        else:
            rng = np.random.RandomState(self.seed)
            self.centroids = rng.randn(self.num_centroids, self.behavioral_dims).astype(np.float32)
            logger.info(f'Initialized {self.num_centroids} CVT centroids from seed {self.seed}')

    def validate_behavioral_space(self, samples: np.ndarray) -> float:
        """
        Validate behavioral space using KMO test.

        Returns:
            KMO statistic (>0.6 acceptable, >0.8 good, >0.9 excellent)
        """
        try:
            from factor_analyzer.factor_analyzer import calculate_kmo
        except ImportError:
            logger.warning('factor_analyzer not installed, skipping KMO validation')
            return 1.0

        if len(samples) < 50:
            logger.warning(f'Only {len(samples)} behavioral samples, need ≥50 for reliable KMO test')
            return 0.0

        try:
            from scipy.stats import bartlett
            kmo_all, kmo_model = calculate_kmo(samples)
            logger.info(f'KMO statistic: {kmo_model:.3f} (>0.6=acceptable, >0.8=good, >0.9=excellent)')

            if kmo_model < self.kmo_threshold:
                logger.error(f'KMO {kmo_model:.3f} < {self.kmo_threshold}: behavioral dimensions are not factorable!')

            corr_matrix = np.corrcoef(samples, rowvar=False)
            stat, p_value = bartlett(*[samples[:, i] for i in range(samples.shape[1])])
            logger.info(f"Bartlett's test: statistic={stat:.2f}, p-value={p_value:.4f}")

            return kmo_model
        except Exception as e:
            logger.error(f'Failed to validate behavioral space: {e}')
            return 0.0

    def _find_nearest_centroid(self, behavior: Union[Tuple[float, ...], np.ndarray]) -> int:
        """Find nearest centroid via Euclidean distance."""
        if self.centroids is None:
            raise ValueError('Centroids not initialized. Call initialize_centroids() first.')

        if isinstance(behavior, tuple):
            behavior_array = np.array(behavior, dtype=np.float32)
        else:
            behavior_array = behavior.astype(np.float32)

        distances = distance.cdist([behavior_array], self.centroids, metric='euclidean')[0]
        return int(np.argmin(distances))

    def _compress_weights(self, state_dict: dict) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Compress weights using SVD low-rank factorization.

        For matrix W ∈ R^(D1×D2):
        W ≈ U @ V where U ∈ R^(D1×k), V ∈ R^(k×D2)

        Memory reduction: D1*D2 → D1*k + k*D2
        For k=64, D=512: 262144 → 65536 (4x reduction)
        """
        weights_u = {}
        weights_v = {}

        for key, tensor in state_dict.items():
            if tensor.ndim == 2 and min(tensor.shape) > self.low_rank_k:
                # SVD decomposition
                U, S, Vt = torch.linalg.svd(tensor, full_matrices=False)
                k = min(self.low_rank_k, len(S))

                # Distribute singular values: W = (U @ sqrt(S)) @ (sqrt(S) @ V^T)
                sqrt_S = torch.sqrt(S[:k])
                weights_u[key] = U[:, :k] * sqrt_S
                weights_v[key] = Vt[:k, :] * sqrt_S.unsqueeze(1)
            else:
                # Matrix too small for factorization, store directly
                k = min(self.low_rank_k, tensor.shape[0] if tensor.ndim >= 1 else 1)
                if tensor.ndim == 2:
                    identity = torch.eye(tensor.shape[0], device=tensor.device, dtype=tensor.dtype)[:, :k]
                    weights_u[key] = tensor @ identity
                    weights_v[key] = identity.T
                else:
                    # 1D or scalar tensor, store as-is
                    weights_u[key] = tensor.unsqueeze(-1) if tensor.ndim == 1 else tensor.view(-1, 1)
                    weights_v[key] = torch.ones(1, 1, device=tensor.device, dtype=tensor.dtype)

        return weights_u, weights_v

    def add(
        self,
        behavior: Tuple[float, ...],
        fitness: float,
        state_dict: dict,
        generation: Optional[int] = None,
        metadata: Optional[dict] = None
    ) -> bool:
        """
        Add elite to archive.

        Args:
            behavior: Behavioral coordinates (tuple of floats)
            fitness: Fitness value (higher is better)
            state_dict: Dictionary of weight tensors
            generation: Generation number (defaults to archive generation)
            metadata: Optional metadata dictionary

        Returns:
            True if elite was added/replaced, False if rejected
        """
        if len(behavior) != self.behavioral_dims:
            raise ValueError(
                f'Behavior dimension mismatch: expected {self.behavioral_dims}, got {len(behavior)}'
            )

        # Store behavioral sample for later KMO validation
        self._behavioral_samples.append(behavior)

        # Auto-initialize centroids if enough samples
        if self.centroids is None:
            if len(self._behavioral_samples) >= 100:
                samples = np.array(self._behavioral_samples, dtype=np.float32)
                kmo = self.validate_behavioral_space(samples)
                if kmo >= self.kmo_threshold:
                    self.initialize_centroids(samples)
                else:
                    logger.warning('Deferring centroid initialization: KMO too low')
            return False

        centroid_id = self._find_nearest_centroid(behavior)

        # Check if should replace existing elite
        if centroid_id in self._archive:
            if fitness <= self._archive[centroid_id].fitness:
                return False  # Lower fitness, reject
            self._total_replacements += 1

        # Compress weights
        weights_u, weights_v = self._compress_weights(state_dict)

        # Create elite
        elite = Elite(
            behavior=behavior,
            fitness=fitness,
            weights_u=weights_u,
            weights_v=weights_v,
            generation=generation if generation is not None else self._generation,
            metadata=metadata or {}
        )

        self._archive[centroid_id] = elite
        self._total_additions += 1

        logger.debug(
            f'Added elite at centroid {centroid_id}: fitness={fitness:.4f}, '
            f'memory={elite.memory_usage() / 1024:.1f}KB'
        )
        return True

    def get(self, behavior: Tuple[float, ...]) -> Optional[Elite]:
        """Get elite at nearest centroid to behavior."""
        if self.centroids is None:
            return None
        centroid_id = self._find_nearest_centroid(behavior)
        return self._archive.get(centroid_id)

    def sample_near(self, behavior: Tuple[float, ...], k: int = 5) -> List[Elite]:
        """Sample k nearest elites to behavior."""
        if self.centroids is None or not self._archive:
            return []

        behavior_array = np.array(behavior, dtype=np.float32)
        centroid_ids = list(self._archive.keys())
        centroid_positions = self.centroids[centroid_ids]

        distances = distance.cdist([behavior_array], centroid_positions, metric='euclidean')[0]
        nearest_indices = np.argsort(distances)[:k]

        return [self._archive[centroid_ids[i]] for i in nearest_indices if i < len(centroid_ids)]

    def bootstrap_component(
        self,
        component_factory: Callable,
        target_behavior: Tuple[float, ...],
        k_neighbors: int = 3,
        mutation_std: float = 0.1
    ):
        """
        Bootstrap component from archive near target behavior.

        Args:
            component_factory: Callable that creates component (may accept state_dict or no args)
            target_behavior: Target behavioral coordinates
            k_neighbors: Number of nearby elites to consider
            mutation_std: Std dev of mutation noise

        Returns:
            Component initialized from mutated elite weights, or None if archive empty
        """
        nearby = self.sample_near(target_behavior, k=k_neighbors)
        if not nearby:
            return None

        # Select best nearby elite
        best = max(nearby, key=lambda e: e.fitness)
        weights = best.reconstruct_weights()

        # Mutate weights
        torch.manual_seed(self.seed + best.generation)
        for key in weights:
            noise = torch.randn_like(weights[key]) * mutation_std
            weights[key] = weights[key] + noise

        # Create component (handle both signatures)
        try:
            component = component_factory()
            component.load_state_dict(weights)
        except TypeError:
            # Factory expects state_dict argument
            component = component_factory(weights)

        logger.debug(
            f'Bootstrapped component from elite at generation {best.generation}, '
            f'fitness={best.fitness:.4f}'
        )
        return component

    def increment_generation(self) -> None:
        """Increment generation counter."""
        self._generation += 1

    def size(self) -> int:
        """Number of elites in archive."""
        return len(self._archive)

    def coverage(self) -> float:
        """Fraction of centroids occupied by elites."""
        if self.centroids is None:
            return 0.0
        return len(self._archive) / self.num_centroids

    def max_fitness(self) -> float:
        """Maximum fitness across all elites (returns -inf if empty)."""
        if not self._archive:
            return float('-inf')
        return max(e.fitness for e in self._archive.values())

    def get_behavioral_variance(self) -> np.ndarray:
        """Variance of behaviors across each dimension."""
        if not self._archive:
            return np.zeros(self.behavioral_dims)

        behaviors = np.array([elite.behavior for elite in self._archive.values()])
        return np.var(behaviors, axis=0)

    def total_memory_usage(self) -> int:
        """Total memory usage in bytes for all compressed elites."""
        return sum(elite.memory_usage() for elite in self._archive.values())

    @property
    def elites(self) -> Dict[int, Elite]:
        """Direct access to archive dictionary (for compatibility)."""
        return self._archive

    def clear(self) -> None:
        """Clear all elites and reset counters."""
        self._archive.clear()
        self._generation = 0
        self._total_additions = 0
        self._total_replacements = 0
        self._behavioral_samples.clear()
