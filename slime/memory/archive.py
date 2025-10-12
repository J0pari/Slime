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
    behavior: Tuple[float, ...]
    fitness: float
    weights_u: Dict[str, torch.Tensor]
    weights_v: Dict[str, torch.Tensor]
    generation: int
    metadata: dict

    def __post_init__(self):
        self.behavior = tuple(self.behavior)

    def reconstruct_weights(self) -> dict:
        weights = {}
        for key in self.weights_u.keys():
            weights[key] = torch.matmul(self.weights_u[key], self.weights_v[key])
        return weights

    def memory_usage(self) -> int:
        mem = 0
        for key in self.weights_u.keys():
            mem += self.weights_u[key].numel() * self.weights_u[key].element_size()
            mem += self.weights_v[key].numel() * self.weights_v[key].element_size()
        return mem

class CVTArchive:

    def __init__(self, num_raw_metrics: int=15, target_dims: int=5, num_centroids: int=1000, low_rank_k: int=64, device: torch.device=None, kmo_threshold: float=0.6, reconstruction_error_threshold: float=0.5, kernel: str='rbf', gamma: float=1.0, seed: int=42):
        self.num_raw_metrics = num_raw_metrics
        self.target_dims = target_dims
        self.num_centroids = num_centroids
        self.low_rank_k = low_rank_k
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kmo_threshold = kmo_threshold
        self.reconstruction_error_threshold = reconstruction_error_threshold
        self.kernel = kernel
        self.gamma = gamma
        self.seed = seed
        self.centroids: Optional[np.ndarray] = None
        self.kpca_transform = None
        self.behavioral_dims: Optional[int] = None
        self._archive: Dict[int, Elite] = {}
        self._generation = 0
        self._total_additions = 0
        self._total_replacements = 0
        self._raw_metrics_samples: List[np.ndarray] = []
        self._discovered = False

    def initialize_centroids(self, initial_samples: Optional[np.ndarray]=None):
        if initial_samples is not None and len(initial_samples) >= self.num_centroids:
            from scipy.cluster.vq import kmeans
            self.centroids, _ = kmeans(initial_samples.astype(np.float32), self.num_centroids)
            logger.info(f'Initialized {self.num_centroids} CVT centroids from {len(initial_samples)} samples')
        else:
            rng = np.random.RandomState(self.seed)
            self.centroids = rng.randn(self.num_centroids, self.behavioral_dims).astype(np.float32)
            logger.info(f'Initialized {self.num_centroids} CVT centroids from seed {self.seed}')

    def discover_dimensions(self) -> bool:
        if self._discovered:
            logger.warning('Dimensions already discovered')
            return True
        if len(self._raw_metrics_samples) < 100:
            raise ValueError(f'Need ≥100 raw metric samples for Kernel PCA, got {len(self._raw_metrics_samples)}')
        raw_matrix = np.array(self._raw_metrics_samples, dtype=np.float32)
        logger.info(f'Running Kernel PCA on {raw_matrix.shape[0]} samples with {raw_matrix.shape[1]} raw metrics')
        try:
            from sklearn.decomposition import KernelPCA
            from factor_analyzer.factor_analyzer import calculate_kmo
            from scipy.stats import bartlett
        except ImportError as e:
            raise ImportError(f'Missing required package for dimension discovery: {e}')
        kmo_all, kmo_model = calculate_kmo(raw_matrix)
        logger.info(f'KMO statistic: {kmo_model:.3f} (>0.6=acceptable, >0.8=good, >0.9=excellent)')
        if kmo_model < self.kmo_threshold:
            raise ValueError(f'KMO {kmo_model:.3f} < {self.kmo_threshold}: raw metrics not factorable')
        stat, p_value = bartlett(*[raw_matrix[:, i] for i in range(raw_matrix.shape[1])])
        logger.info(f"Bartlett's test: statistic={stat:.2f}, p-value={p_value:.4f}")
        if p_value > 0.05:
            raise ValueError(f"Bartlett's test p={p_value:.3f} > 0.05: metrics are uncorrelated")
        kpca = KernelPCA(n_components=self.target_dims, kernel=self.kernel, gamma=self.gamma, random_state=self.seed, fit_inverse_transform=True)
        transformed_samples = kpca.fit_transform(raw_matrix)
        reconstructed = kpca.inverse_transform(transformed_samples)
        reconstruction_error = np.mean((raw_matrix - reconstructed) ** 2)
        logger.info(f'Kernel PCA reconstruction error: {reconstruction_error:.3f}')
        if reconstruction_error > self.reconstruction_error_threshold:
            raise ValueError(f'Kernel PCA reconstruction error {reconstruction_error:.3f} > {self.reconstruction_error_threshold}')
        self.kpca_transform = kpca
        self.behavioral_dims = self.target_dims
        self._discovered = True
        self.initialize_centroids(transformed_samples)
        logger.info(f'Discovered {self.behavioral_dims} behavioral dimensions from {self.num_raw_metrics} raw metrics using Kernel PCA (kernel={self.kernel}, gamma={self.gamma})')
        return True

    def validate_behavioral_space(self, samples: np.ndarray) -> float:
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
        if self.centroids is None:
            raise ValueError('Centroids not initialized. Call initialize_centroids() first.')
        if isinstance(behavior, tuple):
            behavior_array = np.array(behavior, dtype=np.float32)
        else:
            behavior_array = behavior.astype(np.float32)
        distances = distance.cdist([behavior_array], self.centroids, metric='euclidean')[0]
        return int(np.argmin(distances))

    def _compress_weights(self, state_dict: dict) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        weights_u = {}
        weights_v = {}
        for key, tensor in state_dict.items():
            if tensor.ndim == 2 and min(tensor.shape) > self.low_rank_k:
                U, S, Vt = torch.linalg.svd(tensor, full_matrices=False)
                k = min(self.low_rank_k, len(S))
                sqrt_S = torch.sqrt(S[:k])
                weights_u[key] = U[:, :k] * sqrt_S
                weights_v[key] = Vt[:k, :] * sqrt_S.unsqueeze(1)
            else:
                k = min(self.low_rank_k, tensor.shape[0] if tensor.ndim >= 1 else 1)
                if tensor.ndim == 2:
                    identity = torch.eye(tensor.shape[0], device=tensor.device, dtype=tensor.dtype)[:, :k]
                    weights_u[key] = tensor @ identity
                    weights_v[key] = identity.T
                else:
                    weights_u[key] = tensor.unsqueeze(-1) if tensor.ndim == 1 else tensor.view(-1, 1)
                    weights_v[key] = torch.ones(1, 1, device=tensor.device, dtype=tensor.dtype)
        return (weights_u, weights_v)

    def add_raw_metrics(self, raw_metrics: np.ndarray) -> bool:
        if self._discovered:
            raise ValueError('Cannot add raw metrics after dimension discovery. Use add() instead.')
        if len(raw_metrics) != self.num_raw_metrics:
            raise ValueError(f'Raw metrics dimension mismatch: expected {self.num_raw_metrics}, got {len(raw_metrics)}')
        self._raw_metrics_samples.append(raw_metrics)
        logger.debug(f'Collected raw metrics sample {len(self._raw_metrics_samples)}/{100} for dimension discovery')
        return False

    def add(self, behavior: Tuple[float, ...], fitness: float, state_dict: dict, generation: Optional[int]=None, metadata: Optional[dict]=None) -> bool:
        if not self._discovered:
            raise ValueError('Cannot add elites before dimension discovery. Use add_raw_metrics() during warmup phase.')
        if self.behavioral_dims is None:
            raise ValueError('Behavioral dimensions not set. Call discover_dimensions() first.')
        if len(behavior) != self.behavioral_dims:
            raise ValueError(f'Behavior dimension mismatch: expected {self.behavioral_dims}, got {len(behavior)}')
        if self.centroids is None:
            raise ValueError('Centroids not initialized. Call discover_dimensions() first.')
        centroid_id = self._find_nearest_centroid(behavior)
        if centroid_id in self._archive:
            if fitness <= self._archive[centroid_id].fitness:
                return False
            self._total_replacements += 1
        weights_u, weights_v = self._compress_weights(state_dict)
        elite = Elite(behavior=behavior, fitness=fitness, weights_u=weights_u, weights_v=weights_v, generation=generation if generation is not None else self._generation, metadata=metadata or {})
        self._archive[centroid_id] = elite
        self._total_additions += 1
        logger.debug(f'Added elite at centroid {centroid_id}: fitness={fitness:.4f}, memory={elite.memory_usage() / 1024:.1f}KB')
        return True

    def get(self, behavior: Tuple[float, ...]) -> Optional[Elite]:
        if self.centroids is None:
            return None
        centroid_id = self._find_nearest_centroid(behavior)
        return self._archive.get(centroid_id)

    def sample_near(self, behavior: Tuple[float, ...], k: int=5) -> List[Elite]:
        if self.centroids is None or not self._archive:
            return []
        behavior_array = np.array(behavior, dtype=np.float32)
        centroid_ids = list(self._archive.keys())
        centroid_positions = self.centroids[centroid_ids]
        distances = distance.cdist([behavior_array], centroid_positions, metric='euclidean')[0]
        nearest_indices = np.argsort(distances)[:k]
        return [self._archive[centroid_ids[i]] for i in nearest_indices if i < len(centroid_ids)]

    def bootstrap_component(self, component_factory: Callable, target_behavior: Tuple[float, ...], k_neighbors: int=3, mutation_std: float=0.1):
        nearby = self.sample_near(target_behavior, k=k_neighbors)
        if not nearby:
            return None
        best = max(nearby, key=lambda e: e.fitness)
        weights = best.reconstruct_weights()
        torch.manual_seed(self.seed + best.generation)
        for key in weights:
            noise = torch.randn_like(weights[key]) * mutation_std
            weights[key] = weights[key] + noise
        try:
            component = component_factory()
            component.load_state_dict(weights)
        except TypeError:
            component = component_factory(weights)
        logger.debug(f'Bootstrapped component from elite at generation {best.generation}, fitness={best.fitness:.4f}')
        return component

    def increment_generation(self) -> None:
        self._generation += 1

    def size(self) -> int:
        return len(self._archive)

    def coverage(self) -> float:
        if self.centroids is None:
            return 0.0
        return len(self._archive) / self.num_centroids

    def max_fitness(self) -> float:
        if not self._archive:
            return float('-inf')
        return max((e.fitness for e in self._archive.values()))

    def get_behavioral_variance(self) -> np.ndarray:
        if not self._archive:
            return np.zeros(self.behavioral_dims)
        behaviors = np.array([elite.behavior for elite in self._archive.values()])
        return np.var(behaviors, axis=0)

    def total_memory_usage(self) -> int:
        return sum((elite.memory_usage() for elite in self._archive.values()))

    @property
    def elites(self) -> Dict[int, Elite]:
        return self._archive

    def clear(self) -> None:
        self._archive.clear()
        self._generation = 0
        self._total_additions = 0
        self._total_replacements = 0
        self._behavioral_samples.clear()