import torch
from typing import Tuple, Optional, Dict, Literal
import logging
import numpy as np
from slime.memory.archive import CVTArchive
logger = logging.getLogger(__name__)

DistanceMetric = Literal['euclidean', 'manhattan', 'mahalanobis', 'cosine']

class Chemotaxis:

    def __init__(self, archive: CVTArchive, device: Optional[torch.device]=None,
                 distance_metric: DistanceMetric = 'euclidean'):
        self.archive = archive
        self.device = device or torch.device('cuda')
        self.distance_metric = distance_metric
        self._sources: Dict[Tuple[int, ...], Tuple[torch.Tensor, float]] = {}
        self._covariance_matrix = None  # For Mahalanobis

    def add_source(self, nutrient: torch.Tensor, location: Tuple[float, ...], concentration: float) -> None:
        if not isinstance(nutrient, torch.Tensor):
            raise TypeError(f'nutrient must be torch.Tensor, got {type(nutrient)}')
        import numpy as np
        behavior_array = np.array(location)
        centroid_id = self.archive._find_nearest_centroid(behavior_array)
        if centroid_id not in self._sources or self._sources[centroid_id][1] < concentration:
            self._sources[centroid_id] = (nutrient.detach().to(self.device), float(concentration))
            logger.debug(f'Added nutrient source at centroid {centroid_id}, concentration={concentration:.4f}')

    def _compute_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Compute distance between two positions using configured metric."""
        if self.distance_metric == 'euclidean':
            return np.linalg.norm(pos1 - pos2)

        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(pos1 - pos2))

        elif self.distance_metric == 'mahalanobis':
            if self._covariance_matrix is None:
                # Fallback to Euclidean if covariance not computed
                logger.warning('Mahalanobis requested but covariance not available, using Euclidean')
                return np.linalg.norm(pos1 - pos2)
            diff = pos1 - pos2
            return np.sqrt(diff @ np.linalg.inv(self._covariance_matrix) @ diff)

        elif self.distance_metric == 'cosine':
            # Cosine distance: 1 - cos(θ)
            dot = np.dot(pos1, pos2)
            norm1 = np.linalg.norm(pos1)
            norm2 = np.linalg.norm(pos2)
            if norm1 < 1e-10 or norm2 < 1e-10:
                return 1.0  # Maximum distance for zero vectors
            return 1.0 - (dot / (norm1 * norm2))

        else:
            raise ValueError(f'Unknown distance metric: {self.distance_metric}')

    def update_covariance(self):
        """Estimate covariance matrix from archive centroids for Mahalanobis distance."""
        if self.archive.centroids is None or len(self.archive.centroids) < 2:
            logger.warning('Not enough centroids to estimate covariance')
            return
        self._covariance_matrix = np.cov(self.archive.centroids.T)
        logger.info(f'Updated covariance matrix for Mahalanobis distance (shape: {self._covariance_matrix.shape})')

    def sample(self, behavior: Tuple[float, ...], metabolic_rate: float=1.0, hunger: float=0.0) -> Optional[torch.Tensor]:
        if not self._sources:
            return None
        behavior_array = np.array(behavior)
        center_centroid = self.archive._find_nearest_centroid(behavior_array)

        # Get center position in behavioral space
        center_pos = self.archive.centroids[center_centroid]

        weights = []
        nutrients_list = []
        for centroid_id, (nutrient, concentration) in self._sources.items():
            # Use configured distance metric in behavioral space
            source_pos = self.archive.centroids[centroid_id]
            distance = self._compute_distance(center_pos, source_pos)

            # Weight by concentration and inverse distance
            weight = (concentration + hunger * 0.1) / (distance + 1.0)
            weight = weight / (metabolic_rate + 1e-10)
            weights.append(weight)
            nutrients_list.append(nutrient)

        if not nutrients_list:
            return None
        weights_tensor = torch.tensor(weights, device=self.device)
        weights_tensor = weights_tensor - weights_tensor.max()
        weights_tensor = torch.softmax(weights_tensor, dim=0)
        nutrients = torch.stack(nutrients_list)
        sampled = torch.sum(nutrients * weights_tensor.view(-1, 1, 1), dim=0)
        return sampled

    def clear(self) -> None:
        self._sources.clear()

    def stats(self) -> dict:
        if not self._sources:
            return {'num_sources': 0, 'avg_concentration': 0.0, 'max_concentration': 0.0}
        concentrations = [c for _, c in self._sources.values()]
        return {'num_sources': len(self._sources), 'avg_concentration': sum(concentrations) / len(concentrations), 'max_concentration': max(concentrations), 'min_concentration': min(concentrations)}