import torch
from typing import Tuple, Optional, Dict
import logging
from slime.memory.archive import BehavioralArchive
logger = logging.getLogger(__name__)

class Chemotaxis:

    def __init__(self, archive: BehavioralArchive, device: Optional[torch.device]=None):
        self.archive = archive
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._sources: Dict[Tuple[int, ...], Tuple[torch.Tensor, float]] = {}

    def add_source(self, nutrient: torch.Tensor, location: Tuple[float, float], concentration: float) -> None:
        if not isinstance(nutrient, torch.Tensor):
            raise TypeError(f'nutrient must be torch.Tensor, got {type(nutrient)}')
        cell = self.archive._quantize(location)
        if cell not in self._sources or self._sources[cell][1] < concentration:
            self._sources[cell] = (nutrient.detach().to(self.device), float(concentration))
            logger.debug(f'Added nutrient source at {cell}, concentration={concentration:.4f}')

    def sample(self, behavior: Tuple[float, float], metabolic_rate: float=1.0, hunger: float=0.0) -> Optional[torch.Tensor]:
        if not self._sources:
            return None
        center_cell = self.archive._quantize(behavior)
        nearby = []
        for cell, (nutrient, concentration) in self._sources.items():
            distance = sum((abs(c - center_cell[i]) for i, c in enumerate(cell)))
            if distance <= 1:
                nearby.append((nutrient, concentration, distance))
        if not nearby:
            nearby = [(nutrient, concentration, sum((abs(c - center_cell[i]) for i, c in enumerate(cell)))) for cell, (nutrient, concentration) in self._sources.items()]
        weights = []
        for nutrient, concentration, distance in nearby:
            weight = (concentration + hunger * 0.1) / (distance + 1.0)
            weight = weight / (metabolic_rate + 1e-10)
            weights.append(weight)
        weights_tensor = torch.tensor(weights, device=self.device)
        weights_tensor = weights_tensor - weights_tensor.max()
        weights_tensor = torch.softmax(weights_tensor, dim=0)
        nutrients = torch.stack([nutrient for nutrient, _, _ in nearby])
        sampled = torch.sum(nutrients * weights_tensor.view(-1, 1, 1), dim=0)
        return sampled

    def clear(self) -> None:
        self._sources.clear()

    def stats(self) -> dict:
        if not self._sources:
            return {'num_sources': 0, 'avg_concentration': 0.0, 'max_concentration': 0.0}
        concentrations = [c for _, c in self._sources.values()]
        return {'num_sources': len(self._sources), 'avg_concentration': sum(concentrations) / len(concentrations), 'max_concentration': max(concentrations), 'min_concentration': min(concentrations)}