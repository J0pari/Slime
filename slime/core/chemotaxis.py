"""Chemotaxis: behavioral space navigator"""

import torch
from typing import Tuple, Optional, Dict
import logging

from slime.memory.archive import BehavioralArchive

logger = logging.getLogger(__name__)


class Chemotaxis:
    """Navigate behavioral space (rank, coherence) to find nutrients.

    Implements proto.model.Chemotaxis protocol.

    Biological inspiration: Slime molds follow chemical gradients to food.
    Here: "food" = high-fitness regions of behavioral space.

    Uses archive for spatial indexing (which elites exist where).
    """

    def __init__(
        self,
        archive: BehavioralArchive,
        device: Optional[torch.device] = None,
    ):
        """Initialize chemotaxis navigator.

        Args:
            archive: Behavioral archive for spatial queries
            device: Computation device
        """
        self.archive = archive
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Nutrient sources: behavior_coords -> (nutrient_tensor, concentration)
        self._sources: Dict[Tuple[int, ...], Tuple[torch.Tensor, float]] = {}

    def add_source(
        self,
        nutrient: torch.Tensor,
        location: Tuple[float, float],
        concentration: float,
    ) -> None:
        """Deposit nutrient at behavioral coordinates.

        Args:
            location: (rank, coherence) coordinates in [0, 1]²
            nutrient: Tensor to store
            concentration: Strength of deposit (fitness)
        """
        if not isinstance(nutrient, torch.Tensor):
            raise TypeError(f"nutrient must be torch.Tensor, got {type(nutrient)}")

        # Quantize to archive cell
        cell = self.archive._quantize(location)

        # Store if better than existing or new cell
        if cell not in self._sources or self._sources[cell][1] < concentration:
            self._sources[cell] = (nutrient.detach().to(self.device), float(concentration))

            logger.debug(f"Added nutrient source at {cell}, concentration={concentration:.4f}")

    def sample(
        self,
        behavior: Tuple[float, float],
        metabolic_rate: float = 1.0,
        hunger: float = 0.0,
    ) -> Optional[torch.Tensor]:
        """Sample nutrient proportional to concentration and proximity.

        Args:
            behavior: Current (rank, coherence) location
            metabolic_rate: Temperature (high = more exploration)
            hunger: Bias toward high-concentration areas

        Returns:
            Weighted sum of nearby nutrients or None if empty
        """
        if not self._sources:
            return None

        # Get cell at current behavior
        center_cell = self.archive._quantize(behavior)

        # Find nearby sources (within 1 cell Manhattan distance)
        nearby = []
        for cell, (nutrient, concentration) in self._sources.items():
            distance = sum(abs(c - center_cell[i]) for i, c in enumerate(cell))
            if distance <= 1:  # Adjacent or same cell
                nearby.append((nutrient, concentration, distance))

        if not nearby:
            # Fallback: use all sources with distance weighting
            nearby = [
                (nutrient, concentration, sum(abs(c - center_cell[i]) for i, c in enumerate(cell)))
                for cell, (nutrient, concentration) in self._sources.items()
            ]

        # Compute weights: concentration / (distance + 1)
        # Higher metabolic_rate = flatter distribution (more exploration)
        # Higher hunger = sharper distribution (exploitation)
        weights = []
        for nutrient, concentration, distance in nearby:
            weight = (concentration + hunger * 0.1) / (distance + 1.0)
            weight = weight / (metabolic_rate + 1e-10)
            weights.append(weight)

        weights_tensor = torch.tensor(weights, device=self.device)

        # Numerically stable softmax
        weights_tensor = weights_tensor - weights_tensor.max()
        weights_tensor = torch.softmax(weights_tensor, dim=0)

        # Weighted sum
        nutrients = torch.stack([nutrient for nutrient, _, _ in nearby])
        sampled = torch.sum(nutrients * weights_tensor.view(-1, 1, 1), dim=0)

        return sampled

    def clear(self) -> None:
        """Clear all nutrient sources."""
        self._sources.clear()

    def stats(self) -> dict:
        """Get statistics about nutrient distribution."""
        if not self._sources:
            return {
                'num_sources': 0,
                'avg_concentration': 0.0,
                'max_concentration': 0.0,
            }

        concentrations = [c for _, c in self._sources.values()]

        return {
            'num_sources': len(self._sources),
            'avg_concentration': sum(concentrations) / len(concentrations),
            'max_concentration': max(concentrations),
            'min_concentration': min(concentrations),
        }
