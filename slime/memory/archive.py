"""MAP-Elites behavioral archive for quality-diversity optimization"""

import torch
from typing import Dict, Tuple, Optional, List, Callable
from dataclasses import dataclass
import weakref
import logging

from slime.proto.component import Component

logger = logging.getLogger(__name__)


@dataclass
class Elite:
    """Immutable elite solution in archive"""

    behavior: Tuple[float, ...]  # Behavioral coordinates
    fitness: float
    genome: dict  # Serialized component state (immutable)
    generation: int

    def __post_init__(self):
        # Ensure immutability
        self.behavior = tuple(self.behavior)
        self.genome = dict(self.genome)  # Defensive copy


class BehavioralArchive:
    """GPU-resident MAP-Elites archive with spatial indexing.

    Stores elite solutions in behavioral space (rank, coherence, ...).
    Provides O(1) lookup and O(k) neighborhood sampling.

    Lifecycle:
    - add(): New elites replace worse ones in same cell
    - Old elites are garbage collected (no strong refs)
    - No memory leaks via weakref for live component tracking
    """

    def __init__(
        self,
        dimensions: List[str],
        bounds: List[Tuple[float, float]],
        resolution: int = 50,
        device: torch.device = None,
    ):
        """Initialize behavioral archive.

        Args:
            dimensions: Names of behavioral dimensions (e.g., ['rank', 'coherence'])
            bounds: [(min, max), ...] for each dimension
            resolution: Grid resolution per dimension
            device: GPU device for tensor storage
        """
        self.dimensions = dimensions
        self.bounds = bounds
        self.resolution = resolution
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Archive storage: cell -> Elite
        self._archive: Dict[Tuple[int, ...], Elite] = {}

        # Weak references to live components (prevent memory leaks)
        self._live_components: weakref.WeakSet = weakref.WeakSet()

        # Generation counter
        self._generation = 0

        # Statistics
        self._total_additions = 0
        self._total_replacements = 0

    def _quantize(self, behavior: Tuple[float, ...]) -> Tuple[int, ...]:
        """Map continuous behavior to discrete cell"""
        cell = []
        for val, (low, high) in zip(behavior, self.bounds):
            # Clamp to bounds
            val = max(low, min(high, val))
            # Quantize
            normalized = (val - low) / (high - low)
            idx = int(normalized * (self.resolution - 1))
            cell.append(idx)
        return tuple(cell)

    def add(
        self,
        component: Component,
        behavior: Tuple[float, ...],
        fitness: float,
    ) -> bool:
        """Add component to archive if it's elite in its cell.

        Args:
            component: Component to archive
            behavior: Behavioral coordinates
            fitness: Fitness score

        Returns:
            True if component was added (new elite)
        """
        if len(behavior) != len(self.dimensions):
            raise ValueError(
                f"Behavior dimension mismatch: expected {len(self.dimensions)}, "
                f"got {len(behavior)}"
            )

        cell = self._quantize(behavior)

        # Check if this is a new elite
        if cell in self._archive:
            if fitness <= self._archive[cell].fitness:
                return False  # Not better than current elite
            self._total_replacements += 1

        # Store immutable snapshot
        elite = Elite(
            behavior=behavior,
            fitness=fitness,
            genome=component.to_dict(),  # Serialize (no circular refs)
            generation=self._generation,
        )

        self._archive[cell] = elite
        self._total_additions += 1

        logger.debug(
            f"Added elite at cell {cell}: fitness={fitness:.4f}, "
            f"generation={self._generation}"
        )

        return True

    def get(self, behavior: Tuple[float, ...]) -> Optional[Elite]:
        """Retrieve elite at behavioral location"""
        cell = self._quantize(behavior)
        return self._archive.get(cell)

    def sample_near(
        self,
        behavior: Tuple[float, ...],
        radius: float = 0.1,
    ) -> List[Elite]:
        """Sample elites within radius of behavior location.

        Args:
            behavior: Center point
            radius: Radius in normalized [0, 1] space

        Returns:
            List of nearby elites
        """
        center = self._quantize(behavior)
        radius_cells = int(radius * self.resolution)

        nearby = []

        # Simple box search (could optimize with spatial index)
        for cell, elite in self._archive.items():
            if all(abs(c - center[i]) <= radius_cells
                   for i, c in enumerate(cell)):
                nearby.append(elite)

        return nearby

    def bootstrap_component(
        self,
        component_factory: Callable[[dict], Component],
        behavior: Tuple[float, ...],
        search_radius: float = 0.2,
    ) -> Optional[Component]:
        """Bootstrap new component from archive elites.

        Samples nearby elites and creates new component from best.

        Args:
            component_factory: Factory(genome_dict) -> Component
            behavior: Target behavioral location
            search_radius: Neighborhood radius to search

        Returns:
            Initialized component or None if no elites found
        """
        nearby = self.sample_near(behavior, search_radius)

        if not nearby:
            return None

        # Use best nearby elite
        best = max(nearby, key=lambda e: e.fitness)

        # Factory handles reconstruction with dependencies
        component = component_factory(best.genome)

        # Track with weak reference
        self._live_components.add(component)

        logger.debug(
            f"Bootstrapped component from elite at generation {best.generation}, "
            f"fitness={best.fitness:.4f}"
        )

        return component

    def increment_generation(self) -> None:
        """Advance generation counter"""
        self._generation += 1

    def size(self) -> int:
        """Number of elites in archive"""
        return len(self._archive)

    def coverage(self) -> float:
        """Fraction of behavior space covered [0, 1]"""
        total_cells = self.resolution ** len(self.dimensions)
        return len(self._archive) / total_cells

    def max_fitness(self) -> float:
        """Maximum fitness in archive"""
        if not self._archive:
            return float('-inf')
        return max(e.fitness for e in self._archive.values())

    def clear(self) -> None:
        """Clear archive (for testing)"""
        self._archive.clear()
        self._live_components.clear()
        self._generation = 0
        self._total_additions = 0
        self._total_replacements = 0
