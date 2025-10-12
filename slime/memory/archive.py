import torch
from typing import Dict, Tuple, Optional, List, Callable
from dataclasses import dataclass
import weakref
import logging
from slime.proto.component import Component
logger = logging.getLogger(__name__)

@dataclass
class Elite:
    behavior: Tuple[float, ...]
    fitness: float
    genome: dict
    generation: int

    def __post_init__(self):
        self.behavior = tuple(self.behavior)
        self.genome = dict(self.genome)

class BehavioralArchive:

    def __init__(self, dimensions: List[str], bounds: List[Tuple[float, float]], resolution: int=50, device: torch.device=None):
        self.dimensions = dimensions
        self.bounds = bounds
        self.resolution = resolution
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._archive: Dict[Tuple[int, ...], Elite] = {}
        self._live_components: weakref.WeakSet = weakref.WeakSet()
        self._generation = 0
        self._total_additions = 0
        self._total_replacements = 0

    def _quantize(self, behavior: Tuple[float, ...]) -> Tuple[int, ...]:
        cell = []
        for val, (low, high) in zip(behavior, self.bounds):
            val = max(low, min(high, val))
            normalized = (val - low) / (high - low)
            idx = int(normalized * (self.resolution - 1))
            cell.append(idx)
        return tuple(cell)

    def add(self, component: Component, behavior: Tuple[float, ...], fitness: float) -> bool:
        if len(behavior) != len(self.dimensions):
            raise ValueError(f'Behavior dimension mismatch: expected {len(self.dimensions)}, got {len(behavior)}')
        cell = self._quantize(behavior)
        if cell in self._archive:
            if fitness <= self._archive[cell].fitness:
                return False
            self._total_replacements += 1
        elite = Elite(behavior=behavior, fitness=fitness, genome=component.to_dict(), generation=self._generation)
        self._archive[cell] = elite
        self._total_additions += 1
        logger.debug(f'Added elite at cell {cell}: fitness={fitness:.4f}, generation={self._generation}')
        return True

    def get(self, behavior: Tuple[float, ...]) -> Optional[Elite]:
        cell = self._quantize(behavior)
        return self._archive.get(cell)

    def sample_near(self, behavior: Tuple[float, ...], radius: float=0.1) -> List[Elite]:
        center = self._quantize(behavior)
        radius_cells = int(radius * self.resolution)
        nearby = []
        for cell, elite in self._archive.items():
            if all((abs(c - center[i]) <= radius_cells for i, c in enumerate(cell))):
                nearby.append(elite)
        return nearby

    def bootstrap_component(self, component_factory: Callable[[dict], Component], behavior: Tuple[float, ...], search_radius: float=0.2) -> Optional[Component]:
        nearby = self.sample_near(behavior, search_radius)
        if not nearby:
            return None
        best = max(nearby, key=lambda e: e.fitness)
        component = component_factory(best.genome)
        self._live_components.add(component)
        logger.debug(f'Bootstrapped component from elite at generation {best.generation}, fitness={best.fitness:.4f}')
        return component

    def increment_generation(self) -> None:
        self._generation += 1

    def size(self) -> int:
        return len(self._archive)

    def coverage(self) -> float:
        total_cells = self.resolution ** len(self.dimensions)
        return len(self._archive) / total_cells

    def max_fitness(self) -> float:
        if not self._archive:
            return float('-inf')
        return max((e.fitness for e in self._archive.values()))

    def clear(self) -> None:
        self._archive.clear()
        self._live_components.clear()
        self._generation = 0
        self._total_additions = 0
        self._total_replacements = 0