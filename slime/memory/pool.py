"""Dynamic component pools with lifecycle management"""

import torch
from typing import List, Optional, Callable, Tuple
from dataclasses import dataclass
import logging
import weakref

from slime.proto.component import Component

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Configuration for dynamic pool"""

    min_size: int = 1
    max_size: Optional[int] = None  # None = unbounded
    birth_threshold: float = 0.8  # Spawn when avg fitness > this
    death_threshold: float = 0.1  # Cull when individual fitness < this
    cull_interval: int = 100  # Check for culling every N steps
    spawn_batch_size: int = 1  # Spawn this many at once


class DynamicPool:
    """Self-organizing pool with apoptosis.

    Manages component lifecycle:
    - Birth: Spawn when pool fitness is high (demand exceeds supply)
    - Death: Cull low-fitness components (apoptosis)
    - No memory leaks: Weak references to external users
    - No cancer: Bounded growth via max_size

    DRY: Single pool implementation used for pseudopods, tubes, etc.
    """

    def __init__(
        self,
        component_factory: Callable[[], Component],
        config: PoolConfig,
        bootstrap_factory: Optional[Callable[[dict], Component]] = None,
        archive: Optional['BehavioralArchive'] = None,
    ):
        """Initialize dynamic pool.

        Args:
            component_factory: Factory function to create new components
            bootstrap_factory: Factory(genome) -> Component for archive spawns
            config: Pool configuration
            archive: Optional archive for bootstrapping
        """
        self.factory = component_factory
        self.bootstrap_factory = bootstrap_factory if bootstrap_factory is not None else component_factory
        self.config = config
        self.archive = archive

        # Active components
        self._components: List[Component] = []

        # Initialize to min_size
        for _ in range(config.min_size):
            self._components.append(self.factory())

        # Statistics
        self._step = 0
        self._total_spawned = config.min_size
        self._total_culled = 0

        # Weak references to consumers (prevent leaks)
        self._consumers = weakref.WeakSet()

    def _spawn_component(
        self,
        behavior_location: Optional[Tuple[float, ...]] = None,
    ) -> Component:
        """Spawn new component, optionally bootstrapping from archive"""

        if self.archive is not None and behavior_location is not None:
            # Try bootstrapping from archive using bootstrap_factory
            component = self.archive.bootstrap_component(
                self.bootstrap_factory,
                behavior_location,
            )
            if component is not None:
                logger.debug("Bootstrapped component from archive")
                return component

        # Create from scratch
        return self.factory()

    def get_all(self) -> List[Component]:
        """Get all active components"""
        return list(self._components)

    def get_at(
        self,
        behavior_location: Tuple[float, ...],
        max_count: Optional[int] = None,
    ) -> List[Component]:
        """Get components relevant to behavioral location.

        For now, returns all components. Future: spatial indexing.

        Args:
            behavior_location: Target location in behavior space
            max_count: Maximum components to return

        Returns:
            List of relevant components
        """
        if behavior_location is None:
            components = self._components
        else:
            # Sort by distance to behavior location
            components_with_dist = []
            for comp in self._components:
                if hasattr(comp, 'last_behavior'):
                    dist = sum((a - b) ** 2 for a, b in zip(comp.last_behavior, behavior_location))
                    components_with_dist.append((dist, comp))
                else:
                    components_with_dist.append((float('inf'), comp))

            components_with_dist.sort(key=lambda x: x[0])
            components = [comp for _, comp in components_with_dist]

        if max_count is not None:
            components = components[:max_count]

        return components

    def step(self, behavior_location: Optional[Tuple[float, ...]] = None) -> None:
        """Execute lifecycle management step.

        Args:
            behavior_location: Current location (for spawning)
        """
        self._step += 1

        # Periodic culling (apoptosis)
        if self._step % self.config.cull_interval == 0:
            self._cull_low_fitness()

        # Check if we should spawn
        if self._should_spawn():
            self._spawn_batch(behavior_location)

    def _cull_low_fitness(self) -> None:
        """Remove components below death threshold"""
        before = len(self._components)

        # Keep components above threshold or minimum count
        surviving = [
            c for c in self._components
            if c.fitness >= self.config.death_threshold
        ]

        # Ensure minimum size
        if len(surviving) < self.config.min_size:
            surviving = sorted(
                self._components,
                key=lambda c: c.fitness,
                reverse=True,
            )[:self.config.min_size]

        culled = before - len(surviving)

        if culled > 0:
            self._components = surviving
            self._total_culled += culled
            logger.debug(f"Culled {culled} components (fitness < {self.config.death_threshold})")

    def _should_spawn(self) -> bool:
        """Check if pool should spawn new components"""
        # Don't exceed max size
        if self.config.max_size is not None:
            if len(self._components) >= self.config.max_size:
                return False

        # Spawn if average fitness is high (demand signal)
        if not self._components:
            return True

        avg_fitness = sum(c.fitness for c in self._components) / len(self._components)
        return avg_fitness >= self.config.birth_threshold

    def _spawn_batch(self, behavior_location: Optional[Tuple[float, ...]] = None) -> None:
        """Spawn batch of new components"""
        for _ in range(self.config.spawn_batch_size):
            if self.config.max_size is not None:
                if len(self._components) >= self.config.max_size:
                    break

            component = self._spawn_component(behavior_location)
            self._components.append(component)
            self._total_spawned += 1

        logger.debug(
            f"Spawned {self.config.spawn_batch_size} components "
            f"(total={len(self._components)})"
        )

    def size(self) -> int:
        """Current pool size"""
        return len(self._components)

    def stats(self) -> dict:
        """Pool statistics"""
        return {
            'size': len(self._components),
            'total_spawned': self._total_spawned,
            'total_culled': self._total_culled,
            'step': self._step,
            'avg_fitness': (
                sum(c.fitness for c in self._components) / len(self._components)
                if self._components else 0.0
            ),
        }

    def clear(self) -> None:
        """Clear pool (for testing)"""
        self._components.clear()
        self._step = 0
        self._total_spawned = 0
        self._total_culled = 0
