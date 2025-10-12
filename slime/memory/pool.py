import torch
from typing import List, Optional, Callable, Tuple
from dataclasses import dataclass
import logging
import weakref
from slime.proto.component import Component
from slime.core.comonad import SpatialContext, behavioral_distance, extract_relative_fitness, extract_behavioral_divergence, extract_gradient_magnitude_rank, extract_attention_coherence
logger = logging.getLogger(__name__)

@dataclass
class PoolConfig:
    min_size: int = 1
    max_size: Optional[int] = None
    birth_threshold: float = 0.8
    death_threshold: float = 0.1
    cull_interval: int = 100
    spawn_batch_size: int = 1

class DynamicPool:

    def __init__(self, component_factory: Callable[[], Component], config: PoolConfig, bootstrap_factory: Optional[Callable[[dict], Component]]=None, archive: Optional['CVTArchive']=None):
        self.factory = component_factory
        self.bootstrap_factory = bootstrap_factory if bootstrap_factory is not None else component_factory
        self.config = config
        self.archive = archive
        self._components: List[Component] = []
        for _ in range(config.min_size):
            self._components.append(self.factory())
        self._step = 0
        self._total_spawned = config.min_size
        self._total_culled = 0
        self._consumers = weakref.WeakSet()

    def _spawn_component(self, behavior_location: Optional[Tuple[float, ...]]=None) -> Component:
        if self.archive is not None and behavior_location is not None:
            component = self.archive.bootstrap_component(self.bootstrap_factory, behavior_location)
            if component is not None:
                logger.debug('Bootstrapped component from archive')
                return component
        return self.factory()

    def get_all(self) -> List[Component]:
        return list(self._components)

    def get_at(self, behavior_location: Tuple[float, ...], max_count: Optional[int]=None) -> List[Component]:
        if behavior_location is None:
            components = self._components
        else:
            components_with_dist = []
            for comp in self._components:
                if hasattr(comp, 'last_behavior'):
                    dist = sum(((a - b) ** 2 for a, b in zip(comp.last_behavior, behavior_location)))
                    components_with_dist.append((dist, comp))
                else:
                    components_with_dist.append((float('inf'), comp))
            components_with_dist.sort(key=lambda x: x[0])
            components = [comp for _, comp in components_with_dist]
        if max_count is not None:
            components = components[:max_count]
        return components

    def step(self, behavior_location: Optional[Tuple[float, ...]]=None) -> None:
        self._step += 1
        if self._step % self.config.cull_interval == 0:
            self._cull_low_fitness()
        if self._should_spawn():
            self._spawn_batch(behavior_location)

    def _cull_low_fitness(self) -> None:
        before = len(self._components)
        surviving = [c for c in self._components if c.fitness >= self.config.death_threshold]
        if len(surviving) < self.config.min_size:
            surviving = sorted(self._components, key=lambda c: c.fitness, reverse=True)[:self.config.min_size]
        culled = before - len(surviving)
        if culled > 0:
            self._components = surviving
            self._total_culled += culled
            logger.debug(f'Culled {culled} components (fitness < {self.config.death_threshold})')

    def _should_spawn(self) -> bool:
        if self.config.max_size is not None:
            if len(self._components) >= self.config.max_size:
                return False
        if not self._components:
            return True
        avg_fitness = sum((c.fitness for c in self._components)) / len(self._components)
        return avg_fitness >= self.config.birth_threshold

    def _spawn_batch(self, behavior_location: Optional[Tuple[float, ...]]=None) -> None:
        for _ in range(self.config.spawn_batch_size):
            if self.config.max_size is not None:
                if len(self._components) >= self.config.max_size:
                    break
            component = self._spawn_component(behavior_location)
            self._components.append(component)
            self._total_spawned += 1
        logger.debug(f'Spawned {self.config.spawn_batch_size} components (total={len(self._components)})')

    def size(self) -> int:
        return len(self._components)

    def stats(self) -> dict:
        return {'size': len(self._components), 'total_spawned': self._total_spawned, 'total_culled': self._total_culled, 'step': self._step, 'avg_fitness': sum((c.fitness for c in self._components)) / len(self._components) if self._components else 0.0}

    def clear(self) -> None:
        self._components.clear()
        self._step = 0
        self._total_spawned = 0
        self._total_culled = 0

    def extract_component_context(self, component: Component) -> SpatialContext:
        neighborhood = [c for c in self._components if c is not component]
        return SpatialContext(focus=component, neighborhood=neighborhood, distance_fn=behavioral_distance)

    def compute_contextual_fitness(self, component: Component) -> float:
        ctx = self.extract_component_context(component)
        return extract_relative_fitness(ctx)

    def compute_behavioral_divergence(self, component: Component) -> torch.Tensor:
        ctx = self.extract_component_context(component)
        return extract_behavioral_divergence(ctx)

    def compute_gradient_rank(self, component: Component) -> float:
        ctx = self.extract_component_context(component)
        return extract_gradient_magnitude_rank(ctx)

    def compute_attention_coherence(self, component: Component) -> float:
        ctx = self.extract_component_context(component)
        return extract_attention_coherence(ctx)