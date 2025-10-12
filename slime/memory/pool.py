import torch
from typing import List, Optional, Callable, Tuple
from dataclasses import dataclass
import logging
import weakref
from slime.proto.component import Component
from slime.core.stencil import SpatialStencil
from slime.config.dimensions import ArchitectureConfig
from slime.gpu.comonad import GPUContext, make_spawn_retire_decisions
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

    def __init__(self, component_factory: Callable[[], Component], config: PoolConfig, arch_config: ArchitectureConfig, bootstrap_factory: Optional[Callable[[dict], Component]]=None, archive: Optional['CVTArchive']=None, device: Optional[torch.device]=None, gpu_context: Optional[GPUContext]=None):
        self.factory = component_factory
        self.bootstrap_factory = bootstrap_factory if bootstrap_factory is not None else component_factory
        self.config = config
        self.archive = archive
        self.device = device or torch.device('cuda')
        self._components: List[Component] = []
        for _ in range(config.min_size):
            self._components.append(self.factory())
        self._step = 0
        self._total_spawned = config.min_size
        self._total_culled = 0
        self._consumers = weakref.WeakSet()
        self.stencil = SpatialStencil(k_neighbors=arch_config.k_neighbors, distance_metric='euclidean', device=self.device)

        # GPU comonadic context for resource-aware spawn/retire
        self.gpu_context = gpu_context or GPUContext(num_warps=32, device=self.device)

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
                if hasattr(comp, 'last_behavior') and comp.last_behavior is not None:
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

        # GPU-aware retire decision: context-aware culling
        should_spawn, retire_count = make_spawn_retire_decisions(
            self.gpu_context,
            pool_size=len(self._components),
            max_pool_size=self.config.max_size or 64
        )

        # Combine GPU retire count with curiosity-driven culling
        # Curiosity-driven: cull components with low learning progress (high hunger)
        # coherence() → learning progress, hunger = 1 - coherence
        components_with_hunger = []
        for c in self._components:
            coherence_val = c.coherence().item() if hasattr(c, 'coherence') else c.fitness
            hunger = 1.0 - coherence_val
            components_with_hunger.append((hunger, c))

        # Sort by hunger (high hunger first = low learning progress)
        components_with_hunger.sort(key=lambda x: x[0], reverse=True)

        # Retire top N by hunger + GPU-recommended retire count
        curiosity_threshold = 1.0 - self.config.death_threshold
        curiosity_cull_count = sum(1 for hunger, _ in components_with_hunger if hunger > curiosity_threshold)
        total_cull = max(curiosity_cull_count, retire_count)

        # Keep at least min_size
        total_cull = min(total_cull, before - self.config.min_size)

        surviving = [c for _, c in components_with_hunger[total_cull:]]

        culled = before - len(surviving)
        if culled > 0:
            self._components = surviving
            self._total_culled += culled
            logger.debug(f'Culled {culled} components (curiosity: {curiosity_cull_count}, GPU context: {retire_count})')

    def _should_spawn(self) -> bool:
        if self.config.max_size is not None:
            if len(self._components) >= self.config.max_size:
                return False
        if not self._components:
            return True

        # GPU-aware spawn decision
        should_spawn_gpu, _ = make_spawn_retire_decisions(
            self.gpu_context,
            pool_size=len(self._components),
            max_pool_size=self.config.max_size or 64
        )

        # Curiosity-driven spawn: high average learning progress
        avg_coherence = sum((c.coherence().item() if hasattr(c, 'coherence') else c.fitness
                            for c in self._components)) / len(self._components)
        should_spawn_curiosity = avg_coherence >= self.config.birth_threshold

        # Spawn if EITHER GPU context or curiosity recommends it
        return should_spawn_gpu or should_spawn_curiosity

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

    def _gather_component_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._components:
            return (torch.zeros((0, 5), device=self.device), torch.zeros(0, device=self.device), torch.zeros(0, device=self.device), torch.zeros((0, 1, 1, 16, 16), device=self.device))
        behaviors = []
        fitnesses = []
        grad_norms = []
        ca_patterns = []
        for comp in self._components:
            if hasattr(comp, 'last_behavior') and comp.last_behavior is not None:
                behaviors.append(comp.last_behavior)
            else:
                behaviors.append(torch.zeros(5, device=self.device))
            fitnesses.append(comp.fitness if hasattr(comp, 'fitness') else 0.0)
            grad_norm_list = []
            for param in comp.parameters():
                if param.grad is not None:
                    grad_norm_list.append(torch.norm(param.grad).item())
            grad_norms.append(sum(grad_norm_list) / len(grad_norm_list) if grad_norm_list else 0.0)
            if hasattr(comp, '_last_ca_pattern') and comp._last_ca_pattern is not None:
                ca_patterns.append(comp._last_ca_pattern)
            else:
                ca_patterns.append(torch.zeros(1, 1, 16, 16, device=self.device))
        behaviors_tensor = torch.stack(behaviors)
        fitnesses_tensor = torch.tensor(fitnesses, device=self.device, dtype=torch.float32)
        grad_norms_tensor = torch.tensor(grad_norms, device=self.device, dtype=torch.float32)
        ca_pattern_tensor = torch.stack(ca_patterns)
        return (behaviors_tensor, fitnesses_tensor, grad_norms_tensor, ca_pattern_tensor)

    def compute_all_contextual_metrics(self) -> dict:
        behaviors, fitnesses, grad_norms, ca_patterns = self._gather_component_tensors()
        return self.stencil.compute_all_contexts(behaviors, fitnesses, grad_norms, ca_patterns)

    def compute_contextual_fitness(self, component: Component) -> float:
        results = self.compute_all_contextual_metrics()
        component_idx = self._components.index(component)
        return results['relative_fitness'][component_idx].item()

    def compute_behavioral_divergence(self, component: Component) -> torch.Tensor:
        results = self.compute_all_contextual_metrics()
        component_idx = self._components.index(component)
        return results['behavioral_divergence'][component_idx]

    def compute_gradient_rank(self, component: Component) -> float:
        results = self.compute_all_contextual_metrics()
        component_idx = self._components.index(component)
        return results['gradient_rank'][component_idx].item()

    def compute_ca_coherence(self, component: Component) -> float:
        results = self.compute_all_contextual_metrics()
        component_idx = self._components.index(component)
        return results['ca_coherence'][component_idx].item()