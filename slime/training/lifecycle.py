import torch
import numpy as np
from typing import List, Optional, Callable
from dataclasses import dataclass
import logging
import hashlib
from slime.core.pseudopod import Pseudopod
from slime.memory.pool import DynamicPool
from slime.memory.archive import CVTArchive
from slime.topology.genealogy import Genealogy
logger = logging.getLogger(__name__)

@dataclass
class LifecycleConfig:
    initial_temp: float = 1.0
    min_temp: float = 0.01
    cooling_schedule: str = 'linear'
    max_pool_size: int = 64
    min_pool_size: int = 4
    max_loss_ratio: float = 10.0
    loss_ema_alpha: float = 0.99
    seed: int = 42

class LifecycleManager:

    def __init__(self, config: Optional[LifecycleConfig]=None):
        self.config = config or LifecycleConfig()
        self._lifecycle = None
        self._lifecycle_frozen = False

    def initialize(self, pool: DynamicPool, archive: CVTArchive):
        self._lifecycle = SimulatedAnnealingLifecycle(pool=pool, archive=archive, initial_temp=self.config.initial_temp, min_temp=self.config.min_temp, cooling_schedule=self.config.cooling_schedule, max_pool_size=self.config.max_pool_size, min_pool_size=self.config.min_pool_size, max_loss_ratio=self.config.max_loss_ratio, loss_ema_alpha=self.config.loss_ema_alpha, seed=self.config.seed)

    def set_max_steps(self, max_steps: int):
        if self._lifecycle:
            self._lifecycle.set_max_steps(max_steps)

    def step_lifecycle(self, current_loss: float):
        if self._lifecycle:
            self._lifecycle.step_lifecycle(current_loss)

    def should_spawn_component(self, behavior: np.ndarray, fitness: float, component_factory: Callable[[dict], Pseudopod]) -> Optional[Pseudopod]:
        if self._lifecycle:
            return self._lifecycle.should_spawn_component(behavior, fitness, component_factory)
        return None

    def should_cull_component(self, component: Pseudopod) -> bool:
        if self._lifecycle:
            return self._lifecycle.should_cull_component(component)
        return False

    def get_statistics(self) -> dict:
        if self._lifecycle:
            return self._lifecycle.get_statistics()
        return {}

class SimulatedAnnealingLifecycle:

    def __init__(self, pool: DynamicPool, archive: CVTArchive, initial_temp: float=1.0, min_temp: float=0.01, cooling_schedule: str='linear', max_pool_size: int=64, min_pool_size: int=4, max_loss_ratio: float=10.0, loss_ema_alpha: float=0.99, seed: int=42):
        self.pool = pool
        self.archive = archive
        self.genealogy = Genealogy()  # Track lineages
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.cooling_schedule = cooling_schedule
        self.max_pool_size = max_pool_size
        self.min_pool_size = min_pool_size
        self.max_loss_ratio = max_loss_ratio
        self.seed = seed
        self.loss_ema = None
        self.loss_ema_alpha = loss_ema_alpha
        self.step = 0
        self.max_steps = None
        self.lifecycle_frozen = False
        self._birth_history = []
        self._death_history = []
        self._temperature_history = []

        # Register initial pool as genesis pseudopods
        for component in self.pool._components:
            self.genealogy.register_genesis(component.component_id)

    def _deterministic_random(self, context: str) -> float:
        hash_input = f'{self.seed}:{self.step}:{context}'.encode('utf-8')
        hash_digest = hashlib.sha256(hash_input).digest()
        random_bytes = int.from_bytes(hash_digest[:8], byteorder='big')
        return random_bytes / (2 ** 64 - 1)

    def set_max_steps(self, max_steps: int):
        self.max_steps = max_steps

    def get_temperature(self) -> float:
        if self.max_steps is None or self.step == 0:
            return self.initial_temp
        progress = min(1.0, self.step / self.max_steps)
        if self.cooling_schedule == 'linear':
            temp = self.initial_temp * (1.0 - progress)
        elif self.cooling_schedule == 'exponential':
            temp = self.initial_temp * 0.95 ** (progress * 100)
        elif self.cooling_schedule == 'logarithmic':
            temp = self.initial_temp / (1.0 + np.log(1.0 + progress * 10))
        else:
            temp = self.initial_temp
        return max(self.min_temp, temp)

    def update_loss_ema(self, current_loss: float):
        if self.loss_ema is None:
            self.loss_ema = current_loss
        else:
            self.loss_ema = self.loss_ema_alpha * self.loss_ema + (1 - self.loss_ema_alpha) * current_loss
        if self.loss_ema is not None and current_loss > self.max_loss_ratio * self.loss_ema:
            if not self.lifecycle_frozen:
                logger.warning(f'Loss diverging: {current_loss:.4f} > {self.max_loss_ratio}x EMA ({self.loss_ema:.4f}). Freezing lifecycle.')
                self.lifecycle_frozen = True
        elif self.lifecycle_frozen and current_loss < 2.0 * self.loss_ema:
            logger.info(f'Loss stabilized. Unfreezing lifecycle.')
            self.lifecycle_frozen = False

    def birth_probability(self, fitness: float, archive_max_fitness: float, temperature: float) -> float:
        if fitness >= archive_max_fitness * 0.8:
            return 1.0
        if archive_max_fitness <= 0:
            return 0.5
        fitness_deficit = archive_max_fitness - fitness
        acceptance_prob = np.exp(-fitness_deficit / (temperature * archive_max_fitness + 1e-06))
        return acceptance_prob

    def should_spawn_component(self, behavior: np.ndarray, fitness: float, component_factory: Callable[[dict], Pseudopod]) -> Optional[Pseudopod]:
        if self.lifecycle_frozen:
            return None
        if len(self.pool._components) >= self.max_pool_size:
            return None
        temperature = self.get_temperature()
        archive_max_fitness = self.archive.max_fitness()
        if archive_max_fitness == float('-inf'):
            return None
        birth_prob = self.birth_probability(fitness, archive_max_fitness, temperature)
        behavior_hash = hashlib.sha256(str(behavior).encode()).hexdigest()[:8]
        random_value = self._deterministic_random(f'birth:{behavior_hash}')
        if random_value < birth_prob:
            behavior_tuple = tuple(behavior.tolist())
            # Find parent (closest behavior in pool)
            parent_id = self.pool._components[0].component_id if len(self.pool._components) > 0 else None
            
            component = self.archive.bootstrap_component(component_factory, behavior_tuple, k_neighbors=3, mutation_std=temperature * 0.5)
            if component is not None:
                # Register spawn in genealogy
                if parent_id is not None:
                    self.genealogy.register_spawn(parent_id, component.component_id)
                else:
                    self.genealogy.register_genesis(component.component_id)
                    
                self._birth_history.append({'step': self.step, 'temperature': temperature, 'fitness': fitness, 'birth_prob': birth_prob, 'behavior': behavior_tuple, 'parent_id': parent_id})
                logger.info(f'Spawned component {component.component_id} from parent {parent_id}: fitness={fitness:.4f}, temp={temperature:.3f}, prob={birth_prob:.3f}')
                return component
        return None

    def should_cull_component(self, component: Pseudopod) -> bool:
        if self.lifecycle_frozen:
            return False
        if len(self.pool._components) <= self.min_pool_size:
            return False
        temperature = self.get_temperature()
        archive_max_fitness = self.archive.max_fitness()
        if archive_max_fitness == float('-inf'):
            return False
        component_fitness = component.fitness
        fitness_gap = archive_max_fitness - component_fitness
        death_prob = 1.0 - np.exp(-fitness_gap / (temperature * archive_max_fitness + 1e-06))
        low_temp_threshold = 0.2
        if temperature < low_temp_threshold and component_fitness < archive_max_fitness * 0.3:
            death_prob = max(death_prob, 0.8)
        component_hash = hashlib.sha256(str(component.component_id).encode()).hexdigest()[:8]
        random_value = self._deterministic_random(f'cull:{component_hash}')
        should_cull = random_value < death_prob
        if should_cull:
            self._death_history.append({'step': self.step, 'temperature': temperature, 'fitness': component_fitness, 'death_prob': death_prob, 'gap': fitness_gap})
            logger.info(f'Culled component: fitness={component_fitness:.4f}, temp={temperature:.3f}, prob={death_prob:.3f}')
        return should_cull

    def refine_archive_centroids(self, behavioral_samples: np.ndarray, num_iterations: int=100):
        if self.archive.centroids is None:
            logger.warning('Archive centroids not initialized, skipping refinement')
            return
        temperature = self.get_temperature()
        centroids = self.archive.centroids.copy()

        def coverage_metric(centroids_candidate):
            from scipy.spatial import distance
            distances = distance.cdist(behavioral_samples, centroids_candidate, metric='euclidean')
            nearest_dists = np.min(distances, axis=1)
            return -np.mean(nearest_dists)
        current_coverage = coverage_metric(centroids)
        for iteration in range(num_iterations):
            iter_seed = self.seed + self.step * 1000 + iteration
            rng = np.random.RandomState(iter_seed)
            perturbation = rng.randn(*centroids.shape) * temperature * 0.1
            new_centroids = centroids + perturbation
            new_coverage = coverage_metric(new_centroids)
            delta_coverage = new_coverage - current_coverage
            if delta_coverage > 0:
                centroids = new_centroids
                current_coverage = new_coverage
            else:
                accept_prob = np.exp(delta_coverage / (temperature + 1e-06))
                random_value = self._deterministic_random(f'centroid_refine:{iteration}')
                if random_value < accept_prob:
                    centroids = new_centroids
                    current_coverage = new_coverage
        self.archive.centroids = centroids
        logger.debug(f'Refined centroids: coverage improvement from annealing')

    def step_lifecycle(self, current_loss: float):
        self.step += 1
        temperature = self.get_temperature()
        self._temperature_history.append(temperature)
        self.update_loss_ema(current_loss)
        if self.step % 1000 == 0 and (not self.lifecycle_frozen):
            logger.info(f'Lifecycle step {self.step}: temp={temperature:.3f}, pool_size={len(self.pool._components)}, archive_size={self.archive.size()}, loss_ema={self.loss_ema:.4f}, frozen={self.lifecycle_frozen}')

    def get_statistics(self) -> dict:
        # Compute genealogy diversity
        pool_ids = [c.component_id for c in self.pool._components]
        phylo_diversity = self.genealogy.phylogenetic_diversity(pool_ids) if len(pool_ids) > 1 else 0.0
        
        return {'step': self.step, 'temperature': self.get_temperature(), 'pool_size': len(self.pool._components), 'archive_size': self.archive.size(), 'archive_coverage': self.archive.coverage(), 'loss_ema': self.loss_ema, 'frozen': self.lifecycle_frozen, 'total_births': len(self._birth_history), 'total_deaths': len(self._death_history), 'archive_max_fitness': self.archive.max_fitness(), 'phylogenetic_diversity': phylo_diversity}

    def clear_history(self):
        self._birth_history.clear()
        self._death_history.clear()
        self._temperature_history.clear()