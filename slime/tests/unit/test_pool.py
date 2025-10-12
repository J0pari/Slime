import pytest
import torch
from slime.memory.pool import DynamicPool, PoolConfig
from slime.core.pseudopod import Pseudopod

class TestDynamicPool:

    @pytest.fixture
    def pool_config(self):
        return PoolConfig(initial_size=4, max_size=8, spawn_threshold=0.7, cull_threshold=0.3)

    @pytest.fixture
    def pool(self, pool_config):
        return DynamicPool(latent_dim=64, head_dim=32, config=pool_config, device=torch.device('cpu'))

    def test_initialization(self, pool, pool_config):
        assert pool.size() == pool_config.initial_size
        assert pool.max_size == pool_config.max_size

    def test_spawn_component(self, pool):
        initial_size = pool.size()
        new_component = pool.spawn()
        assert new_component is not None
        assert isinstance(new_component, Pseudopod)
        assert pool.size() == initial_size + 1

    def test_max_size_enforcement(self, pool):
        while pool.size() < pool.max_size:
            pool.spawn()
        assert pool.size() == pool.max_size
        new_component = pool.spawn()
        assert new_component is None
        assert pool.size() == pool.max_size

    def test_cull_low_fitness(self, pool):
        for i, component in enumerate(pool.components):
            pool.fitness_scores[component.component_id] = float(i) / pool.size()
        initial_size = pool.size()
        culled = pool._cull_low_fitness()
        assert culled > 0
        assert pool.size() < initial_size
        for component in pool.components:
            fitness = pool.fitness_scores.get(component.component_id, 0.0)
            assert fitness >= pool.config.cull_threshold

    def test_update_fitness(self, pool):
        component = pool.components[0]
        component_id = component.component_id
        new_fitness = 0.85
        pool.update_fitness(component_id, new_fitness)
        assert pool.fitness_scores[component_id] == new_fitness

    def test_get_fitness(self, pool):
        component = pool.components[0]
        component_id = component.component_id
        pool.fitness_scores[component_id] = 0.75
        fitness = pool.get_fitness(component_id)
        assert fitness == 0.75
        fitness = pool.get_fitness(99999)
        assert fitness == 0.0

    def test_get_component_by_id(self, pool):
        component = pool.components[0]
        component_id = component.component_id
        retrieved = pool.get_component(component_id)
        assert retrieved is not None
        assert retrieved.component_id == component_id
        retrieved = pool.get_component(99999)
        assert retrieved is None

    def test_spawn_batch(self, pool):
        initial_size = pool.size()
        batch_size = 2
        spawned = pool._spawn_batch(batch_size)
        assert spawned == batch_size
        assert pool.size() == initial_size + batch_size

    def test_spawn_batch_respects_max_size(self, pool):
        while pool.size() < pool.max_size - 1:
            pool.spawn()
        spawned = pool._spawn_batch(3)
        assert spawned == 1
        assert pool.size() == pool.max_size

    def test_get_top_performers(self, pool):
        for i, component in enumerate(pool.components):
            pool.fitness_scores[component.component_id] = float(i)
        top = pool.get_top_performers(k=2)
        assert len(top) == 2
        assert top[0][1] >= top[1][1]

    def test_lifecycle_frozen(self, pool):
        pool.freeze_lifecycle()
        assert pool._lifecycle_frozen is True
        new_component = pool.spawn()
        assert new_component is None
        culled = pool._cull_low_fitness()
        assert culled == 0

    def test_lifecycle_unfreeze(self, pool):
        pool.freeze_lifecycle()
        pool.unfreeze_lifecycle()
        assert pool._lifecycle_frozen is False
        new_component = pool.spawn()
        assert new_component is not None

    def test_clear_pool(self, pool):
        assert pool.size() > 0
        pool.clear()
        assert pool.size() == 0
        assert len(pool.fitness_scores) == 0

    def test_get_statistics(self, pool):
        for i, component in enumerate(pool.components):
            pool.fitness_scores[component.component_id] = float(i) / pool.size()
        stats = pool.get_statistics()
        assert 'size' in stats
        assert 'max_size' in stats
        assert 'mean_fitness' in stats
        assert 'min_fitness' in stats
        assert 'max_fitness' in stats
        assert stats['size'] == pool.size()
        assert stats['max_size'] == pool.max_size

    def test_component_uniqueness(self, pool):
        for _ in range(3):
            pool.spawn()
        ids = [c.component_id for c in pool.components]
        assert len(ids) == len(set(ids))