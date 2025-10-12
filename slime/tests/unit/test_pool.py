"""Unit tests for DynamicPool (component lifecycle management)"""

import pytest
import torch
from slime.memory.pool import DynamicPool, PoolConfig
from slime.core.pseudopod import Pseudopod


class TestDynamicPool:
    """Test dynamic pool lifecycle management"""

    @pytest.fixture
    def pool_config(self):
        """Create test pool configuration"""
        return PoolConfig(
            initial_size=4,
            max_size=8,
            spawn_threshold=0.7,
            cull_threshold=0.3,
        )

    @pytest.fixture
    def pool(self, pool_config):
        """Create test pool"""
        return DynamicPool(
            latent_dim=64,
            head_dim=32,
            config=pool_config,
            device=torch.device('cpu'),
        )

    def test_initialization(self, pool, pool_config):
        """Test pool initializes with correct size"""
        assert pool.size() == pool_config.initial_size
        assert pool.max_size == pool_config.max_size

    def test_spawn_component(self, pool):
        """Test spawning new component"""
        initial_size = pool.size()

        # Spawn new component
        new_component = pool.spawn()
        assert new_component is not None
        assert isinstance(new_component, Pseudopod)
        assert pool.size() == initial_size + 1

    def test_max_size_enforcement(self, pool):
        """Test pool respects max size"""
        # Spawn until max size
        while pool.size() < pool.max_size:
            pool.spawn()

        assert pool.size() == pool.max_size

        # Try to spawn beyond max size
        new_component = pool.spawn()
        assert new_component is None
        assert pool.size() == pool.max_size

    def test_cull_low_fitness(self, pool):
        """Test culling low-fitness components"""
        # Set fitness values
        for i, component in enumerate(pool.components):
            pool.fitness_scores[component.component_id] = float(i) / pool.size()

        initial_size = pool.size()

        # Cull low fitness components
        culled = pool._cull_low_fitness()

        # Should have culled at least one component
        assert culled > 0
        assert pool.size() < initial_size

        # Remaining components should have higher fitness
        for component in pool.components:
            fitness = pool.fitness_scores.get(component.component_id, 0.0)
            assert fitness >= pool.config.cull_threshold

    def test_update_fitness(self, pool):
        """Test updating fitness scores"""
        component = pool.components[0]
        component_id = component.component_id

        # Update fitness
        new_fitness = 0.85
        pool.update_fitness(component_id, new_fitness)

        assert pool.fitness_scores[component_id] == new_fitness

    def test_get_fitness(self, pool):
        """Test retrieving fitness scores"""
        component = pool.components[0]
        component_id = component.component_id

        # Set fitness
        pool.fitness_scores[component_id] = 0.75

        # Get fitness
        fitness = pool.get_fitness(component_id)
        assert fitness == 0.75

        # Get fitness for non-existent component
        fitness = pool.get_fitness(99999)
        assert fitness == 0.0

    def test_get_component_by_id(self, pool):
        """Test retrieving component by ID"""
        component = pool.components[0]
        component_id = component.component_id

        retrieved = pool.get_component(component_id)
        assert retrieved is not None
        assert retrieved.component_id == component_id

        # Non-existent component
        retrieved = pool.get_component(99999)
        assert retrieved is None

    def test_spawn_batch(self, pool):
        """Test batch spawning"""
        initial_size = pool.size()
        batch_size = 2

        spawned = pool._spawn_batch(batch_size)

        assert spawned == batch_size
        assert pool.size() == initial_size + batch_size

    def test_spawn_batch_respects_max_size(self, pool):
        """Test batch spawning respects max size"""
        # Fill to near max
        while pool.size() < pool.max_size - 1:
            pool.spawn()

        # Try to spawn batch of 3 (should only spawn 1)
        spawned = pool._spawn_batch(3)
        assert spawned == 1
        assert pool.size() == pool.max_size

    def test_get_top_performers(self, pool):
        """Test retrieving top performers"""
        # Set fitness values
        for i, component in enumerate(pool.components):
            pool.fitness_scores[component.component_id] = float(i)

        # Get top 2
        top = pool.get_top_performers(k=2)
        assert len(top) == 2

        # Should be sorted by fitness (descending)
        assert top[0][1] >= top[1][1]

    def test_lifecycle_frozen(self, pool):
        """Test freezing lifecycle dynamics"""
        pool.freeze_lifecycle()
        assert pool._lifecycle_frozen is True

        # Spawning should fail
        new_component = pool.spawn()
        assert new_component is None

        # Culling should do nothing
        culled = pool._cull_low_fitness()
        assert culled == 0

    def test_lifecycle_unfreeze(self, pool):
        """Test unfreezing lifecycle"""
        pool.freeze_lifecycle()
        pool.unfreeze_lifecycle()
        assert pool._lifecycle_frozen is False

        # Spawning should work again
        new_component = pool.spawn()
        assert new_component is not None

    def test_clear_pool(self, pool):
        """Test clearing all components"""
        assert pool.size() > 0

        pool.clear()

        assert pool.size() == 0
        assert len(pool.fitness_scores) == 0

    def test_get_statistics(self, pool):
        """Test getting pool statistics"""
        # Set some fitness values
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
        """Test all components have unique IDs"""
        # Spawn several components
        for _ in range(3):
            pool.spawn()

        ids = [c.component_id for c in pool.components]
        assert len(ids) == len(set(ids))  # All unique
