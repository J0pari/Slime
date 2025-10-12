import pytest
import torch
import numpy as np
from slime.training.lifecycle import SimulatedAnnealingLifecycle
from slime.memory.archive import CVTArchive

@pytest.fixture
def archive():
    arc = CVTArchive(num_raw_metrics=10, min_dims=5, max_dims=5, num_centroids=50, low_rank_k=32, seed=42)
    for i in range(150):
        arc.add_raw_metrics(np.random.randn(10).astype(np.float32))
    arc.discover_dimensions()
    return arc

@pytest.fixture
def lifecycle(archive):
    return SimulatedAnnealingLifecycle(archive=archive, initial_temp=1.0, min_temp=0.01, cooling_schedule='linear', max_pool_size=32, min_pool_size=4, max_loss_ratio=10.0, seed=42)

def test_lifecycle_initialization(lifecycle):
    assert lifecycle.initial_temp == 1.0
    assert lifecycle.min_temp == 0.01
    assert lifecycle.cooling_schedule == 'linear'
    assert lifecycle.step == 0
    assert not lifecycle.frozen

def test_temperature_linear_cooling(lifecycle):
    lifecycle.step = 0
    lifecycle.max_steps = 1000
    temp_start = lifecycle.get_temperature()
    assert temp_start == 1.0
    lifecycle.step = 500
    temp_mid = lifecycle.get_temperature()
    assert 0.4 < temp_mid < 0.6
    lifecycle.step = 1000
    temp_end = lifecycle.get_temperature()
    assert temp_end == 0.01

def test_temperature_exponential_cooling():
    archive = CVTArchive(behavioral_dims=3, num_centroids=10, seed=42)
    lifecycle = SimulatedAnnealingLifecycle(archive=archive, initial_temp=1.0, min_temp=0.01, cooling_schedule='exponential', seed=42)
    lifecycle.step = 0
    lifecycle.max_steps = 1000
    temp_start = lifecycle.get_temperature()
    lifecycle.step = 500
    temp_mid = lifecycle.get_temperature()
    lifecycle.step = 1000
    temp_end = lifecycle.get_temperature()
    assert temp_start > temp_mid > temp_end
    assert temp_start == 1.0
    assert temp_end == 0.01

def test_temperature_logarithmic_cooling():
    archive = CVTArchive(behavioral_dims=3, num_centroids=10, seed=42)
    lifecycle = SimulatedAnnealingLifecycle(archive=archive, initial_temp=1.0, min_temp=0.01, cooling_schedule='logarithmic', seed=42)
    lifecycle.step = 0
    lifecycle.max_steps = 1000
    temp_start = lifecycle.get_temperature()
    lifecycle.step = 500
    temp_mid = lifecycle.get_temperature()
    lifecycle.step = 1000
    temp_end = lifecycle.get_temperature()
    assert temp_start > temp_mid > temp_end

def test_birth_probability_high_fitness(lifecycle):
    fitness = 0.95
    archive_max = 1.0
    temperature = 0.5
    prob = lifecycle.birth_probability(fitness, archive_max, temperature)
    assert prob == 1.0

def test_birth_probability_low_fitness_high_temp(lifecycle):
    fitness = 0.3
    archive_max = 1.0
    temperature = 1.0
    prob = lifecycle.birth_probability(fitness, archive_max, temperature)
    assert 0.3 < prob < 0.8

def test_birth_probability_low_fitness_low_temp(lifecycle):
    fitness = 0.3
    archive_max = 1.0
    temperature = 0.01
    prob = lifecycle.birth_probability(fitness, archive_max, temperature)
    assert prob < 0.01

def test_death_probability_low_fitness(lifecycle):
    fitness = 0.1
    archive_max = 1.0
    temperature = 0.5
    prob = lifecycle.death_probability(fitness, archive_max, temperature)
    assert prob > 0.5

def test_death_probability_high_fitness(lifecycle):
    fitness = 0.95
    archive_max = 1.0
    temperature = 0.5
    prob = lifecycle.death_probability(fitness, archive_max, temperature)
    assert prob < 0.1

def test_deterministic_random_consistency(lifecycle):
    val1 = lifecycle._deterministic_random('test_context')
    lifecycle.step = 0
    val2 = lifecycle._deterministic_random('test_context')
    assert val1 == val2

def test_deterministic_random_different_steps(lifecycle):
    lifecycle.step = 0
    val1 = lifecycle._deterministic_random('test')
    lifecycle.step = 1
    val2 = lifecycle._deterministic_random('test')
    assert val1 != val2

def test_deterministic_random_different_contexts(lifecycle):
    lifecycle.step = 0
    val1 = lifecycle._deterministic_random('context_a')
    val2 = lifecycle._deterministic_random('context_b')
    assert val1 != val2

def test_deterministic_random_range(lifecycle):
    for i in range(100):
        lifecycle.step = i
        val = lifecycle._deterministic_random(f'test_{i}')
        assert 0.0 <= val <= 1.0

def test_should_spawn_component_deterministic(lifecycle):

    class MockComponent:

        def __init__(self):
            pass

        def load_state_dict(self, state_dict):
            pass
    lifecycle.step = 10
    lifecycle.max_steps = 1000
    behavior = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    fitness = 0.7
    result1 = lifecycle.should_spawn_component(behavior, fitness, MockComponent)
    lifecycle.step = 10
    result2 = lifecycle.should_spawn_component(behavior, fitness, MockComponent)
    assert (result1 is None) == (result2 is None)

def test_lifecycle_freeze_on_high_loss(lifecycle):
    lifecycle.loss_history = [1.0, 1.0, 1.0, 1.0, 1.0]
    lifecycle.check_loss_and_freeze(loss=15.0)
    assert lifecycle.frozen

def test_lifecycle_no_freeze_normal_loss(lifecycle):
    lifecycle.loss_history = [1.0, 1.0, 1.0, 1.0, 1.0]
    lifecycle.check_loss_and_freeze(loss=1.2)
    assert not lifecycle.frozen

def test_lifecycle_warmup_phase():
    archive = CVTArchive(num_raw_metrics=10, min_dims=3, max_dims=3, num_centroids=10, seed=42)
    for i in range(150):
        archive.add_raw_metrics(np.random.randn(10).astype(np.float32))
    archive.discover_dimensions()
    lifecycle = SimulatedAnnealingLifecycle(archive=archive, seed=42)
    lifecycle.step = 50
    lifecycle.loss_history = [1.0]
    lifecycle.check_loss_and_freeze(loss=100.0)
    assert not lifecycle.frozen

def test_spawn_component_with_archive_elite(lifecycle, archive):
    state_dict = {'W': torch.randn(32, 32)}
    behavior = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    archive.add(behavior, 0.8, state_dict, generation=0, metadata={})

    class MockComponent:

        def __init__(self):
            self.state_dict_loaded = None

        def load_state_dict(self, state_dict):
            self.state_dict_loaded = state_dict
    lifecycle.step = 500
    lifecycle.max_steps = 1000
    component = lifecycle.should_spawn_component(behavior, 0.9, MockComponent)
    if component is not None:
        assert component.state_dict_loaded is not None

def test_temperature_monotonic_decrease(lifecycle):
    lifecycle.max_steps = 1000
    temps = []
    for step in range(0, 1001, 100):
        lifecycle.step = step
        temps.append(lifecycle.get_temperature())
    for i in range(len(temps) - 1):
        assert temps[i] >= temps[i + 1]

def test_annealing_exploration_exploitation_transition(lifecycle):
    lifecycle.max_steps = 1000
    lifecycle.step = 0
    early_temp = lifecycle.get_temperature()
    early_prob = lifecycle.birth_probability(0.5, 1.0, early_temp)
    lifecycle.step = 1000
    late_temp = lifecycle.get_temperature()
    late_prob = lifecycle.birth_probability(0.5, 1.0, late_temp)
    assert early_prob > late_prob

def test_seed_reproducibility():
    archive1 = CVTArchive(num_raw_metrics=10, min_dims=3, max_dims=3, num_centroids=10, seed=42)
    for i in range(150):
        archive1.add_raw_metrics(np.random.randn(10).astype(np.float32))
    archive1.discover_dimensions()
    lifecycle1 = SimulatedAnnealingLifecycle(archive=archive1, seed=42)

    archive2 = CVTArchive(num_raw_metrics=10, min_dims=3, max_dims=3, num_centroids=10, seed=42)
    for i in range(150):
        archive2.add_raw_metrics(np.random.randn(10).astype(np.float32))
    archive2.discover_dimensions()
    lifecycle2 = SimulatedAnnealingLifecycle(archive=archive2, seed=42)

    lifecycle1.step = 123
    lifecycle2.step = 123
    val1 = lifecycle1._deterministic_random('test')
    val2 = lifecycle2._deterministic_random('test')
    assert val1 == val2

def test_different_seeds_different_behavior():
    archive1 = CVTArchive(num_raw_metrics=10, min_dims=3, max_dims=3, num_centroids=10, seed=42)
    for i in range(150):
        archive1.add_raw_metrics(np.random.randn(10).astype(np.float32))
    archive1.discover_dimensions()
    lifecycle1 = SimulatedAnnealingLifecycle(archive=archive1, seed=42)

    archive2 = CVTArchive(num_raw_metrics=10, min_dims=3, max_dims=3, num_centroids=10, seed=99)
    for i in range(150):
        archive2.add_raw_metrics(np.random.randn(10).astype(np.float32))
    archive2.discover_dimensions()
    lifecycle2 = SimulatedAnnealingLifecycle(archive=archive2, seed=99)

    lifecycle1.step = 123
    lifecycle2.step = 123
    val1 = lifecycle1._deterministic_random('test')
    val2 = lifecycle2._deterministic_random('test')
    assert val1 != val2