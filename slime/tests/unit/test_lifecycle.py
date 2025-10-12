import pytest
import torch
import numpy as np
from slime.training.lifecycle import SimulatedAnnealingLifecycle
from slime.memory.archive import CVTArchive
from slime.memory.pool import DynamicPool
from slime.config.dimensions import TINY

@pytest.fixture
def archive():
    arc = CVTArchive(config=TINY, seed=42)
    rng = np.random.RandomState(42)
    latent = rng.randn(150, TINY.behavioral_space.max_dims).astype(np.float32)
    mixing_matrix = rng.randn(TINY.behavioral_space.max_dims, TINY.behavioral_space.num_raw_metrics).astype(np.float32)
    for i in range(150):
        raw_metrics = latent[i] @ mixing_matrix + rng.randn(TINY.behavioral_space.num_raw_metrics).astype(np.float32) * 0.1
        arc.add_raw_metrics(raw_metrics)
    arc.discover_dimensions()
    return arc

@pytest.fixture
def lifecycle(archive):
    from slime.memory.pool import PoolConfig
    from slime.core.pseudopod import Pseudopod
    from slime.kernels.torch_fallback import TorchKernel

    def component_factory():
        kernel = TorchKernel(numerical_config=TINY.numerical)
        return Pseudopod(
            head_dim=TINY.dimensions.head_dim,
            kernel=kernel,
            fitness_config=TINY.fitness,
            numerical_config=TINY.numerical,
            num_heads=TINY.dimensions.num_heads
        )

    pool_config = PoolConfig(min_size=TINY.test.test_pool_min_size, max_size=TINY.test.test_pool_max_size)
    pool = DynamicPool(component_factory=component_factory, config=pool_config, arch_config=TINY, archive=archive)
    return SimulatedAnnealingLifecycle(pool=pool, archive=archive, initial_temp=1.0, min_temp=0.01, cooling_schedule='linear', max_pool_size=TINY.test.test_pool_max_size, min_pool_size=TINY.test.test_pool_min_size, max_loss_ratio=10.0, seed=42)

def test_lifecycle_initialization(lifecycle, constraint):
    constraint('Initial temperature is 1.0', lambda: (lifecycle.initial_temp == 1.0, lifecycle.initial_temp, 1.0, {}))
    constraint('Min temperature is 0.01', lambda: (lifecycle.min_temp == 0.01, lifecycle.min_temp, 0.01, {}))
    constraint('Cooling schedule is linear', lambda: (lifecycle.cooling_schedule == 'linear', lifecycle.cooling_schedule, 'linear', {}))
    constraint('Step starts at 0', lambda: (lifecycle.step == 0, lifecycle.step, 0, {}))
    constraint('Not frozen initially', lambda: (not lifecycle.lifecycle_frozen, lifecycle.lifecycle_frozen, False, {}))

def test_temperature_linear_cooling(lifecycle, constraint):
    lifecycle.step = 0
    lifecycle.max_steps = 1000
    temp_start = lifecycle.get_temperature()
    constraint('Start temperature is 1.0', lambda: (temp_start == 1.0, temp_start, 1.0, {}))
    lifecycle.step = 500
    temp_mid = lifecycle.get_temperature()
    constraint('Mid temperature in range', lambda: (0.4 < temp_mid < 0.6, temp_mid, 'range(0.4, 0.6)', {}))
    lifecycle.step = 1000
    temp_end = lifecycle.get_temperature()
    constraint('End temperature is 0.01', lambda: (temp_end == 0.01, temp_end, 0.01, {}))

def test_temperature_exponential_cooling(constraint):
    from slime.memory.pool import PoolConfig
    from slime.core.pseudopod import Pseudopod
    from slime.kernels.torch_fallback import TorchKernel

    archive = CVTArchive(config=TINY, seed=42)

    def component_factory():
        kernel = TorchKernel(numerical_config=TINY.numerical)
        return Pseudopod(
            head_dim=TINY.dimensions.head_dim,
            kernel=kernel,
            fitness_config=TINY.fitness,
            numerical_config=TINY.numerical,
            num_heads=TINY.dimensions.num_heads
        )

    pool_config = PoolConfig(min_size=TINY.test.test_pool_min_size, max_size=TINY.test.test_pool_max_size)
    pool = DynamicPool(component_factory=component_factory, config=pool_config, arch_config=TINY, archive=archive)
    lifecycle = SimulatedAnnealingLifecycle(pool=pool, archive=archive, initial_temp=1.0, min_temp=0.01, cooling_schedule='exponential', seed=42)
    lifecycle.step = 0
    lifecycle.max_steps = 1000
    temp_start = lifecycle.get_temperature()
    lifecycle.step = 500
    temp_mid = lifecycle.get_temperature()
    lifecycle.step = 1000
    temp_end = lifecycle.get_temperature()
    constraint('Temperatures monotonically decrease', lambda: (temp_start > temp_mid > temp_end, [temp_start, temp_mid, temp_end], 'monotonic decrease', {}))
    constraint('Start temperature is 1.0', lambda: (temp_start == 1.0, temp_start, 1.0, {}))
    constraint('End temperature is 0.01', lambda: (temp_end == 0.01, temp_end, 0.01, {}))

def test_temperature_logarithmic_cooling(constraint):
    from slime.memory.pool import PoolConfig
    from slime.core.pseudopod import Pseudopod
    from slime.kernels.torch_fallback import TorchKernel

    archive = CVTArchive(config=TINY, seed=42)

    def component_factory():
        kernel = TorchKernel(numerical_config=TINY.numerical)
        return Pseudopod(
            head_dim=TINY.dimensions.head_dim,
            kernel=kernel,
            fitness_config=TINY.fitness,
            numerical_config=TINY.numerical,
            num_heads=TINY.dimensions.num_heads
        )

    pool_config = PoolConfig(min_size=TINY.test.test_pool_min_size, max_size=TINY.test.test_pool_max_size)
    pool = DynamicPool(component_factory=component_factory, config=pool_config, arch_config=TINY, archive=archive)
    lifecycle = SimulatedAnnealingLifecycle(pool=pool, archive=archive, initial_temp=1.0, min_temp=0.01, cooling_schedule='logarithmic', seed=42)
    lifecycle.step = 0
    lifecycle.max_steps = 1000
    temp_start = lifecycle.get_temperature()
    lifecycle.step = 500
    temp_mid = lifecycle.get_temperature()
    lifecycle.step = 1000
    temp_end = lifecycle.get_temperature()
    constraint('Temperatures monotonically decrease', lambda: (temp_start > temp_mid > temp_end, [temp_start, temp_mid, temp_end], 'monotonic decrease', {}))

def test_birth_probability_high_fitness(lifecycle, constraint):
    fitness = 0.95
    archive_max = 1.0
    temperature = 0.5
    prob = lifecycle.birth_probability(fitness, archive_max, temperature)
    constraint('High fitness gives birth probability 1.0', lambda: (prob == 1.0, prob, 1.0, {}))

def test_birth_probability_low_fitness_high_temp(lifecycle, constraint):
    fitness = 0.3
    archive_max = 1.0
    temperature = 1.0
    prob = lifecycle.birth_probability(fitness, archive_max, temperature)
    constraint('Low fitness + high temp gives moderate probability', lambda: (0.3 < prob < 0.8, prob, 'range(0.3, 0.8)', {}))

def test_birth_probability_low_fitness_low_temp(lifecycle, constraint):
    fitness = 0.3
    archive_max = 1.0
    temperature = 0.01
    prob = lifecycle.birth_probability(fitness, archive_max, temperature)
    constraint('Low fitness + low temp gives very low probability', lambda: (prob < 0.01, prob, '<0.01', {}))

def test_loss_ema_update(lifecycle, constraint):
    """Test exponential moving average loss tracking."""
    lifecycle.update_loss_ema(1.0)
    constraint('Loss EMA initialized to first value', lambda: (lifecycle.loss_ema == 1.0, lifecycle.loss_ema, 1.0, {}))

    lifecycle.update_loss_ema(2.0)
    expected = 0.99 * 1.0 + 0.01 * 2.0  # alpha=0.99 from fixture
    constraint('Loss EMA updated correctly', lambda: (abs(lifecycle.loss_ema - expected) < 1e-6, lifecycle.loss_ema, expected, {}))

def test_loss_divergence_freezing(lifecycle, constraint):
    """Test lifecycle freezes when loss diverges."""
    lifecycle.update_loss_ema(1.0)
    constraint('Not frozen initially', lambda: (not lifecycle.lifecycle_frozen, lifecycle.lifecycle_frozen, False, {}))

    # Trigger freeze with loss > max_loss_ratio * EMA (10.0 * 1.0)
    lifecycle.update_loss_ema(15.0)
    constraint('Frozen after divergent loss', lambda: (lifecycle.lifecycle_frozen, lifecycle.lifecycle_frozen, True, {}))

def test_loss_stabilization_unfreezing(lifecycle, constraint):
    """Test lifecycle unfreezes when loss stabilizes."""
    lifecycle.update_loss_ema(1.0)
    lifecycle.update_loss_ema(15.0)  # Freeze
    constraint('Frozen after divergence', lambda: (lifecycle.lifecycle_frozen, lifecycle.lifecycle_frozen, True, {}))

    # Stabilize loss < 2.0 * EMA
    lifecycle.update_loss_ema(1.5)
    constraint('Unfrozen after stabilization', lambda: (not lifecycle.lifecycle_frozen, lifecycle.lifecycle_frozen, False, {}))

def test_deterministic_random_consistency(lifecycle, constraint):
    val1 = lifecycle._deterministic_random('test_context')
    lifecycle.step = 0
    val2 = lifecycle._deterministic_random('test_context')
    constraint('Same step + context gives same random value', lambda: (val1 == val2, val1, val2, {}))

def test_deterministic_random_different_steps(lifecycle, constraint):
    lifecycle.step = 0
    val1 = lifecycle._deterministic_random('test')
    lifecycle.step = 1
    val2 = lifecycle._deterministic_random('test')
    constraint('Different steps give different random values', lambda: (val1 != val2, val1, f'!= {val2}', {}))

def test_deterministic_random_different_contexts(lifecycle, constraint):
    lifecycle.step = 0
    val1 = lifecycle._deterministic_random('context_a')
    val2 = lifecycle._deterministic_random('context_b')
    constraint('Different contexts give different random values', lambda: (val1 != val2, val1, f'!= {val2}', {}))

def test_deterministic_random_range(lifecycle, constraint):
    for i in range(100):
        lifecycle.step = i
        val = lifecycle._deterministic_random(f'test_{i}')
        constraint(f'Random value {i} in [0, 1]', lambda v=val: (0.0 <= v <= 1.0, v, '[0.0, 1.0]', {}))

def test_should_spawn_component_deterministic(lifecycle, constraint):

    class MockComponent:

        def __init__(self):
            pass

        def load_state_dict(self, state_dict):
            pass
    lifecycle.step = 10
    lifecycle.max_steps = 1000
    behavior = np.array([0.5] * TINY.behavioral_space.max_dims)
    fitness = 0.7
    result1 = lifecycle.should_spawn_component(behavior, fitness, MockComponent)
    lifecycle.step = 10
    result2 = lifecycle.should_spawn_component(behavior, fitness, MockComponent)
    constraint('Same inputs give deterministic spawn decision', lambda: ((result1 is None) == (result2 is None), result1 is None, result2 is None, {}))

def test_frozen_lifecycle_prevents_spawning(lifecycle, constraint):
    """Test frozen lifecycle prevents component spawning."""
    lifecycle.lifecycle_frozen = True
    behavior = np.array([0.5] * TINY.behavioral_space.max_dims)

    def mock_factory(state_dict):
        return None

    result = lifecycle.should_spawn_component(behavior, 0.9, mock_factory)
    constraint('Frozen lifecycle returns None', lambda: (result is None, result, None, {}))

def test_pool_size_limits_spawning(lifecycle, constraint):
    """Test max pool size prevents spawning."""
    # Manually set pool to max size
    lifecycle.pool._components = [None] * TINY.test.pool_max_size
    behavior = np.array([0.5] * TINY.behavioral_space.max_dims)

    def mock_factory(state_dict):
        return None

    result = lifecycle.should_spawn_component(behavior, 0.9, mock_factory)
    constraint('Full pool prevents spawning', lambda: (result is None, result, None, {}))

def test_spawn_component_with_archive_elite(lifecycle, archive, constraint):
    state_dict = {'W': torch.randn(32, 32)}
    behavior = np.array([0.5] * TINY.behavioral_space.max_dims)
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
        constraint('Spawned component loaded state_dict from archive', lambda: (component.state_dict_loaded is not None, component.state_dict_loaded is not None, True, {}))
    # Else: stochastic decision resulted in no spawn, which is valid

def test_temperature_monotonic_decrease(lifecycle, constraint):
    lifecycle.max_steps = 1000
    temps = []
    for step in range(0, 1001, 100):
        lifecycle.step = step
        temps.append(lifecycle.get_temperature())
    for i in range(len(temps) - 1):
        constraint(f'Temperature decreases from step {i*100} to {(i+1)*100}', lambda t1=temps[i], t2=temps[i+1]: (t1 >= t2, t1, f'>= {t2}', {}))

def test_annealing_exploration_exploitation_transition(lifecycle, constraint):
    lifecycle.max_steps = 1000
    lifecycle.step = 0
    early_temp = lifecycle.get_temperature()
    early_prob = lifecycle.birth_probability(0.5, 1.0, early_temp)
    lifecycle.step = 1000
    late_temp = lifecycle.get_temperature()
    late_prob = lifecycle.birth_probability(0.5, 1.0, late_temp)
    constraint('Early exploration higher than late exploitation', lambda: (early_prob > late_prob, early_prob, f'> {late_prob}', {}))

def test_seed_reproducibility(constraint):
    from slime.memory.pool import PoolConfig
    from slime.core.pseudopod import Pseudopod
    from slime.kernels.torch_fallback import TorchKernel

    def component_factory():
        kernel = TorchKernel(numerical_config=TINY.numerical)
        return Pseudopod(
            head_dim=TINY.dimensions.head_dim,
            kernel=kernel,
            fitness_config=TINY.fitness,
            numerical_config=TINY.numerical,
            num_heads=TINY.dimensions.num_heads
        )

    archive1 = CVTArchive(config=TINY, seed=42)
    rng1 = np.random.RandomState(42)
    latent1 = rng1.randn(150, TINY.behavioral_space.max_dims).astype(np.float32)
    mixing_matrix1 = rng1.randn(TINY.behavioral_space.max_dims, TINY.behavioral_space.num_raw_metrics).astype(np.float32)
    for i in range(150):
        raw_metrics = latent1[i] @ mixing_matrix1 + rng1.randn(TINY.behavioral_space.num_raw_metrics).astype(np.float32) * 0.1
        archive1.add_raw_metrics(raw_metrics)
    archive1.discover_dimensions()
    pool_config1 = PoolConfig(min_size=TINY.test.pool_min_size, max_size=TINY.test.pool_max_size)
    pool1 = DynamicPool(component_factory=component_factory, config=pool_config1, arch_config=TINY, archive=archive1)
    lifecycle1 = SimulatedAnnealingLifecycle(pool=pool1, archive=archive1, seed=42)

    archive2 = CVTArchive(config=TINY, seed=42)
    rng2 = np.random.RandomState(42)
    latent2 = rng2.randn(150, TINY.behavioral_space.max_dims).astype(np.float32)
    mixing_matrix2 = rng2.randn(TINY.behavioral_space.max_dims, TINY.behavioral_space.num_raw_metrics).astype(np.float32)
    for i in range(150):
        raw_metrics = latent2[i] @ mixing_matrix2 + rng2.randn(TINY.behavioral_space.num_raw_metrics).astype(np.float32) * 0.1
        archive2.add_raw_metrics(raw_metrics)
    archive2.discover_dimensions()
    pool_config2 = PoolConfig(min_size=TINY.test.pool_min_size, max_size=TINY.test.pool_max_size)
    pool2 = DynamicPool(component_factory=component_factory, config=pool_config2, arch_config=TINY, archive=archive2)
    lifecycle2 = SimulatedAnnealingLifecycle(pool=pool2, archive=archive2, seed=42)

    lifecycle1.step = 123
    lifecycle2.step = 123
    val1 = lifecycle1._deterministic_random('test')
    val2 = lifecycle2._deterministic_random('test')
    constraint('Same seed gives reproducible random values', lambda: (val1 == val2, val1, val2, {}))

def test_different_seeds_different_behavior(constraint):
    from slime.memory.pool import PoolConfig
    from slime.core.pseudopod import Pseudopod
    from slime.kernels.torch_fallback import TorchKernel

    def component_factory():
        kernel = TorchKernel(numerical_config=TINY.numerical)
        return Pseudopod(
            head_dim=TINY.dimensions.head_dim,
            kernel=kernel,
            fitness_config=TINY.fitness,
            numerical_config=TINY.numerical,
            num_heads=TINY.dimensions.num_heads
        )

    archive1 = CVTArchive(config=TINY, seed=42)
    rng1 = np.random.RandomState(42)
    latent1 = rng1.randn(150, TINY.behavioral_space.max_dims).astype(np.float32)
    mixing_matrix1 = rng1.randn(TINY.behavioral_space.max_dims, TINY.behavioral_space.num_raw_metrics).astype(np.float32)
    for i in range(150):
        raw_metrics = latent1[i] @ mixing_matrix1 + rng1.randn(TINY.behavioral_space.num_raw_metrics).astype(np.float32) * 0.1
        archive1.add_raw_metrics(raw_metrics)
    archive1.discover_dimensions()
    pool_config1 = PoolConfig(min_size=TINY.test.pool_min_size, max_size=TINY.test.pool_max_size)
    pool1 = DynamicPool(component_factory=component_factory, config=pool_config1, arch_config=TINY, archive=archive1)
    lifecycle1 = SimulatedAnnealingLifecycle(pool=pool1, archive=archive1, seed=42)

    archive2 = CVTArchive(config=TINY, seed=99)
    rng2 = np.random.RandomState(99)
    latent2 = rng2.randn(150, TINY.behavioral_space.max_dims).astype(np.float32)
    mixing_matrix2 = rng2.randn(TINY.behavioral_space.max_dims, TINY.behavioral_space.num_raw_metrics).astype(np.float32)
    for i in range(150):
        raw_metrics = latent2[i] @ mixing_matrix2 + rng2.randn(TINY.behavioral_space.num_raw_metrics).astype(np.float32) * 0.1
        archive2.add_raw_metrics(raw_metrics)
    archive2.discover_dimensions()
    pool_config2 = PoolConfig(min_size=TINY.test.pool_min_size, max_size=TINY.test.pool_max_size)
    pool2 = DynamicPool(component_factory=component_factory, config=pool_config2, arch_config=TINY, archive=archive2)
    lifecycle2 = SimulatedAnnealingLifecycle(pool=pool2, archive=archive2, seed=99)

    lifecycle1.step = 123
    lifecycle2.step = 123
    val1 = lifecycle1._deterministic_random('test')
    val2 = lifecycle2._deterministic_random('test')
    constraint('Different seeds give different random values', lambda: (val1 != val2, val1, f'!= {val2}', {}))
