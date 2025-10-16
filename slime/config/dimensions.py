from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class DimensionConfig:
    head_dim: int
    num_heads: int
    hidden_dim: int

    def __post_init__(self):
        assert self.hidden_dim == self.head_dim * self.num_heads, f'hidden_dim ({self.hidden_dim}) must equal head_dim ({self.head_dim}) * num_heads ({self.num_heads})'
        assert self.head_dim > 0 and self.head_dim % 8 == 0, f'head_dim must be positive multiple of 8, got {self.head_dim}'
        assert self.num_heads > 0 and self.num_heads & (self.num_heads - 1) == 0, f'num_heads must be power of 2, got {self.num_heads}'

@dataclass(frozen=True)
class BehavioralSpaceConfig:
    num_raw_metrics: int
    min_dims: int
    max_dims: int
    num_centroids: int

    def __post_init__(self):
        assert 0 < self.min_dims <= self.max_dims <= self.num_raw_metrics, f'Must satisfy: 0 < min_dims ({self.min_dims}) <= max_dims ({self.max_dims}) <= num_raw_metrics ({self.num_raw_metrics})'
        assert self.num_centroids > 0, f'num_centroids must be positive, got {self.num_centroids}'
        assert self.max_dims <= 7, f'max_dims > 7 causes exponential explosion, got {self.max_dims}'

@dataclass(frozen=True)
class CompressionConfig:
    low_rank_k: int
    delta_rank: int

    def __post_init__(self):
        assert self.low_rank_k > 0 and self.low_rank_k % 4 == 0, f'low_rank_k must be positive multiple of 4, got {self.low_rank_k}'
        assert self.delta_rank > 0 and self.delta_rank % 4 == 0, f'delta_rank must be positive multiple of 4, got {self.delta_rank}'
        assert self.delta_rank <= self.low_rank_k, f'delta_rank ({self.delta_rank}) should be <= low_rank_k ({self.low_rank_k}) for efficient deltas'

@dataclass(frozen=True)
class FitnessConfig:
    ema_decay: float
    entropy_weight: float
    magnitude_weight: float

    def __post_init__(self):
        assert 0.0 < self.ema_decay < 1.0, f'ema_decay must be in (0, 1), got {self.ema_decay}'
        assert self.entropy_weight >= 0.0, f'entropy_weight must be non-negative, got {self.entropy_weight}'
        assert self.magnitude_weight >= 0.0, f'magnitude_weight must be non-negative, got {self.magnitude_weight}'
        total = self.entropy_weight + self.magnitude_weight
        assert total > 0.0, f'At least one weight must be positive'

@dataclass(frozen=True)
class TestConfig:
    batch_size: int
    seq_len: int
    pool_max_size: int
    pool_min_size: int
    test_pool_max_size: int  # Smaller pool size for unit tests
    test_pool_min_size: int

    def __post_init__(self):
        assert self.batch_size > 0, f'batch_size must be positive, got {self.batch_size}'
        assert self.seq_len > 0, f'seq_len must be positive, got {self.seq_len}'
        assert self.pool_max_size > 0, f'pool_max_size must be positive, got {self.pool_max_size}'
        assert self.pool_min_size > 0, f'pool_min_size must be positive, got {self.pool_min_size}'
        assert self.pool_min_size <= self.pool_max_size, f'pool_min_size ({self.pool_min_size}) must be <= pool_max_size ({self.pool_max_size})'
        assert self.test_pool_max_size > 0, f'test_pool_max_size must be positive, got {self.test_pool_max_size}'
        assert self.test_pool_min_size > 0, f'test_pool_min_size must be positive, got {self.test_pool_min_size}'
        assert self.test_pool_min_size <= self.test_pool_max_size, f'test_pool_min_size ({self.test_pool_min_size}) must be <= test_pool_max_size ({self.test_pool_max_size})'

@dataclass(frozen=True)
class NumericalConfig:
    epsilon: float
    svd_threshold: float
    attention_temperature: float

    def __post_init__(self):
        assert self.epsilon > 0, f'epsilon must be positive, got {self.epsilon}'
        assert self.svd_threshold > 0, f'svd_threshold must be positive, got {self.svd_threshold}'
        assert self.attention_temperature > 0, f'attention_temperature must be positive, got {self.attention_temperature}'

@dataclass(frozen=True)
class StatisticalGeometryConfig:
    """Configuration for statistical geometry computations."""
    # Fisher information computation
    fisher_num_samples: int
    fisher_regularization: float
    
    # Fisher-Rao distance interpretation thresholds
    fr_very_close: float
    fr_similar: float
    fr_different: float
    
    # Relative information threshold
    rel_info_threshold: float
    
    # Jacobian correction coefficient
    jacobian_coefficient: float
    
    # Recommendation thresholds
    overparameterization_ratio: float  # intrinsic_dim < nominal/X is overparameterized
    low_info_gain_threshold: float
    high_phase_transition_count: int
    high_loss_threshold: float
    phase_transition_std_multiplier: float
    
    # Convergence speed thresholds
    fast_convergence_ratio: float
    moderate_convergence_ratio: float
    
    # Stability thresholds
    very_stable_std: float
    stable_std: float
    moderate_std: float

    def __post_init__(self):
        assert self.fisher_num_samples > 0, f'fisher_num_samples must be positive, got {self.fisher_num_samples}'
        assert self.fisher_regularization > 0, f'fisher_regularization must be positive, got {self.fisher_regularization}'
        assert 0 < self.fr_very_close < self.fr_similar < self.fr_different, 'Fisher-Rao thresholds must be increasing'
        assert self.rel_info_threshold > 0, f'rel_info_threshold must be positive, got {self.rel_info_threshold}'
        assert self.jacobian_coefficient > 0, f'jacobian_coefficient must be positive, got {self.jacobian_coefficient}'
        assert self.overparameterization_ratio > 1, f'overparameterization_ratio must be > 1, got {self.overparameterization_ratio}'
        assert self.low_info_gain_threshold > 0, f'low_info_gain_threshold must be positive, got {self.low_info_gain_threshold}'
        assert self.high_phase_transition_count > 0, f'high_phase_transition_count must be positive, got {self.high_phase_transition_count}'
        assert self.high_loss_threshold > 0, f'high_loss_threshold must be positive, got {self.high_loss_threshold}'
        assert self.phase_transition_std_multiplier > 0, f'phase_transition_std_multiplier must be positive, got {self.phase_transition_std_multiplier}'
        assert 0 < self.moderate_convergence_ratio < self.fast_convergence_ratio < 1, 'Convergence ratios must be increasing and < 1'
        assert 0 < self.very_stable_std < self.stable_std < self.moderate_std, 'Stability thresholds must be increasing'

@dataclass(frozen=True)
class ArchitectureConfig:
    dimensions: DimensionConfig
    behavioral_space: BehavioralSpaceConfig
    compression: CompressionConfig
    fitness: FitnessConfig
    test: TestConfig
    numerical: NumericalConfig
    statistical_geometry: StatisticalGeometryConfig
    k_neighbors: int = 5

    def __post_init__(self):
        assert self.k_neighbors > 0, f'k_neighbors must be positive, got {self.k_neighbors}'
        assert self.k_neighbors < self.behavioral_space.num_centroids, f'k_neighbors ({self.k_neighbors}) must be < num_centroids ({self.behavioral_space.num_centroids})'

TINY = ArchitectureConfig(
    dimensions=DimensionConfig(head_dim=16, num_heads=4, hidden_dim=64),
    behavioral_space=BehavioralSpaceConfig(num_raw_metrics=65, min_dims=3, max_dims=5, num_centroids=50),
    compression=CompressionConfig(low_rank_k=16, delta_rank=8),
    fitness=FitnessConfig(ema_decay=0.9, entropy_weight=1.0, magnitude_weight=1.0),
    test=TestConfig(batch_size=2, seq_len=16, pool_max_size=32, pool_min_size=4, test_pool_max_size=4, test_pool_min_size=2),
    numerical=NumericalConfig(epsilon=1e-10, svd_threshold=1e-06, attention_temperature=1.0),
    statistical_geometry=StatisticalGeometryConfig(
        fisher_num_samples=100,
        fisher_regularization=1e-6,
        fr_very_close=0.1,
        fr_similar=1.0,
        fr_different=10.0,
        rel_info_threshold=1.0,
        jacobian_coefficient=0.5,
        overparameterization_ratio=10.0,
        low_info_gain_threshold=0.01,
        high_phase_transition_count=5,
        high_loss_threshold=1.0,
        phase_transition_std_multiplier=2.0,
        fast_convergence_ratio=0.9,
        moderate_convergence_ratio=0.5,
        very_stable_std=0.01,
        stable_std=0.1,
        moderate_std=1.0
    ),
    k_neighbors=5
)

SMALL = ArchitectureConfig(
    dimensions=DimensionConfig(head_dim=32, num_heads=8, hidden_dim=256),
    behavioral_space=BehavioralSpaceConfig(num_raw_metrics=65, min_dims=4, max_dims=6, num_centroids=200),
    compression=CompressionConfig(low_rank_k=32, delta_rank=16),
    fitness=FitnessConfig(ema_decay=0.9, entropy_weight=1.0, magnitude_weight=1.0),
    test=TestConfig(batch_size=4, seq_len=32, pool_max_size=64, pool_min_size=8, test_pool_max_size=8, test_pool_min_size=4),
    numerical=NumericalConfig(epsilon=1e-10, svd_threshold=1e-06, attention_temperature=1.0),
    statistical_geometry=StatisticalGeometryConfig(
        fisher_num_samples=100,
        fisher_regularization=1e-6,
        fr_very_close=0.1,
        fr_similar=1.0,
        fr_different=10.0,
        rel_info_threshold=1.0,
        jacobian_coefficient=0.5,
        overparameterization_ratio=10.0,
        low_info_gain_threshold=0.01,
        high_phase_transition_count=5,
        high_loss_threshold=1.0,
        phase_transition_std_multiplier=2.0,
        fast_convergence_ratio=0.9,
        moderate_convergence_ratio=0.5,
        very_stable_std=0.01,
        stable_std=0.1,
        moderate_std=1.0
    ),
    k_neighbors=10
)

MEDIUM = ArchitectureConfig(
    dimensions=DimensionConfig(head_dim=64, num_heads=8, hidden_dim=512),
    behavioral_space=BehavioralSpaceConfig(num_raw_metrics=65, min_dims=5, max_dims=7, num_centroids=500),
    compression=CompressionConfig(low_rank_k=64, delta_rank=16),
    fitness=FitnessConfig(ema_decay=0.95, entropy_weight=1.0, magnitude_weight=1.0),
    test=TestConfig(batch_size=8, seq_len=64, pool_max_size=128, pool_min_size=16, test_pool_max_size=16, test_pool_min_size=8),
    numerical=NumericalConfig(epsilon=1e-10, svd_threshold=1e-06, attention_temperature=1.0),
    statistical_geometry=StatisticalGeometryConfig(
        fisher_num_samples=100,
        fisher_regularization=1e-6,
        fr_very_close=0.1,
        fr_similar=1.0,
        fr_different=10.0,
        rel_info_threshold=1.0,
        jacobian_coefficient=0.5,
        overparameterization_ratio=10.0,
        low_info_gain_threshold=0.01,
        high_phase_transition_count=5,
        high_loss_threshold=1.0,
        phase_transition_std_multiplier=2.0,
        fast_convergence_ratio=0.9,
        moderate_convergence_ratio=0.5,
        very_stable_std=0.01,
        stable_std=0.1,
        moderate_std=1.0
    ),
    k_neighbors=10
)

LARGE = ArchitectureConfig(
    dimensions=DimensionConfig(head_dim=128, num_heads=16, hidden_dim=2048),
    behavioral_space=BehavioralSpaceConfig(num_raw_metrics=65, min_dims=5, max_dims=7, num_centroids=1000),
    compression=CompressionConfig(low_rank_k=128, delta_rank=32),
    fitness=FitnessConfig(ema_decay=0.95, entropy_weight=1.0, magnitude_weight=1.0),
    test=TestConfig(batch_size=16, seq_len=128, pool_max_size=256, pool_min_size=32, test_pool_max_size=32, test_pool_min_size=16),
    numerical=NumericalConfig(epsilon=1e-10, svd_threshold=1e-06, attention_temperature=1.0),
    statistical_geometry=StatisticalGeometryConfig(
        fisher_num_samples=100,
        fisher_regularization=1e-6,
        fr_very_close=0.1,
        fr_similar=1.0,
        fr_different=10.0,
        rel_info_threshold=1.0,
        jacobian_coefficient=0.5,
        overparameterization_ratio=10.0,
        low_info_gain_threshold=0.01,
        high_phase_transition_count=5,
        high_loss_threshold=1.0,
        phase_transition_std_multiplier=2.0,
        fast_convergence_ratio=0.9,
        moderate_convergence_ratio=0.5,
        very_stable_std=0.01,
        stable_std=0.1,
        moderate_std=1.0
    ),
    k_neighbors=15
)
