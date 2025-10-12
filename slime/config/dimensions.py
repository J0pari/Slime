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

    def __post_init__(self):
        assert self.batch_size > 0, f'batch_size must be positive, got {self.batch_size}'
        assert self.seq_len > 0, f'seq_len must be positive, got {self.seq_len}'

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
class ArchitectureConfig:
    dimensions: DimensionConfig
    behavioral_space: BehavioralSpaceConfig
    compression: CompressionConfig
    fitness: FitnessConfig
    test: TestConfig
    numerical: NumericalConfig
    k_neighbors: int = 5

    def __post_init__(self):
        assert self.k_neighbors > 0, f'k_neighbors must be positive, got {self.k_neighbors}'
        assert self.k_neighbors < self.behavioral_space.num_centroids, f'k_neighbors ({self.k_neighbors}) must be < num_centroids ({self.behavioral_space.num_centroids})'

TINY = ArchitectureConfig(
    dimensions=DimensionConfig(head_dim=16, num_heads=4, hidden_dim=64),
    behavioral_space=BehavioralSpaceConfig(num_raw_metrics=50, min_dims=3, max_dims=5, num_centroids=50),
    compression=CompressionConfig(low_rank_k=16, delta_rank=8),
    fitness=FitnessConfig(ema_decay=0.9, entropy_weight=1.0, magnitude_weight=1.0),
    test=TestConfig(batch_size=2, seq_len=16),
    numerical=NumericalConfig(epsilon=1e-10, svd_threshold=1e-06, attention_temperature=1.0),
    k_neighbors=5
)

SMALL = ArchitectureConfig(
    dimensions=DimensionConfig(head_dim=32, num_heads=8, hidden_dim=256),
    behavioral_space=BehavioralSpaceConfig(num_raw_metrics=50, min_dims=4, max_dims=6, num_centroids=200),
    compression=CompressionConfig(low_rank_k=32, delta_rank=16),
    fitness=FitnessConfig(ema_decay=0.9, entropy_weight=1.0, magnitude_weight=1.0),
    test=TestConfig(batch_size=4, seq_len=32),
    numerical=NumericalConfig(epsilon=1e-10, svd_threshold=1e-06, attention_temperature=1.0),
    k_neighbors=10
)

MEDIUM = ArchitectureConfig(
    dimensions=DimensionConfig(head_dim=64, num_heads=8, hidden_dim=512),
    behavioral_space=BehavioralSpaceConfig(num_raw_metrics=50, min_dims=5, max_dims=7, num_centroids=500),
    compression=CompressionConfig(low_rank_k=64, delta_rank=16),
    fitness=FitnessConfig(ema_decay=0.95, entropy_weight=1.0, magnitude_weight=1.0),
    test=TestConfig(batch_size=8, seq_len=64),
    numerical=NumericalConfig(epsilon=1e-10, svd_threshold=1e-06, attention_temperature=1.0),
    k_neighbors=10
)

LARGE = ArchitectureConfig(
    dimensions=DimensionConfig(head_dim=128, num_heads=16, hidden_dim=2048),
    behavioral_space=BehavioralSpaceConfig(num_raw_metrics=50, min_dims=5, max_dims=7, num_centroids=1000),
    compression=CompressionConfig(low_rank_k=128, delta_rank=32),
    fitness=FitnessConfig(ema_decay=0.95, entropy_weight=1.0, magnitude_weight=1.0),
    test=TestConfig(batch_size=16, seq_len=128),
    numerical=NumericalConfig(epsilon=1e-10, svd_threshold=1e-06, attention_temperature=1.0),
    k_neighbors=15
)
