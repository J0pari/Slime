"""Configuration loader with validation"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Dynamic pool configuration"""
    min_size: int = 4
    max_size: Optional[int] = 32
    birth_threshold: float = 0.8
    death_threshold: float = 0.1
    cull_interval: int = 100


@dataclass
class ArchiveConfig:
    """MAP-Elites archive configuration"""
    dimensions: List[str] = field(default_factory=lambda: ['rank', 'coherence'])
    bounds: List[tuple] = field(default_factory=lambda: [(0.0, 1.0), (0.0, 1.0)])
    resolution: int = 50


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    sensory_dim: int = 512
    latent_dim: int = 512
    head_dim: int = 64
    pool: PoolConfig = field(default_factory=PoolConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)


@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    learning_rate: float = 1e-4
    max_epochs: int = 100
    gradient_clip: float = 1.0
    checkpoint_interval: int = 1000


@dataclass
class ConfigSchema:
    """Complete configuration schema"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(config_path: Path) -> ConfigSchema:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Validated configuration schema
    """
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}, using defaults")
        return ConfigSchema()

    with open(config_path) as f:
        data = yaml.safe_load(f)

    if not data:
        return ConfigSchema()

    # Validate required fields
    if 'model' in data:
        model = data['model']
        required = ['sensory_dim', 'latent_dim', 'head_dim']
        for field in required:
            if field not in model:
                raise ValueError(f"Missing required model field: {field}")
            if not isinstance(model[field], int) or model[field] <= 0:
                raise ValueError(f"Invalid {field}: must be positive integer")

    if 'archive' in data:
        archive = data['archive']
        if 'grid_size' in archive:
            if not isinstance(archive['grid_size'], list):
                raise ValueError("archive.grid_size must be a list")
            if not all(isinstance(x, int) and x > 0 for x in archive['grid_size']):
                raise ValueError("archive.grid_size must contain positive integers")

    return ConfigSchema(**data)
