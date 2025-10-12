"""Benchmarking utilities and datasets for slime mold evaluation"""

from slime.bench.datasets import ToyDataset, SinDataset, XORDataset, ParityDataset
from slime.bench.transformer import TransformerBaseline
from slime.bench.profile import profile_model, ProfileResult

__all__ = [
    'ToyDataset',
    'SinDataset',
    'XORDataset',
    'ParityDataset',
    'TransformerBaseline',
    'profile_model',
    'ProfileResult',
]
