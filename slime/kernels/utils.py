import torch
from typing import Tuple, Optional
import logging
try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
logger = logging.getLogger(__name__)

def cdiv(a: int, b: int) -> int:
    if HAS_TRITON:
        return triton.cdiv(a, b)
    return (a + b - 1) // b

def next_power_of_2(n: int) -> int:
    if HAS_TRITON:
        return triton.next_power_of_2(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

def validate_tensor(tensor: torch.Tensor, expected_dim: Optional[int]=None, expected_device: Optional[torch.device]=None, name: str='tensor') -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f'{name} must be torch.Tensor, got {type(tensor)}')
    if expected_dim is not None and tensor.dim() != expected_dim:
        raise ValueError(f'{name} expected {expected_dim}D, got {tensor.dim()}D')
    if expected_device is not None and tensor.device != expected_device:
        raise ValueError(f'{name} expected device {expected_device}, got {tensor.device}')
    if not tensor.is_contiguous():
        raise ValueError(f'{name} must be contiguous')
    if torch.isnan(tensor).any():
        raise ValueError(f'{name} contains NaN values')
    if torch.isinf(tensor).any():
        raise ValueError(f'{name} contains Inf values')

def safe_grid_config(size: int, block_size: int, max_grid: int=65535) -> Tuple[int, int]:
    if size <= 0:
        raise ValueError(f'size must be positive, got {size}')
    if block_size <= 0 or block_size & block_size - 1 != 0:
        raise ValueError(f'block_size must be power of 2, got {block_size}')
    grid_size = cdiv(size, block_size)
    if grid_size > max_grid:
        new_block_size = next_power_of_2(cdiv(size, max_grid))
        grid_size = cdiv(size, new_block_size)
        logger.warning(f'Adjusted block_size from {block_size} to {new_block_size}')
        block_size = new_block_size
    return (grid_size, block_size)

def optimal_num_warps(block_size: int) -> int:
    num_warps = max(1, block_size // 32)
    num_warps = next_power_of_2(num_warps)
    num_warps = min(8, num_warps)
    return num_warps