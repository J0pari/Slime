import torch
from typing import Optional, Deque
from collections import deque
import logging
from slime.proto.memory import Memory
logger = logging.getLogger(__name__)

class TubeNetwork:

    def __init__(self, decay: float=0.95, capacity: int=100, device: Optional[torch.device]=None):
        if not 0 <= decay <= 1:
            raise ValueError(f'decay must be in [0, 1], got {decay}')
        if capacity < 1:
            raise ValueError(f'capacity must be >= 1, got {capacity}')
        self.decay = decay
        self.capacity = capacity
        self.device = device or torch.device('cuda')
        self._tubes: Deque[tuple[torch.Tensor, float]] = deque(maxlen=capacity)
        self._decay_cache: Optional[torch.Tensor] = None

    def store(self, data: torch.Tensor, weight: float) -> None:
        if not isinstance(data, torch.Tensor):
            raise TypeError(f'data must be torch.Tensor, got {type(data)}')
        if not data.is_contiguous():
            data = data.contiguous()
        if data.device != self.device:
            data = data.to(self.device)
        self._tubes.append((data.detach(), float(weight)))
        self._decay_cache = None

    def recall(self) -> Optional[torch.Tensor]:
        if not self._tubes:
            return None
        first_data, _ = self._tubes[0]
        shapes = [data.shape for data, _ in self._tubes]
        if len(set(shapes)) > 1:
            logger.warning(f'Incompatible shapes in tubes: {shapes}, using first only')
            return first_data
        data_stack = torch.stack([data for data, _ in self._tubes])
        weights = torch.tensor([w for _, w in self._tubes], device=self.device)
        if self._decay_cache is None or len(self._decay_cache) != len(self._tubes):
            decay_factors = torch.tensor([self.decay ** i for i in range(len(self._tubes) - 1, -1, -1)], device=self.device, dtype=data_stack.dtype)
            self._decay_cache = decay_factors
        else:
            decay_factors = self._decay_cache
        combined_weights = weights * decay_factors
        combined_weights = combined_weights / (combined_weights.sum() + 1e-10)
        result = torch.sum(data_stack * combined_weights.view(-1, *[1] * (data_stack.ndim - 1)), dim=0)
        return result

    def clear(self) -> None:
        self._tubes.clear()
        self._decay_cache = None

    def __len__(self) -> int:
        return len(self._tubes)

    def stats(self) -> dict:
        if not self._tubes:
            return {'size': 0, 'capacity': self.capacity, 'utilization': 0.0, 'avg_weight': 0.0}
        weights = [w for _, w in self._tubes]
        return {'size': len(self._tubes), 'capacity': self.capacity, 'utilization': len(self._tubes) / self.capacity, 'avg_weight': sum(weights) / len(weights), 'min_weight': min(weights), 'max_weight': max(weights)}