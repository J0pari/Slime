"""Tube network: temporal memory with exponential decay"""

import torch
from typing import Optional, Deque
from collections import deque
import logging

from slime.proto.memory import Memory

logger = logging.getLogger(__name__)


class TubeNetwork:
    """Temporal memory with exponential decay.

    Implements proto.memory.Memory protocol.

    Biological inspiration: Slime molds reinforce pathways to food sources.
    Tubes persist based on conductance (weight), decaying over time.

    Properties:
    - Exponential decay (older memories fade)
    - Weighted storage (important memories weighted higher)
    - GPU-resident (all tensors on device)
    - Bounded capacity (no unbounded growth)
    """

    def __init__(
        self,
        decay: float = 0.95,
        capacity: int = 100,
        device: Optional[torch.device] = None,
    ):
        """Initialize tube network.

        Args:
            decay: Exponential decay factor per timestep [0, 1]
            capacity: Maximum number of stored memories
            device: GPU device for storage
        """
        if not 0 <= decay <= 1:
            raise ValueError(f"decay must be in [0, 1], got {decay}")

        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")

        self.decay = decay
        self.capacity = capacity
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Storage: deque of (tensor, weight) tuples
        self._tubes: Deque[tuple[torch.Tensor, float]] = deque(maxlen=capacity)

        # Precomputed decay factors (cached for efficiency)
        self._decay_cache: Optional[torch.Tensor] = None

    def store(self, data: torch.Tensor, weight: float) -> None:
        """Store data with associated weight.

        Args:
            data: Tensor to store (any shape)
            weight: Importance weight (conductance)
        """
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"data must be torch.Tensor, got {type(data)}")

        if not data.is_contiguous():
            data = data.contiguous()

        # Move to device if needed
        if data.device != self.device:
            data = data.to(self.device)

        # Store with weight
        self._tubes.append((data.detach(), float(weight)))

        # Invalidate decay cache (length changed)
        self._decay_cache = None

    def recall(self) -> Optional[torch.Tensor]:
        """Recall stored data with temporal decay.

        Returns weighted sum of stored tensors with exponential decay.
        Older memories have less weight.

        Returns:
            Aggregated tensor or None if empty
        """
        if not self._tubes:
            return None

        # Get first tensor for shape reference
        first_data, _ = self._tubes[0]

        # Ensure all tensors have compatible shapes
        shapes = [data.shape for data, _ in self._tubes]
        if len(set(shapes)) > 1:
            logger.warning(f"Incompatible shapes in tubes: {shapes}, using first only")
            return first_data

        # Stack all data
        data_stack = torch.stack([data for data, _ in self._tubes])
        weights = torch.tensor([w for _, w in self._tubes], device=self.device)

        # Compute decay factors (most recent = 1.0, oldest = decay^(n-1))
        if self._decay_cache is None or len(self._decay_cache) != len(self._tubes):
            # Recent to old ordering
            decay_factors = torch.tensor(
                [self.decay ** i for i in range(len(self._tubes) - 1, -1, -1)],
                device=self.device,
                dtype=data_stack.dtype,
            )
            self._decay_cache = decay_factors
        else:
            decay_factors = self._decay_cache

        # Combine weights with decay
        combined_weights = weights * decay_factors

        # Normalize
        combined_weights = combined_weights / (combined_weights.sum() + 1e-10)

        # Weighted sum
        result = torch.sum(
            data_stack * combined_weights.view(-1, *([1] * (data_stack.ndim - 1))),
            dim=0,
        )

        return result

    def clear(self) -> None:
        """Clear all stored memories."""
        self._tubes.clear()
        self._decay_cache = None

    def __len__(self) -> int:
        """Return number of stored memories."""
        return len(self._tubes)

    def stats(self) -> dict:
        """Get statistics about tube network."""
        if not self._tubes:
            return {
                'size': 0,
                'capacity': self.capacity,
                'utilization': 0.0,
                'avg_weight': 0.0,
            }

        weights = [w for _, w in self._tubes]

        return {
            'size': len(self._tubes),
            'capacity': self.capacity,
            'utilization': len(self._tubes) / self.capacity,
            'avg_weight': sum(weights) / len(weights),
            'min_weight': min(weights),
            'max_weight': max(weights),
        }
