"""Memory interface protocol"""

from typing import Protocol, Optional
import torch


class Memory(Protocol):
    """Protocol for temporal memory systems.

    Enables swapping memory implementations (GPU-resident, distributed, etc.)
    without changing core organism logic.
    """

    def store(
        self,
        data: torch.Tensor,
        weight: float,
    ) -> None:
        """Store data with associated weight.

        Args:
            data: Tensor to store
            weight: Importance weight (e.g., conductance, fitness)
        """
        ...

    def recall(self) -> Optional[torch.Tensor]:
        """Recall stored data with temporal decay.

        Returns:
            Aggregated memory tensor, or None if empty
        """
        ...

    def clear(self) -> None:
        """Clear all stored data."""
        ...

    def __len__(self) -> int:
        """Return number of stored items."""
        ...
