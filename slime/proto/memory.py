from typing import Protocol, Optional
import torch

class Memory(Protocol):

    def store(self, data: torch.Tensor, weight: float) -> None:
        ...

    def recall(self) -> Optional[torch.Tensor]:
        ...

    def clear(self) -> None:
        ...

    def __len__(self) -> int:
        ...