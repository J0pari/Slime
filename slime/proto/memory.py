from typing import Protocol, Optional
import torch

class Memory(Protocol):

    def store(self, data: torch.Tensor, weight: float) -> None:
        pass

    def recall(self) -> Optional[torch.Tensor]:
        pass

    def clear(self) -> None:
        pass

    def __len__(self) -> int:
        pass