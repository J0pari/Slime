from typing import Protocol, Tuple
import torch

class Kernel(Protocol):

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, temperature: float) -> torch.Tensor:
        ...

    def correlation(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        ...

    def effective_rank(self, matrix: torch.Tensor) -> torch.Tensor:
        ...