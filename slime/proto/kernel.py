"""Kernel interface protocol"""

from typing import Protocol, Tuple
import torch


class Kernel(Protocol):
    """Protocol for GPU/CPU compute kernels.

    All kernel implementations must satisfy this interface.
    Enables swapping Triton/CUDA/PyTorch without changing core logic.
    """

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Compute attention with temperature scaling.

        Args:
            query: [batch, seq_len, dim]
            key: [batch, seq_len, dim]
            value: [batch, seq_len, dim]
            temperature: Softmax temperature

        Returns:
            output: [batch, seq_len, dim]
        """
        ...

    def correlation(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Compute normalized correlation matrix.

        Args:
            key: [batch, dim]
            value: [batch, dim]

        Returns:
            correlation: [batch, batch]
        """
        ...

    def effective_rank(
        self,
        matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Compute effective rank via entropy of singular values.

        Args:
            matrix: [n, n] square matrix

        Returns:
            rank: scalar tensor
        """
        ...
