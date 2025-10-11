"""PyTorch fallback kernel implementation (CPU/GPU compatible)"""

import torch
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TorchKernel:
    """Pure PyTorch kernel implementation.

    Implements proto.kernel.Kernel protocol.

    Used when Triton is unavailable (CPU, unsupported GPU, testing).
    Functionally equivalent to Triton kernels but slower.
    """

    def __init__(self, device: Optional[torch.device] = None):
        """Initialize kernel.

        Args:
            device: Target device for computation
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        batch_size, seq_len, dim = query.shape

        # Normalize query and key
        q_norm = query / (torch.norm(query, dim=-1, keepdim=True) + 1e-10)
        k_norm = key / (torch.norm(key, dim=-1, keepdim=True) + 1e-10)

        # Compute scores
        scores = torch.bmm(q_norm, k_norm.transpose(1, 2))  # [batch, seq, seq]
        scores = scores / (torch.sqrt(torch.tensor(dim, dtype=torch.float32)) * temperature)

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply to values
        output = torch.bmm(attn_weights, value)

        return output

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
        # Normalize
        k_norm = key / (torch.norm(key, dim=-1, keepdim=True) + 1e-10)
        v_norm = value / (torch.norm(value, dim=-1, keepdim=True) + 1e-10)

        # Correlation
        correlation = k_norm @ v_norm.T

        return correlation

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
        # Compute singular values
        s = torch.linalg.svdvals(matrix)

        # Filter near-zero values
        s = s[s > 1e-6]

        if s.numel() == 0:
            return torch.tensor(1.0, device=matrix.device)

        # Normalize to probabilities
        s_norm = s / (s.sum() + 1e-10)

        # Shannon entropy
        entropy = -torch.sum(s_norm * torch.log(s_norm + 1e-10))

        # Effective rank = exp(entropy)
        return torch.exp(entropy)
