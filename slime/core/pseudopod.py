"""Pseudopod: sensory probe implementation"""

import torch
import torch.nn as nn
from typing import Optional
import logging

from slime.proto.kernel import Kernel
from slime.proto.component import Component

logger = logging.getLogger(__name__)


class Pseudopod(nn.Module):
    """Sensory probe that explores information space.

    Implements:
    - proto.model.Pseudopod (model interface)
    - proto.component.Component (lifecycle interface)

    Supports serialization for archive storage.
    Tracks fitness for pool lifecycle management.
    """

    def __init__(
        self,
        head_dim: int,
        kernel: Kernel,
        device: Optional[torch.device] = None,
    ):
        """Initialize pseudopod.

        Args:
            head_dim: Dimension of attention head
            kernel: Injected kernel for GPU compute
            device: Computation device
        """
        super().__init__()
        self.head_dim = head_dim
        self.kernel = kernel  # INJECTED (Decision #1)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Learnable parameters (evolved via backprop)
        self.key_weight = nn.Parameter(torch.randn(head_dim, head_dim, device=self.device))
        self.value_weight = nn.Parameter(torch.randn(head_dim, head_dim, device=self.device))
        self.query_weight = nn.Parameter(torch.randn(head_dim, head_dim, device=self.device))

        # Internal state
        self._correlation: Optional[torch.Tensor] = None
        self._fitness = 0.0

    def forward(
        self,
        latent: torch.Tensor,
        stimulus: torch.Tensor,
    ) -> torch.Tensor:
        """Extend pseudopod and compute attended output.

        Args:
            latent: Current body state [batch, head_dim]
            stimulus: External stimulus [batch, head_dim]

        Returns:
            Attended output [batch, head_dim]
        """
        # Project to key/value/query
        k = latent @ self.key_weight
        v = latent @ self.value_weight
        q = stimulus @ self.query_weight

        # Compute and cache correlation
        self._correlation = self._compute_correlation(k, v)

        # Simple attention (will be replaced by fused kernel)
        scores = (q @ k.T) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn = torch.softmax(scores, dim=-1)
        output = attn @ v

        # Update fitness based on attention entropy (diversity metric)
        self._update_fitness(attn)

        return output

    def _compute_correlation(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Compute normalized correlation matrix via injected kernel"""
        return self.kernel.correlation(k, v)

    def _update_fitness(self, attn: torch.Tensor) -> None:
        """Update fitness based on gradient magnitude (task-relevant).

        High gradient = component affects loss = high fitness
        Low gradient = component irrelevant = low fitness

        NOTE: Actual gradient computed externally during backward pass.
        This is a proxy using attention variance as surrogate.
        Real implementation should use: fitness = grad_norm of parameters.
        """
        # Proxy: attention variance (high variance = diverse attention)
        # TODO: Replace with actual gradient magnitude after backward pass
        variance = torch.var(attn, dim=-1).mean()

        # Exponential moving average
        self._fitness = 0.9 * self._fitness + 0.1 * variance.item()

    def get_attention_distance(self, attn: torch.Tensor) -> float:
        """Compute average attention distance (hardware-relevant metric).

        Short distance → local memory access patterns → cache-friendly
        Long distance → global memory access patterns → cache-unfriendly

        Args:
            attn: Attention weights [batch, seq_len] or [seq_len, seq_len]

        Returns:
            Average attention distance in [0, 1]
        """
        seq_len = attn.shape[-1]
        positions = torch.arange(seq_len, device=attn.device, dtype=torch.float32)

        # Compute weighted average distance
        # For each query position, compute average distance to attended keys
        distances = torch.abs(positions.unsqueeze(-1) - positions.unsqueeze(0))
        weighted_distances = (attn * distances).sum(dim=-1).mean()

        # Normalize by max possible distance
        normalized = weighted_distances / seq_len
        return normalized.item()

    def get_activation_sparsity(self, output: torch.Tensor) -> float:
        """Compute activation sparsity (hardware-relevant metric).

        High sparsity → fewer operations → compute-efficient
        Low sparsity → more operations → compute-intensive

        Args:
            output: Component output [batch, seq_len, dim]

        Returns:
            Sparsity ratio in [0, 1]
        """
        # L1/L2 ratio approximates sparsity
        l1 = torch.abs(output).sum()
        l2 = torch.sqrt((output ** 2).sum())

        # Normalize by dimension
        dim = output.numel()
        sparsity = 1.0 - (l1 / (l2 * torch.sqrt(torch.tensor(dim, dtype=torch.float32))))

        return sparsity.clamp(0.0, 1.0).item()

    def get_behavioral_coordinates(self, attn: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Compute behavioral coordinates for MAP-Elites archive.

        These coordinates MUST correlate with hardware costs:
        - attention_distance → memory access patterns
        - activation_sparsity → compute efficiency

        Args:
            attn: Attention weights [batch, seq_len, seq_len]
            output: Component output [batch, seq_len, dim]

        Returns:
            Behavioral coordinates [2] in [0, 1]²
        """
        attention_distance = self.get_attention_distance(attn)
        activation_sparsity = self.get_activation_sparsity(output)

        return torch.tensor([attention_distance, activation_sparsity], device=self.device)

    @property
    def correlation(self) -> torch.Tensor:
        """Get cached correlation matrix"""
        if self._correlation is None:
            raise RuntimeError("Must call forward() before accessing correlation")
        return self._correlation

    def effective_rank(self) -> torch.Tensor:
        """Compute effective rank via injected kernel"""
        return self.kernel.effective_rank(self.correlation)

    def coherence(self) -> torch.Tensor:
        """Measure correlation structure preservation under inversion"""
        eye = torch.eye(self.correlation.shape[0], device=self.device)
        partial = torch.linalg.solve(self.correlation + eye * 1e-3, eye)

        corr_sq = torch.sum(self.correlation ** 2)
        partial_sq = torch.sum(partial ** 2)

        return corr_sq / (corr_sq + partial_sq + 1e-10)

    @property
    def fitness(self) -> float:
        """Current fitness (for pool management)"""
        return self._fitness

    def reset(self) -> None:
        """Reset internal state"""
        self._correlation = None
        self._fitness = 0.0

    def to_dict(self) -> dict:
        """Serialize for archive storage"""
        return {
            'head_dim': self.head_dim,
            'key_weight': self.key_weight.detach().cpu().numpy().tolist(),
            'value_weight': self.value_weight.detach().cpu().numpy().tolist(),
            'query_weight': self.query_weight.detach().cpu().numpy().tolist(),
            'fitness': self._fitness,
        }

    @classmethod
    def from_dict(cls, data: dict, kernel: Kernel, device: Optional[torch.device] = None) -> 'Pseudopod':
        """Deserialize from archive.

        Args:
            data: Serialized state
            kernel: Injected kernel (required for reconstruction)
            device: Target device

        Returns:
            Reconstructed pseudopod
        """
        pod = cls(data['head_dim'], kernel, device)
        pod.key_weight.data = torch.tensor(data['key_weight'], device=pod.device)
        pod.value_weight.data = torch.tensor(data['value_weight'], device=pod.device)
        pod.query_weight.data = torch.tensor(data['query_weight'], device=pod.device)
        pod._fitness = data['fitness']
        return pod
