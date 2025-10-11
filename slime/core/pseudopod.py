"""Pseudopod: sensory probe implementation"""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class Pseudopod(nn.Module):
    """Sensory probe that explores information space.

    Implements proto.model.Pseudopod protocol.
    Supports serialization for archive storage.
    Tracks fitness for pool lifecycle management.
    """

    def __init__(self, head_dim: int, device: torch.device = None):
        super().__init__()
        self.head_dim = head_dim
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
        """Compute normalized correlation matrix"""
        # Normalize
        k_norm = k / (torch.norm(k, dim=-1, keepdim=True) + 1e-10)
        v_norm = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-10)

        # Correlation
        corr = k_norm @ v_norm.T

        return corr

    def _update_fitness(self, attn: torch.Tensor) -> None:
        """Update fitness based on attention pattern.

        High entropy = exploring diverse space = high fitness
        Low entropy = collapsed to single point = low fitness
        """
        # Attention entropy
        entropy = -torch.sum(attn * torch.log(attn + 1e-10), dim=-1).mean()

        # Exponential moving average
        self._fitness = 0.9 * self._fitness + 0.1 * entropy.item()

    @property
    def correlation(self) -> torch.Tensor:
        """Get cached correlation matrix"""
        if self._correlation is None:
            raise RuntimeError("Must call forward() before accessing correlation")
        return self._correlation

    def effective_rank(self) -> torch.Tensor:
        """Compute effective rank via entropy of singular values"""
        s = torch.linalg.svdvals(self.correlation)
        s = s[s > 1e-6]

        if s.numel() == 0:
            return torch.tensor(1.0, device=self.device)

        s_norm = s / (s.sum() + 1e-10)
        entropy = -torch.sum(s_norm * torch.log(s_norm + 1e-10))

        return torch.exp(entropy)

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
    def from_dict(cls, data: dict, device: torch.device = None) -> 'Pseudopod':
        """Deserialize from archive"""
        pod = cls(data['head_dim'], device)
        pod.key_weight.data = torch.tensor(data['key_weight'], device=pod.device)
        pod.value_weight.data = torch.tensor(data['value_weight'], device=pod.device)
        pod.query_weight.data = torch.tensor(data['query_weight'], device=pod.device)
        pod._fitness = data['fitness']
        return pod
