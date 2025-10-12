import torch
import torch.nn as nn
from typing import Optional
import logging
from slime.proto.kernel import Kernel
from slime.proto.component import Component
logger = logging.getLogger(__name__)

class Pseudopod(nn.Module):

    def __init__(self, head_dim: int, kernel: Kernel, device: Optional[torch.device]=None):
        super().__init__()
        self.head_dim = head_dim
        self.kernel = kernel
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.key_weight = nn.Parameter(torch.randn(head_dim, head_dim, device=self.device))
        self.value_weight = nn.Parameter(torch.randn(head_dim, head_dim, device=self.device))
        self.query_weight = nn.Parameter(torch.randn(head_dim, head_dim, device=self.device))
        self._correlation: Optional[torch.Tensor] = None
        self._fitness = 0.0

    def forward(self, latent: torch.Tensor, stimulus: torch.Tensor) -> torch.Tensor:
        k = latent @ self.key_weight
        v = latent @ self.value_weight
        q = stimulus @ self.query_weight
        self._correlation = self._compute_correlation(k, v)
        scores = q @ k.T / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn = torch.softmax(scores, dim=-1)
        output = attn @ v
        self._update_fitness(attn)
        return output

    def _compute_correlation(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.kernel.correlation(k, v)

    def update_fitness_from_gradients(self) -> None:
        grad_norms = []
        for param in self.parameters():
            if param.grad is not None:
                grad_norms.append(torch.norm(param.grad).item())
        if not grad_norms:
            return
        mean_grad = sum(grad_norms) / len(grad_norms)
        self._fitness = 0.9 * self._fitness + 0.1 * mean_grad

    def get_attention_distance(self, attn: torch.Tensor) -> float:
        seq_len = attn.shape[-1]
        positions = torch.arange(seq_len, device=attn.device, dtype=torch.float32)
        distances = torch.abs(positions.unsqueeze(-1) - positions.unsqueeze(0))
        weighted_distances = (attn * distances).sum(dim=-1).mean()
        normalized = weighted_distances / seq_len
        return normalized.item()

    def get_activation_sparsity(self, output: torch.Tensor) -> float:
        l1 = torch.abs(output).sum()
        l2 = torch.sqrt((output ** 2).sum())
        dim = output.numel()
        sparsity = 1.0 - l1 / (l2 * torch.sqrt(torch.tensor(dim, dtype=torch.float32)))
        return sparsity.clamp(0.0, 1.0).item()

    def get_behavioral_coordinates(self, attn: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        attention_distance = self.get_attention_distance(attn)
        activation_sparsity = self.get_activation_sparsity(output)
        return torch.tensor([attention_distance, activation_sparsity], device=self.device)

    @property
    def correlation(self) -> torch.Tensor:
        if self._correlation is None:
            raise RuntimeError('Must call forward() before accessing correlation')
        return self._correlation

    def effective_rank(self) -> torch.Tensor:
        return self.kernel.effective_rank(self.correlation)

    def coherence(self) -> torch.Tensor:
        eye = torch.eye(self.correlation.shape[0], device=self.device)
        partial = torch.linalg.solve(self.correlation + eye * 0.001, eye)
        corr_sq = torch.sum(self.correlation ** 2)
        partial_sq = torch.sum(partial ** 2)
        return corr_sq / (corr_sq + partial_sq + 1e-10)

    @property
    def fitness(self) -> float:
        return self._fitness

    def reset(self) -> None:
        self._correlation = None
        self._fitness = 0.0

    def to_dict(self) -> dict:
        return {'head_dim': self.head_dim, 'key_weight': self.key_weight.detach().cpu().numpy().tolist(), 'value_weight': self.value_weight.detach().cpu().numpy().tolist(), 'query_weight': self.query_weight.detach().cpu().numpy().tolist(), 'fitness': self._fitness}

    @classmethod
    def from_dict(cls, data: dict, kernel: Kernel, device: Optional[torch.device]=None) -> 'Pseudopod':
        pod = cls(data['head_dim'], kernel, device)
        pod.key_weight.data = torch.tensor(data['key_weight'], device=pod.device)
        pod.value_weight.data = torch.tensor(data['value_weight'], device=pod.device)
        pod.query_weight.data = torch.tensor(data['query_weight'], device=pod.device)
        pod._fitness = data['fitness']
        return pod