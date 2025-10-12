import torch
import torch.nn as nn
from typing import Optional
import logging
from slime.proto.kernel import Kernel
from slime.proto.component import Component
from slime.config.dimensions import FitnessConfig, NumericalConfig
logger = logging.getLogger(__name__)

class Pseudopod(nn.Module):

    def __init__(self, head_dim: int, kernel: Kernel, fitness_config: FitnessConfig, numerical_config: NumericalConfig, device: Optional[torch.device]=None, component_id: int=0, latent_dim: int=None, stimulus_dim: int=None, num_heads: int=1):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.kernel = kernel
        self.fitness_config = fitness_config
        self.numerical_config = numerical_config
        self.device = device or torch.device('cuda')
        self.component_id = component_id
        self.latent_dim = latent_dim if latent_dim is not None else head_dim
        self.stimulus_dim = stimulus_dim if stimulus_dim is not None else head_dim
        self.input_dim = self.latent_dim + self.stimulus_dim
        self.key_weight = nn.Parameter(torch.randn(self.num_heads, self.input_dim, head_dim, device=self.device))
        self.value_weight = nn.Parameter(torch.randn(self.num_heads, self.input_dim, head_dim, device=self.device))
        self.query_weight = nn.Parameter(torch.randn(self.num_heads, self.input_dim, head_dim, device=self.device))
        self.output_proj = nn.Parameter(torch.randn(self.num_heads, head_dim, head_dim, device=self.device))
        self._correlation: Optional[torch.Tensor] = None
        self._fitness = 0.0
        self.last_behavior: Optional[torch.Tensor] = None
        self._last_attention_pattern: Optional[torch.Tensor] = None

    def forward(self, latent: torch.Tensor, stimulus: torch.Tensor) -> torch.Tensor:
        x = torch.cat([latent, stimulus], dim=-1)
        batch_size, seq_len, input_dim = x.shape
        x_expanded = x.unsqueeze(1).expand(batch_size, self.num_heads, seq_len, input_dim)
        k = torch.einsum('bhsi,hid->bhsd', x_expanded, self.key_weight)
        v = torch.einsum('bhsi,hid->bhsd', x_expanded, self.value_weight)
        q = torch.einsum('bhsi,hid->bhsd', x_expanded, self.query_weight)
        self._correlation = self._compute_correlation(k, v)
        scores = torch.einsum('bhqd,bhkd->bhqk', q, k) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=x.device))
        attn = torch.softmax(scores, dim=-1)
        output = torch.einsum('bhqk,bhvd->bhqd', attn, v)
        output = torch.einsum('bhqd,hdo->bhqo', output, self.output_proj)
        self._last_attention_pattern = attn.detach()
        self.last_behavior = self.get_behavioral_coordinates(attn, output)
        attention_entropy = -(attn * torch.log(attn + self.numerical_config.epsilon)).sum(dim=-1).mean()
        output_magnitude = torch.norm(output) / torch.sqrt(torch.tensor(output.numel(), dtype=torch.float32, device=output.device))
        fitness_signal = (self.fitness_config.entropy_weight * attention_entropy +
                         self.fitness_config.magnitude_weight * output_magnitude)
        self._fitness = self.fitness_config.ema_decay * self._fitness + (1.0 - self.fitness_config.ema_decay) * fitness_signal.item()
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

    def get_gradient_flow_magnitude(self) -> float:
        grad_norms = []
        for param in self.parameters():
            if param.grad is not None:
                grad_norms.append(torch.norm(param.grad).item())
        if not grad_norms:
            return 0.0
        return sum(grad_norms) / len(grad_norms)

    def get_memory_access_locality(self, attn: torch.Tensor) -> float:
        seq_len = attn.shape[-1]
        positions = torch.arange(seq_len, device=attn.device, dtype=torch.float32)
        position_variance = []
        for i in range(seq_len):
            attn_weights = attn[..., i, :]
            mean_pos = (attn_weights * positions).sum(dim=-1)
            variance = (attn_weights * (positions - mean_pos.unsqueeze(-1)) ** 2).sum(dim=-1)
            position_variance.append(variance.mean().item())
        avg_variance = sum(position_variance) / len(position_variance)
        normalized = avg_variance / seq_len ** 2
        return min(1.0, normalized)

    def get_computational_intensity(self, output: torch.Tensor, seq_len: int) -> float:
        batch_size = output.shape[0]
        d = self.head_dim
        attn_flops = 2 * batch_size * seq_len * seq_len * d
        linear_flops = 2 * batch_size * seq_len * d * d * 3
        total_flops = attn_flops + linear_flops
        normalized_flops = total_flops / 1000000000.0
        return min(1.0, normalized_flops)

    def get_behavioral_coordinates(self, attn: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        attention_span = self.get_attention_distance(attn)
        activation_sparsity = self.get_activation_sparsity(output)
        gradient_flow = self.get_gradient_flow_magnitude()
        memory_locality = self.get_memory_access_locality(attn)
        compute_intensity = self.get_computational_intensity(output, attn.shape[-1])
        return torch.tensor([attention_span, activation_sparsity, gradient_flow, memory_locality, compute_intensity], device=self.device)

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
        return {'head_dim': self.head_dim, 'num_heads': self.num_heads, 'key_weight': self.key_weight.detach().cpu().numpy().tolist(), 'value_weight': self.value_weight.detach().cpu().numpy().tolist(), 'query_weight': self.query_weight.detach().cpu().numpy().tolist(), 'output_proj': self.output_proj.detach().cpu().numpy().tolist(), 'fitness': self._fitness}

    @classmethod
    def from_dict(cls, data: dict, kernel: Kernel, fitness_config: FitnessConfig, numerical_config: NumericalConfig, device: Optional[torch.device]=None) -> 'Pseudopod':
        pod = cls(data['head_dim'], kernel, fitness_config, numerical_config, device, num_heads=data['num_heads'])
        pod.key_weight.data = torch.tensor(data['key_weight'], device=pod.device)
        pod.value_weight.data = torch.tensor(data['value_weight'], device=pod.device)
        pod.query_weight.data = torch.tensor(data['query_weight'], device=pod.device)
        pod.output_proj.data = torch.tensor(data['output_proj'], device=pod.device)
        pod._fitness = data['fitness']
        return pod