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
        self._raw_metrics: Optional[torch.Tensor] = None

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
        # Store raw metrics for kernel PCA (50+ GPU-aware metrics)
        self._raw_metrics = self.compute_raw_metrics(attn, output)
        # Keep behavioral coordinates for backward compatibility during transition
        self.last_behavior = self._raw_metrics[:5]  # Use first 5 metrics temporarily
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

    def compute_raw_metrics(self, attn: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Compute 50+ GPU-aware raw metrics that kernel PCA will reduce to 3-5D.

        These capture actual GPU execution characteristics:
        - Memory access patterns (coalescing, bandwidth, cache behavior)
        - Compute utilization (SM occupancy, warp efficiency, divergence)
        - Data movement (global/shared memory, register usage)
        - Tensor operations (contiguity, stride patterns, alignment)
        """
        batch_size, num_heads, seq_len, head_dim = output.shape
        metrics = []

        # === Core 5 metrics (preserved for compatibility) ===
        attention_span = self.get_attention_distance(attn)
        activation_sparsity = self.get_activation_sparsity(output)
        gradient_flow = self.get_gradient_flow_magnitude()
        memory_locality = self.get_memory_access_locality(attn)
        compute_intensity = self.get_computational_intensity(output, attn.shape[-1])
        metrics.extend([attention_span, activation_sparsity, gradient_flow, memory_locality, compute_intensity])

        # === Memory Access Patterns (12 metrics) ===
        # Stride patterns - strided access kills GPU perf
        q_strides = sum([abs(s) for s in self.query_weight.stride()])
        k_strides = sum([abs(s) for s in self.key_weight.stride()])
        v_strides = sum([abs(s) for s in self.value_weight.stride()])
        metrics.extend([q_strides, k_strides, v_strides])

        # Memory contiguity - non-contiguous = extra copies
        attn_contiguous = 1.0 if attn.is_contiguous() else 0.0
        output_contiguous = 1.0 if output.is_contiguous() else 0.0
        metrics.extend([attn_contiguous, output_contiguous])

        # Attention sparsity - affects memory bandwidth
        attn_sparsity = (attn < 0.01).float().mean().item()
        metrics.append(attn_sparsity)

        # Memory footprint per batch element
        attn_mem_mb = (attn.element_size() * attn.nelement()) / (1024**2)
        output_mem_mb = (output.element_size() * output.nelement()) / (1024**2)
        metrics.extend([attn_mem_mb, output_mem_mb])

        # Alignment - unaligned access = performance penalty
        attn_aligned = 1.0 if (attn.data_ptr() % 128 == 0) else 0.0
        output_aligned = 1.0 if (output.data_ptr() % 128 == 0) else 0.0
        metrics.extend([attn_aligned, output_aligned])

        # === Compute Utilization (15 metrics) ===
        # Warp divergence proxy - how uniform are operations
        attn_variance = torch.var(attn).item()
        output_variance = torch.var(output).item()
        metrics.extend([attn_variance, output_variance])

        # Activation density - affects SM occupancy
        output_activation_density = (torch.abs(output) > self.numerical_config.epsilon).float().mean().item()
        metrics.append(output_activation_density)

        # Output magnitude distribution
        output_mean = torch.mean(output).item()
        output_std = torch.std(output).item()
        output_max = torch.max(torch.abs(output)).item()
        output_min = torch.min(torch.abs(output)).item()
        metrics.extend([output_mean, output_std, output_max, output_min])

        # Entropy - measure of information content
        attn_entropy = -(attn * torch.log(attn + self.numerical_config.epsilon)).sum(dim=-1).mean().item()
        metrics.append(attn_entropy)

        # Correlation rank - numerical stability indicator
        corr_rank = torch.linalg.matrix_rank(self._correlation).item() if self._correlation is not None else 0.0
        metrics.append(corr_rank)

        # Attention head utilization
        attn_head_variance = torch.var(attn.mean(dim=-1), dim=1).mean().item()
        metrics.append(attn_head_variance)

        # Dynamic range
        attn_dynamic_range = (attn.max() - attn.min()).item()
        output_dynamic_range = (output.max() - output.min()).item()
        metrics.extend([attn_dynamic_range, output_dynamic_range])

        # === Tensor Core Utilization (8 metrics) ===
        # Optimal for tensor cores: multiples of 8/16
        seq_len_tc_friendly = 1.0 if (seq_len % 16 == 0) else 0.0
        head_dim_tc_friendly = 1.0 if (head_dim % 8 == 0) else 0.0
        batch_tc_friendly = 1.0 if (batch_size % 8 == 0) else 0.0
        metrics.extend([seq_len_tc_friendly, head_dim_tc_friendly, batch_tc_friendly])

        # Matrix shape ratios - square matrices utilize tensor cores better
        attn_aspect_ratio = seq_len / max(seq_len, 1)
        output_aspect_ratio = seq_len / max(head_dim, 1)
        metrics.extend([attn_aspect_ratio, output_aspect_ratio])

        # Data type efficiency (fp16/bf16 better than fp32)
        dtype_efficiency = 1.0 if output.dtype in [torch.float16, torch.bfloat16] else 0.5
        metrics.append(dtype_efficiency)

        # Operation fusion potential
        softmax_fusion_potential = attention_span  # local attention = more fuseable
        metrics.append(softmax_fusion_potential)

        # === Register Pressure (8 metrics) ===
        # More live tensors = more register spills
        num_params = sum(p.numel() for p in self.parameters())
        metrics.append(num_params / 1000.0)  # Normalize

        # Intermediate tensor sizes
        qkv_intermediate_size = (batch_size * num_heads * seq_len * head_dim * 3) / 1000.0
        metrics.append(qkv_intermediate_size)

        # Weight matrix sizes
        weight_size = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1000.0
        metrics.append(weight_size)

        # Activation checkpointing benefit estimate
        activation_mem = (output.numel() * output.element_size()) / (1024**2)
        metrics.append(activation_mem)

        # Gradient accumulation memory
        if any(p.grad is not None for p in self.parameters()):
            grad_mem = sum(p.grad.numel() * p.grad.element_size() for p in self.parameters() if p.grad is not None) / (1024**2)
        else:
            grad_mem = 0.0
        metrics.append(grad_mem)

        # Sparsity benefits for memory
        param_sparsity = sum((torch.abs(p) < 1e-6).float().mean().item() for p in self.parameters()) / max(1, len(list(self.parameters())))
        metrics.append(param_sparsity)

        # Layer norm stats
        output_mean_abs = torch.mean(torch.abs(output)).item()
        metrics.append(output_mean_abs)

        # Numerical stability indicators
        output_has_nan = torch.isnan(output).any().float().item()
        output_has_inf = torch.isinf(output).any().float().item()
        metrics.extend([output_has_nan, output_has_inf])

        return torch.tensor(metrics, device=self.device)

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