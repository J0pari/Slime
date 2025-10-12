import torch
from typing import Optional
import logging
from slime.config.dimensions import NumericalConfig
logger = logging.getLogger(__name__)

class TorchKernel:

    def __init__(self, numerical_config: NumericalConfig, device: Optional[torch.device]=None):
        self.numerical_config = numerical_config
        self.device = device or torch.device('cuda')

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = query.shape
        eps = self.numerical_config.epsilon
        temp = self.numerical_config.attention_temperature
        q_norm = query / (torch.norm(query, dim=-1, keepdim=True) + eps)
        k_norm = key / (torch.norm(key, dim=-1, keepdim=True) + eps)
        scores = torch.bmm(q_norm, k_norm.transpose(1, 2))
        scores = scores / (torch.sqrt(torch.tensor(dim, dtype=torch.float32)) * temp)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.bmm(attn_weights, value)
        return output

    def correlation(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        eps = self.numerical_config.epsilon
        k_norm = key / (torch.norm(key, dim=-1, keepdim=True) + eps)
        v_norm = value / (torch.norm(value, dim=-1, keepdim=True) + eps)
        if k_norm.ndim == 4:
            correlation = torch.einsum('bhsd,bhtd->bhst', k_norm, v_norm)
        else:
            correlation = torch.bmm(k_norm, v_norm.transpose(1, 2))
        return correlation

    def effective_rank(self, matrix: torch.Tensor) -> torch.Tensor:
        eps = self.numerical_config.epsilon
        threshold = self.numerical_config.svd_threshold
        if matrix.ndim == 4:
            batch, num_heads, seq1, seq2 = matrix.shape
            matrix_reshaped = matrix.reshape(batch * num_heads, seq1, seq2)
            s = torch.linalg.svdvals(matrix_reshaped)
            s = s[s > threshold]
            if s.numel() == 0:
                return torch.tensor(1.0, device=matrix.device)
            s_norm = s / (s.sum() + eps)
            entropy = -torch.sum(s_norm * torch.log(s_norm + eps))
            return torch.exp(entropy)
        else:
            s = torch.linalg.svdvals(matrix)
            s = s[s > threshold]
            if s.numel() == 0:
                return torch.tensor(1.0, device=matrix.device)
            s_norm = s / (s.sum() + eps)
            entropy = -torch.sum(s_norm * torch.log(s_norm + eps))
            return torch.exp(entropy)