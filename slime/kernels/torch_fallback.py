import torch
from typing import Optional
import logging
logger = logging.getLogger(__name__)

class TorchKernel:

    def __init__(self, device: Optional[torch.device]=None):
        self.device = device or torch.device('cuda')

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, temperature: float) -> torch.Tensor:
        batch_size, seq_len, dim = query.shape
        q_norm = query / (torch.norm(query, dim=-1, keepdim=True) + 1e-10)
        k_norm = key / (torch.norm(key, dim=-1, keepdim=True) + 1e-10)
        scores = torch.bmm(q_norm, k_norm.transpose(1, 2))
        scores = scores / (torch.sqrt(torch.tensor(dim, dtype=torch.float32)) * temperature)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.bmm(attn_weights, value)
        return output

    def correlation(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        k_norm = key / (torch.norm(key, dim=-1, keepdim=True) + 1e-10)
        v_norm = value / (torch.norm(value, dim=-1, keepdim=True) + 1e-10)
        correlation = k_norm @ v_norm.T
        return correlation

    def effective_rank(self, matrix: torch.Tensor) -> torch.Tensor:
        s = torch.linalg.svdvals(matrix)
        s = s[s > 1e-06]
        if s.numel() == 0:
            return torch.tensor(1.0, device=matrix.device)
        s_norm = s / (s.sum() + 1e-10)
        entropy = -torch.sum(s_norm * torch.log(s_norm + 1e-10))
        return torch.exp(entropy)