import torch
import torch.nn as nn
from typing import Optional

class TransformerBaseline(nn.Module):

    def __init__(self, d_model: int=128, nhead: int=8, num_layers: int=6, dim_feedforward: int=512, dropout: float=0.1, device: Optional[torch.device]=None):
        super().__init__()
        self.d_model = d_model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, device=self.device)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embedding = nn.Linear(d_model, d_model, device=self.device)
        self.output_proj = nn.Linear(d_model, d_model, device=self.device)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        x = self.embedding(x)
        x = self.transformer(x, mask=mask)
        x = self.output_proj(x)
        return x

    def count_parameters(self) -> int:
        return sum((p.numel() for p in self.parameters() if p.requires_grad))

class FlashAttentionBaseline(nn.Module):

    def __init__(self, d_model: int=128, nhead: int=8, num_layers: int=6, dim_feedforward: int=512, dropout: float=0.1, device: Optional[torch.device]=None):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding = nn.Linear(d_model, d_model, device=self.device)
        self.layers = nn.ModuleList([FlashAttentionLayer(d_model, nhead, dim_feedforward, dropout, self.device) for _ in range(num_layers)])
        self.output_proj = nn.Linear(d_model, d_model, device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_proj(x)
        return x

    def count_parameters(self) -> int:
        return sum((p.numel() for p in self.parameters() if p.requires_grad))

class FlashAttentionLayer(nn.Module):

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, device: torch.device):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.device = device
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, device=device)
        self.out_proj = nn.Linear(d_model, d_model, device=device)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward, device=device), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model, device=device))
        self.norm1 = nn.LayerNorm(d_model, device=device)
        self.norm2 = nn.LayerNorm(d_model, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        residual = x
        x = self.norm1(x)
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0, is_causal=False)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        x = self.out_proj(attn_output)
        x = self.dropout(x)
        x = x + residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual
        return x

def create_baseline(baseline_type: str, **kwargs) -> nn.Module:
    baselines = {'transformer': TransformerBaseline, 'flash_attention': FlashAttentionBaseline}
    if baseline_type not in baselines:
        raise ValueError(f'Unknown baseline: {baseline_type}. Available: {list(baselines.keys())}')
    return baselines[baseline_type](**kwargs)