import torch
import torch.nn as nn
from typing import Optional
import logging
from slime.core.organism import Organism
from slime.memory.pool import PoolConfig
from slime.proto.kernel import Kernel
from slime.kernels.torch_fallback import TorchKernel
from slime.config.dimensions import ArchitectureConfig
logger = logging.getLogger(__name__)

class SlimeMoldEncoder(nn.Module):

    def __init__(self, d_model: int, nhead: int=8, dim_feedforward: int=2048, dropout: float=0.1, activation: str='relu', layer_norm_eps: float=1e-05, batch_first: bool=False, norm_first: bool=False, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None, pool_config: Optional[PoolConfig]=None, kernel: Optional[Kernel]=None, arch_config: Optional[ArchitectureConfig]=None):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        self.device = device or torch.device('cuda')
        if d_model % nhead != 0:
            raise ValueError(f'd_model ({d_model}) must be divisible by nhead ({nhead})')
        head_dim = d_model // nhead
        if pool_config is None:
            pool_config = PoolConfig(min_size=nhead, max_size=nhead * 4, birth_threshold=0.8, death_threshold=0.1)
        if kernel is None:
            kernel = TorchKernel(self.device)
        if arch_config is None:
            arch_config = ArchitectureConfig.TINY()
        self.organism = Organism(sensory_dim=d_model, latent_dim=d_model, head_dim=head_dim, arch_config=arch_config, device=self.device, kernel=kernel, pool_config=pool_config)
        self._state: Optional[dict] = None

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor]=None, src_key_padding_mask: Optional[torch.Tensor]=None, is_causal: bool=False) -> torch.Tensor:
        # Handle 2D input (e.g. flattened images) by reshaping to 3D
        if src.dim() == 2:
            batch_size, features = src.shape
            # Reshape to [batch, seq_len, d_model] where seq_len * d_model <= features
            seq_len = features // self.d_model
            if seq_len * self.d_model < features:
                src = src[:, :seq_len * self.d_model]  # Truncate to fit
            src = src.reshape(batch_size, seq_len, self.d_model)

        if not self.batch_first:
            src = src.transpose(0, 1)
        batch_size, seq_len, d_model = src.shape
        if d_model != self.d_model:
            raise ValueError(f'Expected d_model={self.d_model}, got {d_model}')
        outputs = []
        for t in range(seq_len):
            stimulus = src[:, t, :]
            state = self._state.get(t - 1) if self._state and t > 0 else None
            output, new_state = self.organism(stimulus, state)
            outputs.append(output)
            if self._state is None:
                self._state = {}
            self._state[t] = new_state
        output = torch.stack(outputs, dim=1)
        if not self.batch_first:
            output = output.transpose(0, 1)
        return output

    def reset_state(self) -> None:
        self._state = None
        self.organism.reset_state()

    def get_stats(self) -> dict:
        return self.organism.stats()