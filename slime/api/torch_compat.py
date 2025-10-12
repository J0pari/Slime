"""PyTorch TransformerEncoder API compatibility"""

import torch
import torch.nn as nn
from typing import Optional
import logging

from slime.core.organism import Organism
from slime.memory.pool import PoolConfig
from slime.proto.kernel import Kernel
from slime.kernels.torch_fallback import TorchKernel

logger = logging.getLogger(__name__)


class SlimeMoldEncoder(nn.Module):
    """Drop-in replacement for nn.TransformerEncoder.

    API-compatible with torch.nn.TransformerEncoder but uses
    slime mold architecture internally.

    Example:
        # Replace this:
        # encoder = nn.TransformerEncoder(layer, num_layers=6)

        # With this:
        encoder = SlimeMoldEncoder(d_model=512, nhead=8)

        # Same API:
        output = encoder(src, mask=mask, src_key_padding_mask=padding_mask)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        pool_config: Optional[PoolConfig] = None,
        kernel: Optional[Kernel] = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        head_dim = d_model // nhead

        if pool_config is None:
            pool_config = PoolConfig(
                min_size=nhead,
                max_size=nhead * 4,
                birth_threshold=0.8,
                death_threshold=0.1,
            )

        if kernel is None:
            kernel = TorchKernel(self.device)

        self.organism = Organism(
            sensory_dim=d_model,
            latent_dim=d_model,
            head_dim=head_dim,
            device=self.device,
            kernel=kernel,
            pool_config=pool_config,
        )

        self._state: Optional[dict] = None

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            src: Input tensor
                If batch_first=False: [seq_len, batch, d_model]
                If batch_first=True: [batch, seq_len, d_model]
            mask: Attention mask (currently ignored)
            src_key_padding_mask: Padding mask (currently ignored)
            is_causal: Whether to use causal mask (currently ignored)

        Returns:
            Output tensor (same shape as input)
        """
        # Handle batch_first convention
        if not self.batch_first:
            src = src.transpose(0, 1)  # [seq, batch, d] -> [batch, seq, d]

        batch_size, seq_len, d_model = src.shape

        if d_model != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {d_model}")

        # Process each sequence position
        outputs = []

        for t in range(seq_len):
            stimulus = src[:, t, :]  # [batch, d_model]

            # Get state from previous timestep
            state = self._state.get(t - 1) if self._state and t > 0 else None

            # Forward through organism
            output, new_state = self.organism(stimulus, state)

            outputs.append(output)

            # Cache state for next timestep
            if self._state is None:
                self._state = {}
            self._state[t] = new_state

        # Stack outputs
        output = torch.stack(outputs, dim=1)  # [batch, seq, d_model]

        # Handle batch_first convention
        if not self.batch_first:
            output = output.transpose(0, 1)  # [batch, seq, d] -> [seq, batch, d]

        return output

    def reset_state(self) -> None:
        """Clear internal state (call between sequences)"""
        self._state = None
        self.organism.reset_state()

    def get_stats(self) -> dict:
        """Get organism statistics"""
        return self.organism.stats()
