"""Native slime mold API (non-PyTorch-compatible)"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from slime.core.organism import Organism
from slime.core.state import FlowState
from slime.memory.pool import PoolConfig


class SlimeModel(nn.Module):
    """Native slime mold model API.

    More flexible than PyTorch compatibility layer.
    Exposes full organism capabilities.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        output_dim: int,
        head_dim: int = 64,
        pool_config: Optional[PoolConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize SlimeModel.

        Args:
            input_dim: Input feature dimension
            latent_dim: Internal latent dimension
            output_dim: Output dimension
            head_dim: Pseudopod head dimension
            pool_config: Dynamic pool configuration
            device: Computation device
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Core organism
        self.organism = Organism(
            sensory_dim=input_dim,
            latent_dim=latent_dim,
            head_dim=head_dim,
            device=self.device,
            pool_config=pool_config,
        )

        # Output projection
        self.project_out = nn.Linear(latent_dim, output_dim).to(self.device)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[FlowState] = None,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[FlowState]]:
        """Forward pass.

        Args:
            x: Input [batch, input_dim]
            state: Optional previous state
            return_state: Whether to return new state

        Returns:
            (output, state) if return_state=True else (output, None)
        """
        # Forward through organism
        latent_out, new_state = self.organism(x, state)

        # Project to output dimension
        output = self.project_out(latent_out)

        if return_state:
            return output, new_state
        else:
            return output, None

    def reset(self) -> None:
        """Reset organism state"""
        self.organism.reset_state()

    def stats(self) -> dict:
        """Get statistics"""
        return self.organism.stats()

    def archive_summary(self) -> dict:
        """Get archive summary"""
        archive = self.organism.archive
        return {
            'size': archive.size(),
            'coverage': archive.coverage(),
            'max_fitness': archive.max_fitness(),
            'generation': archive._generation,
        }
