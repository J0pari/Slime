"""Model component protocols"""

from typing import Protocol, Tuple, Optional
import torch


class Pseudopod(Protocol):
    """Protocol for sensory probes.

    Pseudopodia extend into information space to sample gradients.
    """

    @property
    def correlation(self) -> torch.Tensor:
        """Correlation matrix between key and value."""
        ...

    def effective_rank(self) -> torch.Tensor:
        """Effective rank of correlation structure."""
        ...

    def coherence(self) -> torch.Tensor:
        """Coherence measure of correlation."""
        ...


class Chemotaxis(Protocol):
    """Protocol for nutrient search in behavioral space.

    Maps (rank, coherence) coordinates to learned representations.
    """

    def deposit(
        self,
        nutrient: torch.Tensor,
        location: Tuple[float, float],
        concentration: float,
    ) -> None:
        """Deposit nutrient at behavioral coordinates.

        Args:
            nutrient: Tensor to store
            location: (rank, coherence) coordinates
            concentration: Strength of deposit
        """
        ...

    def forage(
        self,
        metabolic_rate: float,
        hunger: float,
    ) -> Optional[torch.Tensor]:
        """Sample nutrient proportional to concentration.

        Args:
            metabolic_rate: Exploration temperature
            hunger: Bias toward high-concentration areas

        Returns:
            Sampled nutrient or None if empty
        """
        ...


class Organism(Protocol):
    """Protocol for the main organism (Plasmodium).

    Top-level component that orchestrates pseudopodia, memory, chemotaxis.
    """

    def forward(
        self,
        stimulus: torch.Tensor,
        state: Optional[dict],
    ) -> Tuple[torch.Tensor, dict]:
        """Single forward pass through organism.

        Args:
            stimulus: Input tensor [batch, seq_len, dim]
            state: Optional persistent state from previous call

        Returns:
            (output, new_state): Output tensor and updated state
        """
        ...

    def reset_state(self) -> None:
        """Clear all internal state (memory, food sources)."""
        ...
