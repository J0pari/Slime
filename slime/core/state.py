"""Flow state dataclass"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass
class FlowState:
    """Immutable snapshot of organism state.

    Used for checkpointing and inter-step communication.
    Does NOT hold references to pools/archives (prevents leaks).
    """

    body: torch.Tensor  # Current latent representation [batch, dim]
    behavior: Tuple[float, float]  # (rank, coherence) coordinates
    generation: int  # Current generation counter
    fitness: float  # Current fitness score

    # Optional serialized state (for checkpointing)
    archive_snapshot: Optional[dict] = None

    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        return {
            'body': self.body.cpu().numpy().tolist(),
            'behavior': self.behavior,
            'generation': self.generation,
            'fitness': self.fitness,
        }

    @classmethod
    def from_dict(cls, data: dict, device: torch.device):
        """Deserialize from dictionary"""
        return cls(
            body=torch.tensor(data['body'], device=device),
            behavior=tuple(data['behavior']),
            generation=data['generation'],
            fitness=data['fitness'],
        )
