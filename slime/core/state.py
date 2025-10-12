from dataclasses import dataclass
from typing import Optional, Tuple
import torch

@dataclass
class FlowState:
    body: torch.Tensor
    behavior: Tuple[float, float]
    generation: int
    fitness: float
    archive_snapshot: Optional[dict] = None

    def to_dict(self) -> dict:
        return {'body': self.body.cpu().numpy().tolist(), 'behavior': self.behavior, 'generation': self.generation, 'fitness': self.fitness}

    @classmethod
    def from_dict(cls, data: dict, device: torch.device):
        return cls(body=torch.tensor(data['body'], device=device), behavior=tuple(data['behavior']), generation=data['generation'], fitness=data['fitness'])