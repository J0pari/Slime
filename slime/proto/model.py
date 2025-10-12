from typing import Protocol, Tuple, Optional
import torch

class Pseudopod(Protocol):

    @property
    def correlation(self) -> torch.Tensor:
        pass

    def effective_rank(self) -> torch.Tensor:
        pass

    def coherence(self) -> torch.Tensor:
        pass

class Chemotaxis(Protocol):

    def deposit(self, nutrient: torch.Tensor, location: Tuple[float, float], concentration: float) -> None:
        pass

    def forage(self, metabolic_rate: float, hunger: float) -> Optional[torch.Tensor]:
        pass

class Organism(Protocol):

    def forward(self, stimulus: torch.Tensor, state: Optional[dict]) -> Tuple[torch.Tensor, dict]:
        pass

    def reset_state(self) -> None:
        pass