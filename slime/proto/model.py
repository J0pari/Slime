from typing import Protocol, Tuple, Optional
import torch

class Pseudopod(Protocol):

    @property
    def correlation(self) -> torch.Tensor:
        ...

    def effective_rank(self) -> torch.Tensor:
        ...

    def coherence(self) -> torch.Tensor:
        ...

class Chemotaxis(Protocol):

    def deposit(self, nutrient: torch.Tensor, location: Tuple[float, float], concentration: float) -> None:
        ...

    def forage(self, metabolic_rate: float, hunger: float) -> Optional[torch.Tensor]:
        ...

class Organism(Protocol):

    def forward(self, stimulus: torch.Tensor, state: Optional[dict]) -> Tuple[torch.Tensor, dict]:
        ...

    def reset_state(self) -> None:
        ...