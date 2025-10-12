from typing import Protocol

class Component(Protocol):

    @property
    def fitness(self) -> float:
        pass

    def reset(self) -> None:
        pass

    def to_dict(self) -> dict:
        pass