from typing import Protocol

class Component(Protocol):

    @property
    def fitness(self) -> float:
        ...

    def reset(self) -> None:
        ...

    def to_dict(self) -> dict:
        ...