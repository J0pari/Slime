"""Component lifecycle protocol for pool management"""

from typing import Protocol


class Component(Protocol):
    """Protocol for components managed by dynamic pools.

    All pooled components (Pseudopods, etc.) must implement this interface.
    Enables:
    - Archive serialization (via to_dict)
    - Pool lifecycle management (via fitness)
    - State reset between uses (via reset)

    Reconstruction handled by factories, not protocol (varying dependencies).
    """

    @property
    def fitness(self) -> float:
        """Current fitness metric for lifecycle management.

        Used by pool to determine:
        - Birth: High average fitness → spawn more
        - Death: Low individual fitness → cull component

        Returns:
            Fitness score in [0, inf), higher is better
        """
        ...

    def reset(self) -> None:
        """Reset internal state without deallocating.

        Called when component is reused from pool.
        Should clear caches, reset accumulators, but keep learned parameters.
        """
        ...

    def to_dict(self) -> dict:
        """Serialize to dictionary for archive storage.

        Must return plain dict (no tensor refs, no object refs).
        Archive stores this immutably, breaking circular references.

        Returns:
            Dictionary with all state needed to reconstruct component
        """
        ...
