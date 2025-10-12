"""
Component protocol - structural subtyping for pool members.

Defines interface that all pool components must satisfy.
Uses Protocol for duck typing (no inheritance required).
"""

from typing import Protocol, Dict, Any


class Component(Protocol):
    """
    Protocol for pool-managed components (e.g., Pseudopods).

    Components must provide:
    - fitness: Scalar quality metric
    - reset(): Reinitialize state
    - to_dict(): Serialize for checkpointing

    Example:
        class MyComponent:
            @property
            def fitness(self) -> float:
                return self._fitness

            def reset(self) -> None:
                self._state = torch.zeros(...)

            def to_dict(self) -> Dict[str, Any]:
                return {'state': self._state.cpu().numpy()}

        # No inheritance needed - structural typing!
        pool = DynamicPool(component_factory=MyComponent, ...)
    """

    @property
    def fitness(self) -> float:
        """Component quality metric (higher = better)."""
        ...

    def reset(self) -> None:
        """Reinitialize component state."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Serialize component for checkpointing."""
        ...


__all__ = ['Component']