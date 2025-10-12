"""
Algebraic effect handlers for compositional feature management.

Emulates algebraic effects via context managers (Python doesn't have native effects).
Inspired by Eff, Koka, OCaml 5 effect systems.

Key Concepts:
- Effect: Capability declaration (like type signature: Effect a)
- Handler: Provides implementation (like value: a)
- Perform: Request effect from current handler context
- Composition: Handlers stack like monadic bind (>>=)

Why This Matters:
Like Julia sets from z² + c - simple compositional rules generate
emergent complexity without enumerating 2^N feature combinations.

Example:
    # Declare effect
    GetConfig = declare_effect('config', dict)

    # Use effect (pure code - no knowledge of source)
    def run_training():
        config = GetConfig.perform()
        model = build_model(config['hidden_dim'])

    # Install handler (provide implementation)
    with GetConfig.handler(lambda: load_yaml('config.yaml')):
        run_training()  # Uses YAML config

    with GetConfig.handler(lambda: {'hidden_dim': 64}):
        run_training()  # Uses inline config

    # No if statements, no flags, full compositionality
"""

from typing import TypeVar, Generic, Callable, Optional, Dict, Any, Iterator
from contextlib import contextmanager, AbstractContextManager
from types import TracebackType
import threading
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

class EffectNotHandled(Exception):
    """
    Raised when effect requested but no handler installed.

    This is NOT an error - it's a signal for graceful degradation.
    Components should catch this and provide fallback behavior.

    Example:
        try:
            hierarchy = GetHierarchy.perform()
            use_hybrid_metric(hierarchy)
        except EffectNotHandled:
            use_euclidean_metric()  # Fallback
    """
    pass


class Effect(Generic[T]):
    """
    Effect signature - declares capability needed.

    Like Haskell type: Effect a = forall r. (a -> r) -> r
    Or OCaml 5: effect Get : unit -> a

    Thread-safe: Each thread has independent handler stack.
    """

    def __init__(self, name: str, return_type: Optional[type] = None):
        """
        Create effect signature.

        Args:
            name: Unique effect identifier (e.g., 'topology.hierarchy')
            return_type: Optional type annotation for documentation
        """
        self.name = name
        self.return_type = return_type
        self._handlers: Dict[int, Callable[[], T]] = {}  # thread_id -> handler
        logger.debug(f"Declared effect: {name}")

    def perform(self) -> T:
        """
        Request effect - handler provides implementation.

        Returns:
            Value provided by current handler

        Raises:
            EffectNotHandled: No handler installed in current context

        Example:
            config = GetConfig.perform()  # Gets value from handler
        """
        thread_id = threading.get_ident()
        if thread_id not in self._handlers:
            raise EffectNotHandled(
                f"Effect '{self.name}' not handled in current context. "
                f"Install handler with: `with {self.name}.handler(impl): ...`"
            )

        logger.debug(f"Performing effect: {self.name}")
        return self._handlers[thread_id]()

    @contextmanager
    def handler(self, implementation: Callable[[], T]) -> Iterator[None]:
        """
        Install handler for this effect.

        Handlers stack like monadic bind - inner handlers shadow outer.
        Thread-local: Only affects current thread.

        Args:
            implementation: Function providing effect value (called lazily)

        Yields:
            None (context manager protocol)

        Example:
            with GetConfig.handler(lambda: {'lr': 0.001}):
                train()  # Can call GetConfig.perform()
        """
        thread_id = threading.get_ident()
        old_handler = self._handlers.get(thread_id)
        self._handlers[thread_id] = implementation

        logger.debug(f"Installed handler for effect: {self.name}")

        try:
            yield
        finally:
            if old_handler is None:
                del self._handlers[thread_id]
            else:
                self._handlers[thread_id] = old_handler

            logger.debug(f"Removed handler for effect: {self.name}")

    def try_perform(self, default: T) -> T:
        """
        Attempt to perform effect, return default if not handled.

        Convenience wrapper for:
            try:
                return effect.perform()
            except EffectNotHandled:
                return default

        Args:
            default: Value to return if effect not handled

        Returns:
            Effect value or default

        Example:
            hierarchy = GetHierarchy.try_perform(None)
            if hierarchy is not None:
                use_hybrid_metric(hierarchy)
        """
        try:
            return self.perform()
        except EffectNotHandled:
            return default

    def is_handled(self) -> bool:
        """Check if handler installed in current context."""
        thread_id = threading.get_ident()
        return thread_id in self._handlers

    def __repr__(self) -> str:
        return f"Effect('{self.name}', handled={self.is_handled()})"


# Global effect registry (like Eff's effect row)
_effects: Dict[str, Effect[Any]] = {}


def declare_effect(name: str, return_type: Optional[type] = None) -> Effect[Any]:
    """
    Declare new effect capability.

    Idempotent: Calling twice with same name returns same Effect instance.

    Args:
        name: Unique effect identifier (e.g., 'topology.hierarchy')
        return_type: Optional type annotation for documentation

    Returns:
        Effect instance

    Example:
        GetHierarchy = declare_effect('topology.hierarchy', BehavioralHierarchy)
        GetGenealogy = declare_effect('topology.genealogy', Genealogy)

        # Later, anywhere in codebase:
        hierarchy = GetHierarchy.perform()
    """
    if name in _effects:
        logger.debug(f"Effect '{name}' already declared, returning existing")
        return _effects[name]

    effect: Effect[Any] = Effect(name, return_type)
    _effects[name] = effect
    logger.info(f"Declared effect: {name}")
    return effect


def get_effect(name: str) -> Effect[Any]:
    """
    Retrieve declared effect by name.

    Args:
        name: Effect identifier

    Returns:
        Effect instance

    Raises:
        KeyError: Effect not declared
    """
    if name not in _effects:
        raise KeyError(
            f"Effect '{name}' not declared. "
            f"Available effects: {list(_effects.keys())}"
        )
    return _effects[name]


@contextmanager
def handle_effects(**handlers: Callable[[], Any]) -> Iterator[None]:
    """
    Install multiple handlers at once.

    Convenience wrapper for nested `with effect.handler(...)` contexts.
    Handlers are installed in order, uninstalled in reverse order.

    Args:
        **handlers: effect_name=implementation pairs

    Yields:
        None (context manager protocol)

    Example:
        with handle_effects(
            hierarchy=lambda: learn_gmm(archive),
            genealogy=lambda: track_lineages(pool),
            p_adic=lambda: True
        ):
            # All three effects available
            run_training()

    Equivalent to:
        with GetHierarchy.handler(lambda: learn_gmm(archive)):
            with GetGenealogy.handler(lambda: track_lineages(pool)):
                with UsePAdicDistance.handler(lambda: True):
                    run_training()
    """
    contexts = []

    # Install handlers
    for name, implementation in handlers.items():
        effect = get_effect(name)
        ctx: AbstractContextManager[None] = effect.handler(implementation)
        contexts.append(ctx)
        ctx.__enter__()

    logger.info(f"Installed {len(handlers)} effect handlers: {list(handlers.keys())}")

    try:
        yield
    finally:
        # Uninstall in reverse order (stack unwinding)
        for ctx in reversed(contexts):
            ctx.__exit__(None, None, None)

        logger.info(f"Removed {len(handlers)} effect handlers")


def list_effects() -> Dict[str, bool]:
    """
    List all declared effects and their handler status.

    Returns:
        Dict mapping effect name to is_handled

    Example:
        >>> list_effects()
        {'topology.hierarchy': False, 'topology.genealogy': False, 'topology.p_adic': False}
        >>> with handle_effects(hierarchy=lambda: gmm):
        ...     list_effects()
        {'topology.hierarchy': True, 'topology.genealogy': False, 'topology.p_adic': False}
    """
    return {name: effect.is_handled() for name, effect in _effects.items()}


# Export public API
__all__ = [
    'Effect',
    'EffectNotHandled',
    'declare_effect',
    'get_effect',
    'handle_effects',
    'list_effects',
]
