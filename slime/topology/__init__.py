"""
Topology effects - hierarchical structure capabilities.

Declares effect signatures for optional p-adic topology features:
- hierarchy: GMM-based behavioral space clustering
- genealogy: Pseudopod lineage tracking with ultrametric distance
- p_adic: Hash-based distance for content-addressable storage

Effects are DECLARATIONS only (no implementations).
Implementations are provided by handlers in run.py (opt-in via --enable-topology).

Why Effects for Topology:
- Optional: Features enabled via handlers, not if statements
- Composable: Topology effects compose with other effects
- Testable: Mock handlers provide fake implementations
- Type-safe: Effect[T] preserves type information

Example:
    # Without handler (fallback):
    try:
        hierarchy = get_hierarchy()
        use_hybrid_metric(hierarchy)
    except EffectNotHandled:
        use_euclidean_metric()  # Graceful degradation

    # With handler (opt-in):
    with handle_effects(**{
        'topology.hierarchy': lambda: learn_gmm(archive)
    }):
        # Topology features available
        run_training()
"""

from typing import TYPE_CHECKING
from slime.effects import Effect, declare_effect, EffectNotHandled

# Forward declarations for type checking (implementations in Phase 2)
if TYPE_CHECKING:
    from slime.topology.hierarchy import BehavioralHierarchy
    from slime.topology.genealogy import Genealogy

# ============================================================================
# Effect Declarations (Signatures Only)
# ============================================================================

# Behavioral clustering via GMM
GetHierarchy: Effect['BehavioralHierarchy'] = declare_effect(
    'topology.hierarchy',
    return_type=None  # Type annotation via TYPE_CHECKING above
)

# Pseudopod lineage tracking
GetGenealogy: Effect['Genealogy'] = declare_effect(
    'topology.genealogy',
    return_type=None
)

# p-Adic distance enabled flag
UsePAdicDistance: Effect[bool] = declare_effect(
    'topology.p_adic',
    return_type=bool
)

# ============================================================================
# Convenience Functions (Hide Effect Machinery)
# ============================================================================

def get_hierarchy() -> 'BehavioralHierarchy':
    """
    Request behavioral hierarchy clustering.

    Returns:
        BehavioralHierarchy with GMM clusters and dendrogram

    Raises:
        EffectNotHandled: No handler installed (topology disabled)

    Example:
        try:
            hierarchy = get_hierarchy()
            cluster_id = hierarchy.predict_cluster(behavioral_coords)
        except EffectNotHandled:
            # Fallback: no clustering
            cluster_id = 0
    """
    return GetHierarchy.perform()  # type: ignore[return-value]


def get_genealogy() -> 'Genealogy':
    """
    Request pseudopod genealogy tracking.

    Returns:
        Genealogy with lineage tree and ultrametric distances

    Raises:
        EffectNotHandled: No handler installed (topology disabled)

    Example:
        try:
            genealogy = get_genealogy()
            distance = genealogy.ultrametric_distance(pod_a, pod_b)
        except EffectNotHandled:
            # Fallback: no lineage tracking
            distance = float('inf')
    """
    return GetGenealogy.perform()  # type: ignore[return-value]


def use_p_adic_distance() -> bool:
    """
    Check if p-adic distance enabled for content-addressable storage.

    Returns:
        True if p-adic distance should be used, False for standard Euclidean

    Raises:
        EffectNotHandled: No handler installed (topology disabled)

    Example:
        try:
            if use_p_adic_distance():
                dist = p_adic_distance(hash1, hash2)
            else:
                dist = euclidean_distance(pos1, pos2)
        except EffectNotHandled:
            dist = euclidean_distance(pos1, pos2)
    """
    return UsePAdicDistance.perform()


def try_get_hierarchy() -> 'BehavioralHierarchy | None':
    """
    Attempt to get hierarchy, return None if not available.

    Convenience wrapper for try/except pattern.

    Returns:
        BehavioralHierarchy if handler installed, None otherwise

    Example:
        hierarchy = try_get_hierarchy()
        if hierarchy is not None:
            use_hybrid_metric(hierarchy)
        else:
            use_euclidean_metric()
    """
    try:
        return get_hierarchy()
    except EffectNotHandled:
        return None


def try_get_genealogy() -> 'Genealogy | None':
    """
    Attempt to get genealogy, return None if not available.

    Returns:
        Genealogy if handler installed, None otherwise
    """
    try:
        return get_genealogy()
    except EffectNotHandled:
        return None


def is_topology_enabled() -> bool:
    """
    Check if any topology effects are enabled.

    Returns:
        True if at least one topology effect has a handler installed

    Example:
        if is_topology_enabled():
            logger.info("Topology features active")
    """
    return (GetHierarchy.is_handled() or
            GetGenealogy.is_handled() or
            UsePAdicDistance.is_handled())


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Effects (for handler installation)
    'GetHierarchy',
    'GetGenealogy',
    'UsePAdicDistance',

    # Convenience functions (for usage)
    'get_hierarchy',
    'get_genealogy',
    'use_p_adic_distance',
    'try_get_hierarchy',
    'try_get_genealogy',
    'is_topology_enabled',

    # Re-export from effects (convenience)
    'EffectNotHandled',
]
