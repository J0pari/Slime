"""
Learning effect declarations for Neural CA.

Declares optional effects for Neural CA learning progress tracking:
- GetLocalUpdateRule: Access spatially-varying CA parameters
- GetLearningProgress: Track Δ prediction error for curiosity

Effects enable optional instrumentation without coupling components.
"""

from typing import TYPE_CHECKING, Dict, Tuple
from slime.effects import Effect, declare_effect, EffectNotHandled
import torch

if TYPE_CHECKING:
    from slime.core.pseudopod import Pseudopod


# ============================================================================
# Effect Declarations
# ============================================================================

# CA local update rule parameters (spatially-varying)
GetLocalUpdateRule: Effect[Dict[str, torch.Tensor]] = declare_effect(
    'learning.local_update_rule',
    return_type=dict
)

# Learning progress metric (Δ prediction error)
GetLearningProgress: Effect[float] = declare_effect(
    'learning.learning_progress',
    return_type=float
)

# CA parameter localization metric (spatial variance)
GetParameterLocalization: Effect[float] = declare_effect(
    'learning.parameter_localization',
    return_type=float
)


# ============================================================================
# Convenience Functions
# ============================================================================

def get_local_update_rule(position: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
    """
    Request CA update rule parameters at specific spatial position.

    Args:
        position: Spatial coordinates (e.g., (x, y) for 2D CA)

    Returns:
        Dict with CA parameters:
            - 'kernel': Convolution kernel weights
            - 'growth_center': Flow-Lenia growth center
            - 'growth_width': Flow-Lenia growth width

    Raises:
        EffectNotHandled: No handler installed (local rules not tracked)

    Example:
        try:
            params = get_local_update_rule((10, 20))
            visualize_local_kernel(params['kernel'])
        except EffectNotHandled:
            # Fallback: global parameters
            pass
    """
    return GetLocalUpdateRule.perform()


def get_learning_progress(component_id: str) -> float:
    """
    Request learning progress metric for specific component.

    Learning progress = -slope(prediction_error_over_time)
    High progress = learning fast, low progress = plateaued

    Args:
        component_id: Component identifier

    Returns:
        Learning progress in [0, 1] (0 = no learning, 1 = rapid learning)

    Raises:
        EffectNotHandled: No handler installed (progress not tracked)

    Example:
        try:
            progress = get_learning_progress('pseudopod_42')
            if progress < 0.1:
                retire_component()  # Not learning, cull
        except EffectNotHandled:
            # Fallback: use fitness instead
            if fitness < threshold:
                retire_component()
    """
    return GetLearningProgress.perform()


def get_parameter_localization(component: 'Pseudopod') -> float:
    """
    Request parameter localization metric.

    Parameter localization = spatial variance of CA update rule parameters
    High localization = parameters vary across space (good)
    Low localization = global parameters (bad)

    Args:
        component: Pseudopod to measure

    Returns:
        Localization score in [0, 1] (0 = global, 1 = fully localized)

    Raises:
        EffectNotHandled: No handler installed

    Example:
        try:
            localization = get_parameter_localization(pseudopod)
            if localization < 0.3:
                logger.warning("CA parameters too global")
        except EffectNotHandled:
            pass
    """
    return GetParameterLocalization.perform()


def try_get_local_update_rule(position: Tuple[int, ...]) -> Dict[str, torch.Tensor] | None:
    """Attempt to get local CA rule, return None if not available."""
    try:
        return get_local_update_rule(position)
    except EffectNotHandled:
        return None


def try_get_learning_progress(component_id: str) -> float | None:
    """Attempt to get learning progress, return None if not available."""
    try:
        return get_learning_progress(component_id)
    except EffectNotHandled:
        return None


def try_get_parameter_localization(component: 'Pseudopod') -> float | None:
    """Attempt to get parameter localization, return None if not available."""
    try:
        return get_parameter_localization(component)
    except EffectNotHandled:
        return None


def is_learning_tracking_enabled() -> bool:
    """
    Check if learning effects are enabled.

    Returns:
        True if at least one learning effect has a handler installed
    """
    return (GetLocalUpdateRule.is_handled() or
            GetLearningProgress.is_handled() or
            GetParameterLocalization.is_handled())


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Effects
    'GetLocalUpdateRule',
    'GetLearningProgress',
    'GetParameterLocalization',

    # Convenience functions
    'get_local_update_rule',
    'get_learning_progress',
    'get_parameter_localization',
    'try_get_local_update_rule',
    'try_get_learning_progress',
    'try_get_parameter_localization',
    'is_learning_tracking_enabled',

    # Re-export
    'EffectNotHandled',
]
