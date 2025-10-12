"""
Unit tests for learning effect handlers using constraint-based verification.
"""

import pytest
import torch
from slime.effects.learning import (
    GetLocalUpdateRule,
    GetLearningProgress,
    GetParameterLocalization,
    get_local_update_rule,
    get_learning_progress,
    get_parameter_localization,
    try_get_local_update_rule,
    try_get_learning_progress,
    try_get_parameter_localization,
    is_learning_tracking_enabled,
    EffectNotHandled
)


def test_local_update_rule_effect_constraint(constraint):
    """Test GetLocalUpdateRule effect."""
    # No handler → raises
    try:
        get_local_update_rule((0, 0))
        raised = False
    except EffectNotHandled:
        raised = True
    constraint('EffectNotHandled raised without handler', lambda: (raised, raised, True, {}))

    # Install handler
    mock_params = {
        'kernel': torch.randn(3, 3),
        'growth_center': torch.tensor(0.15),
        'growth_width': torch.tensor(0.015)
    }

    with GetLocalUpdateRule.handler(lambda: mock_params):
        params = get_local_update_rule((10, 20))
        constraint('kernel in params', lambda: ('kernel' in params, 'kernel' in params, True, {}))
        constraint('growth_center in params', lambda: ('growth_center' in params, 'growth_center' in params, True, {}))
        constraint('growth_width in params', lambda: ('growth_width' in params, 'growth_width' in params, True, {}))
        constraint('kernel has shape (3, 3)', lambda: (params['kernel'].shape == (3, 3), params['kernel'].shape, (3, 3), {}))


def test_learning_progress_effect_constraint(constraint):
    """Test GetLearningProgress effect."""
    # No handler → raises
    try:
        get_learning_progress('pseudopod_1')
        raised = False
    except EffectNotHandled:
        raised = True
    constraint('EffectNotHandled raised without handler', lambda: (raised, raised, True, {}))

    # Install handler
    mock_progress = 0.75

    with GetLearningProgress.handler(lambda: mock_progress):
        progress = get_learning_progress('pseudopod_1')
        constraint('Progress equals expected', lambda: (progress == 0.75, progress, 0.75, {}))
        constraint('Progress in range [0, 1]', lambda: (0.0 <= progress <= 1.0, progress, '[0,1]', {}))


def test_parameter_localization_effect_constraint(constraint):
    """Test GetParameterLocalization effect."""
    # Mock pseudopod (don't need real one for effect test)
    mock_pseudopod = object()

    # No handler → raises
    try:
        get_parameter_localization(mock_pseudopod)
        raised = False
    except EffectNotHandled:
        raised = True
    constraint('EffectNotHandled raised without handler', lambda: (raised, raised, True, {}))

    # Install handler
    mock_localization = 0.85

    with GetParameterLocalization.handler(lambda: mock_localization):
        loc = get_parameter_localization(mock_pseudopod)
        constraint('Localization equals expected', lambda: (loc == 0.85, loc, 0.85, {}))
        constraint('Localization in range [0, 1]', lambda: (0.0 <= loc <= 1.0, loc, '[0,1]', {}))


def test_try_get_functions_constraint(constraint):
    """Test try_get_* convenience functions."""
    # No handlers → return None
    result1 = try_get_local_update_rule((0, 0))
    result2 = try_get_learning_progress('id')
    result3 = try_get_parameter_localization(object())
    constraint('try_get_local_update_rule returns None', lambda: (result1 is None, result1, None, {}))
    constraint('try_get_learning_progress returns None', lambda: (result2 is None, result2, None, {}))
    constraint('try_get_parameter_localization returns None', lambda: (result3 is None, result3, None, {}))

    # With handlers → return values
    mock_params = {'kernel': torch.randn(3, 3)}
    mock_progress = 0.5
    mock_localization = 0.6

    with GetLocalUpdateRule.handler(lambda: mock_params):
        result = try_get_local_update_rule((0, 0))
        constraint('try_get_local_update_rule returns value', lambda: (result is not None, result is not None, True, {}))

    with GetLearningProgress.handler(lambda: mock_progress):
        result = try_get_learning_progress('id')
        constraint('try_get_learning_progress returns 0.5', lambda: (result == 0.5, result, 0.5, {}))

    with GetParameterLocalization.handler(lambda: mock_localization):
        result = try_get_parameter_localization(object())
        constraint('try_get_parameter_localization returns 0.6', lambda: (result == 0.6, result, 0.6, {}))


def test_is_learning_tracking_enabled_constraint(constraint):
    """Test learning tracking status check."""
    # No handlers → disabled
    result = is_learning_tracking_enabled()
    constraint('Learning tracking disabled without handlers', lambda: (result is False, result, False, {}))

    # With any handler → enabled
    with GetLocalUpdateRule.handler(lambda: {}):
        result1 = is_learning_tracking_enabled()
        constraint('Learning tracking enabled with GetLocalUpdateRule', lambda: (result1 is True, result1, True, {}))

    with GetLearningProgress.handler(lambda: 0.0):
        result2 = is_learning_tracking_enabled()
        constraint('Learning tracking enabled with GetLearningProgress', lambda: (result2 is True, result2, True, {}))

    with GetParameterLocalization.handler(lambda: 0.0):
        result3 = is_learning_tracking_enabled()
        constraint('Learning tracking enabled with GetParameterLocalization', lambda: (result3 is True, result3, True, {}))


def test_effect_stacking_constraint(constraint):
    """Test multiple handlers stack correctly."""
    # Outer handler
    with GetLearningProgress.handler(lambda: 0.1):
        outer = get_learning_progress('id')
        constraint('Outer handler returns 0.1', lambda: (outer == 0.1, outer, 0.1, {}))

        # Inner handler shadows outer
        with GetLearningProgress.handler(lambda: 0.9):
            inner = get_learning_progress('id')
            constraint('Inner handler shadows with 0.9', lambda: (inner == 0.9, inner, 0.9, {}))

        # Back to outer handler
        restored = get_learning_progress('id')
        constraint('Outer handler restored', lambda: (restored == 0.1, restored, 0.1, {}))

    # No handlers
    try:
        get_learning_progress('id')
        raised = False
    except EffectNotHandled:
        raised = True
    constraint('EffectNotHandled after handler removed', lambda: (raised, raised, True, {}))


def test_multiple_effects_compose_constraint(constraint):
    """Test multiple learning effects compose."""
    mock_params = {'kernel': torch.randn(3, 3)}
    mock_progress = 0.7
    mock_localization = 0.8

    with GetLocalUpdateRule.handler(lambda: mock_params):
        with GetLearningProgress.handler(lambda: mock_progress):
            with GetParameterLocalization.handler(lambda: mock_localization):
                # All three effects available
                params = get_local_update_rule((0, 0))
                progress = get_learning_progress('id')
                loc = get_parameter_localization(object())

                constraint('Params returned', lambda: (params is not None, params is not None, True, {}))
                constraint('Progress is 0.7', lambda: (progress == 0.7, progress, 0.7, {}))
                constraint('Localization is 0.8', lambda: (loc == 0.8, loc, 0.8, {}))


def test_effect_laziness_constraint(constraint):
    """Test effect handlers are called lazily."""
    call_count = [0]

    def handler_impl():
        call_count[0] += 1
        return 0.5

    with GetLearningProgress.handler(handler_impl):
        # Not called yet
        constraint('Handler not called initially', lambda: (call_count[0] == 0, call_count[0], 0, {}))

        # Called on first perform
        get_learning_progress('id')
        constraint('Handler called once after first perform', lambda: (call_count[0] == 1, call_count[0], 1, {}))

        # Called again on second perform (not memoized)
        get_learning_progress('id')
        constraint('Handler called twice (not memoized)', lambda: (call_count[0] == 2, call_count[0], 2, {}))


def test_effect_thread_safety_constraint(constraint):
    """Test effects are thread-local."""
    import threading
    import time

    results = []

    def thread_a():
        with GetLearningProgress.handler(lambda: 0.1):
            time.sleep(0.01)  # Let thread B install handler
            results.append(('A', get_learning_progress('id')))

    def thread_b():
        with GetLearningProgress.handler(lambda: 0.9):
            results.append(('B', get_learning_progress('id')))

    ta = threading.Thread(target=thread_a)
    tb = threading.Thread(target=thread_b)

    ta.start()
    tb.start()
    ta.join()
    tb.join()

    # Each thread sees its own handler
    results_dict = dict(results)
    constraint('Thread A sees handler 0.1', lambda: (results_dict['A'] == 0.1, results_dict['A'], 0.1, {}))
    constraint('Thread B sees handler 0.9', lambda: (results_dict['B'] == 0.9, results_dict['B'], 0.9, {}))
