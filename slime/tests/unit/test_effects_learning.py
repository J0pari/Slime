"""
Unit tests for learning effect handlers.
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


def test_local_update_rule_effect():
    """Test GetLocalUpdateRule effect."""
    # No handler → raises
    with pytest.raises(EffectNotHandled):
        get_local_update_rule((0, 0))

    # Install handler
    mock_params = {
        'kernel': torch.randn(3, 3),
        'growth_center': torch.tensor(0.15),
        'growth_width': torch.tensor(0.015)
    }

    with GetLocalUpdateRule.handler(lambda: mock_params):
        params = get_local_update_rule((10, 20))
        assert 'kernel' in params
        assert 'growth_center' in params
        assert 'growth_width' in params
        assert params['kernel'].shape == (3, 3)


def test_learning_progress_effect():
    """Test GetLearningProgress effect."""
    # No handler → raises
    with pytest.raises(EffectNotHandled):
        get_learning_progress('pseudopod_1')

    # Install handler
    mock_progress = 0.75

    with GetLearningProgress.handler(lambda: mock_progress):
        progress = get_learning_progress('pseudopod_1')
        assert progress == 0.75
        assert 0.0 <= progress <= 1.0


def test_parameter_localization_effect():
    """Test GetParameterLocalization effect."""
    # Mock pseudopod (don't need real one for effect test)
    mock_pseudopod = object()

    # No handler → raises
    with pytest.raises(EffectNotHandled):
        get_parameter_localization(mock_pseudopod)

    # Install handler
    mock_localization = 0.85

    with GetParameterLocalization.handler(lambda: mock_localization):
        loc = get_parameter_localization(mock_pseudopod)
        assert loc == 0.85
        assert 0.0 <= loc <= 1.0


def test_try_get_functions():
    """Test try_get_* convenience functions."""
    # No handlers → return None
    assert try_get_local_update_rule((0, 0)) is None
    assert try_get_learning_progress('id') is None
    assert try_get_parameter_localization(object()) is None

    # With handlers → return values
    mock_params = {'kernel': torch.randn(3, 3)}
    mock_progress = 0.5
    mock_localization = 0.6

    with GetLocalUpdateRule.handler(lambda: mock_params):
        assert try_get_local_update_rule((0, 0)) is not None

    with GetLearningProgress.handler(lambda: mock_progress):
        assert try_get_learning_progress('id') == 0.5

    with GetParameterLocalization.handler(lambda: mock_localization):
        assert try_get_parameter_localization(object()) == 0.6


def test_is_learning_tracking_enabled():
    """Test learning tracking status check."""
    # No handlers → disabled
    assert is_learning_tracking_enabled() is False

    # With any handler → enabled
    with GetLocalUpdateRule.handler(lambda: {}):
        assert is_learning_tracking_enabled() is True

    with GetLearningProgress.handler(lambda: 0.0):
        assert is_learning_tracking_enabled() is True

    with GetParameterLocalization.handler(lambda: 0.0):
        assert is_learning_tracking_enabled() is True


def test_effect_stacking():
    """Test multiple handlers stack correctly."""
    # Outer handler
    with GetLearningProgress.handler(lambda: 0.1):
        assert get_learning_progress('id') == 0.1

        # Inner handler shadows outer
        with GetLearningProgress.handler(lambda: 0.9):
            assert get_learning_progress('id') == 0.9

        # Back to outer handler
        assert get_learning_progress('id') == 0.1

    # No handlers
    with pytest.raises(EffectNotHandled):
        get_learning_progress('id')


def test_multiple_effects_compose():
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

                assert params is not None
                assert progress == 0.7
                assert loc == 0.8


def test_effect_laziness():
    """Test effect handlers are called lazily."""
    call_count = 0

    def handler_impl():
        nonlocal call_count
        call_count += 1
        return 0.5

    with GetLearningProgress.handler(handler_impl):
        # Not called yet
        assert call_count == 0

        # Called on first perform
        get_learning_progress('id')
        assert call_count == 1

        # Called again on second perform (not memoized)
        get_learning_progress('id')
        assert call_count == 2


def test_effect_thread_safety():
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
    assert results_dict['A'] == 0.1
    assert results_dict['B'] == 0.9
