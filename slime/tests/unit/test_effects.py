"""
Unit tests for algebraic effect handler system.

Tests compositional properties using constraint-based verification.
"""

import pytest
import threading
from slime.effects import (
    Effect,
    EffectNotHandled,
    declare_effect,
    get_effect,
    handle_effects,
    list_effects,
)


def test_declare_effect_constraint(constraint):
    """Effect can be declared with name and type."""
    effect = declare_effect('test.simple', int)
    constraint('Effect has correct name', lambda: (effect.name == 'test.simple', effect.name, 'test.simple', {}))
    constraint('Effect has correct type', lambda: (effect.return_type == int, effect.return_type, int, {}))
    constraint('Effect not handled initially', lambda: (not effect.is_handled(), effect.is_handled(), False, {}))


def test_declare_effect_idempotent_constraint(constraint):
    """Declaring same effect twice returns same instance."""
    effect1 = declare_effect('test.idempotent', str)
    effect2 = declare_effect('test.idempotent', str)
    constraint('Same effect instance returned', lambda: (effect1 is effect2, id(effect1), id(effect2), {}))


def test_get_effect_constraint(constraint):
    """Can retrieve declared effect by name."""
    original = declare_effect('test.retrieve', float)
    retrieved = get_effect('test.retrieve')
    constraint('Retrieved effect is same instance', lambda: (retrieved is original, id(retrieved), id(original), {}))


def test_get_effect_not_declared_constraint(constraint):
    """Getting undeclared effect raises KeyError."""
    try:
        get_effect('test.nonexistent')
        raised = False
    except KeyError as e:
        raised = True
        has_message = 'not declared' in str(e)
    constraint('KeyError raised for undeclared effect', lambda: (raised, raised, True, {}))
    constraint('Error message mentions not declared', lambda: (has_message, has_message, True, {}))


def test_perform_without_handler_constraint(constraint):
    """Performing effect without handler raises EffectNotHandled."""
    effect = declare_effect('test.no_handler', int)
    try:
        effect.perform()
        raised = False
    except EffectNotHandled as e:
        raised = True
        has_message = 'not handled' in str(e)
    constraint('EffectNotHandled raised', lambda: (raised, raised, True, {}))
    constraint('Error message mentions not handled', lambda: (has_message, has_message, True, {}))


def test_perform_with_handler_constraint(constraint):
    """Performing effect with handler returns handler value."""
    effect = declare_effect('test.with_handler', int)

    with effect.handler(lambda: 42):
        result = effect.perform()

    constraint('Handler returns correct value', lambda: (result == 42, result, 42, {}))


def test_try_perform_with_default_constraint(constraint):
    """try_perform returns default if not handled."""
    effect = declare_effect('test.try_perform', int)

    # Without handler
    result_no_handler = effect.try_perform(default=99)
    constraint('Default returned without handler', lambda: (result_no_handler == 99, result_no_handler, 99, {}))

    # With handler
    with effect.handler(lambda: 42):
        result_with_handler = effect.try_perform(default=99)
    constraint('Handler value returned with handler', lambda: (result_with_handler == 42, result_with_handler, 42, {}))


def test_is_handled_constraint(constraint):
    """is_handled returns True only when handler installed."""
    effect = declare_effect('test.is_handled', int)

    before_handler = effect.is_handled()
    constraint('Not handled before handler', lambda: (not before_handler, before_handler, False, {}))

    with effect.handler(lambda: 42):
        during_handler = effect.is_handled()
        constraint('Handled during handler context', lambda: (during_handler, during_handler, True, {}))

    after_handler = effect.is_handled()
    constraint('Not handled after handler', lambda: (not after_handler, after_handler, False, {}))


def test_inner_handler_shadows_outer_constraint(constraint):
    """Inner handler shadows outer handler (like monadic bind)."""
    effect = declare_effect('test.shadow', int)

    with effect.handler(lambda: 1):
        outer_result = effect.perform()
        constraint('Outer handler returns 1', lambda: (outer_result == 1, outer_result, 1, {}))

        with effect.handler(lambda: 2):
            inner_result = effect.perform()
            constraint('Inner handler shadows with 2', lambda: (inner_result == 2, inner_result, 2, {}))

        restored_result = effect.perform()
        constraint('Outer handler restored after inner', lambda: (restored_result == 1, restored_result, 1, {}))


def test_handler_cleanup_on_exit_constraint(constraint):
    """Handler removed on context exit."""
    effect = declare_effect('test.cleanup', int)

    with effect.handler(lambda: 42):
        during = effect.is_handled()
        constraint('Handler installed during context', lambda: (during, during, True, {}))

    after = effect.is_handled()
    constraint('Handler removed after context', lambda: (not after, after, False, {}))


def test_handler_cleanup_on_exception_constraint(constraint):
    """Handler removed even if exception raised."""
    effect = declare_effect('test.exception', int)

    try:
        with effect.handler(lambda: 42):
            raise ValueError("Test exception")
    except ValueError:
        pass

    after_exception = effect.is_handled()
    constraint('Handler removed despite exception', lambda: (not after_exception, after_exception, False, {}))


def test_handle_multiple_effects_constraint(constraint):
    """Can install multiple handlers at once."""
    effect1 = declare_effect('test.multi1', int)
    effect2 = declare_effect('test.multi2', str)
    effect3 = declare_effect('test.multi3', float)

    with handle_effects(**{
        'test.multi1': lambda: 42,
        'test.multi2': lambda: "hello",
        'test.multi3': lambda: 3.14
    }):
        r1 = effect1.perform()
        r2 = effect2.perform()
        r3 = effect3.perform()

    constraint('Effect1 returns 42', lambda: (r1 == 42, r1, 42, {}))
    constraint('Effect2 returns hello', lambda: (r2 == "hello", r2, "hello", {}))
    constraint('Effect3 returns 3.14', lambda: (abs(r3 - 3.14) < 0.001, r3, 3.14, {}))


def test_handle_effects_cleanup_constraint(constraint):
    """handle_effects removes all handlers on exit."""
    effect1 = declare_effect('test.cleanup1', int)
    effect2 = declare_effect('test.cleanup2', int)

    with handle_effects(**{
        'test.cleanup1': lambda: 1,
        'test.cleanup2': lambda: 2
    }):
        during1 = effect1.is_handled()
        during2 = effect2.is_handled()
        constraint('Effect1 handled during context', lambda: (during1, during1, True, {}))
        constraint('Effect2 handled during context', lambda: (during2, during2, True, {}))

    after1 = effect1.is_handled()
    after2 = effect2.is_handled()
    constraint('Effect1 not handled after', lambda: (not after1, after1, False, {}))
    constraint('Effect2 not handled after', lambda: (not after2, after2, False, {}))


def test_handlers_thread_local_constraint(constraint):
    """Handlers in one thread don't affect other threads."""
    effect = declare_effect('test.threads', int)

    results = []

    def thread1_work():
        with effect.handler(lambda: 1):
            results.append(effect.perform())

    def thread2_work():
        with effect.handler(lambda: 2):
            results.append(effect.perform())

    t1 = threading.Thread(target=thread1_work)
    t2 = threading.Thread(target=thread2_work)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    sorted_results = sorted(results)
    constraint('Each thread got its own handler', lambda: (sorted_results == [1, 2], sorted_results, [1, 2], {}))


def test_handler_not_visible_to_other_threads_constraint(constraint):
    """Handler in one thread not visible to other thread."""
    effect = declare_effect('test.isolation', int)

    barrier = threading.Barrier(2)
    thread2_saw_handler = [False]

    def thread1_work():
        with effect.handler(lambda: 42):
            barrier.wait()
            barrier.wait()

    def thread2_work():
        barrier.wait()
        thread2_saw_handler[0] = effect.is_handled()
        barrier.wait()

    t1 = threading.Thread(target=thread1_work)
    t2 = threading.Thread(target=thread2_work)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    constraint('Thread2 did not see thread1 handler', lambda: (not thread2_saw_handler[0], thread2_saw_handler[0], False, {}))


def test_effect_composition_constraint(constraint):
    """Effects compose like Kleisli arrows."""
    GetA = declare_effect('test.comp_a', int)
    GetB = declare_effect('test.comp_b', str)

    def use_a():
        a = GetA.perform()
        return a * 2

    def use_b():
        b = GetB.perform()
        return f"Result: {b}"

    def composed():
        intermediate = use_a()
        with GetB.handler(lambda: str(intermediate)):
            return use_b()

    with GetA.handler(lambda: 21):
        result = composed()

    constraint('Composed effects produce correct result', lambda: (result == "Result: 42", result, "Result: 42", {}))


def test_optional_effects_constraint(constraint):
    """Effects can be optionally provided for graceful degradation."""
    OptionalEffect = declare_effect('test.optional', bool)

    def use_feature():
        try:
            enabled = OptionalEffect.perform()
            return f"Feature: {enabled}"
        except EffectNotHandled:
            return "Feature: disabled"

    result_without = use_feature()
    constraint('Without handler returns disabled', lambda: (result_without == "Feature: disabled", result_without, "Feature: disabled", {}))

    with OptionalEffect.handler(lambda: True):
        result_with = use_feature()
    constraint('With handler returns enabled', lambda: (result_with == "Feature: True", result_with, "Feature: True", {}))


def test_list_effects_shows_all_declared_constraint(constraint):
    """list_effects returns all declared effects."""
    declare_effect('test.list1', int)
    declare_effect('test.list2', str)

    effects = list_effects()

    constraint('test.list1 in effects', lambda: ('test.list1' in effects, 'test.list1' in effects, True, {}))
    constraint('test.list2 in effects', lambda: ('test.list2' in effects, 'test.list2' in effects, True, {}))


def test_list_effects_shows_handler_status_constraint(constraint):
    """list_effects shows which effects have handlers."""
    effect1 = declare_effect('test.status1', int)
    effect2 = declare_effect('test.status2', int)

    with effect1.handler(lambda: 42):
        effects = list_effects()
        status1 = effects['test.status1']
        status2 = effects['test.status2']
        constraint('Effect1 shows as handled', lambda: (status1 is True, status1, True, {}))
        constraint('Effect2 shows as not handled', lambda: (status2 is False, status2, False, {}))


def test_feature_flag_pattern_constraint(constraint):
    """Effect handlers as feature flags."""
    UseTopology = declare_effect('feature.topology', bool)

    def run_training():
        if UseTopology.try_perform(False):
            return "Training with topology"
        else:
            return "Training without topology"

    result_disabled = run_training()
    constraint('Feature disabled by default', lambda: (result_disabled == "Training without topology", result_disabled, "Training without topology", {}))

    with UseTopology.handler(lambda: True):
        result_enabled = run_training()
    constraint('Feature enabled with handler', lambda: (result_enabled == "Training with topology", result_enabled, "Training with topology", {}))


def test_dependency_injection_pattern_constraint(constraint):
    """Effect handlers as dependency injection."""
    GetConfig = declare_effect('di.config', dict)

    class Model:
        def __init__(self):
            config = GetConfig.perform()
            self.hidden_dim = config['hidden_dim']

    with GetConfig.handler(lambda: {'hidden_dim': 64}):
        model1 = Model()
    constraint('Model1 has hidden_dim 64', lambda: (model1.hidden_dim == 64, model1.hidden_dim, 64, {}))

    with GetConfig.handler(lambda: {'hidden_dim': 128}):
        model2 = Model()
    constraint('Model2 has hidden_dim 128', lambda: (model2.hidden_dim == 128, model2.hidden_dim, 128, {}))


def test_lazy_initialization_pattern_constraint(constraint):
    """Effect handlers enable lazy initialization."""
    GetExpensiveResource = declare_effect('lazy.resource', str)

    expensive_calls = [0]

    def expensive_computation():
        expensive_calls[0] += 1
        return "computed"

    def use_resource():
        resource = GetExpensiveResource.perform()
        return f"Using {resource}"

    with GetExpensiveResource.handler(expensive_computation):
        result1 = use_resource()
        constraint('First call returns correct result', lambda: (result1 == "Using computed", result1, "Using computed", {}))
        constraint('Computation called once', lambda: (expensive_calls[0] == 1, expensive_calls[0], 1, {}))

        result2 = use_resource()
        constraint('Second call also works', lambda: (result2 == "Using computed", result2, "Using computed", {}))
        constraint('Computation called twice (not cached)', lambda: (expensive_calls[0] == 2, expensive_calls[0], 2, {}))


def test_test_mocking_pattern_constraint(constraint):
    """Effect handlers as test mocks."""
    GetDatabase = declare_effect('db.connection', object)

    def fetch_user(user_id: int):
        db = GetDatabase.perform()
        return db.query(user_id)

    class RealDB:
        def query(self, user_id):
            return f"Real user {user_id}"

    class MockDB:
        def query(self, user_id):
            return f"Mock user {user_id}"

    with GetDatabase.handler(lambda: RealDB()):
        real_result = fetch_user(1)
    constraint('Production handler returns real user', lambda: (real_result == "Real user 1", real_result, "Real user 1", {}))

    with GetDatabase.handler(lambda: MockDB()):
        mock_result = fetch_user(1)
    constraint('Test handler returns mock user', lambda: (mock_result == "Mock user 1", mock_result, "Mock user 1", {}))
