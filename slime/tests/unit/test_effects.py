"""
Unit tests for algebraic effect handler system.

Tests compositional properties:
- Effect declaration and registration
- Handler installation and stacking
- Thread-safety
- Graceful degradation (EffectNotHandled)
- Kleisli composition
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


class TestEffectBasics:
    """Test basic effect declaration and performance."""

    def test_declare_effect(self):
        """Effect can be declared with name and type."""
        effect = declare_effect('test.simple', int)
        assert effect.name == 'test.simple'
        assert effect.return_type == int
        assert not effect.is_handled()

    def test_declare_effect_idempotent(self):
        """Declaring same effect twice returns same instance."""
        effect1 = declare_effect('test.idempotent', str)
        effect2 = declare_effect('test.idempotent', str)
        assert effect1 is effect2

    def test_get_effect(self):
        """Can retrieve declared effect by name."""
        original = declare_effect('test.retrieve', float)
        retrieved = get_effect('test.retrieve')
        assert retrieved is original

    def test_get_effect_not_declared(self):
        """Getting undeclared effect raises KeyError."""
        with pytest.raises(KeyError, match='not declared'):
            get_effect('test.nonexistent')

    def test_perform_without_handler(self):
        """Performing effect without handler raises EffectNotHandled."""
        effect = declare_effect('test.no_handler', int)
        with pytest.raises(EffectNotHandled, match='not handled'):
            effect.perform()

    def test_perform_with_handler(self):
        """Performing effect with handler returns handler value."""
        effect = declare_effect('test.with_handler', int)

        with effect.handler(lambda: 42):
            result = effect.perform()

        assert result == 42

    def test_try_perform_with_default(self):
        """try_perform returns default if not handled."""
        effect = declare_effect('test.try_perform', int)

        # Without handler
        result = effect.try_perform(default=99)
        assert result == 99

        # With handler
        with effect.handler(lambda: 42):
            result = effect.try_perform(default=99)
            assert result == 42

    def test_is_handled(self):
        """is_handled returns True only when handler installed."""
        effect = declare_effect('test.is_handled', int)

        assert not effect.is_handled()

        with effect.handler(lambda: 42):
            assert effect.is_handled()

        assert not effect.is_handled()


class TestHandlerStacking:
    """Test handler composition and shadowing."""

    def test_inner_handler_shadows_outer(self):
        """Inner handler shadows outer handler (like monadic bind)."""
        effect = declare_effect('test.shadow', int)

        with effect.handler(lambda: 1):
            assert effect.perform() == 1

            with effect.handler(lambda: 2):
                assert effect.perform() == 2  # Inner shadows

            assert effect.perform() == 1  # Outer restored

    def test_handler_cleanup_on_exit(self):
        """Handler removed on context exit."""
        effect = declare_effect('test.cleanup', int)

        with effect.handler(lambda: 42):
            assert effect.is_handled()

        assert not effect.is_handled()

    def test_handler_cleanup_on_exception(self):
        """Handler removed even if exception raised."""
        effect = declare_effect('test.exception', int)

        try:
            with effect.handler(lambda: 42):
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert not effect.is_handled()


class TestMultipleEffects:
    """Test handle_effects convenience function."""

    def test_handle_multiple_effects(self):
        """Can install multiple handlers at once."""
        effect1 = declare_effect('test.multi1', int)
        effect2 = declare_effect('test.multi2', str)
        effect3 = declare_effect('test.multi3', float)

        with handle_effects(**{
            'test.multi1': lambda: 42,
            'test.multi2': lambda: "hello",
            'test.multi3': lambda: 3.14
        }):
            assert effect1.perform() == 42
            assert effect2.perform() == "hello"
            assert effect3.perform() == 3.14

    def test_handle_effects_cleanup(self):
        """handle_effects removes all handlers on exit."""
        effect1 = declare_effect('test.cleanup1', int)
        effect2 = declare_effect('test.cleanup2', int)

        with handle_effects(**{
            'test.cleanup1': lambda: 1,
            'test.cleanup2': lambda: 2
        }):
            assert effect1.is_handled()
            assert effect2.is_handled()

        assert not effect1.is_handled()
        assert not effect2.is_handled()


class TestThreadSafety:
    """Test thread-local handler isolation."""

    def test_handlers_thread_local(self):
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

        # Each thread got its own handler value
        assert sorted(results) == [1, 2]

    def test_handler_not_visible_to_other_threads(self):
        """Handler in one thread not visible to other thread."""
        effect = declare_effect('test.isolation', int)

        barrier = threading.Barrier(2)
        thread2_saw_handler = [False]

        def thread1_work():
            with effect.handler(lambda: 42):
                barrier.wait()  # Sync with thread2
                barrier.wait()  # Wait for thread2 to check

        def thread2_work():
            barrier.wait()  # Sync with thread1 (handler installed)
            thread2_saw_handler[0] = effect.is_handled()
            barrier.wait()  # Signal done

        t1 = threading.Thread(target=thread1_work)
        t2 = threading.Thread(target=thread2_work)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not thread2_saw_handler[0]


class TestComposition:
    """Test Kleisli composition properties."""

    def test_effect_composition(self):
        """Effects compose like Kleisli arrows."""
        # Effect A -> B
        GetA = declare_effect('test.comp_a', int)
        # Effect B -> C
        GetB = declare_effect('test.comp_b', str)

        def use_a():
            """A -> B"""
            a = GetA.perform()
            return a * 2

        def use_b():
            """B -> C"""
            b = GetB.perform()
            return f"Result: {b}"

        def composed():
            """A -> C (composition)"""
            intermediate = use_a()
            # Inject intermediate as effect
            with GetB.handler(lambda: str(intermediate)):
                return use_b()

        with GetA.handler(lambda: 21):
            result = composed()

        assert result == "Result: 42"

    def test_optional_effects(self):
        """Effects can be optionally provided for graceful degradation."""
        OptionalEffect = declare_effect('test.optional', bool)

        def use_feature():
            try:
                enabled = OptionalEffect.perform()
                return f"Feature: {enabled}"
            except EffectNotHandled:
                return "Feature: disabled"

        # Without handler
        assert use_feature() == "Feature: disabled"

        # With handler
        with OptionalEffect.handler(lambda: True):
            assert use_feature() == "Feature: True"


class TestListEffects:
    """Test effect introspection."""

    def test_list_effects_shows_all_declared(self):
        """list_effects returns all declared effects."""
        declare_effect('test.list1', int)
        declare_effect('test.list2', str)

        effects = list_effects()

        assert 'test.list1' in effects
        assert 'test.list2' in effects

    def test_list_effects_shows_handler_status(self):
        """list_effects shows which effects have handlers."""
        effect1 = declare_effect('test.status1', int)
        effect2 = declare_effect('test.status2', int)

        with effect1.handler(lambda: 42):
            effects = list_effects()
            assert effects['test.status1'] is True
            assert effects['test.status2'] is False


class TestRealWorldScenarios:
    """Test realistic usage patterns."""

    def test_feature_flag_pattern(self):
        """Effect handlers as feature flags."""
        UseTopology = declare_effect('feature.topology', bool)

        def run_training():
            if UseTopology.try_perform(False):
                return "Training with topology"
            else:
                return "Training without topology"

        # Feature disabled
        assert run_training() == "Training without topology"

        # Feature enabled
        with UseTopology.handler(lambda: True):
            assert run_training() == "Training with topology"

    def test_dependency_injection_pattern(self):
        """Effect handlers as dependency injection."""
        GetConfig = declare_effect('di.config', dict)

        class Model:
            def __init__(self):
                config = GetConfig.perform()
                self.hidden_dim = config['hidden_dim']

        # Different configurations via different handlers
        with GetConfig.handler(lambda: {'hidden_dim': 64}):
            model1 = Model()
            assert model1.hidden_dim == 64

        with GetConfig.handler(lambda: {'hidden_dim': 128}):
            model2 = Model()
            assert model2.hidden_dim == 128

    def test_lazy_initialization_pattern(self):
        """Effect handlers enable lazy initialization."""
        GetExpensiveResource = declare_effect('lazy.resource', str)

        expensive_calls = [0]

        def expensive_computation():
            expensive_calls[0] += 1
            return "computed"

        def use_resource():
            # Resource only computed if actually needed
            resource = GetExpensiveResource.perform()
            return f"Using {resource}"

        # Handler provides lazy computation
        with GetExpensiveResource.handler(expensive_computation):
            result = use_resource()
            assert result == "Using computed"
            assert expensive_calls[0] == 1

            # Second call triggers computation again (not cached in this example)
            result = use_resource()
            assert expensive_calls[0] == 2

    def test_test_mocking_pattern(self):
        """Effect handlers as test mocks."""
        GetDatabase = declare_effect('db.connection', object)

        def fetch_user(user_id: int):
            db = GetDatabase.perform()
            return db.query(user_id)

        # Production handler (real DB)
        class RealDB:
            def query(self, user_id):
                return f"Real user {user_id}"

        # Test handler (mock DB)
        class MockDB:
            def query(self, user_id):
                return f"Mock user {user_id}"

        # Production
        with GetDatabase.handler(lambda: RealDB()):
            assert fetch_user(1) == "Real user 1"

        # Test
        with GetDatabase.handler(lambda: MockDB()):
            assert fetch_user(1) == "Mock user 1"
