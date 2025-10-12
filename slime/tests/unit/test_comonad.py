"""Unit tests for comonadic spatial context extraction."""
import pytest
import torch
from slime.core.comonad import (
    SpatialContext,
    behavioral_distance,
    extract_relative_fitness,
    extract_behavioral_divergence,
    extract_gradient_magnitude_rank,
    extract_attention_coherence
)


class MockComponent:
    """Mock component for testing comonadic extractors."""
    def __init__(self, component_id, fitness=0.5, behavior=None, gradient_norm=1.0, attention_pattern=None):
        self.component_id = component_id
        self.fitness = fitness
        self.last_behavior = behavior if behavior is not None else torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
        self.gradient_norm = gradient_norm
        self._last_attention_pattern = attention_pattern if attention_pattern is not None else torch.randn(1, 1, 16, 16)


def test_spatial_context_initialization():
    """Verify SpatialContext initializes correctly."""
    focus = MockComponent('focus', fitness=0.8)
    neighbors = [MockComponent(f'neighbor_{i}', fitness=0.5 + i * 0.1) for i in range(5)]

    ctx = SpatialContext(
        focus=focus,
        neighborhood=neighbors,
        distance_fn=behavioral_distance
    )

    assert ctx.focus.component_id == 'focus'
    assert len(ctx.neighborhood) == 5


def test_spatial_context_extract():
    """Verify extract returns focal component."""
    focus = MockComponent('focus', fitness=0.8)
    neighbors = [MockComponent(f'neighbor_{i}') for i in range(3)]

    ctx = SpatialContext(
        focus=focus,
        neighborhood=neighbors,
        distance_fn=behavioral_distance
    )

    extracted = ctx.extract()
    assert extracted.component_id == 'focus'
    assert extracted.fitness == 0.8


def test_get_k_nearest():
    """Verify get_k_nearest returns correct neighbors."""
    focus = MockComponent('focus', behavior=torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5]))
    neighbors = [
        MockComponent('near_1', behavior=torch.tensor([0.51, 0.51, 0.51, 0.51, 0.51])),
        MockComponent('near_2', behavior=torch.tensor([0.52, 0.52, 0.52, 0.52, 0.52])),
        MockComponent('far', behavior=torch.tensor([0.9, 0.9, 0.9, 0.9, 0.9])),
    ]

    ctx = SpatialContext(
        focus=focus,
        neighborhood=neighbors,
        distance_fn=behavioral_distance
    )

    k_nearest = ctx.get_k_nearest(k=2)

    assert len(k_nearest) == 2
    assert k_nearest[0].component_id in ['near_1', 'near_2']
    assert k_nearest[1].component_id in ['near_1', 'near_2']


def test_get_k_nearest_empty_neighborhood():
    """Verify get_k_nearest returns empty list for empty neighborhood."""
    focus = MockComponent('focus')
    ctx = SpatialContext(
        focus=focus,
        neighborhood=[],
        distance_fn=behavioral_distance
    )

    k_nearest = ctx.get_k_nearest(k=5)
    assert k_nearest == []


def test_behavioral_distance():
    """Verify behavioral distance metric."""
    comp1 = MockComponent('comp1', behavior=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]))
    comp2 = MockComponent('comp2', behavior=torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0]))

    distance = behavioral_distance(comp1, comp2)

    expected = (5 ** 0.5)
    assert abs(distance - expected) < 1e-5


def test_behavioral_distance_same_component():
    """Verify behavioral distance is zero for same behavior."""
    behavior = torch.tensor([0.3, 0.4, 0.5, 0.6, 0.7])
    comp1 = MockComponent('comp1', behavior=behavior)
    comp2 = MockComponent('comp2', behavior=behavior.clone())

    distance = behavioral_distance(comp1, comp2)

    assert distance < 1e-6


def test_extract_relative_fitness_above_mean():
    """Verify relative fitness extraction for above-average component."""
    focus = MockComponent('focus', fitness=0.9)
    neighbors = [MockComponent(f'neighbor_{i}', fitness=0.5) for i in range(5)]

    ctx = SpatialContext(
        focus=focus,
        neighborhood=neighbors,
        distance_fn=behavioral_distance
    )

    relative_fitness = extract_relative_fitness(ctx)

    assert relative_fitness > 0.0


def test_extract_relative_fitness_below_mean():
    """Verify relative fitness extraction for below-average component."""
    focus = MockComponent('focus', fitness=0.3)
    neighbors = [MockComponent(f'neighbor_{i}', fitness=0.7) for i in range(5)]

    ctx = SpatialContext(
        focus=focus,
        neighborhood=neighbors,
        distance_fn=behavioral_distance
    )

    relative_fitness = extract_relative_fitness(ctx)

    assert relative_fitness < 0.0


def test_extract_relative_fitness_no_neighbors():
    """Verify relative fitness returns 0 for no neighbors."""
    focus = MockComponent('focus', fitness=0.8)
    ctx = SpatialContext(
        focus=focus,
        neighborhood=[],
        distance_fn=behavioral_distance
    )

    relative_fitness = extract_relative_fitness(ctx)

    assert relative_fitness == 0.0


def test_extract_behavioral_divergence():
    """Verify behavioral divergence extraction."""
    focus = MockComponent('focus', behavior=torch.tensor([0.8, 0.8, 0.8, 0.8, 0.8]))
    neighbors = [
        MockComponent(f'neighbor_{i}', behavior=torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5]))
        for i in range(5)
    ]

    ctx = SpatialContext(
        focus=focus,
        neighborhood=neighbors,
        distance_fn=behavioral_distance
    )

    divergence = extract_behavioral_divergence(ctx)

    assert divergence.shape == (5,)
    assert (divergence > 0.0).all()


def test_extract_behavioral_divergence_similar_neighbors():
    """Verify behavioral divergence is near zero for similar neighbors."""
    behavior = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
    focus = MockComponent('focus', behavior=behavior)
    neighbors = [
        MockComponent(f'neighbor_{i}', behavior=behavior + torch.randn(5) * 0.01)
        for i in range(5)
    ]

    ctx = SpatialContext(
        focus=focus,
        neighborhood=neighbors,
        distance_fn=behavioral_distance
    )

    divergence = extract_behavioral_divergence(ctx)

    assert torch.norm(divergence) < 0.1


def test_extract_behavioral_divergence_no_neighbors():
    """Verify behavioral divergence returns zeros for no neighbors."""
    focus = MockComponent('focus', behavior=torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5]))
    ctx = SpatialContext(
        focus=focus,
        neighborhood=[],
        distance_fn=behavioral_distance
    )

    divergence = extract_behavioral_divergence(ctx)

    assert (divergence == 0.0).all()


def test_extract_gradient_magnitude_rank_high():
    """Verify gradient rank extraction for high gradient component."""
    focus = MockComponent('focus', gradient_norm=10.0)
    neighbors = [MockComponent(f'neighbor_{i}', gradient_norm=1.0) for i in range(5)]

    ctx = SpatialContext(
        focus=focus,
        neighborhood=neighbors,
        distance_fn=behavioral_distance
    )

    rank = extract_gradient_magnitude_rank(ctx)

    assert 0.8 < rank <= 1.0


def test_extract_gradient_magnitude_rank_low():
    """Verify gradient rank extraction for low gradient component."""
    focus = MockComponent('focus', gradient_norm=0.5)
    neighbors = [MockComponent(f'neighbor_{i}', gradient_norm=5.0) for i in range(5)]

    ctx = SpatialContext(
        focus=focus,
        neighborhood=neighbors,
        distance_fn=behavioral_distance
    )

    rank = extract_gradient_magnitude_rank(ctx)

    assert 0.0 <= rank < 0.2


def test_extract_gradient_magnitude_rank_no_neighbors():
    """Verify gradient rank returns 0.5 for no neighbors."""
    focus = MockComponent('focus', gradient_norm=5.0)
    ctx = SpatialContext(
        focus=focus,
        neighborhood=[],
        distance_fn=behavioral_distance
    )

    rank = extract_gradient_magnitude_rank(ctx)

    assert rank == 0.5


def test_extract_attention_coherence_high():
    """Verify attention coherence for similar patterns."""
    pattern = torch.randn(1, 1, 16, 16)
    focus = MockComponent('focus', attention_pattern=pattern)
    neighbors = [
        MockComponent(f'neighbor_{i}', attention_pattern=pattern + torch.randn(1, 1, 16, 16) * 0.1)
        for i in range(5)
    ]

    ctx = SpatialContext(
        focus=focus,
        neighborhood=neighbors,
        distance_fn=behavioral_distance
    )

    coherence = extract_attention_coherence(ctx)

    assert coherence > 0.7


def test_extract_attention_coherence_low():
    """Verify attention coherence for dissimilar patterns."""
    focus = MockComponent('focus', attention_pattern=torch.randn(1, 1, 16, 16))
    neighbors = [
        MockComponent(f'neighbor_{i}', attention_pattern=torch.randn(1, 1, 16, 16))
        for i in range(5)
    ]

    ctx = SpatialContext(
        focus=focus,
        neighborhood=neighbors,
        distance_fn=behavioral_distance
    )

    coherence = extract_attention_coherence(ctx)

    assert -1.0 <= coherence <= 1.0


def test_extract_attention_coherence_no_neighbors():
    """Verify attention coherence returns 0 for no neighbors."""
    focus = MockComponent('focus', attention_pattern=torch.randn(1, 1, 16, 16))
    ctx = SpatialContext(
        focus=focus,
        neighborhood=[],
        distance_fn=behavioral_distance
    )

    coherence = extract_attention_coherence(ctx)

    assert coherence == 0.0


def test_spatial_context_extend():
    """Verify extend applies context-aware function."""
    focus = MockComponent('focus', fitness=0.8)
    neighbors = [MockComponent(f'neighbor_{i}', fitness=0.5 + i * 0.1) for i in range(3)]

    ctx = SpatialContext(
        focus=focus,
        neighborhood=neighbors,
        distance_fn=behavioral_distance
    )

    def extract_fitness_squared(c: SpatialContext) -> float:
        return c.focus.fitness ** 2

    new_ctx = ctx.extend(extract_fitness_squared)

    assert new_ctx.focus == 0.64
    assert len(new_ctx.neighborhood) == 3


def test_comonad_laws_left_identity():
    """Verify comonad left identity law: extract . duplicate = id."""
    focus = MockComponent('focus', fitness=0.8)
    neighbors = [MockComponent(f'neighbor_{i}') for i in range(3)]

    ctx = SpatialContext(
        focus=focus,
        neighborhood=neighbors,
        distance_fn=behavioral_distance
    )

    duplicated = ctx.duplicate()
    extracted = duplicated.extract()

    assert extracted.focus.component_id == ctx.focus.component_id


def test_comonad_laws_right_identity():
    """Verify comonad right identity law: extend extract = id."""
    focus = MockComponent('focus', fitness=0.8)
    neighbors = [MockComponent(f'neighbor_{i}') for i in range(3)]

    ctx = SpatialContext(
        focus=focus,
        neighborhood=neighbors,
        distance_fn=behavioral_distance
    )

    extended = ctx.extend(lambda c: c.extract())

    assert extended.focus.component_id == ctx.focus.component_id


def test_multiple_extractors_composition():
    """Verify multiple extractors can be composed."""
    focus = MockComponent('focus', fitness=0.9, gradient_norm=8.0)
    neighbors = [
        MockComponent(f'neighbor_{i}', fitness=0.5, gradient_norm=2.0)
        for i in range(5)
    ]

    ctx = SpatialContext(
        focus=focus,
        neighborhood=neighbors,
        distance_fn=behavioral_distance
    )

    relative_fitness = extract_relative_fitness(ctx)
    gradient_rank = extract_gradient_magnitude_rank(ctx)

    assert relative_fitness > 0.0
    assert gradient_rank > 0.8
