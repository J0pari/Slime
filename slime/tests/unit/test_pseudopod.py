"""Unit tests for Pseudopod 5D behavioral metrics."""
import pytest
import torch
from slime.core.pseudopod import Pseudopod


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def pseudopod(device):
    return Pseudopod(
        component_id='test_component_0',
        sensory_dim=64,
        latent_dim=32,
        head_dim=16,
        device=device
    )


def test_pseudopod_initialization(pseudopod, device):
    """Verify pseudopod initializes correctly."""
    assert pseudopod.component_id == 'test_component_0'
    assert pseudopod.device == device
    assert pseudopod.last_behavior is None


def test_forward_pass_shape(pseudopod, device):
    """Verify forward pass produces correct output shape."""
    batch_size, seq_len, sensory_dim = 2, 32, 64
    x = torch.randn(batch_size, seq_len, sensory_dim, device=device)

    output = pseudopod(x)

    assert output.shape == (batch_size, seq_len, 32)


def test_attention_distance_metric(pseudopod, device):
    """Verify attention distance metric."""
    batch_size, seq_len = 2, 16
    attn = torch.softmax(torch.randn(batch_size, 1, seq_len, seq_len, device=device), dim=-1)

    distance = pseudopod.get_attention_distance(attn)

    assert isinstance(distance, float)
    assert 0.0 <= distance <= seq_len


def test_activation_sparsity_metric(pseudopod, device):
    """Verify activation sparsity metric."""
    output = torch.randn(2, 16, 32, device=device)
    output[output.abs() < 0.5] = 0.0

    sparsity = pseudopod.get_activation_sparsity(output)

    assert isinstance(sparsity, float)
    assert 0.0 <= sparsity <= 1.0


def test_gradient_flow_magnitude_no_grad(pseudopod):
    """Verify gradient flow returns 0 when no gradients."""
    magnitude = pseudopod.get_gradient_flow_magnitude()
    assert magnitude == 0.0


def test_gradient_flow_magnitude_with_grad(pseudopod, device):
    """Verify gradient flow computes magnitude with gradients."""
    x = torch.randn(2, 16, 64, device=device, requires_grad=True)
    output = pseudopod(x)
    loss = output.sum()
    loss.backward()

    magnitude = pseudopod.get_gradient_flow_magnitude()

    assert magnitude > 0.0


def test_memory_access_locality(pseudopod, device):
    """Verify memory access locality metric."""
    seq_len = 32
    attn = torch.zeros(2, 1, seq_len, seq_len, device=device)
    for i in range(seq_len):
        start = max(0, i - 2)
        end = min(seq_len, i + 3)
        attn[:, :, i, start:end] = 1.0 / (end - start)

    locality = pseudopod.get_memory_access_locality(attn)

    assert isinstance(locality, float)
    assert 0.0 <= locality <= 1.0


def test_computational_intensity(pseudopod, device):
    """Verify computational intensity metric."""
    output = torch.randn(2, 32, 32, device=device)
    seq_len = 32

    intensity = pseudopod.get_computational_intensity(output, seq_len)

    assert isinstance(intensity, float)
    assert intensity >= 0.0


def test_behavioral_coordinates_5d(pseudopod, device):
    """Verify behavioral coordinates returns 5D tensor."""
    x = torch.randn(2, 16, 64, device=device, requires_grad=True)
    output = pseudopod(x)
    loss = output.sum()
    loss.backward()

    behavior = pseudopod.last_behavior

    assert behavior is not None
    assert behavior.shape == (5,)
    assert not torch.isnan(behavior).any()
    assert not torch.isinf(behavior).any()


def test_behavioral_coordinates_ranges(pseudopod, device):
    """Verify behavioral coordinates are in valid ranges."""
    x = torch.randn(2, 16, 64, device=device)
    output = pseudopod(x)

    behavior = pseudopod.last_behavior

    assert behavior[0] >= 0.0
    assert 0.0 <= behavior[1] <= 1.0
    assert behavior[2] >= 0.0
    assert 0.0 <= behavior[3] <= 1.0
    assert behavior[4] >= 0.0


def test_behavioral_consistency_same_input(pseudopod, device):
    """Verify behavioral coordinates are consistent for same input."""
    x = torch.randn(2, 16, 64, device=device)

    pseudopod(x)
    behavior1 = pseudopod.last_behavior.clone()

    pseudopod(x)
    behavior2 = pseudopod.last_behavior.clone()

    assert torch.allclose(behavior1, behavior2, atol=1e-3)


def test_behavioral_diversity_different_inputs(pseudopod, device):
    """Verify different inputs produce different behaviors."""
    x1 = torch.randn(2, 16, 64, device=device)
    x2 = torch.randn(2, 16, 64, device=device) * 10.0

    pseudopod(x1)
    behavior1 = pseudopod.last_behavior.clone()

    pseudopod(x2)
    behavior2 = pseudopod.last_behavior.clone()

    distance = torch.norm(behavior1 - behavior2)
    assert distance > 0.01


def test_attention_pattern_storage(pseudopod, device):
    """Verify attention pattern is stored for coherence computation."""
    x = torch.randn(2, 16, 64, device=device)
    pseudopod(x)

    assert pseudopod._last_attention_pattern is not None
    assert pseudopod._last_attention_pattern.ndim == 4


def test_different_pseudopod_ids(device):
    """Verify different component IDs produce different pseudopods."""
    pod1 = Pseudopod('component_0', 64, 32, 16, device)
    pod2 = Pseudopod('component_1', 64, 32, 16, device)

    assert pod1.component_id != pod2.component_id


def test_behavioral_metrics_deterministic(pseudopod, device):
    """Verify behavioral metrics are deterministic given same state."""
    torch.manual_seed(42)
    x = torch.randn(2, 16, 64, device=device)

    pseudopod(x)
    behavior1 = pseudopod.last_behavior.clone()

    torch.manual_seed(42)
    x = torch.randn(2, 16, 64, device=device)
    pseudopod(x)
    behavior2 = pseudopod.last_behavior.clone()

    assert torch.allclose(behavior1, behavior2, atol=1e-5)


def test_attention_span_local_vs_global(device):
    """Verify attention span distinguishes local vs global attention."""
    pod = Pseudopod('test', 64, 32, 16, device)

    seq_len = 32
    local_attn = torch.zeros(2, 1, seq_len, seq_len, device=device)
    for i in range(seq_len):
        start = max(0, i - 2)
        end = min(seq_len, i + 3)
        local_attn[:, :, i, start:end] = 1.0 / (end - start)

    global_attn = torch.ones(2, 1, seq_len, seq_len, device=device) / seq_len

    local_distance = pod.get_attention_distance(local_attn)
    global_distance = pod.get_attention_distance(global_attn)

    assert local_distance < global_distance


def test_sparsity_dense_vs_sparse_activations(pseudopod, device):
    """Verify sparsity metric distinguishes dense vs sparse activations."""
    dense_output = torch.randn(2, 16, 32, device=device)
    sparse_output = torch.randn(2, 16, 32, device=device)
    sparse_output[sparse_output.abs() < 1.0] = 0.0

    dense_sparsity = pseudopod.get_activation_sparsity(dense_output)
    sparse_sparsity = pseudopod.get_activation_sparsity(sparse_output)

    assert sparse_sparsity > dense_sparsity
