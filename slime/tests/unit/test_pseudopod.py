import pytest
import torch
from slime.core.pseudopod import Pseudopod
from slime.kernels.torch_fallback import TorchKernel
from slime.config.dimensions import TINY

@pytest.fixture
def device():
    return torch.device('cuda')

@pytest.fixture
def kernel(config, device):
    return TorchKernel(numerical_config=config.numerical, device=device)

@pytest.fixture
def config():
    return TINY

@pytest.fixture
def pseudopod(config, kernel, device):
    return Pseudopod(head_dim=config.dimensions.head_dim, kernel=kernel, fitness_config=config.fitness, numerical_config=config.numerical, device=device, component_id=0, num_heads=config.dimensions.num_heads)

def test_pseudopod_initialization(pseudopod, device):
    assert pseudopod.component_id == 0
    assert pseudopod.device == device
    assert pseudopod.last_behavior is None

def test_forward_pass_shape(pseudopod, config, device):
    batch_size = config.test.batch_size
    seq_len = config.test.seq_len
    head_dim = config.dimensions.head_dim
    num_heads = config.dimensions.num_heads
    latent = torch.randn(batch_size, seq_len, head_dim, device=device)
    stimulus = torch.randn(batch_size, seq_len, head_dim, device=device)
    output = pseudopod(latent, stimulus)
    assert output.shape == (batch_size, num_heads, seq_len, head_dim)

def test_attention_distance_metric(pseudopod, config, device):
    batch_size = config.test.batch_size
    seq_len = config.test.seq_len
    attn = torch.softmax(torch.randn(batch_size, 1, seq_len, seq_len, device=device), dim=-1)
    distance = pseudopod.get_attention_distance(attn)
    assert isinstance(distance, float)
    assert 0.0 <= distance <= seq_len

def test_activation_sparsity_metric(pseudopod, config, device):
    batch_size = config.test.batch_size
    seq_len = config.test.seq_len
    head_dim = config.dimensions.head_dim
    output = torch.randn(batch_size, seq_len, head_dim, device=device)
    output[output.abs() < 0.5] = 0.0
    sparsity = pseudopod.get_activation_sparsity(output)
    assert isinstance(sparsity, float)
    assert 0.0 <= sparsity <= 1.0

def test_gradient_flow_magnitude_no_grad(pseudopod):
    magnitude = pseudopod.get_gradient_flow_magnitude()
    assert magnitude == 0.0

def test_gradient_flow_magnitude_with_grad(pseudopod, config, device):
    batch_size = config.test.batch_size
    seq_len = config.test.seq_len
    head_dim = config.dimensions.head_dim
    latent = torch.randn(batch_size, seq_len, head_dim, device=device, requires_grad=True)
    stimulus = torch.randn(batch_size, seq_len, head_dim, device=device, requires_grad=True)
    output = pseudopod(latent, stimulus)
    loss = output.sum()
    loss.backward()
    magnitude = pseudopod.get_gradient_flow_magnitude()
    assert magnitude > 0.0

def test_memory_access_locality(pseudopod, config, device):
    batch_size = config.test.batch_size
    seq_len = config.test.seq_len
    attn = torch.zeros(batch_size, 1, seq_len, seq_len, device=device)
    for i in range(seq_len):
        start = max(0, i - 2)
        end = min(seq_len, i + 3)
        attn[:, :, i, start:end] = 1.0 / (end - start)
    locality = pseudopod.get_memory_access_locality(attn)
    assert isinstance(locality, float)
    assert 0.0 <= locality <= 1.0

def test_computational_intensity(pseudopod, config, device):
    batch_size = config.test.batch_size
    seq_len = config.test.seq_len
    head_dim = config.dimensions.head_dim
    output = torch.randn(batch_size, seq_len, head_dim, device=device)
    intensity = pseudopod.get_computational_intensity(output, seq_len)
    assert isinstance(intensity, float)
    assert intensity >= 0.0

def test_behavioral_coordinates_5d(pseudopod, config, device):
    batch_size = config.test.batch_size
    seq_len = config.test.seq_len
    head_dim = config.dimensions.head_dim
    latent = torch.randn(batch_size, seq_len, head_dim, device=device, requires_grad=True)
    stimulus = torch.randn(batch_size, seq_len, head_dim, device=device, requires_grad=True)
    output = pseudopod(latent, stimulus)
    loss = output.sum()
    loss.backward()
    behavior = pseudopod.last_behavior
    assert behavior is not None
    assert behavior.shape == (5,)
    assert not torch.isnan(behavior).any()
    assert not torch.isinf(behavior).any()

def test_behavioral_coordinates_ranges(pseudopod, config, device):
    batch_size = config.test.batch_size
    seq_len = config.test.seq_len
    head_dim = config.dimensions.head_dim
    latent = torch.randn(batch_size, seq_len, head_dim, device=device)
    stimulus = torch.randn(batch_size, seq_len, head_dim, device=device)
    output = pseudopod(latent, stimulus)
    behavior = pseudopod.last_behavior
    assert behavior[0] >= 0.0
    assert 0.0 <= behavior[1] <= 1.0
    assert behavior[2] >= 0.0
    assert 0.0 <= behavior[3] <= 1.0
    assert behavior[4] >= 0.0

def test_behavioral_consistency_same_input(pseudopod, config, device):
    batch_size = config.test.batch_size
    seq_len = config.test.seq_len
    head_dim = config.dimensions.head_dim
    latent = torch.randn(batch_size, seq_len, head_dim, device=device)
    stimulus = torch.randn(batch_size, seq_len, head_dim, device=device)
    pseudopod(latent, stimulus)
    behavior1 = pseudopod.last_behavior.clone()
    pseudopod(latent, stimulus)
    behavior2 = pseudopod.last_behavior.clone()
    assert torch.allclose(behavior1, behavior2, atol=0.001)

def test_behavioral_diversity_different_inputs(pseudopod, config, device):
    batch_size = config.test.batch_size
    seq_len = config.test.seq_len
    head_dim = config.dimensions.head_dim
    latent1 = torch.randn(batch_size, seq_len, head_dim, device=device)
    stimulus1 = torch.randn(batch_size, seq_len, head_dim, device=device)
    latent2 = torch.randn(batch_size, seq_len, head_dim, device=device) * 10.0
    stimulus2 = torch.randn(batch_size, seq_len, head_dim, device=device) * 10.0
    pseudopod(latent1, stimulus1)
    behavior1 = pseudopod.last_behavior.clone()
    pseudopod(latent2, stimulus2)
    behavior2 = pseudopod.last_behavior.clone()
    distance = torch.norm(behavior1 - behavior2)
    assert distance > 0.01

def test_attention_pattern_storage(pseudopod, config, device):
    batch_size = config.test.batch_size
    seq_len = config.test.seq_len
    head_dim = config.dimensions.head_dim
    num_heads = config.dimensions.num_heads
    latent = torch.randn(batch_size, seq_len, head_dim, device=device)
    stimulus = torch.randn(batch_size, seq_len, head_dim, device=device)
    pseudopod(latent, stimulus)
    assert pseudopod._last_attention_pattern is not None
    assert pseudopod._last_attention_pattern.ndim == 4
    assert pseudopod._last_attention_pattern.shape == (batch_size, num_heads, seq_len, seq_len)

def test_different_pseudopod_ids(config, kernel, device):
    head_dim = config.dimensions.head_dim
    num_heads = config.dimensions.num_heads
    fitness_config = config.fitness
    numerical_config = config.numerical
    pod1 = Pseudopod(head_dim=head_dim, kernel=kernel, fitness_config=fitness_config, numerical_config=numerical_config, device=device, component_id=0, num_heads=num_heads)
    pod2 = Pseudopod(head_dim=head_dim, kernel=kernel, fitness_config=fitness_config, numerical_config=numerical_config, device=device, component_id=1, num_heads=num_heads)
    assert pod1.component_id != pod2.component_id

def test_behavioral_metrics_deterministic(pseudopod, config, device):
    batch_size = config.test.batch_size
    seq_len = config.test.seq_len
    head_dim = config.dimensions.head_dim
    torch.manual_seed(42)
    latent = torch.randn(batch_size, seq_len, head_dim, device=device)
    stimulus = torch.randn(batch_size, seq_len, head_dim, device=device)
    pseudopod(latent, stimulus)
    behavior1 = pseudopod.last_behavior.clone()
    torch.manual_seed(42)
    latent = torch.randn(batch_size, seq_len, head_dim, device=device)
    stimulus = torch.randn(batch_size, seq_len, head_dim, device=device)
    pseudopod(latent, stimulus)
    behavior2 = pseudopod.last_behavior.clone()
    assert torch.allclose(behavior1, behavior2, atol=1e-05)

def test_attention_span_local_vs_global(config, kernel, device):
    head_dim = config.dimensions.head_dim
    num_heads = config.dimensions.num_heads
    fitness_config = config.fitness
    numerical_config = config.numerical
    batch_size = config.test.batch_size
    seq_len = config.test.seq_len
    pod = Pseudopod(head_dim=head_dim, kernel=kernel, fitness_config=fitness_config, numerical_config=numerical_config, device=device, component_id=0, num_heads=num_heads)
    local_attn = torch.zeros(batch_size, num_heads, seq_len, seq_len, device=device)
    for i in range(seq_len):
        start = max(0, i - 2)
        end = min(seq_len, i + 3)
        local_attn[:, :, i, start:end] = 1.0 / (end - start)
    global_attn = torch.ones(batch_size, num_heads, seq_len, seq_len, device=device) / seq_len
    local_distance = pod.get_attention_distance(local_attn)
    global_distance = pod.get_attention_distance(global_attn)
    assert local_distance < global_distance

def test_sparsity_dense_vs_sparse_activations(pseudopod, config, device):
    batch_size = config.test.batch_size
    seq_len = config.test.seq_len
    head_dim = config.dimensions.head_dim
    num_heads = config.dimensions.num_heads
    dense_output = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    sparse_output = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    sparse_output[sparse_output.abs() < 1.0] = 0.0
    dense_sparsity = pseudopod.get_activation_sparsity(dense_output)
    sparse_sparsity = pseudopod.get_activation_sparsity(sparse_output)
    assert sparse_sparsity > dense_sparsity
