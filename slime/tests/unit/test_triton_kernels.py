import pytest
import torch
from slime.kernels.triton_impl import TritonKernel
from slime.config.dimensions import TINY

@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    return torch.device('cuda')

@pytest.fixture
def kernel(device):
    return TritonKernel(device, TINY.numerical)

def test_attention_power_of_two_shapes_constraint(constraint, kernel, device):
    for size in [64, 128, 256, 512]:
        Q = torch.randn(2, 4, size, 64, device=device, dtype=torch.float16)
        K = torch.randn(2, 4, size, 64, device=device, dtype=torch.float16)
        V = torch.randn(2, 4, size, 64, device=device, dtype=torch.float16)
        output = kernel.attention(Q, K, V, temperature=1.0)
        constraint(f'Output shape matches input for size {size}', lambda o=output, q=Q: (o.shape == q.shape, o.shape, q.shape, {}))
        constraint(f'No NaN values for size {size}', lambda o=output: (not torch.isnan(o).any(), 'no_nan', 'no_nan', {}))
        constraint(f'No Inf values for size {size}', lambda o=output: (not torch.isinf(o).any(), 'no_inf', 'no_inf', {}))

def test_attention_non_power_of_two_shapes_constraint(constraint, kernel, device):
    for size in [100, 200, 300]:
        Q = torch.randn(2, 4, size, 64, device=device, dtype=torch.float16)
        K = torch.randn(2, 4, size, 64, device=device, dtype=torch.float16)
        V = torch.randn(2, 4, size, 64, device=device, dtype=torch.float16)
        output = kernel.attention(Q, K, V, temperature=1.0)
        constraint(f'Output shape matches for non-power-of-2 size {size}', lambda o=output, q=Q: (o.shape == q.shape, o.shape, q.shape, {}))

def test_attention_temperature_extremes_constraint(constraint, kernel, device):
    Q = torch.randn(1, 1, 64, 32, device=device, dtype=torch.float16)
    K = torch.randn(1, 1, 64, 32, device=device, dtype=torch.float16)
    V = torch.randn(1, 1, 64, 32, device=device, dtype=torch.float16)
    cold = kernel.attention(Q, K, V, temperature=0.01)
    hot = kernel.attention(Q, K, V, temperature=100.0)
    cold_entropy = -(cold * torch.log(cold + 1e-09)).sum().item()
    hot_entropy = -(hot * torch.log(hot + 1e-09)).sum().item()
    constraint('Cold temperature has lower entropy than hot', lambda: (cold_entropy < hot_entropy, cold_entropy, f'<{hot_entropy}', {}))

def test_attention_single_head_constraint(constraint, kernel, device):
    Q = torch.randn(1, 1, 128, 64, device=device, dtype=torch.float16)
    K = torch.randn(1, 1, 128, 64, device=device, dtype=torch.float16)
    V = torch.randn(1, 1, 128, 64, device=device, dtype=torch.float16)
    output = kernel.attention(Q, K, V, temperature=1.0)
    constraint('Single head output shape correct', lambda: (output.shape == (1, 1, 128, 64), output.shape, (1, 1, 128, 64), {}))

def test_attention_many_heads_constraint(constraint, kernel, device):
    Q = torch.randn(2, 16, 128, 32, device=device, dtype=torch.float16)
    K = torch.randn(2, 16, 128, 32, device=device, dtype=torch.float16)
    V = torch.randn(2, 16, 128, 32, device=device, dtype=torch.float16)
    output = kernel.attention(Q, K, V, temperature=1.0)
    constraint('Many heads output shape correct', lambda: (output.shape == (2, 16, 128, 32), output.shape, (2, 16, 128, 32), {}))

def test_attention_asymmetric_query_key_constraint(constraint, kernel, device):
    Q = torch.randn(2, 4, 100, 64, device=device, dtype=torch.float16)
    K = torch.randn(2, 4, 200, 64, device=device, dtype=torch.float16)
    V = torch.randn(2, 4, 200, 64, device=device, dtype=torch.float16)
    output = kernel.attention(Q, K, V, temperature=1.0)
    constraint('Asymmetric Q/K output shape matches Q', lambda: (output.shape == (2, 4, 100, 64), output.shape, (2, 4, 100, 64), {}))

def test_correlation_symmetry_constraint(constraint, kernel, device):
    K = torch.randn(2, 64, 32, device=device, dtype=torch.float16)
    V = torch.randn(2, 64, 32, device=device, dtype=torch.float16)
    corr = kernel.correlation(K, V)
    is_symmetric = torch.allclose(corr, corr.transpose(-2, -1), atol=0.01)
    constraint('Correlation matrix is symmetric', lambda: (is_symmetric, is_symmetric, True, {}))

def test_correlation_bounds_constraint(constraint, kernel, device):
    K = torch.randn(2, 64, 32, device=device, dtype=torch.float16)
    V = torch.randn(2, 64, 32, device=device, dtype=torch.float16)
    corr = kernel.correlation(K, V)
    lower_bound = (corr >= -1.1).all().item()
    upper_bound = (corr <= 1.1).all().item()
    constraint('Correlation values >= -1.1', lambda: (lower_bound, lower_bound, True, {}))
    constraint('Correlation values <= 1.1', lambda: (upper_bound, upper_bound, True, {}))

def test_correlation_large_batch_constraint(constraint, kernel, device):
    K = torch.randn(16, 128, 64, device=device, dtype=torch.float16)
    V = torch.randn(16, 128, 64, device=device, dtype=torch.float16)
    corr = kernel.correlation(K, V)
    constraint('Large batch correlation shape correct', lambda: (corr.shape == (16, 128, 128), corr.shape, (16, 128, 128), {}))

def test_effective_rank_range_constraint(constraint, kernel, device):
    identity = torch.eye(64, device=device, dtype=torch.float16).unsqueeze(0)
    ones = torch.ones(1, 64, 64, device=device, dtype=torch.float16)
    random = torch.randn(1, 64, 64, device=device, dtype=torch.float16)
    rank_identity = kernel.effective_rank(identity).item()
    rank_ones = kernel.effective_rank(ones).item()
    rank_random = kernel.effective_rank(random).item()
    constraint('Identity matrix has higher rank than ones matrix', lambda: (rank_identity > rank_ones, rank_identity, f'>{rank_ones}', {}))

def test_effective_rank_batch_constraint(constraint, kernel, device):
    M = torch.randn(8, 64, 64, device=device, dtype=torch.float16)
    rank = kernel.effective_rank(M)
    shape_correct = rank.shape == (8,)
    all_positive = (rank > 0).all().item()
    all_bounded = (rank <= 64).all().item()
    constraint('Batch rank shape correct', lambda: (shape_correct, rank.shape, (8,), {}))
    constraint('All ranks positive', lambda: (all_positive, all_positive, True, {}))
    constraint('All ranks <= matrix dimension', lambda: (all_bounded, all_bounded, True, {}))

def test_attention_gradient_flow_constraint(constraint, kernel, device):
    """Test that gradients flow through CA activation and value propagation."""
    Q = torch.randn(2, 4, 32, 16, device=device, dtype=torch.float32, requires_grad=True)
    K = torch.randn(2, 4, 32, 16, device=device, dtype=torch.float32, requires_grad=True)
    V = torch.randn(2, 4, 32, 16, device=device, dtype=torch.float32, requires_grad=True)

    output = kernel.attention(Q, K, V, temperature=1.0)
    loss = output.sum()
    loss.backward()

    q_has_grad = Q.grad is not None and Q.grad.abs().sum() > 0
    k_has_grad = K.grad is not None and K.grad.abs().sum() > 0
    v_has_grad = V.grad is not None and V.grad.abs().sum() > 0

    constraint('Q receives gradients', lambda: (q_has_grad, Q.grad.abs().sum().item() if Q.grad is not None else 0, '>0', {}))
    constraint('K receives gradients', lambda: (k_has_grad, K.grad.abs().sum().item() if K.grad is not None else 0, '>0', {}))
    constraint('V receives gradients', lambda: (v_has_grad, V.grad.abs().sum().item() if V.grad is not None else 0, '>0', {}))

def test_attention_numerical_stability_zeros_constraint(constraint, kernel, device):
    Q = torch.zeros(1, 1, 64, 32, device=device, dtype=torch.float16)
    K = torch.zeros(1, 1, 64, 32, device=device, dtype=torch.float16)
    V = torch.randn(1, 1, 64, 32, device=device, dtype=torch.float16)
    output = kernel.attention(Q, K, V, temperature=1.0)
    constraint('No NaN with zero Q/K', lambda: (not torch.isnan(output).any(), 'no_nan', 'no_nan', {}))

def test_attention_numerical_stability_large_values_constraint(constraint, kernel, device):
    Q = torch.randn(1, 1, 64, 32, device=device, dtype=torch.float16) * 100
    K = torch.randn(1, 1, 64, 32, device=device, dtype=torch.float16) * 100
    V = torch.randn(1, 1, 64, 32, device=device, dtype=torch.float16)
    output = kernel.attention(Q, K, V, temperature=1.0)
    constraint('No NaN with large values', lambda: (not torch.isnan(output).any(), 'no_nan', 'no_nan', {}))
    constraint('No Inf with large values', lambda: (not torch.isinf(output).any(), 'no_inf', 'no_inf', {}))
