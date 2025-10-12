import pytest
import torch
from slime.kernels.triton_impl import TritonKernel

@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    return torch.device('cuda')

@pytest.fixture
def kernel(device):
    return TritonKernel(device)

def test_attention_power_of_two_shapes(kernel, device):
    for size in [64, 128, 256, 512]:
        Q = torch.randn(2, 4, size, 64, device=device, dtype=torch.float16)
        K = torch.randn(2, 4, size, 64, device=device, dtype=torch.float16)
        V = torch.randn(2, 4, size, 64, device=device, dtype=torch.float16)
        output = kernel.attention(Q, K, V, temperature=1.0)
        assert output.shape == Q.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

def test_attention_non_power_of_two_shapes(kernel, device):
    for size in [100, 200, 300]:
        Q = torch.randn(2, 4, size, 64, device=device, dtype=torch.float16)
        K = torch.randn(2, 4, size, 64, device=device, dtype=torch.float16)
        V = torch.randn(2, 4, size, 64, device=device, dtype=torch.float16)
        output = kernel.attention(Q, K, V, temperature=1.0)
        assert output.shape == Q.shape

def test_attention_temperature_extremes(kernel, device):
    Q = torch.randn(1, 1, 64, 32, device=device, dtype=torch.float16)
    K = torch.randn(1, 1, 64, 32, device=device, dtype=torch.float16)
    V = torch.randn(1, 1, 64, 32, device=device, dtype=torch.float16)
    cold = kernel.attention(Q, K, V, temperature=0.01)
    hot = kernel.attention(Q, K, V, temperature=100.0)
    cold_entropy = -(cold * torch.log(cold + 1e-09)).sum()
    hot_entropy = -(hot * torch.log(hot + 1e-09)).sum()
    assert cold_entropy < hot_entropy

def test_attention_single_head(kernel, device):
    Q = torch.randn(1, 1, 128, 64, device=device, dtype=torch.float16)
    K = torch.randn(1, 1, 128, 64, device=device, dtype=torch.float16)
    V = torch.randn(1, 1, 128, 64, device=device, dtype=torch.float16)
    output = kernel.attention(Q, K, V, temperature=1.0)
    assert output.shape == (1, 1, 128, 64)

def test_attention_many_heads(kernel, device):
    Q = torch.randn(2, 16, 128, 32, device=device, dtype=torch.float16)
    K = torch.randn(2, 16, 128, 32, device=device, dtype=torch.float16)
    V = torch.randn(2, 16, 128, 32, device=device, dtype=torch.float16)
    output = kernel.attention(Q, K, V, temperature=1.0)
    assert output.shape == (2, 16, 128, 32)

def test_attention_asymmetric_query_key(kernel, device):
    Q = torch.randn(2, 4, 100, 64, device=device, dtype=torch.float16)
    K = torch.randn(2, 4, 200, 64, device=device, dtype=torch.float16)
    V = torch.randn(2, 4, 200, 64, device=device, dtype=torch.float16)
    output = kernel.attention(Q, K, V, temperature=1.0)
    assert output.shape == (2, 4, 100, 64)

def test_correlation_symmetry(kernel, device):
    K = torch.randn(2, 64, 32, device=device, dtype=torch.float16)
    V = torch.randn(2, 64, 32, device=device, dtype=torch.float16)
    corr = kernel.correlation(K, V)
    assert torch.allclose(corr, corr.transpose(-2, -1), atol=0.01)

def test_correlation_bounds(kernel, device):
    K = torch.randn(2, 64, 32, device=device, dtype=torch.float16)
    V = torch.randn(2, 64, 32, device=device, dtype=torch.float16)
    corr = kernel.correlation(K, V)
    assert (corr >= -1.1).all()
    assert (corr <= 1.1).all()

def test_correlation_large_batch(kernel, device):
    K = torch.randn(16, 128, 64, device=device, dtype=torch.float16)
    V = torch.randn(16, 128, 64, device=device, dtype=torch.float16)
    corr = kernel.correlation(K, V)
    assert corr.shape == (16, 128, 128)

def test_effective_rank_range(kernel, device):
    matrices = [torch.eye(64, device=device, dtype=torch.float16).unsqueeze(0), torch.ones(1, 64, 64, device=device, dtype=torch.float16), torch.randn(1, 64, 64, device=device, dtype=torch.float16)]
    ranks = [kernel.effective_rank(m) for m in matrices]
    assert ranks[0] > ranks[1]

def test_effective_rank_batch(kernel, device):
    M = torch.randn(8, 64, 64, device=device, dtype=torch.float16)
    rank = kernel.effective_rank(M)
    assert rank.shape == (8,)
    assert (rank > 0).all()
    assert (rank <= 64).all()

def test_attention_gradient_flow(kernel, device):
    Q = torch.randn(1, 1, 64, 32, device=device, dtype=torch.float32, requires_grad=True)
    K = torch.randn(1, 1, 64, 32, device=device, dtype=torch.float32, requires_grad=True)
    V = torch.randn(1, 1, 64, 32, device=device, dtype=torch.float32, requires_grad=True)
    output = kernel.attention(Q, K, V, temperature=1.0)
    loss = output.sum()
    loss.backward()
    assert Q.grad is not None
    assert K.grad is not None
    assert V.grad is not None
    assert not torch.isnan(Q.grad).any()

def test_attention_numerical_stability_zeros(kernel, device):
    Q = torch.zeros(1, 1, 64, 32, device=device, dtype=torch.float16)
    K = torch.zeros(1, 1, 64, 32, device=device, dtype=torch.float16)
    V = torch.randn(1, 1, 64, 32, device=device, dtype=torch.float16)
    output = kernel.attention(Q, K, V, temperature=1.0)
    assert not torch.isnan(output).any()

def test_attention_numerical_stability_large_values(kernel, device):
    Q = torch.randn(1, 1, 64, 32, device=device, dtype=torch.float16) * 100
    K = torch.randn(1, 1, 64, 32, device=device, dtype=torch.float16) * 100
    V = torch.randn(1, 1, 64, 32, device=device, dtype=torch.float16)
    output = kernel.attention(Q, K, V, temperature=1.0)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()