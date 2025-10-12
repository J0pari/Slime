import pytest
import torch
from slime.kernels.triton_impl import TritonKernel
from slime.kernels.torch_fallback import TorchKernel

class TestTritonKernel:
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.fixture
    def triton_kernel(self, device):
        if not torch.cuda.is_available():
            pytest.skip('CUDA not available')
        return TritonKernel(device)

    def test_attention_shape(self, triton_kernel):
        B, H, M, N, D = 2, 4, 128, 128, 64
        Q = torch.randn(B, H, M, D, device=triton_kernel.device, dtype=torch.float16)
        K = torch.randn(B, H, N, D, device=triton_kernel.device, dtype=torch.float16)
        V = torch.randn(B, H, N, D, device=triton_kernel.device, dtype=torch.float16)
        output = triton_kernel.attention(Q, K, V, temperature=1.0)
        assert output.shape == (B, H, M, D)

    def test_correlation_shape(self, triton_kernel):
        B, N, D = 4, 128, 64
        K = torch.randn(B, N, D, device=triton_kernel.device, dtype=torch.float16)
        V = torch.randn(B, N, D, device=triton_kernel.device, dtype=torch.float16)
        corr = triton_kernel.correlation(K, V)
        assert corr.shape == (B, N, N)

    def test_effective_rank(self, triton_kernel):
        B, N = 4, 128
        matrix = torch.randn(B, N, N, device=triton_kernel.device, dtype=torch.float16)
        rank = triton_kernel.effective_rank(matrix)
        assert rank.shape == (B,)
        assert (rank > 0).all()

    def test_attention_temperature(self, triton_kernel):
        B, H, M, N, D = 1, 1, 64, 64, 32
        Q = torch.randn(B, H, M, D, device=triton_kernel.device, dtype=torch.float16)
        K = torch.randn(B, H, N, D, device=triton_kernel.device, dtype=torch.float16)
        V = torch.randn(B, H, N, D, device=triton_kernel.device, dtype=torch.float16)
        output_low = triton_kernel.attention(Q, K, V, temperature=0.1)
        output_high = triton_kernel.attention(Q, K, V, temperature=10.0)
        assert not torch.allclose(output_low, output_high, atol=0.1)


class TestTorchKernel:
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.fixture
    def torch_kernel(self, device):
        return TorchKernel(device)

    def test_attention_shape(self, torch_kernel):
        B, H, M, N, D = 2, 4, 128, 128, 64
        Q = torch.randn(B, H, M, D, device=torch_kernel.device)
        K = torch.randn(B, H, N, D, device=torch_kernel.device)
        V = torch.randn(B, H, N, D, device=torch_kernel.device)
        output = torch_kernel.attention(Q, K, V, temperature=1.0)
        assert output.shape == (B, H, M, D)

    def test_correlation_shape(self, torch_kernel):
        B, N, D = 4, 128, 64
        K = torch.randn(B, N, D, device=torch_kernel.device)
        V = torch.randn(B, N, D, device=torch_kernel.device)
        corr = torch_kernel.correlation(K, V)
        assert corr.shape == (B, N, N)

    def test_effective_rank(self, torch_kernel):
        B, N = 4, 128
        matrix = torch.randn(B, N, N, device=torch_kernel.device)
        rank = torch_kernel.effective_rank(matrix)
        assert rank.shape == (B,)
        assert (rank > 0).all()

    def test_correlation_normalized(self, torch_kernel):
        B, N, D = 2, 64, 32
        K = torch.randn(B, N, D, device=torch_kernel.device)
        V = torch.randn(B, N, D, device=torch_kernel.device)
        corr = torch_kernel.correlation(K, V)
        assert (corr >= -1.0).all()
        assert (corr <= 1.0).all()
