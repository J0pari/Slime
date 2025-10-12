"""
Unit tests for warp-level Neural CA kernels.

Note: These tests require CUDA hardware with compute capability >= 7.0.
Tests will be skipped if CUDA not available or tensor cores not present.
"""

import pytest
import torch
from slime.kernels.warp_ca_bindings import (
    neural_ca_1d_warp,
    neural_ca_2d_tensor_core,
    enforce_mass_conservation,
    is_warp_ca_available,
    get_tensor_core_info
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


def test_warp_ca_availability():
    """Test CUDA and tensor core detection."""
    available = is_warp_ca_available()
    info = get_tensor_core_info()

    assert isinstance(available, bool)
    assert 'compute_capability' in info
    assert 'tensor_cores' in info
    assert 'flops_per_cycle' in info

    if torch.cuda.is_available():
        assert info['available'] is True
        major, minor = info['compute_capability']
        assert major > 0


@pytest.mark.skipif(
    not is_warp_ca_available(),
    reason="Tensor cores not available (compute capability < 7.0)"
)
def test_neural_ca_1d_warp_shapes():
    """Test 1D warp CA kernel output shapes."""
    batch = 2
    seq_len = 64  # Multiple of warp size (32)
    channels = 4
    kernel_size = 3

    state = torch.randn(batch, seq_len, channels, device='cuda')
    kernel_weights = torch.randn(channels, channels, kernel_size, device='cuda')
    growth_center = torch.randn(channels, device='cuda')
    growth_width = torch.abs(torch.randn(channels, device='cuda'))

    output = neural_ca_1d_warp(state, kernel_weights, growth_center, growth_width)

    assert output.shape == state.shape
    assert output.device == state.device
    assert output.dtype == state.dtype


@pytest.mark.skipif(
    not is_warp_ca_available(),
    reason="Tensor cores not available"
)
def test_neural_ca_1d_warp_mass_conservation():
    """Test mass conservation in 1D warp CA."""
    batch = 1
    seq_len = 32
    channels = 2
    kernel_size = 3

    state = torch.randn(batch, seq_len, channels, device='cuda')
    kernel_weights = torch.randn(channels, channels, kernel_size, device='cuda') * 0.1
    growth_center = torch.full((channels,), 0.15, device='cuda')
    growth_width = torch.full((channels,), 0.015, device='cuda')

    output = neural_ca_1d_warp(state, kernel_weights, growth_center, growth_width)

    # Check approximate mass conservation (within 5% due to growth function)
    input_mass = state.sum(dim=1)
    output_mass = output.sum(dim=1)

    mass_ratio = output_mass / (input_mass + 1e-7)
    assert torch.allclose(mass_ratio, torch.ones_like(mass_ratio), atol=0.05)


@pytest.mark.skipif(
    not is_warp_ca_available(),
    reason="Tensor cores not available"
)
def test_neural_ca_2d_tensor_core_shapes():
    """Test 2D tensor core CA kernel output shapes."""
    batch = 2
    height = 16  # Multiple of 16 for tensor cores
    width = 16
    channels = 4

    state = torch.randn(batch, height, width, channels, device='cuda', dtype=torch.float16)
    kernel_weights = torch.randn(channels, channels, 3, 3, device='cuda', dtype=torch.float16)
    growth_center = torch.randn(channels, device='cuda', dtype=torch.float16)
    growth_width = torch.abs(torch.randn(channels, device='cuda', dtype=torch.float16))

    output = neural_ca_2d_tensor_core(state, kernel_weights, growth_center, growth_width)

    assert output.shape == state.shape
    assert output.device == state.device
    assert output.dtype == torch.float16


@pytest.mark.skipif(
    not is_warp_ca_available(),
    reason="Tensor cores not available"
)
def test_enforce_mass_conservation_1d():
    """Test mass conservation enforcement in 1D."""
    batch = 2
    seq_len = 32
    channels = 4

    input_state = torch.randn(batch, seq_len, channels, device='cuda')
    output_state = torch.randn(batch, seq_len, channels, device='cuda')

    conserved = enforce_mass_conservation(input_state, output_state)

    # Check exact mass conservation
    input_mass = input_state.sum(dim=1)
    conserved_mass = conserved.sum(dim=1)

    assert torch.allclose(input_mass, conserved_mass, rtol=1e-5)


@pytest.mark.skipif(
    not is_warp_ca_available(),
    reason="Tensor cores not available"
)
def test_enforce_mass_conservation_2d():
    """Test mass conservation enforcement in 2D."""
    batch = 2
    height = 16
    width = 16
    channels = 4

    input_state = torch.randn(batch, height, width, channels, device='cuda')
    output_state = torch.randn(batch, height, width, channels, device='cuda')

    conserved = enforce_mass_conservation(input_state, output_state)

    # Check exact mass conservation per channel
    input_mass = input_state.sum(dim=(1, 2))  # [batch, channels]
    conserved_mass = conserved.sum(dim=(1, 2))

    assert torch.allclose(input_mass, conserved_mass, rtol=1e-5)


@pytest.mark.skipif(
    not is_warp_ca_available(),
    reason="Tensor cores not available"
)
def test_warp_ca_gradient_flow():
    """Test gradient backpropagation through warp CA."""
    batch = 1
    seq_len = 32
    channels = 2
    kernel_size = 3

    state = torch.randn(batch, seq_len, channels, device='cuda', requires_grad=True)
    kernel_weights = torch.randn(
        channels, channels, kernel_size,
        device='cuda',
        requires_grad=True
    )
    growth_center = torch.randn(channels, device='cuda', requires_grad=True)
    growth_width = torch.abs(torch.randn(channels, device='cuda'))

    output = neural_ca_1d_warp(state, kernel_weights, growth_center, growth_width)

    # Compute loss and backprop
    loss = output.sum()
    loss.backward()

    # Check gradients exist and are non-zero
    assert state.grad is not None
    assert kernel_weights.grad is not None
    assert growth_center.grad is not None

    assert torch.any(state.grad != 0)
    assert torch.any(kernel_weights.grad != 0)


@pytest.mark.skipif(
    not is_warp_ca_available(),
    reason="Tensor cores not available"
)
def test_warp_shuffle_boundary_conditions():
    """Test boundary handling in warp shuffles."""
    batch = 1
    seq_len = 32
    channels = 1
    kernel_size = 3

    # Create state with known pattern
    state = torch.zeros(batch, seq_len, channels, device='cuda')
    state[0, 0, 0] = 1.0  # Left boundary
    state[0, -1, 0] = 1.0  # Right boundary

    kernel_weights = torch.ones(channels, channels, kernel_size, device='cuda') * 0.1
    growth_center = torch.zeros(channels, device='cuda')
    growth_width = torch.ones(channels, device='cuda') * 0.1

    output = neural_ca_1d_warp(state, kernel_weights, growth_center, growth_width)

    # Boundaries should be handled (zero-pad)
    # Output should not have NaN or inf
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_get_tensor_core_info_structure():
    """Test tensor core info returns correct structure."""
    info = get_tensor_core_info()

    required_keys = [
        'available',
        'compute_capability',
        'tensor_cores',
        'flops_per_cycle'
    ]

    for key in required_keys:
        assert key in info

    assert isinstance(info['available'], bool)
    assert isinstance(info['compute_capability'], tuple)
    assert isinstance(info['tensor_cores'], bool)
    assert isinstance(info['flops_per_cycle'], int)

    if info['tensor_cores']:
        assert info['flops_per_cycle'] == 256
    elif info['available']:
        assert info['flops_per_cycle'] == 32
    else:
        assert info['flops_per_cycle'] == 0
