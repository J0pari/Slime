"""
Triton-accelerated kernels for Multi-Head Neural CA operations.

Implements fused Neural CA operations:
- Multi-head CA projections (Perception/Interaction/Value)
- CA neighborhood convolution with learned kernels
- Flow-Lenia growth function (bell curve modulation)
- CA pattern computation (like attention scores but for CA)
- Mass conservation enforcement
- Parameter localization (spatially-varying CA rules)

Replaces attention mechanisms with Neural Cellular Automaton dynamics.
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple
import math
from slime.proto.kernel import Kernel


@triton.jit
def fused_ca_projection_kernel(
    X,  # [batch, num_heads, seq_len, input_dim]
    W_perception, W_interaction, W_value,  # [num_heads, input_dim, head_dim]
    Perception, Interaction, Values,  # Outputs
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_wh, stride_wi, stride_wd,
    stride_pb, stride_ph, stride_ps, stride_pd,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    D_in: tl.constexpr, D_head: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr, BLOCK_SIZE_D: tl.constexpr,
):
    """
    Fused multi-head CA projection kernel.

    Computes three projections in parallel:
    - Perception = X @ W_perception
    - Interaction = X @ W_interaction
    - Values = X @ W_value

    Each head has its own weight matrices.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_d_in = tl.arange(0, D_in)
    offs_d_head = tl.arange(0, D_head)

    # Load input vector: [input_dim]
    x_ptrs = X + (pid_b * stride_xb + pid_h * stride_xh + pid_s * stride_xs + offs_d_in * stride_xd)
    x = tl.load(x_ptrs, mask=offs_d_in < D_in, other=0.0)

    # Load weight matrices for this head: [input_dim, head_dim]
    w_perc_ptrs = W_perception + (pid_h * stride_wh + offs_d_in[:, None] * stride_wi + offs_d_head[None, :] * stride_wd)
    w_inter_ptrs = W_interaction + (pid_h * stride_wh + offs_d_in[:, None] * stride_wi + offs_d_head[None, :] * stride_wd)
    w_val_ptrs = W_value + (pid_h * stride_wh + offs_d_in[:, None] * stride_wi + offs_d_head[None, :] * stride_wd)

    w_perc = tl.load(w_perc_ptrs, mask=(offs_d_in[:, None] < D_in) & (offs_d_head[None, :] < D_head), other=0.0)
    w_inter = tl.load(w_inter_ptrs, mask=(offs_d_in[:, None] < D_in) & (offs_d_head[None, :] < D_head), other=0.0)
    w_val = tl.load(w_val_ptrs, mask=(offs_d_in[:, None] < D_in) & (offs_d_head[None, :] < D_head), other=0.0)

    # Compute projections: [head_dim]
    perception = tl.sum(x[:, None] * w_perc, axis=0)
    interaction = tl.sum(x[:, None] * w_inter, axis=0)
    values = tl.sum(x[:, None] * w_val, axis=0)

    # Store outputs
    p_ptrs = Perception + (pid_b * stride_pb + pid_h * stride_ph + pid_s * stride_ps + offs_d_head * stride_pd)
    i_ptrs = Interaction + (pid_b * stride_pb + pid_h * stride_ph + pid_s * stride_ps + offs_d_head * stride_pd)
    v_ptrs = Values + (pid_b * stride_pb + pid_h * stride_ph + pid_s * stride_ps + offs_d_head * stride_pd)

    tl.store(p_ptrs, perception, mask=offs_d_head < D_head)
    tl.store(i_ptrs, interaction, mask=offs_d_head < D_head)
    tl.store(v_ptrs, values, mask=offs_d_head < D_head)


@triton.jit
def flow_lenia_growth_kernel(
    Potential,  # [batch, num_heads, seq_len, 1]
    Growth_Center, Growth_Width,  # [num_heads]
    Growth_Output,  # [batch, num_heads, seq_len, 1]
    stride_pb, stride_ph, stride_ps,
    stride_gb, stride_gh, stride_gs,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Flow-Lenia growth function: bell curve modulation per head.

    growth(x) = exp(-0.5 * ((x - center) / width)^2)

    Each head has learnable center and width parameters.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_s = pid_s * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Load potential values
    pot_ptrs = Potential + (pid_b * stride_pb + pid_h * stride_ph + offs_s * stride_ps)
    potential = tl.load(pot_ptrs, mask=offs_s < S, other=0.0)

    # Load growth parameters for this head
    center = tl.load(Growth_Center + pid_h)
    width = tl.load(Growth_Width + pid_h)

    # Compute Gaussian bell curve
    diff = potential - center
    z = diff / (width + 1e-10)
    growth = tl.exp(-0.5 * z * z)

    # Store output
    growth_ptrs = Growth_Output + (pid_b * stride_gb + pid_h * stride_gh + offs_s * stride_gs)
    tl.store(growth_ptrs, growth, mask=offs_s < S)


@triton.jit
def ca_activation_kernel(
    Perception, Interaction,  # [batch, num_heads, seq_len, head_dim]
    CA_Activation,  # [batch, num_heads, seq_len, seq_len]
    stride_pb, stride_ph, stride_ps, stride_pd,
    stride_ib, stride_ih, stride_is, stride_id,
    stride_cb, stride_ch, stride_cq, stride_ck,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    CA activation pattern computation: perception ⊗ interaction.

    Similar to Q·K^T in attention, but for CA neighborhood interactions.
    Computes scores[q, k] = (CA_conv(perception)[q] · interaction[k]) / sqrt(D)

    For now, this kernel computes the interaction scores after CA convolution
    is applied externally. Full CA conv+interaction fusion requires Conv1d in Triton.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, D)

    # Load perception for query positions: [BLOCK_Q, D]
    perc_ptrs = Perception + (pid_b * stride_pb + pid_h * stride_ph + offs_q[:, None] * stride_ps + offs_d[None, :] * stride_pd)
    perception_q = tl.load(perc_ptrs, mask=(offs_q[:, None] < S) & (offs_d[None, :] < D), other=0.0)

    # Initialize scores accumulator
    scores = tl.zeros([BLOCK_Q, BLOCK_K], dtype=tl.float32)

    # Loop over key positions
    for k_start in range(0, S, BLOCK_K):
        offs_k_block = k_start + offs_k

        # Load interaction for key positions: [BLOCK_K, D]
        inter_ptrs = Interaction + (pid_b * stride_ib + pid_h * stride_ih + offs_k_block[:, None] * stride_is + offs_d[None, :] * stride_id)
        interaction_k = tl.load(inter_ptrs, mask=(offs_k_block[:, None] < S) & (offs_d[None, :] < D), other=0.0)

        # Compute scores: perception[q] · interaction[k]^T
        scores_block = tl.dot(perception_q, tl.trans(interaction_k))

        # Scale by sqrt(D) (similar to attention temperature)
        scale = 1.0 / tl.sqrt(float(D))
        scores_block = scores_block * scale

        # Store scores
        ca_ptrs = CA_Activation + (pid_b * stride_cb + pid_h * stride_ch + offs_q[:, None] * stride_cq + offs_k_block[None, :] * stride_ck)
        tl.store(ca_ptrs, scores_block, mask=(offs_q[:, None] < S) & (offs_k_block[None, :] < S))


@triton.jit
def ca_value_propagation_kernel(
    CA_Pattern, Values,  # [B, H, S, S], [B, H, S, D]
    CA_Output,  # [B, H, S, D]
    stride_cb, stride_ch, stride_cq, stride_ck,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """
    CA value propagation: output = CA_pattern @ values.

    Similar to attention output but using CA activation pattern.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)

    # Initialize output accumulator: [BLOCK_Q, D]
    output_acc = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)

    # Loop over key positions
    for k_start in range(0, S, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load CA pattern: [BLOCK_Q, BLOCK_K]
        ca_ptrs = CA_Pattern + (pid_b * stride_cb + pid_h * stride_ch + offs_q[:, None] * stride_cq + offs_k[None, :] * stride_ck)
        ca_pattern = tl.load(ca_ptrs, mask=(offs_q[:, None] < S) & (offs_k[None, :] < S), other=0.0)

        # Load values: [BLOCK_K, D]
        val_ptrs = Values + (pid_b * stride_vb + pid_h * stride_vh + offs_k[:, None] * stride_vs + offs_d[None, :] * stride_vd)
        values = tl.load(val_ptrs, mask=(offs_k[:, None] < S) & (offs_d[None, :] < D), other=0.0)

        # Accumulate: pattern @ values
        output_acc += tl.dot(ca_pattern, values)

    # Store output
    out_ptrs = CA_Output + (pid_b * stride_ob + pid_h * stride_oh + offs_q[:, None] * stride_os + offs_d[None, :] * stride_od)
    tl.store(out_ptrs, output_acc, mask=(offs_q[:, None] < S) & (offs_d[None, :] < D))


@triton.jit
def mass_conservation_kernel(
    Input_State, Output_State,  # [batch, spatial_size, channels]
    Output_Conserved,  # [batch, spatial_size, channels]
    stride_ib, stride_is, stride_ic,
    stride_ob, stride_os, stride_oc,
    B: tl.constexpr, Spatial: tl.constexpr, C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Mass conservation: enforce ∑ output = ∑ input per batch-channel.

    Computes:
    - input_mass = ∑ input[b, :, c]
    - output_mass = ∑ output[b, :, c]
    - output_conserved = output * (input_mass / output_mass)
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    # Compute mass sums using reduction
    input_mass = 0.0
    output_mass = 0.0

    for s_start in range(0, Spatial, BLOCK_SIZE):
        offs_s = s_start + tl.arange(0, BLOCK_SIZE)

        # Load input and output
        in_ptrs = Input_State + (pid_b * stride_ib + offs_s * stride_is + pid_c * stride_ic)
        out_ptrs = Output_State + (pid_b * stride_ob + offs_s * stride_os + pid_c * stride_oc)

        input_vals = tl.load(in_ptrs, mask=offs_s < Spatial, other=0.0)
        output_vals = tl.load(out_ptrs, mask=offs_s < Spatial, other=0.0)

        input_mass += tl.sum(input_vals)
        output_mass += tl.sum(output_vals)

    # Compute mass ratio
    mass_ratio = input_mass / (output_mass + 1e-10)

    # Apply mass conservation
    for s_start in range(0, Spatial, BLOCK_SIZE):
        offs_s = s_start + tl.arange(0, BLOCK_SIZE)

        out_ptrs = Output_State + (pid_b * stride_ob + offs_s * stride_os + pid_c * stride_oc)
        conserved_ptrs = Output_Conserved + (pid_b * stride_ob + offs_s * stride_os + pid_c * stride_oc)

        output_vals = tl.load(out_ptrs, mask=offs_s < Spatial, other=0.0)
        conserved_vals = output_vals * mass_ratio

        tl.store(conserved_ptrs, conserved_vals, mask=offs_s < Spatial)


@triton.jit
def correlation_kernel(
    K_mat, V,  # [batch, num_heads, seq_len, dim]
    Corr,  # [batch, num_heads, seq_len, seq_len]
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_cb, stride_ch, stride_ck1, stride_ck2,
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    eps: tl.constexpr,
):
    """
    Correlation computation: normalized K @ K^T.

    Used for behavioral metrics and effective rank computation.
    Grid is flattened to 3D: (B, H, N_BLOCKS_I * N_BLOCKS_J)
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_ij = tl.program_id(2)

    # Unflatten pid_ij into i and j
    pid_i = pid_ij // N_BLOCKS
    pid_j = pid_ij % N_BLOCKS

    offs_i = pid_i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_j = pid_j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, D)

    # Load K vectors: [BLOCK_SIZE, D]
    k_i_ptrs = K_mat + (pid_b * stride_kb + pid_h * stride_kh + offs_i[:, None] * stride_kn + offs_d[None, :] * stride_kk)
    k_j_ptrs = K_mat + (pid_b * stride_kb + pid_h * stride_kh + offs_j[:, None] * stride_kn + offs_d[None, :] * stride_kk)

    k_i = tl.load(k_i_ptrs, mask=(offs_i[:, None] < N) & (offs_d[None, :] < D), other=0.0).to(tl.float32)
    k_j = tl.load(k_j_ptrs, mask=(offs_j[:, None] < N) & (offs_d[None, :] < D), other=0.0).to(tl.float32)

    # Compute correlation: K_i @ K_j^T
    corr_block = tl.dot(k_i, tl.trans(k_j))

    # Normalize by norms
    k_i_norm = tl.sqrt(tl.sum(k_i * k_i, axis=1) + eps)
    k_j_norm = tl.sqrt(tl.sum(k_j * k_j, axis=1) + eps)
    norm_factor = k_i_norm[:, None] * k_j_norm[None, :]

    corr_block = corr_block / norm_factor

    # Store correlation
    corr_ptrs = Corr + (pid_b * stride_cb + pid_h * stride_ch + offs_i[:, None] * stride_ck1 + offs_j[None, :] * stride_ck2)
    mask = (offs_i[:, None] < N) & (offs_j[None, :] < N)
    tl.store(corr_ptrs, corr_block, mask=mask)


@triton.jit
def effective_rank_kernel(
    Matrix,  # [batch, seq_len, seq_len]
    Rank,  # [batch]
    stride_mb, stride_m1, stride_m2,
    B: tl.constexpr, N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    threshold: tl.constexpr,
    eps: tl.constexpr,
):
    """
    Effective rank via trace^2 / frobenius norm^2.

    Approximates rank without full SVD.
    """
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)

    # Compute trace and Frobenius norm
    trace = 0.0
    frobenius_sq = 0.0

    for i_start in range(0, N, BLOCK_SIZE):
        for j_start in range(0, N, BLOCK_SIZE):
            offs_i = i_start + offs
            offs_j = j_start + offs

            matrix_ptrs = Matrix + (pid * stride_mb + offs_i[:, None] * stride_m1 + offs_j[None, :] * stride_m2)
            matrix_block = tl.load(matrix_ptrs, mask=(offs_i[:, None] < N) & (offs_j[None, :] < N), other=0.0)

            # Diagonal elements for trace
            if i_start == j_start:
                diag_mask = offs[:, None] == offs[None, :]
                trace += tl.sum(tl.where(diag_mask, matrix_block, 0.0))

            # All elements for Frobenius norm
            frobenius_sq += tl.sum(matrix_block * matrix_block)

    # Effective rank = trace^2 / frobenius_sq
    approx_rank = (trace * trace) / (frobenius_sq + eps)

    tl.store(Rank + pid, approx_rank)


class CAAttentionAutograd(torch.autograd.Function):
    """
    Autograd wrapper for Triton CA activation + value propagation.

    Enables gradient flow through Triton-compiled kernels by falling back
    to PyTorch operations for backward pass.
    """

    @staticmethod
    def forward(ctx, query, key, value, device, numerical_config):
        """Forward: use Triton kernels for speed."""
        B, H, M, D = query.shape
        _, _, N, _ = key.shape

        # Compute CA activation using Triton kernel
        ca_activation = torch.empty(B, H, M, N, device=device, dtype=query.dtype)

        BLOCK_Q = 64
        BLOCK_K = 64
        grid = (B, H, triton.cdiv(M, BLOCK_Q))

        ca_activation_kernel[grid](
            query, key, ca_activation,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            ca_activation.stride(0), ca_activation.stride(1), ca_activation.stride(2), ca_activation.stride(3),
            B, H, M, D,
            BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K
        )

        # Apply CA pattern to values using Triton kernel
        output = torch.empty_like(query)

        BLOCK_D = D
        grid = (B, H, triton.cdiv(M, BLOCK_Q))

        ca_value_propagation_kernel[grid](
            ca_activation, value, output,
            ca_activation.stride(0), ca_activation.stride(1), ca_activation.stride(2), ca_activation.stride(3),
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            B, H, M, D,
            BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D
        )

        # Save for backward
        ctx.save_for_backward(query, key, value, ca_activation)
        ctx.device = device
        ctx.numerical_config = numerical_config

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward: use PyTorch for autograd compatibility."""
        query, key, value, ca_activation = ctx.saved_tensors
        device = ctx.device
        eps = ctx.numerical_config.epsilon

        B, H, M, D = query.shape
        _, _, N, _ = key.shape

        # Recompute CA activation pattern for gradients using PyTorch
        # perception ⊗ interaction / sqrt(D)
        scale = 1.0 / math.sqrt(float(D))
        scores = torch.einsum('bhqd,bhkd->bhqk', query, key) * scale

        # Gradient through value propagation: d(scores @ V) / d(scores, V)
        grad_scores = torch.einsum('bhqd,bhkd->bhqk', grad_output, value)
        grad_value = torch.einsum('bhqk,bhqd->bhkd', scores, grad_output)

        # Gradient through scores = Q @ K^T
        grad_query = torch.einsum('bhqk,bhkd->bhqd', grad_scores, key) * scale
        grad_key = torch.einsum('bhqk,bhqd->bhkd', grad_scores.transpose(-2, -1), query) * scale

        return grad_query, grad_key, grad_value, None, None


class TritonKernel(Kernel):
    """
    Triton-accelerated implementation of Kernel protocol for Neural CA.

    Provides GPU-optimized kernels for:
    - Multi-head CA projections
    - Flow-Lenia growth function
    - CA activation patterns
    - Value propagation
    - Mass conservation
    - Correlation and effective rank (for behavioral metrics)

    Uses autograd wrappers for gradient flow during training.
    """

    def __init__(self, device: torch.device, numerical_config):
        self.device = device
        self.numerical_config = numerical_config

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Legacy attention interface - redirects to correlation-based CA.

        For Neural CA, this computes CA activation pattern and value propagation.
        Uses autograd wrapper to enable gradient flow.
        """
        assert query.is_cuda and key.is_cuda and value.is_cuda

        # Use autograd-enabled function
        return CAAttentionAutograd.apply(query, key, value, self.device, self.numerical_config)

    def correlation(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Compute normalized correlation K @ K^T."""
        if key.ndim == 4:
            B, H, N, D = key.shape
            assert key.is_cuda and value.is_cuda

            corr = torch.empty(B, H, N, N, device=self.device, dtype=key.dtype)

            BLOCK_SIZE = 32
            N_BLOCKS = triton.cdiv(N, BLOCK_SIZE)
            grid = (B, H, N_BLOCKS * N_BLOCKS)

            correlation_kernel[grid](
                key, value, corr,
                key.stride(0), key.stride(1), key.stride(2), key.stride(3),
                value.stride(0), value.stride(1), value.stride(2), value.stride(3),
                corr.stride(0), corr.stride(1), corr.stride(2), corr.stride(3),
                B, H, N, D,
                BLOCK_SIZE=BLOCK_SIZE,
                N_BLOCKS=N_BLOCKS,
                eps=self.numerical_config.epsilon
            )

            return corr
        else:
            B, N, D = key.shape
            assert key.is_cuda and value.is_cuda

            corr = torch.empty(B, N, N, device=self.device, dtype=key.dtype)

            BLOCK_SIZE = 32
            N_BLOCKS = triton.cdiv(N, BLOCK_SIZE)

            # Expand to 4D for kernel
            key_expanded = key.unsqueeze(1)
            value_expanded = value.unsqueeze(1)
            corr_expanded = corr.unsqueeze(1)

            grid = (B, 1, N_BLOCKS * N_BLOCKS)

            correlation_kernel[grid](
                key_expanded, value_expanded, corr_expanded,
                key_expanded.stride(0), key_expanded.stride(1), key_expanded.stride(2), key_expanded.stride(3),
                value_expanded.stride(0), value_expanded.stride(1), value_expanded.stride(2), value_expanded.stride(3),
                corr_expanded.stride(0), corr_expanded.stride(1), corr_expanded.stride(2), corr_expanded.stride(3),
                B, 1, N, D,
                BLOCK_SIZE=BLOCK_SIZE,
                N_BLOCKS=N_BLOCKS,
                eps=self.numerical_config.epsilon
            )

            return corr.squeeze(1)

    def effective_rank(self, matrix: torch.Tensor) -> torch.Tensor:
        """Compute effective rank via trace^2 / frobenius norm^2."""
        if matrix.ndim == 4:
            B, H, N, _ = matrix.shape
            assert matrix.is_cuda
            assert matrix.shape[2] == matrix.shape[3]

            matrix_reshaped = matrix.reshape(B * H, N, N)
            rank = torch.empty(B * H, device=self.device, dtype=torch.float32)

            BLOCK_SIZE = 128
            grid = (B * H,)

            effective_rank_kernel[grid](
                matrix_reshaped, rank,
                matrix_reshaped.stride(0), matrix_reshaped.stride(1), matrix_reshaped.stride(2),
                B * H, N,
                BLOCK_SIZE=BLOCK_SIZE,
                threshold=self.numerical_config.svd_threshold,
                eps=self.numerical_config.epsilon
            )

            return rank.reshape(B, H).mean(dim=1)
        else:
            B, N, _ = matrix.shape
            assert matrix.is_cuda
            assert matrix.shape[1] == matrix.shape[2]

            rank = torch.empty(B, device=self.device, dtype=torch.float32)

            BLOCK_SIZE = 128
            grid = (B,)

            effective_rank_kernel[grid](
                matrix, rank,
                matrix.stride(0), matrix.stride(1), matrix.stride(2),
                B, N,
                BLOCK_SIZE=BLOCK_SIZE,
                threshold=self.numerical_config.svd_threshold,
                eps=self.numerical_config.epsilon
            )

            return rank
