import torch
import triton
import triton.language as tl
from typing import Optional
import math
from slime.proto.kernel import Kernel

@triton.jit
def fused_attention_kernel(
    Q, K_mat, V, Out,
    softmax_scale,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    B, H, M, N, D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q + (pid_b * stride_qb + pid_h * stride_qh +
                  offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    k_ptrs = K_mat + (pid_b * stride_kb + pid_h * stride_kh +
                      offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk)
    v_ptrs = V + (pid_b * stride_vb + pid_h * stride_vh +
                  offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)

    q = tl.load(q_ptrs, mask=offs_m[:, None] < M, other=0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        offs_n_block = start_n + offs_n

        k = tl.load(k_ptrs + start_n * stride_kn,
                   mask=offs_n_block[None, :] < N, other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vn,
                   mask=offs_n_block[:, None] < N, other=0.0)

        qk = tl.dot(q, k)
        qk = qk * softmax_scale

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    acc = acc / l_i[:, None]

    o_ptrs = Out + (pid_b * stride_ob + pid_h * stride_oh +
                   offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < M)

@triton.jit
def correlation_kernel(K_mat, V, Corr, stride_kb, stride_kn, stride_kk, stride_vb, stride_vn, stride_vk, stride_cb, stride_ck1, stride_ck2, B, N, BLOCK_SIZE: tl.constexpr, D: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_j = tl.program_id(2)
    offs_i = pid_i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_j = pid_j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, D)
    k_i_ptrs = K_mat + (pid_b * stride_kb + offs_i[:, None] * stride_kn + offs_d[None, :] * stride_kk)
    k_j_ptrs = K_mat + (pid_b * stride_kb + offs_j[:, None] * stride_kn + offs_d[None, :] * stride_kk)
    k_i = tl.load(k_i_ptrs, mask=offs_i[:, None] < N, other=0.0).to(tl.float32)
    k_j = tl.load(k_j_ptrs, mask=offs_j[:, None] < N, other=0.0).to(tl.float32)
    corr_block = tl.dot(k_i, tl.trans(k_j))
    k_i_norm = tl.sqrt(tl.sum(k_i * k_i, axis=1))
    k_j_norm = tl.sqrt(tl.sum(k_j * k_j, axis=1))
    norm_factor = k_i_norm[:, None] * k_j_norm[None, :] + 1e-10
    corr_block = corr_block / norm_factor
    corr_ptrs = Corr + (pid_b * stride_cb + offs_i[:, None] * stride_ck1 + offs_j[None, :] * stride_ck2)
    mask = (offs_i[:, None] < N) & (offs_j[None, :] < N)
    tl.store(corr_ptrs, corr_block, mask=mask)

@triton.jit
def effective_rank_kernel(Matrix, Rank, stride_mb, stride_m1, stride_m2, B, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    matrix_ptrs = Matrix + pid * stride_mb + offs[:, None] * stride_m1 + offs[None, :] * stride_m2
    matrix = tl.load(matrix_ptrs, mask=(offs[:, None] < N) & (offs[None, :] < N), other=0.0)
    trace = tl.sum(tl.where(offs[:, None] == offs[None, :], matrix, 0.0))
    frobenius_sq = tl.sum(matrix * matrix)
    approx_rank = trace * trace / (frobenius_sq + 1e-10)
    tl.store(Rank + pid, approx_rank)

@triton.jit
def sparse_routing_kernel(Scores, TopK_Indices, TopK_Values, stride_sb, stride_sn, stride_ib, stride_ik, stride_vb, stride_vk, B, N, K, BLOCK_SIZE: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_SIZE)
    score_ptrs = Scores + pid_b * stride_sb + pid_n * stride_sn + offs_m
    scores = tl.load(score_ptrs, mask=offs_m < N, other=float('-inf'))
    topk_vals = tl.full([K], float('-inf'), dtype=tl.float32)
    topk_idx = tl.full([K], -1, dtype=tl.int32)
    for i in range(N):
        if i < BLOCK_SIZE:
            val = scores[i]
            idx = i
            for k in range(K):
                if val > topk_vals[k]:
                    for j in range(K - 1, k, -1):
                        topk_vals[j] = topk_vals[j - 1]
                        topk_idx[j] = topk_idx[j - 1]
                    topk_vals[k] = val
                    topk_idx[k] = idx
                    break
    idx_ptrs = TopK_Indices + pid_b * stride_ib + pid_n * stride_ik + tl.arange(0, K)
    val_ptrs = TopK_Values + pid_b * stride_vb + pid_n * stride_vk + tl.arange(0, K)
    tl.store(idx_ptrs, topk_idx)
    tl.store(val_ptrs, topk_vals)

class TritonKernel(Kernel):

    def __init__(self, device: torch.device):
        self.device = device

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, temperature: float=1.0) -> torch.Tensor:
        B, H, M, D = query.shape
        _, _, N, _ = key.shape
        assert query.shape[-1] == key.shape[-1] == value.shape[-1]
        assert query.is_cuda and key.is_cuda and value.is_cuda
        output = torch.empty_like(query)
        softmax_scale = 1.0 / (math.sqrt(D) * temperature)
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_D = D
        grid = (B, H, triton.cdiv(M, BLOCK_M))
        fused_attention_kernel[grid](query, key, value, output, softmax_scale, query.stride(0), query.stride(1), query.stride(2), query.stride(3), key.stride(0), key.stride(1), key.stride(2), key.stride(3), value.stride(0), value.stride(1), value.stride(2), value.stride(3), output.stride(0), output.stride(1), output.stride(2), output.stride(3), B, H, M, N, D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D)
        return output

    def correlation(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        B, N, D = key.shape
        assert key.is_cuda and value.is_cuda
        corr = torch.empty(B, N, N, device=self.device, dtype=key.dtype)
        BLOCK_SIZE = 32
        grid = (B, triton.cdiv(N, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))
        correlation_kernel[grid](key, value, corr, key.stride(0), key.stride(1), key.stride(2), value.stride(0), value.stride(1), value.stride(2), corr.stride(0), corr.stride(1), corr.stride(2), B, N, BLOCK_SIZE=BLOCK_SIZE, D=D)
        return corr

    def effective_rank(self, matrix: torch.Tensor) -> torch.Tensor:
        B, N, _ = matrix.shape
        assert matrix.is_cuda
        assert matrix.shape[1] == matrix.shape[2]
        rank = torch.empty(B, device=self.device, dtype=torch.float32)
        BLOCK_SIZE = 128
        grid = (B,)
        effective_rank_kernel[grid](matrix, rank, matrix.stride(0), matrix.stride(1), matrix.stride(2), B, N, BLOCK_SIZE=BLOCK_SIZE)
        return rank

    def sparse_top_k(self, scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        B, N = scores.shape
        assert scores.is_cuda
        indices = torch.empty(B, N, k, device=self.device, dtype=torch.int32)
        values = torch.empty(B, N, k, device=self.device, dtype=scores.dtype)
        BLOCK_SIZE = 256
        grid = (B, N)
        sparse_routing_kernel[grid](scores, indices, values, scores.stride(0), scores.stride(1), indices.stride(0), indices.stride(1), values.stride(0), values.stride(1), B, N, k, BLOCK_SIZE=BLOCK_SIZE)
        return (indices, values)