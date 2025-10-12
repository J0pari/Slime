"""
DIRESA: Distance-Regularized Siamese Twin Autoencoder

Learns behavioral embeddings online with adaptive dimensionality (2-10D).
Distance-preserving nonlinear dimension reduction via autoencoder with:
- Reconstruction loss (MSE)
- Distance preservation loss (pairwise distance matching)
- KL regularization (prevent collapse)
- Learned gating for adaptive dimensionality

Reference: Zhang et al. (2025). "DIRESA: Distance-preserving nonlinear dimension
reduction via regularized autoencoders." arXiv:2404.18314
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class DIRESABehavioralEncoder(nn.Module):
    """
    DIRESA autoencoder for learning behavioral embeddings online.

    Architecture:
    - Encoder: raw_metrics (10-20D) → hidden → bottleneck (2-10D)
    - Decoder: bottleneck → hidden → reconstructed_metrics
    - Gating: learned mask for adaptive dimensionality

    Loss = reconstruction_loss + λ_dist * distance_loss + λ_kl * kl_loss

    Args:
        input_dim: Raw behavioral metrics dimensionality (10-20)
        min_dims: Minimum embedding dimensions (2)
        max_dims: Maximum embedding dimensions (10)
        hidden_dim: Hidden layer size (default: 64)
        lambda_dist: Distance preservation weight (default: 1.0)
        lambda_kl: KL regularization weight (default: 0.01)
        device: torch device
    """

    def __init__(
        self,
        input_dim: int,
        min_dims: int = 2,
        max_dims: int = 10,
        hidden_dim: int = 64,
        lambda_dist: float = 1.0,
        lambda_kl: float = 0.01,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.min_dims = min_dims
        self.max_dims = max_dims
        self.hidden_dim = hidden_dim
        self.lambda_dist = lambda_dist
        self.lambda_kl = lambda_kl
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Encoder: raw_metrics → hidden → bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_dims),
            nn.Tanh()  # Bounded activations
        ).to(self.device)

        # Decoder: bottleneck → hidden → reconstructed
        self.decoder = nn.Sequential(
            nn.Linear(max_dims, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        ).to(self.device)

        # Learned gating for adaptive dimensionality
        # Starts with all dimensions active, learns to suppress unused ones
        self.gate_logits = nn.Parameter(torch.ones(max_dims, device=self.device))

        # Track active dimensions via exponential moving average
        self.register_buffer('gate_ema', torch.ones(max_dims, device=self.device))
        self.gate_ema_decay = 0.99

    def forward(
        self,
        x: torch.Tensor,
        return_gate: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through DIRESA encoder-decoder.

        Args:
            x: Raw behavioral metrics [batch, input_dim]
            return_gate: If True, return gating mask

        Returns:
            embedding: Gated bottleneck representation [batch, max_dims]
            reconstruction: Decoded output [batch, input_dim]
            gate: Optional gating mask [max_dims] if return_gate=True
        """
        # Encode to bottleneck
        z = self.encoder(x)

        # Apply learned gating (soft during training, hard at inference)
        gate = torch.sigmoid(self.gate_logits)

        if self.training:
            # Soft gating during training
            z_gated = z * gate.unsqueeze(0)
        else:
            # Hard gating at inference (binary mask)
            gate_hard = (gate > 0.5).float()
            z_gated = z * gate_hard.unsqueeze(0)

        # Update gate EMA
        if self.training:
            self.gate_ema = self.gate_ema_decay * self.gate_ema + (1 - self.gate_ema_decay) * gate.detach()

        # Decode
        x_recon = self.decoder(z_gated)

        if return_gate:
            return z_gated, x_recon, gate
        return z_gated, x_recon, None

    def compute_loss(
        self,
        x: torch.Tensor,
        pairwise_distances: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute DIRESA loss: reconstruction + distance preservation + KL regularization.

        Args:
            x: Raw behavioral metrics [batch, input_dim]
            pairwise_distances: Optional precomputed distances [batch, batch]

        Returns:
            loss: Total loss scalar
            metrics: Dict of individual loss components
        """
        z_gated, x_recon, gate = self.forward(x, return_gate=True)

        # 1. Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x)

        # 2. Distance preservation loss
        if pairwise_distances is not None:
            # Compute pairwise distances in embedding space
            z_dist = torch.cdist(z_gated, z_gated, p=2)

            # Normalize both distance matrices to [0, 1]
            z_dist_norm = z_dist / (z_dist.max() + 1e-10)
            high_dist_norm = pairwise_distances / (pairwise_distances.max() + 1e-10)

            # MSE between normalized distance matrices
            dist_loss = F.mse_loss(z_dist_norm, high_dist_norm)
        else:
            dist_loss = torch.tensor(0.0, device=self.device)

        # 3. KL regularization on gate (encourage sparsity)
        # KL(gate || uniform) = gate * log(gate * max_dims) + (1-gate) * log((1-gate) * max_dims)
        gate_uniform = 1.0 / self.max_dims
        kl_loss = gate * torch.log((gate + 1e-10) / gate_uniform) + \
                  (1 - gate) * torch.log(((1 - gate) + 1e-10) / (1 - gate_uniform))
        kl_loss = kl_loss.sum()

        # Total loss
        total_loss = recon_loss + self.lambda_dist * dist_loss + self.lambda_kl * kl_loss

        metrics = {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'dist_loss': dist_loss.item() if isinstance(dist_loss, torch.Tensor) else dist_loss,
            'kl_loss': kl_loss.item(),
            'active_dims': self.get_active_dims()
        }

        return total_loss, metrics

    def get_active_dims(self) -> int:
        """
        Get current number of active dimensions via warp vote mechanism.

        Uses exponential moving average of gate activations.
        Threshold: gate_ema > 0.5 → dimension active

        Returns:
            Number of active dimensions (clamped to [min_dims, max_dims])
        """
        active = (self.gate_ema > 0.5).sum().item()
        return int(np.clip(active, self.min_dims, self.max_dims))

    def get_embedding_dim(self) -> int:
        """Get current embedding dimensionality."""
        return self.get_active_dims()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode raw metrics to behavioral embedding.

        Args:
            x: Raw metrics [batch, input_dim] or [input_dim]

        Returns:
            embedding: [batch, max_dims] or [max_dims]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        z_gated, _, _ = self.forward(x, return_gate=False)

        if squeeze:
            z_gated = z_gated.squeeze(0)

        return z_gated

    def validate_embeddings(
        self,
        raw_samples: np.ndarray,
        embedded_samples: np.ndarray
    ) -> dict:
        """
        Validate learned embeddings using Trustworthiness, Continuity, Procrustes.

        Args:
            raw_samples: Raw behavioral metrics [n_samples, input_dim]
            embedded_samples: DIRESA embeddings [n_samples, active_dims]

        Returns:
            metrics: Dict with trustworthiness, continuity, procrustes_distance
        """
        from scipy.spatial import procrustes

        n_samples = len(raw_samples)
        k = min(30, n_samples - 1)

        # Convert to GPU tensors for ALL metric computation
        raw_tensor = torch.from_numpy(raw_samples).to(self.device).float()
        emb_tensor = torch.from_numpy(embedded_samples).to(self.device).float()

        # GPU-accelerated pairwise distances using torch.cdist
        dist_high = torch.cdist(raw_tensor, raw_tensor)  # [N, N]
        dist_low = torch.cdist(emb_tensor, emb_tensor)    # [N, N]

        # Trustworthiness: preservation of k-nearest neighbors (GPU)
        # For each point, check if its k-NN in low-dim are also k-NN in high-dim
        neighbors_high = torch.argsort(dist_high, dim=1)[:, 1:k+1]  # [N, k] - exclude self
        neighbors_low = torch.argsort(dist_low, dim=1)[:, 1:k+1]    # [N, k]

        trust_sum = 0.0
        for i in range(n_samples):
            # Count how many of low-dim neighbors are in high-dim neighborhood
            nh_set = set(neighbors_high[i].cpu().numpy())
            nl_set = set(neighbors_low[i].cpu().numpy())
            trust_sum += len(nl_set & nh_set) / k
        trust_score = trust_sum / n_samples

        # Continuity: preservation of neighborhood structure (GPU)
        continuity_sum = 0.0
        for i in range(n_samples):
            nh_set = set(neighbors_high[i].cpu().numpy())
            nl_set = set(neighbors_low[i].cpu().numpy())
            continuity_sum += len(nl_set & nh_set) / k
        continuity_score = continuity_sum / n_samples

        # Procrustes distance: shape similarity
        active_dims = self.get_active_dims()

        # Center and compute SVD for projection to active_dims (raw_tensor already on GPU)
        raw_centered_tensor = raw_tensor - raw_tensor.mean(dim=0, keepdim=True)
        U, S, Vt = torch.linalg.svd(raw_centered_tensor, full_matrices=False)
        raw_projected = (U[:, :active_dims] * S[:active_dims].unsqueeze(0)).cpu().numpy()

        # Center both
        raw_centered = raw_projected - raw_projected.mean(axis=0)
        emb_centered = embedded_samples[:, :active_dims] - embedded_samples[:, :active_dims].mean(axis=0)

        # Procrustes alignment
        mtx1, mtx2, disparity = procrustes(raw_centered, emb_centered)
        procrustes_dist = np.sqrt(disparity)

        return {
            'trustworthiness': trust_score,
            'continuity': continuity_score,
            'procrustes_distance': procrustes_dist,
            'active_dims': active_dims
        }
