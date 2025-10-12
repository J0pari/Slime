import torch
from typing import Tuple, Optional
import logging
logger = logging.getLogger(__name__)

def pairwise_behavioral_distance(behaviors: torch.Tensor, metric: str='euclidean') -> torch.Tensor:
    if metric == 'euclidean':
        diff = behaviors.unsqueeze(1) - behaviors.unsqueeze(0)
        distances = torch.sqrt((diff ** 2).sum(dim=-1))
        return distances
    elif metric == 'cosine':
        normalized = torch.nn.functional.normalize(behaviors, p=2, dim=-1)
        similarity = torch.matmul(normalized, normalized.T)
        distances = 1.0 - similarity
        return distances.clamp(min=0.0)
    else:
        raise ValueError(f'Unknown metric: {metric}')

def topk_neighbors_mask(distances: torch.Tensor, k: int, include_self: bool=False) -> torch.Tensor:
    if not include_self:
        distances = distances + torch.eye(distances.shape[0], device=distances.device) * 1e9
    _, indices = torch.topk(distances, k=k, dim=1, largest=False)
    mask = torch.zeros_like(distances, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return mask

def vmap_relative_fitness(fitnesses: torch.Tensor, neighbor_mask: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    neighbor_counts = neighbor_mask.sum(dim=1, keepdim=True).clamp(min=1)
    neighbor_fitnesses = fitnesses.unsqueeze(1) * neighbor_mask
    neighbor_sum = neighbor_fitnesses.sum(dim=1, keepdim=True)
    neighbor_mean = neighbor_sum / neighbor_counts
    neighbor_sq_sum = ((fitnesses.unsqueeze(1) ** 2) * neighbor_mask).sum(dim=1, keepdim=True)
    neighbor_variance = (neighbor_sq_sum / neighbor_counts) - (neighbor_mean ** 2)
    neighbor_std = torch.sqrt(neighbor_variance.clamp(min=0.0)) + eps
    z_scores = (fitnesses.unsqueeze(1) - neighbor_mean) / neighbor_std
    return z_scores.squeeze(1)

def vmap_behavioral_divergence(behaviors: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
    neighbor_counts = neighbor_mask.sum(dim=1, keepdim=True).clamp(min=1)
    neighbor_mask_expanded = neighbor_mask.unsqueeze(-1)
    neighbor_behaviors = behaviors.unsqueeze(1) * neighbor_mask_expanded
    neighbor_sum = neighbor_behaviors.sum(dim=1)
    neighbor_mean = neighbor_sum / neighbor_counts
    divergence = behaviors - neighbor_mean
    return divergence

def vmap_gradient_rank(grad_norms: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
    neighbor_mask_expanded = neighbor_mask | torch.eye(neighbor_mask.shape[0], device=neighbor_mask.device, dtype=torch.bool)
    neighbor_grads = grad_norms.unsqueeze(1) * neighbor_mask_expanded.float()
    neighbor_counts = neighbor_mask_expanded.sum(dim=1, keepdim=True)
    my_grad_expanded = grad_norms.unsqueeze(1)
    greater_than = (neighbor_grads > my_grad_expanded).float()
    rank_sum = greater_than.sum(dim=1)
    percentile = rank_sum / neighbor_counts.squeeze(1)
    return percentile

def vmap_ca_coherence(ca_patterns: torch.Tensor, neighbor_mask: torch.Tensor, eps: float=1e-8) -> torch.Tensor:
    N, *ca_dims = ca_patterns.shape
    flat_ca = ca_patterns.reshape(N, -1)
    ca_norm = torch.nn.functional.normalize(flat_ca, p=2, dim=-1)
    cosine_sim = torch.matmul(ca_norm, ca_norm.T)
    neighbor_sim = cosine_sim * neighbor_mask.float()
    neighbor_counts = neighbor_mask.sum(dim=1).clamp(min=1)
    coherence = neighbor_sim.sum(dim=1) / neighbor_counts
    return coherence

class SpatialStencil:

    def __init__(self, k_neighbors: int=5, distance_metric: str='euclidean', device: Optional[torch.device]=None):
        self.k_neighbors = k_neighbors
        self.distance_metric = distance_metric
        self.device = device or torch.device('cuda')

    def compute_all_contexts(self, behaviors: torch.Tensor, fitnesses: torch.Tensor, grad_norms: Optional[torch.Tensor]=None, ca_patterns: Optional[torch.Tensor]=None) -> dict:
        N = behaviors.shape[0]
        if N == 0:
            return {'relative_fitness': torch.tensor([], device=self.device), 'behavioral_divergence': torch.zeros((0, behaviors.shape[-1]), device=self.device), 'gradient_rank': torch.tensor([], device=self.device), 'ca_coherence': torch.tensor([], device=self.device)}
        distances = pairwise_behavioral_distance(behaviors, metric=self.distance_metric)
        neighbor_mask = topk_neighbors_mask(distances, k=min(self.k_neighbors, N - 1), include_self=False)
        relative_fitness = vmap_relative_fitness(fitnesses, neighbor_mask)
        behavioral_divergence = vmap_behavioral_divergence(behaviors, neighbor_mask)
        if grad_norms is not None:
            gradient_rank = vmap_gradient_rank(grad_norms, neighbor_mask)
        else:
            gradient_rank = torch.zeros(N, device=self.device)
        if ca_patterns is not None:
            ca_coherence = vmap_ca_coherence(ca_patterns, neighbor_mask)
        else:
            ca_coherence = torch.zeros(N, device=self.device)
        return {'relative_fitness': relative_fitness, 'behavioral_divergence': behavioral_divergence, 'gradient_rank': gradient_rank, 'ca_coherence': ca_coherence, 'neighbor_mask': neighbor_mask, 'distances': distances}
