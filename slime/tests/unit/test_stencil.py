import pytest
import torch
from slime.core.stencil import pairwise_behavioral_distance, topk_neighbors_mask, vmap_relative_fitness, vmap_behavioral_divergence, vmap_gradient_rank, vmap_attention_coherence, SpatialStencil

def test_pairwise_behavioral_distance_euclidean_constraint(constraint):
    behaviors = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], device='cuda')
    distances = pairwise_behavioral_distance(behaviors, metric='euclidean')
    constraint('Distance matrix shape is N×N', lambda: (distances.shape == (3, 3), distances.shape, (3, 3), {}))
    constraint('Self-distance is zero', lambda: (torch.allclose(torch.diag(distances), torch.zeros(3, device='cuda')), float(torch.diag(distances).max()), 0.0, {}))
    constraint('Distance [0,1] is 1.0', lambda: (torch.isclose(distances[0, 1], torch.tensor(1.0, device='cuda')), float(distances[0, 1]), 1.0, {}))
    constraint('Distance is symmetric', lambda: (torch.allclose(distances, distances.T), float((distances - distances.T).abs().max()), 0.0, {}))

def test_pairwise_behavioral_distance_cosine_constraint(constraint):
    behaviors = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], device='cuda')
    distances = pairwise_behavioral_distance(behaviors, metric='cosine')
    constraint('Distance matrix shape is N×N', lambda: (distances.shape == (3, 3), distances.shape, (3, 3), {}))
    constraint('Parallel vectors have zero cosine distance', lambda: (distances[0, 0] < 0.01, float(distances[0, 0]), '<0.01', {}))
    constraint('Orthogonal vectors have distance ~1.0', lambda: (0.9 < distances[0, 1] < 1.1, float(distances[0, 1]), '~1.0', {}))

def test_topk_neighbors_mask_constraint(constraint):
    distances = torch.tensor([[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 1.5, 2.5], [2.0, 1.5, 0.0, 1.0], [3.0, 2.5, 1.0, 0.0]], device='cuda')
    mask = topk_neighbors_mask(distances, k=2, include_self=False)
    constraint('Mask shape is N×N', lambda: (mask.shape == (4, 4), mask.shape, (4, 4), {}))
    constraint('Each row has exactly k neighbors', lambda: (mask.sum(dim=1).eq(2).all().item(), mask.sum(dim=1).tolist(), '[2, 2, 2, 2]', {}))
    constraint('Self is excluded from mask', lambda: (not torch.diag(mask).any().item(), 'self_excluded', 'self_excluded', {}))

def test_vmap_relative_fitness_above_mean_constraint(constraint):
    fitnesses = torch.tensor([0.9, 0.5, 0.5, 0.5, 0.5], device='cuda')
    mask = torch.tensor([[False, True, True, True, True], [True, False, True, True, True], [True, True, False, True, True], [True, True, True, False, True], [True, True, True, True, False]], device='cuda')
    z_scores = vmap_relative_fitness(fitnesses, mask)
    constraint('Z-score shape matches fitness shape', lambda: (z_scores.shape == fitnesses.shape, z_scores.shape, fitnesses.shape, {}))
    constraint('High fitness component has positive z-score', lambda: (z_scores[0] > 0.0, float(z_scores[0]), '>0', {}))
    constraint('Average fitness components have z-score near 0', lambda: (z_scores[1:].abs().mean() < 1.0, float(z_scores[1:].abs().mean()), '<1.0', {}))

def test_vmap_behavioral_divergence_constraint(constraint):
    behaviors = torch.tensor([[0.8, 0.8], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], device='cuda')
    mask = torch.tensor([[False, True, True, True], [True, False, True, True], [True, True, False, True], [True, True, True, False]], device='cuda')
    divergence = vmap_behavioral_divergence(behaviors, mask)
    constraint('Divergence shape is (N, D)', lambda: (divergence.shape == (4, 2), divergence.shape, (4, 2), {}))
    constraint('Outlier has positive divergence', lambda: ((divergence[0] > 0.0).all().item(), divergence[0].tolist(), '[>0, >0]', {}))
    constraint('Average components have near-zero divergence', lambda: (divergence[1:].abs().mean() < 0.2, float(divergence[1:].abs().mean()), '<0.2', {}))

def test_vmap_gradient_rank_constraint(constraint):
    grad_norms = torch.tensor([10.0, 1.0, 2.0, 3.0], device='cuda')
    mask = torch.tensor([[False, True, True, True], [True, False, True, True], [True, True, False, True], [True, True, True, False]], device='cuda')
    ranks = vmap_gradient_rank(grad_norms, mask)
    constraint('Rank shape matches grad_norms shape', lambda: (ranks.shape == grad_norms.shape, ranks.shape, grad_norms.shape, {}))
    constraint('Highest gradient has high rank', lambda: (ranks[0] > 0.7, float(ranks[0]), '>0.7', {}))
    constraint('Lowest gradient has low rank', lambda: (ranks[1] < 0.3, float(ranks[1]), '<0.3', {}))
    constraint('Ranks are in [0, 1]', lambda: ((ranks >= 0.0).all().item() and (ranks <= 1.0).all().item(), [float(ranks.min()), float(ranks.max())], '[0, 1]', {}))

def test_vmap_attention_coherence_constraint(constraint):
    shared_pattern = torch.randn(1, 1, 8, 8, device='cuda')
    attention_patterns = torch.stack([shared_pattern + torch.randn(1, 1, 8, 8, device='cuda') * 0.1 for _ in range(4)])
    mask = torch.tensor([[False, True, True, True], [True, False, True, True], [True, True, False, True], [True, True, True, False]], device='cuda')
    coherence = vmap_attention_coherence(attention_patterns, mask)
    constraint('Coherence shape matches N', lambda: (coherence.shape == (4,), coherence.shape, (4,), {}))
    constraint('High similarity patterns have high coherence', lambda: (coherence.mean() > 0.5, float(coherence.mean()), '>0.5', {}))
    constraint('Coherence is in [-1, 1]', lambda: ((coherence >= -1.0).all().item() and (coherence <= 1.0).all().item(), [float(coherence.min()), float(coherence.max())], '[-1, 1]', {}))

def test_spatial_stencil_integration_constraint(constraint):
    stencil = SpatialStencil(k_neighbors=3, distance_metric='euclidean', device='cuda')
    behaviors = torch.randn(10, 5, device='cuda')
    fitnesses = torch.rand(10, device='cuda')
    grad_norms = torch.rand(10, device='cuda')
    attention_patterns = torch.randn(10, 1, 1, 16, 16, device='cuda')
    results = stencil.compute_all_contexts(behaviors, fitnesses, grad_norms, attention_patterns)
    constraint('Results contain relative_fitness', lambda: ('relative_fitness' in results, 'relative_fitness' in results, True, {}))
    constraint('Results contain behavioral_divergence', lambda: ('behavioral_divergence' in results, 'behavioral_divergence' in results, True, {}))
    constraint('Results contain gradient_rank', lambda: ('gradient_rank' in results, 'gradient_rank' in results, True, {}))
    constraint('Results contain attention_coherence', lambda: ('attention_coherence' in results, 'attention_coherence' in results, True, {}))
    constraint('Relative fitness shape is (N,)', lambda: (results['relative_fitness'].shape == (10,), results['relative_fitness'].shape, (10,), {}))
    constraint('Behavioral divergence shape is (N, D)', lambda: (results['behavioral_divergence'].shape == (10, 5), results['behavioral_divergence'].shape, (10, 5), {}))
    constraint('All outputs on GPU', lambda: (results['relative_fitness'].device.type == 'cuda', results['relative_fitness'].device.type, 'cuda', {}))

def test_spatial_stencil_empty_population_constraint(constraint):
    stencil = SpatialStencil(k_neighbors=3, device='cuda')
    behaviors = torch.zeros((0, 5), device='cuda')
    fitnesses = torch.zeros(0, device='cuda')
    results = stencil.compute_all_contexts(behaviors, fitnesses)
    constraint('Empty population returns empty tensors', lambda: (results['relative_fitness'].shape[0] == 0, results['relative_fitness'].shape[0], 0, {}))
    constraint('Behavioral divergence has correct dims', lambda: (results['behavioral_divergence'].shape == (0, 5), results['behavioral_divergence'].shape, (0, 5), {}))

def test_spatial_stencil_single_component_constraint(constraint):
    stencil = SpatialStencil(k_neighbors=3, device='cuda')
    behaviors = torch.randn(1, 5, device='cuda')
    fitnesses = torch.tensor([0.8], device='cuda')
    results = stencil.compute_all_contexts(behaviors, fitnesses)
    constraint('Single component returns valid shape', lambda: (results['relative_fitness'].shape == (1,), results['relative_fitness'].shape, (1,), {}))
    constraint('Z-score is 0 (no neighbors)', lambda: (abs(results['relative_fitness'][0].item()) < 1e-5, float(results['relative_fitness'][0]), '~0', {}))
    constraint('Divergence is zeros', lambda: (results['behavioral_divergence'].abs().sum() < 1e-5, float(results['behavioral_divergence'].abs().sum()), '~0', {}))

def test_stencil_determinism_constraint(constraint):
    torch.manual_seed(42)
    behaviors1 = torch.randn(20, 5, device='cuda')
    fitnesses1 = torch.rand(20, device='cuda')
    stencil1 = SpatialStencil(k_neighbors=5, device='cuda')
    results1 = stencil1.compute_all_contexts(behaviors1, fitnesses1)
    torch.manual_seed(42)
    behaviors2 = torch.randn(20, 5, device='cuda')
    fitnesses2 = torch.rand(20, device='cuda')
    stencil2 = SpatialStencil(k_neighbors=5, device='cuda')
    results2 = stencil2.compute_all_contexts(behaviors2, fitnesses2)
    constraint('Deterministic relative fitness', lambda: (torch.allclose(results1['relative_fitness'], results2['relative_fitness']), float((results1['relative_fitness'] - results2['relative_fitness']).abs().max()), '~0', {}))
    constraint('Deterministic behavioral divergence', lambda: (torch.allclose(results1['behavioral_divergence'], results2['behavioral_divergence']), float((results1['behavioral_divergence'] - results2['behavioral_divergence']).abs().max()), '~0', {}))
    constraint('Deterministic neighbor mask', lambda: (torch.equal(results1['neighbor_mask'], results2['neighbor_mask']), 'masks_match', 'masks_match', {}))
