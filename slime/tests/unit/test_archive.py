import pytest
import torch
import numpy as np
from slime.memory.archive import CVTArchive

def test_warmup_phase_constraint(constraint):
    archive = CVTArchive(num_raw_metrics=10, min_dims=3, max_dims=3, num_centroids=50, low_rank_k=16, seed=42)
    rng = np.random.RandomState(42)
    for i in range(100):
        raw_metrics = rng.randn(10).astype(np.float32)
        archive.add_raw_metrics(raw_metrics)
    constraint('Archive collected 100 raw metric samples during warmup', lambda: (len(archive._raw_metrics_samples) == 100, len(archive._raw_metrics_samples), 100, {}))
    constraint('Archive not yet discovered (centroids=None)', lambda: (archive.centroids is None, archive.centroids, None, {'discovered': archive._discovered}))
    constraint('Behavioral dims not yet set', lambda: (archive.behavioral_dims is None, archive.behavioral_dims, None, {}))

def test_dimension_discovery_constraint(constraint):
    archive = CVTArchive(num_raw_metrics=10, min_dims=3, max_dims=3, num_centroids=50, low_rank_k=16, kmo_threshold=0.5, reconstruction_error_threshold=1.0, seed=42)
    rng = np.random.RandomState(42)
    latent = rng.randn(150, 3).astype(np.float32)
    mixing_matrix = rng.randn(3, 10).astype(np.float32)
    for i in range(150):
        raw_metrics = latent[i] @ mixing_matrix + rng.randn(10).astype(np.float32) * 0.1
        archive.add_raw_metrics(raw_metrics)
    success = archive.discover_dimensions()
    constraint('Dimension discovery succeeded', lambda: (success == True, success, True, {}))
    constraint('Behavioral dims set to target_dims after discovery', lambda: (archive.behavioral_dims == 3, archive.behavioral_dims, 3, {}))
    constraint('Centroids initialized with shape (num_centroids, target_dims)', lambda: (archive.centroids is not None and archive.centroids.shape == (50, 3), archive.centroids.shape if archive.centroids is not None else None, (50, 3), {}))
    constraint('Kernel PCA transform exists', lambda: (archive.kpca_transform is not None, type(archive.kpca_transform).__name__, 'KernelPCA', {}))
    constraint('Archive marked as discovered', lambda: (archive._discovered == True, archive._discovered, True, {}))

def test_add_after_discovery_constraint(constraint):
    archive = CVTArchive(num_raw_metrics=10, min_dims=3, max_dims=3, num_centroids=50, low_rank_k=16, kmo_threshold=0.5, reconstruction_error_threshold=1.0, seed=42)
    rng = np.random.RandomState(42)
    latent = rng.randn(150, 3).astype(np.float32)
    mixing_matrix = rng.randn(3, 10).astype(np.float32)
    for i in range(150):
        raw_metrics = latent[i] @ mixing_matrix + rng.randn(10).astype(np.float32) * 0.1
        archive.add_raw_metrics(raw_metrics)
    archive.discover_dimensions()
    torch.manual_seed(42)
    state_dict = {'W_q': torch.randn(64, 64, device=archive.device), 'W_k': torch.randn(64, 64, device=archive.device), 'W_v': torch.randn(64, 64, device=archive.device), 'W_o': torch.randn(64, 64, device=archive.device)}
    behavior = (0.1, 0.2, 0.3)
    fitness = 0.8
    added = archive.add(behavior, fitness, state_dict, generation=1, metadata={'test': True})
    constraint('Elite added successfully after discovery', lambda: (added == True, added, True, {}))
    constraint('Archive contains 1 elite', lambda: (len(archive.centroid_refs) == 1, len(archive.centroid_refs), 1, {}))
    centroid_id = archive._find_nearest_centroid(np.array(behavior))
    elite = archive.elites[centroid_id]
    constraint('Elite has correct fitness', lambda: (elite.fitness == 0.8, elite.fitness, 0.8, {}))
    constraint('Elite has correct behavior', lambda: (elite.behavior == (0.1, 0.2, 0.3), elite.behavior, (0.1, 0.2, 0.3), {}))

def test_low_rank_compression_constraint(constraint):
    archive = CVTArchive(num_raw_metrics=10, min_dims=3, max_dims=3, num_centroids=50, low_rank_k=32, kmo_threshold=0.5, reconstruction_error_threshold=1.0, seed=42)
    rng = np.random.RandomState(42)
    latent = rng.randn(150, 3).astype(np.float32)
    mixing_matrix = rng.randn(3, 10).astype(np.float32)
    for i in range(150):
        raw_metrics = latent[i] @ mixing_matrix + rng.randn(10).astype(np.float32) * 0.1
        archive.add_raw_metrics(raw_metrics)
    archive.discover_dimensions()
    torch.manual_seed(42)
    state_dict = {'W_q': torch.randn(64, 64, device=archive.device), 'W_k': torch.randn(64, 64, device=archive.device), 'W_v': torch.randn(64, 64, device=archive.device), 'W_o': torch.randn(64, 64, device=archive.device)}
    behavior = (0.1, 0.2, 0.3)
    fitness = 0.8
    archive.add(behavior, fitness, state_dict)
    centroid_id = archive._find_nearest_centroid(np.array(behavior))
    elite = archive.elites[centroid_id]
    reconstructed = elite.reconstruct_weights(archive)
    constraint('Reconstructed weights have same keys as original', lambda: (set(reconstructed.keys()) == set(state_dict.keys()), set(reconstructed.keys()), set(state_dict.keys()), {}))
    for key in state_dict.keys():
        orig = state_dict[key]
        recon = reconstructed[key]
        error = torch.norm(recon - orig) / torch.norm(orig)
        constraint(f'Reconstruction error for {key} < 0.5 (low-rank k=32)', lambda e=error, k=key: (e < 0.5, float(e), '<0.5', {'key': k, 'relative_error': float(e), 'orig_norm': float(torch.norm(orig)), 'recon_norm': float(torch.norm(recon))}))

def test_centroid_determinism_constraint(constraint):
    archive1 = CVTArchive(num_raw_metrics=10, min_dims=3, max_dims=3, num_centroids=50, seed=42)
    rng1 = np.random.RandomState(42)
    latent1 = rng1.randn(150, 3).astype(np.float32)
    mixing_matrix1 = rng1.randn(3, 10).astype(np.float32)
    for i in range(150):
        raw_metrics = latent1[i] @ mixing_matrix1 + rng1.randn(10).astype(np.float32) * 0.1
        archive1.add_raw_metrics(raw_metrics)
    archive1.discover_dimensions()
    archive2 = CVTArchive(num_raw_metrics=10, min_dims=3, max_dims=3, num_centroids=50, seed=42)
    rng2 = np.random.RandomState(42)
    latent2 = rng2.randn(150, 3).astype(np.float32)
    mixing_matrix2 = rng2.randn(3, 10).astype(np.float32)
    for i in range(150):
        raw_metrics = latent2[i] @ mixing_matrix2 + rng2.randn(10).astype(np.float32) * 0.1
        archive2.add_raw_metrics(raw_metrics)
    archive2.discover_dimensions()
    constraint('Deterministic centroid initialization (seed=42)', lambda: (np.allclose(archive1.centroids, archive2.centroids, atol=0.01), 'centroids_match', 'centroids_match', {'max_diff': float(np.max(np.abs(archive1.centroids - archive2.centroids))), 'mean_diff': float(np.mean(np.abs(archive1.centroids - archive2.centroids)))}))

def test_warmup_phase_errors_constraint(constraint):
    archive = CVTArchive(num_raw_metrics=10, min_dims=3, max_dims=3, num_centroids=50, kmo_threshold=0.5, reconstruction_error_threshold=1.0, seed=42)
    rng = np.random.RandomState(42)
    latent = rng.randn(150, 3).astype(np.float32)
    mixing_matrix = rng.randn(3, 10).astype(np.float32)
    for i in range(150):
        raw_metrics = latent[i] @ mixing_matrix + rng.randn(10).astype(np.float32) * 0.1
        archive.add_raw_metrics(raw_metrics)
    archive.discover_dimensions()
    try:
        archive.add_raw_metrics(rng.randn(10).astype(np.float32))
        constraint('Cannot add raw metrics after discovery', lambda: (False, 'allowed', 'ValueError', {}))
    except ValueError as e:
        constraint('Cannot add raw metrics after discovery', lambda: (True, 'ValueError', 'ValueError', {'error_msg': str(e)}))

def test_add_before_discovery_errors_constraint(constraint):
    archive = CVTArchive(num_raw_metrics=10, min_dims=3, max_dims=3, num_centroids=50, seed=42)
    torch.manual_seed(42)
    state_dict = {'W_q': torch.randn(64, 64, device=archive.device)}
    try:
        archive.add((0.1, 0.2, 0.3), 0.8, state_dict)
        constraint('Cannot add elites before discovery', lambda: (False, 'allowed', 'ValueError', {}))
    except ValueError as e:
        constraint('Cannot add elites before discovery', lambda: (True, 'ValueError', 'ValueError', {'error_msg': str(e)}))
