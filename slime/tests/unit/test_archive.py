import pytest
import torch
import numpy as np
from slime.memory.archive import CVTArchive
from slime.config.dimensions import ArchitectureConfig, TINY

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
    archive = CVTArchive(num_raw_metrics=10, min_dims=3, max_dims=3, num_centroids=50, low_rank_k=16, trustworthiness_threshold=0.5, reconstruction_error_threshold=1.0, seed=42)
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
    archive = CVTArchive(num_raw_metrics=10, min_dims=3, max_dims=3, num_centroids=50, low_rank_k=16, trustworthiness_threshold=0.5, reconstruction_error_threshold=1.0, seed=42)
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
    archive = CVTArchive(num_raw_metrics=10, min_dims=3, max_dims=3, num_centroids=50, low_rank_k=32, trustworthiness_threshold=0.5, reconstruction_error_threshold=1.0, seed=42)
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
    archive = CVTArchive(num_raw_metrics=10, min_dims=3, max_dims=3, num_centroids=50, trustworthiness_threshold=0.5, reconstruction_error_threshold=1.0, seed=42)
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

def test_elite_replacement_consistency_constraint(constraint):
    archive = CVTArchive(num_raw_metrics=10, min_dims=3, max_dims=3, num_centroids=50, low_rank_k=16, trustworthiness_threshold=0.5, reconstruction_error_threshold=1.0, seed=42)
    rng = np.random.RandomState(42)
    latent = rng.randn(150, 3).astype(np.float32)
    mixing_matrix = rng.randn(3, 10).astype(np.float32)
    for i in range(150):
        raw_metrics = latent[i] @ mixing_matrix + rng.randn(10).astype(np.float32) * 0.1
        archive.add_raw_metrics(raw_metrics)
    archive.discover_dimensions()
    torch.manual_seed(42)
    state_dict1 = {'W_q': torch.randn(64, 64, device=archive.device)}
    behavior = (0.1, 0.2, 0.3)
    added1 = archive.add(behavior, 0.5, state_dict1, generation=1)
    constraint('First elite added to empty centroid', lambda: (added1 == True, added1, True, {}))
    centroid_id = archive._find_nearest_centroid(np.array(behavior))
    elite1 = archive.elites[centroid_id]
    sha1 = elite1.elite_sha
    constraint('First elite has fitness 0.5', lambda: (elite1.fitness == 0.5, elite1.fitness, 0.5, {}))
    torch.manual_seed(43)
    state_dict2 = {'W_q': torch.randn(64, 64, device=archive.device)}
    added2 = archive.add(behavior, 0.7, state_dict2, generation=2)
    constraint('Second elite replaces first (higher fitness)', lambda: (added2 == True, added2, True, {}))
    elite2 = archive.elites[centroid_id]
    sha2 = elite2.elite_sha
    constraint('Second elite has fitness 0.7', lambda: (elite2.fitness == 0.7, elite2.fitness, 0.7, {}))
    constraint('Second elite has different SHA (different weights)', lambda: (sha2 != sha1, sha2, f'!={sha1}', {}))
    constraint('Archive still contains exactly 1 elite', lambda: (len(archive.centroid_refs) == 1, len(archive.centroid_refs), 1, {}))
    torch.manual_seed(44)
    state_dict3 = {'W_q': torch.randn(64, 64, device=archive.device)}
    added3 = archive.add(behavior, 0.6, state_dict3, generation=3)
    constraint('Third elite rejected (lower fitness than current)', lambda: (added3 == False, added3, False, {}))
    elite3 = archive.elites[centroid_id]
    constraint('Elite still has fitness 0.7 after rejection', lambda: (elite3.fitness == 0.7, elite3.fitness, 0.7, {}))
    constraint('Elite still has SHA from generation 2', lambda: (elite3.elite_sha == sha2, elite3.elite_sha, sha2, {}))

def test_centroid_voronoi_partitioning_constraint(constraint):
    archive = CVTArchive(num_raw_metrics=10, min_dims=2, max_dims=2, num_centroids=4, low_rank_k=16, trustworthiness_threshold=0.5, reconstruction_error_threshold=1.0, seed=42)
    rng = np.random.RandomState(42)
    latent = rng.randn(150, 2).astype(np.float32)
    mixing_matrix = rng.randn(2, 10).astype(np.float32)
    for i in range(150):
        raw_metrics = latent[i] @ mixing_matrix + rng.randn(10).astype(np.float32) * 0.1
        archive.add_raw_metrics(raw_metrics)
    archive.discover_dimensions()
    torch.manual_seed(42)
    state_dict = {'W': torch.randn(16, 16, device=archive.device)}
    behaviors = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)]
    for i, behavior in enumerate(behaviors):
        archive.add(behavior, 0.5 + i * 0.1, state_dict, generation=i)
    constraint('Each behavior maps to exactly one centroid', lambda: (len(set([archive._find_nearest_centroid(np.array(b)) for b in behaviors])) <= 4, True, True, {}))
    for behavior in behaviors:
        centroid_id = archive._find_nearest_centroid(np.array(behavior))
        distances = [np.linalg.norm(np.array(behavior) - archive.centroids[i]) for i in range(archive.num_centroids)]
        min_dist_id = np.argmin(distances)
        constraint(f'Behavior {behavior} maps to nearest centroid', lambda cid=centroid_id, mid=min_dist_id: (cid == mid, cid, mid, {}))

def test_content_addressable_deduplication_constraint(constraint):
    archive = CVTArchive(num_raw_metrics=10, min_dims=3, max_dims=3, num_centroids=50, low_rank_k=16, trustworthiness_threshold=0.5, reconstruction_error_threshold=1.0, seed=42)
    rng = np.random.RandomState(42)
    latent = rng.randn(150, 3).astype(np.float32)
    mixing_matrix = rng.randn(3, 10).astype(np.float32)
    for i in range(150):
        raw_metrics = latent[i] @ mixing_matrix + rng.randn(10).astype(np.float32) * 0.1
        archive.add_raw_metrics(raw_metrics)
    archive.discover_dimensions()
    torch.manual_seed(42)
    state_dict = {'W_q': torch.randn(64, 64, device=archive.device), 'W_k': torch.randn(64, 64, device=archive.device)}
    behavior1 = (0.1, 0.2, 0.3)
    behavior2 = (0.9, 0.8, 0.7)
    archive.add(behavior1, 0.8, state_dict, generation=1)
    archive.add(behavior2, 0.9, state_dict, generation=2)
    centroid_id1 = archive._find_nearest_centroid(np.array(behavior1))
    centroid_id2 = archive._find_nearest_centroid(np.array(behavior2))
    elite1 = archive.elites[centroid_id1]
    elite2 = archive.elites[centroid_id2]
    constraint('Both elites reference same SHA (identical weights)', lambda: (elite1.elite_sha == elite2.elite_sha, elite1.elite_sha, elite2.elite_sha, {}))
    objects_before = len(archive.object_store)
    torch.manual_seed(42)
    state_dict_dup = {'W_q': torch.randn(64, 64, device=archive.device), 'W_k': torch.randn(64, 64, device=archive.device)}
    behavior3 = (0.5, 0.5, 0.5)
    archive.add(behavior3, 0.85, state_dict_dup, generation=3)
    objects_after = len(archive.object_store)
    constraint('No new objects stored (deduplicated by SHA)', lambda: (objects_after == objects_before, objects_after, objects_before, {}))

def test_metadata_preservation_constraint(constraint):
    archive = CVTArchive(num_raw_metrics=10, min_dims=3, max_dims=3, num_centroids=50, low_rank_k=16, trustworthiness_threshold=0.5, reconstruction_error_threshold=1.0, seed=42)
    rng = np.random.RandomState(42)
    latent = rng.randn(150, 3).astype(np.float32)
    mixing_matrix = rng.randn(3, 10).astype(np.float32)
    for i in range(150):
        raw_metrics = latent[i] @ mixing_matrix + rng.randn(10).astype(np.float32) * 0.1
        archive.add_raw_metrics(raw_metrics)
    archive.discover_dimensions()
    torch.manual_seed(42)
    state_dict = {'W': torch.randn(32, 32, device=archive.device)}
    behavior = (0.1, 0.2, 0.3)
    metadata = {'experiment': 'test', 'mutation_type': 'crossover', 'parent_ids': [1, 2], 'custom_field': 42}
    archive.add(behavior, 0.8, state_dict, generation=5, metadata=metadata)
    centroid_id = archive._find_nearest_centroid(np.array(behavior))
    elite = archive.elites[centroid_id]
    constraint('Elite has correct generation', lambda: (elite.generation == 5, elite.generation, 5, {}))
    constraint('Elite metadata has experiment field', lambda: (elite.metadata.get('experiment') == 'test', elite.metadata.get('experiment'), 'test', {}))
    constraint('Elite metadata has mutation_type field', lambda: (elite.metadata.get('mutation_type') == 'crossover', elite.metadata.get('mutation_type'), 'crossover', {}))
    constraint('Elite metadata has parent_ids list', lambda: (elite.metadata.get('parent_ids') == [1, 2], elite.metadata.get('parent_ids'), [1, 2], {}))
    constraint('Elite metadata has custom_field', lambda: (elite.metadata.get('custom_field') == 42, elite.metadata.get('custom_field'), 42, {}))

def test_behavioral_dimension_bounds_constraint(constraint):
    archive = CVTArchive(num_raw_metrics=10, min_dims=2, max_dims=5, num_centroids=50, low_rank_k=16, trustworthiness_threshold=0.5, reconstruction_error_threshold=2.0, seed=42)
    rng = np.random.RandomState(42)
    latent = rng.randn(150, 3).astype(np.float32)
    mixing_matrix = rng.randn(3, 10).astype(np.float32)
    for i in range(150):
        raw_metrics = latent[i] @ mixing_matrix + rng.randn(10).astype(np.float32) * 0.1
        archive.add_raw_metrics(raw_metrics)
    success = archive.discover_dimensions()
    constraint('Discovery succeeded with min_dims=2, max_dims=5', lambda: (success == True, success, True, {}))
    constraint('Behavioral dims within bounds', lambda: (2 <= archive.behavioral_dims <= 5, archive.behavioral_dims, 'in [2,5]', {}))
    constraint('Centroids have correct dimension', lambda: (archive.centroids.shape[1] == archive.behavioral_dims, archive.centroids.shape[1], archive.behavioral_dims, {}))

def test_archive_deterministic_across_operations_constraint(constraint):
    def build_and_fill_archive(seed):
        archive = CVTArchive(num_raw_metrics=10, min_dims=3, max_dims=3, num_centroids=50, low_rank_k=16, trustworthiness_threshold=0.5, reconstruction_error_threshold=1.0, seed=seed)
        rng = np.random.RandomState(seed)
        latent = rng.randn(150, 3).astype(np.float32)
        mixing_matrix = rng.randn(3, 10).astype(np.float32)
        for i in range(150):
            raw_metrics = latent[i] @ mixing_matrix + rng.randn(10).astype(np.float32) * 0.1
            archive.add_raw_metrics(raw_metrics)
        archive.discover_dimensions()
        torch.manual_seed(seed)
        for i in range(10):
            state_dict = {'W': torch.randn(32, 32, device=archive.device)}
            behavior = tuple(rng.randn(3).astype(np.float32))
            fitness = float(rng.rand())
            archive.add(behavior, fitness, state_dict, generation=i)
        return archive
    archive1 = build_and_fill_archive(123)
    archive2 = build_and_fill_archive(123)
    constraint('Same number of elites', lambda: (len(archive1.centroid_refs) == len(archive2.centroid_refs), len(archive1.centroid_refs), len(archive2.centroid_refs), {}))
    for centroid_id in archive1.centroid_refs:
        if centroid_id in archive2.centroid_refs:
            elite1 = archive1.elites[centroid_id]
            elite2 = archive2.elites[centroid_id]
            constraint(f'Elite at centroid {centroid_id} has same SHA', lambda e1=elite1, e2=elite2: (e1.elite_sha == e2.elite_sha, e1.elite_sha, e2.elite_sha, {}))
            constraint(f'Elite at centroid {centroid_id} has same fitness', lambda e1=elite1, e2=elite2: (abs(e1.fitness - e2.fitness) < 1e-6, e1.fitness, e2.fitness, {}))
            constraint(f'Elite at centroid {centroid_id} has same generation', lambda e1=elite1, e2=elite2: (e1.generation == e2.generation, e1.generation, e2.generation, {}))
