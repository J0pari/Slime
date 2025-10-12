"""Unit tests for CVT-MAP-Elites archive with causal constraint verification."""
import pytest
import torch
import numpy as np
from slime.memory.archive import CVTArchive


def test_archive_initialization_constraint(constraint):
    """Verify CVT archive initializes with correct structure."""
    archive = CVTArchive(
        behavioral_dims=5,
        num_centroids=100,
        low_rank_k=32,
        kmo_threshold=0.6,
        seed=42
    )

    # Initialize centroids
    archive.initialize_centroids()

    # Constraint: Centroids must exist with correct shape
    result = constraint(
        "Archive centroids exist with shape (num_centroids, behavioral_dims)",
        lambda: (
            archive.centroids is not None and archive.centroids.shape == (100, 5),
            archive.centroids.shape if archive.centroids is not None else None,
            (100, 5),
            {'centroids_dtype': str(archive.centroids.dtype) if archive.centroids is not None else None}
        )
    )

    # Constraint: Archive starts empty
    constraint(
        "Archive starts with zero elites",
        lambda: (
            len(archive._archive) == 0,
            len(archive._archive),
            0,
            {}
        )
    )

    assert result.right_value is not None


def test_centroid_determinism_constraint(constraint):
    """Verify centroid initialization is deterministic with same seed."""
    archive1 = CVTArchive(behavioral_dims=3, num_centroids=50, seed=42)
    archive1.initialize_centroids()

    archive2 = CVTArchive(behavioral_dims=3, num_centroids=50, seed=42)
    archive2.initialize_centroids()

    # Constraint: Same seed produces identical centroids
    constraint(
        "Deterministic centroid initialization (seed=42)",
        lambda: (
            np.allclose(archive1.centroids, archive2.centroids),
            'centroids_match',
            'centroids_match',
            {
                'max_diff': float(np.max(np.abs(archive1.centroids - archive2.centroids))),
                'mean_diff': float(np.mean(np.abs(archive1.centroids - archive2.centroids)))
            }
        )
    )

    # Constraint: Different seeds produce different centroids
    archive3 = CVTArchive(behavioral_dims=3, num_centroids=50, seed=99)
    archive3.initialize_centroids()

    constraint(
        "Different seeds produce different centroids",
        lambda: (
            not np.allclose(archive1.centroids, archive3.centroids),
            'centroids_differ',
            'centroids_differ',
            {
                'max_diff': float(np.max(np.abs(archive1.centroids - archive3.centroids))),
                'similarity': float(np.mean(np.abs(archive1.centroids - archive3.centroids)))
            }
        )
    )


def test_nearest_centroid_constraint(constraint):
    """Verify nearest centroid search returns valid index."""
    archive = CVTArchive(behavioral_dims=5, num_centroids=100, seed=42)
    archive.initialize_centroids()

    behavior = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    centroid_id = archive._find_nearest_centroid(behavior)

    # Constraint: Centroid ID in valid range
    constraint(
        "Nearest centroid ID is in valid range [0, num_centroids)",
        lambda: (
            0 <= centroid_id < 100,
            centroid_id,
            'range(0, 100)',
            {'centroid_id': centroid_id, 'is_int': isinstance(centroid_id, (int, np.integer))}
        )
    )

    # Constraint: Centroid ID is truly nearest
    distances = np.linalg.norm(archive.centroids - behavior, axis=1)
    nearest_id = int(np.argmin(distances))

    constraint(
        "Returned centroid is actually nearest by Euclidean distance",
        lambda: (
            centroid_id == nearest_id,
            centroid_id,
            nearest_id,
            {
                'distance_to_returned': float(distances[centroid_id]),
                'min_distance': float(distances[nearest_id])
            }
        )
    )


def test_low_rank_compression_constraint(constraint):
    """Verify low-rank compression and reconstruction."""
    archive = CVTArchive(behavioral_dims=3, num_centroids=10, low_rank_k=8, seed=42)
    archive.initialize_centroids()

    state_dict = {
        'W_q': torch.randn(64, 64),
        'W_k': torch.randn(64, 64),
        'W_v': torch.randn(64, 64),
        'W_o': torch.randn(64, 64),
    }

    behavior = (0.1, 0.2, 0.3)
    fitness = 0.8

    archive.add(behavior, fitness, state_dict)

    centroid_id = archive._find_nearest_centroid(np.array(behavior))
    elite = archive._archive[centroid_id]

    # Constraint: Elite was stored
    constraint(
        "Elite exists at nearest centroid after add()",
        lambda: (
            centroid_id in archive._archive,
            'elite_exists',
            'elite_exists',
            {'centroid_id': centroid_id, 'elite_fitness': elite.fitness}
        )
    )

    # Constraint: Weights can be reconstructed
    reconstructed = elite.reconstruct_weights()

    constraint(
        "Reconstructed weights have same keys as original",
        lambda: (
            set(reconstructed.keys()) == set(state_dict.keys()),
            set(reconstructed.keys()),
            set(state_dict.keys()),
            {}
        )
    )

    # Constraint: Reconstruction error is bounded
    for key in state_dict.keys():
        orig = state_dict[key]
        recon = reconstructed[key]
        error = torch.norm(recon - orig) / torch.norm(orig)

        constraint(
            f"Reconstruction error for {key} < 0.5 (low-rank k=8)",
            lambda e=error, k=key: (
                e < 0.5,
                float(e),
                '<0.5',
                {
                    'key': k,
                    'relative_error': float(e),
                    'orig_norm': float(torch.norm(orig)),
                    'recon_norm': float(torch.norm(recon))
                }
            )
        )


def test_fitness_replacement_constraint(constraint):
    """Verify archive replaces lower-fitness elites."""
    archive = CVTArchive(behavioral_dims=2, num_centroids=10, seed=42)
    archive.initialize_centroids()

    behavior = (0.5, 0.5)
    state_dict_low = {'W': torch.randn(32, 32)}
    state_dict_high = {'W': torch.randn(32, 32)}

    # Add low fitness elite
    archive.add(behavior, 0.5, state_dict_low)
    centroid_id = archive._find_nearest_centroid(np.array(behavior))

    # Constraint: Low fitness elite stored
    constraint(
        "Archive stores low-fitness elite initially",
        lambda: (
            archive._archive[centroid_id].fitness == 0.5,
            archive._archive[centroid_id].fitness,
            0.5,
            {'centroid_id': centroid_id}
        )
    )

    # Add high fitness elite at same behavior
    archive.add(behavior, 0.9, state_dict_high)

    # Constraint: Archive replaced with higher fitness
    constraint(
        "Archive replaces with higher-fitness elite",
        lambda: (
            archive._archive[centroid_id].fitness == 0.9,
            archive._archive[centroid_id].fitness,
            0.9,
            {'centroid_id': centroid_id, 'replacement_occurred': True}
        )
    )

    # Add lower fitness again
    archive.add(behavior, 0.4, state_dict_low)

    # Constraint: Archive keeps higher fitness
    constraint(
        "Archive rejects lower-fitness elite",
        lambda: (
            archive._archive[centroid_id].fitness == 0.9,
            archive._archive[centroid_id].fitness,
            0.9,
            {'attempted_fitness': 0.4, 'kept_fitness': 0.9}
        )
    )


def test_max_fitness_constraint(constraint):
    """Verify max_fitness computation."""
    archive = CVTArchive(behavioral_dims=2, num_centroids=10, seed=42)
    archive.initialize_centroids()

    # Constraint: Empty archive has -inf max fitness
    constraint(
        "Empty archive max_fitness = -inf",
        lambda: (
            archive.max_fitness() == float('-inf'),
            archive.max_fitness(),
            float('-inf'),
            {'num_elites': len(archive._archive)}
        )
    )

    # Add elites
    archive.add((0.1, 0.1), 0.5, {'W': torch.randn(16, 16)})
    archive.add((0.9, 0.9), 0.9, {'W': torch.randn(16, 16)})
    archive.add((0.5, 0.5), 0.3, {'W': torch.randn(16, 16)})

    # Constraint: Max fitness is maximum across all elites
    constraint(
        "max_fitness returns highest fitness across archive",
        lambda: (
            archive.max_fitness() == 0.9,
            archive.max_fitness(),
            0.9,
            {
                'num_elites': len(archive._archive),
                'all_fitnesses': [e.fitness for e in archive._archive.values()]
            }
        )
    )


def test_coverage_constraint(constraint):
    """Verify coverage metric (fraction of centroids occupied)."""
    archive = CVTArchive(behavioral_dims=2, num_centroids=10, seed=42)
    archive.initialize_centroids()

    # Constraint: Empty archive has 0% coverage
    constraint(
        "Empty archive has zero coverage",
        lambda: (
            archive.coverage() == 0.0,
            archive.coverage(),
            0.0,
            {}
        )
    )

    # Add one elite
    archive.add((0.1, 0.1), 0.5, {'W': torch.randn(16, 16)})

    # Constraint: Coverage increases with elites
    cov1 = archive.coverage()
    constraint(
        "Coverage = 1/10 = 0.1 after one elite",
        lambda: (
            cov1 == 0.1,
            cov1,
            0.1,
            {'num_elites': len(archive._archive), 'num_centroids': 10}
        )
    )

    # Add another elite
    archive.add((0.9, 0.9), 0.6, {'W': torch.randn(16, 16)})
    cov2 = archive.coverage()

    constraint(
        "Coverage increases after second elite",
        lambda: (
            cov2 >= cov1,
            cov2,
            f'>={cov1}',
            {'cov_before': cov1, 'cov_after': cov2}
        )
    )


def test_memory_efficiency_constraint(constraint):
    """Verify low-rank storage reduces memory usage."""
    archive_full = CVTArchive(behavioral_dims=2, num_centroids=10, low_rank_k=64, seed=42)
    archive_full.initialize_centroids()

    archive_compressed = CVTArchive(behavioral_dims=2, num_centroids=10, low_rank_k=8, seed=42)
    archive_compressed.initialize_centroids()

    state_dict = {'W': torch.randn(64, 64)}
    behavior = (0.5, 0.5)

    archive_full.add(behavior, 0.8, state_dict)
    archive_compressed.add(behavior, 0.8, state_dict)

    centroid_id_full = archive_full._find_nearest_centroid(np.array(behavior))
    centroid_id_comp = archive_compressed._find_nearest_centroid(np.array(behavior))

    elite_full = archive_full._archive[centroid_id_full]
    elite_compressed = archive_compressed._archive[centroid_id_comp]

    full_params = elite_full.weights_u['W'].numel() + elite_full.weights_v['W'].numel()
    compressed_params = elite_compressed.weights_u['W'].numel() + elite_compressed.weights_v['W'].numel()

    # Constraint: Compression reduces parameter count
    constraint(
        "Low-rank compression reduces parameter count",
        lambda: (
            compressed_params < full_params,
            compressed_params,
            f'<{full_params}',
            {
                'full_params': full_params,
                'compressed_params': compressed_params,
                'compression_ratio': full_params / compressed_params
            }
        )
    )

    # Constraint: Compression ratio is significant (>2x)
    compression_ratio = full_params / compressed_params
    constraint(
        "Compression ratio > 2x for k=8 vs k=64",
        lambda: (
            compression_ratio > 2.0,
            compression_ratio,
            '>2.0',
            {
                'compression_ratio': float(compression_ratio),
                'full_rank_k': 64,
                'compressed_rank_k': 8
            }
        )
    )
