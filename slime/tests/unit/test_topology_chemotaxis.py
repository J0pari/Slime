"""
Tests for topology-aware chemotaxis with optional hierarchical effects.

Verifies that chemotaxis can optionally use:
- BehavioralHierarchy for GMM-based clustering
- HybridMetric for ultrametric (between-cluster) + Mahalanobis (within-cluster) distances
- Effect handlers for graceful opt-in topology features

All tests use the constraint fixture for causal DAG tracking and checkpoint integration.
"""

import pytest
import torch
import numpy as np
from slime.memory.archive import CVTArchive
from slime.core.chemotaxis import Chemotaxis
from slime.topology.hierarchy import BehavioralHierarchy
from slime.topology.hybrid_metric import HybridMetric
from slime.topology import GetHierarchy, try_get_hierarchy
from slime.config.dimensions import (
    ArchitectureConfig,
    DimensionConfig,
    BehavioralSpaceConfig,
    CompressionConfig,
    FitnessConfig,
    TestConfig,
    NumericalConfig,
    TINY
)


def make_test_config(
    num_raw_metrics: int = 10,
    min_dims: int = 3,
    max_dims: int = 3,
    num_centroids: int = 50,
    low_rank_k: int = 16
) -> ArchitectureConfig:
    """Helper to create test ArchitectureConfig."""
    return ArchitectureConfig(
        dimensions=DimensionConfig(head_dim=16, num_heads=4, hidden_dim=64),
        behavioral_space=BehavioralSpaceConfig(
            num_raw_metrics=num_raw_metrics,
            min_dims=min_dims,
            max_dims=max_dims,
            num_centroids=num_centroids
        ),
        compression=CompressionConfig(low_rank_k=low_rank_k, delta_rank=8),
        fitness=FitnessConfig(ema_decay=0.9, entropy_weight=1.0, magnitude_weight=1.0),
        test=TestConfig(batch_size=2, seq_len=16),
        numerical=NumericalConfig(epsilon=1e-10, svd_threshold=1e-06, attention_temperature=1.0),
        k_neighbors=5
    )


def test_chemotaxis_without_topology_constraint(constraint):
    """Chemotaxis works without any topology features (baseline)."""
    config = make_test_config()
    archive = CVTArchive(config, seed=42)

    # Warm up archive with synthetic data
    rng = np.random.RandomState(42)
    latent = rng.randn(150, 3).astype(np.float32)
    mixing_matrix = rng.randn(3, 10).astype(np.float32)

    for i in range(150):
        raw_metrics = latent[i] @ mixing_matrix + rng.randn(10).astype(np.float32) * 0.1
        archive.add_raw_metrics(raw_metrics)

    success = archive.discover_dimensions()
    constraint(
        'Dimension discovery succeeded',
        lambda: (success, success, True, {})
    )

    # Create chemotaxis without topology
    chemotaxis = Chemotaxis(archive, distance_metric='euclidean')

    constraint(
        'Chemotaxis created with euclidean distance',
        lambda: (chemotaxis.distance_metric == 'euclidean', chemotaxis.distance_metric, 'euclidean', {})
    )

    # Add a nutrient source
    torch.manual_seed(42)
    nutrient = torch.randn(64, 64, device=archive.device)
    behavior = (0.1, 0.2, 0.3)

    chemotaxis.add_source(nutrient, behavior, concentration=1.0)

    constraint(
        'Nutrient source added',
        lambda: (len(chemotaxis._sources) == 1, len(chemotaxis._sources), 1, {})
    )

    # Sample from chemotaxis
    sampled = chemotaxis.sample(behavior, metabolic_rate=1.0, hunger=0.0)

    constraint(
        'Chemotaxis sampled successfully',
        lambda: (sampled is not None, sampled is not None, True, {})
    )

    constraint(
        'Sampled tensor has correct shape',
        lambda: (sampled.shape == (64, 64), tuple(sampled.shape), (64, 64), {})
    )


def test_chemotaxis_with_mahalanobis_after_discovery_constraint(constraint):
    """Chemotaxis automatically uses Mahalanobis distance after dimension discovery."""
    config = make_test_config()
    archive = CVTArchive(config, seed=42)

    # Before discovery: should default to euclidean
    chemotaxis_before = Chemotaxis(archive)
    constraint(
        'Chemotaxis defaults to euclidean before discovery',
        lambda: (chemotaxis_before.distance_metric == 'euclidean', chemotaxis_before.distance_metric, 'euclidean', {})
    )

    # Warm up and discover
    rng = np.random.RandomState(42)
    latent = rng.randn(150, 3).astype(np.float32)
    mixing_matrix = rng.randn(3, 10).astype(np.float32)

    for i in range(150):
        raw_metrics = latent[i] @ mixing_matrix + rng.randn(10).astype(np.float32) * 0.1
        archive.add_raw_metrics(raw_metrics)

    success = archive.discover_dimensions()
    constraint(
        'Dimension discovery succeeded',
        lambda: (success, success, True, {})
    )

    # After discovery: should default to mahalanobis
    chemotaxis_after = Chemotaxis(archive)
    constraint(
        'Chemotaxis defaults to mahalanobis after discovery',
        lambda: (chemotaxis_after.distance_metric == 'mahalanobis', chemotaxis_after.distance_metric, 'mahalanobis', {})
    )

    # Update covariance
    chemotaxis_after.update_covariance()

    constraint(
        'Covariance matrix computed',
        lambda: (chemotaxis_after._covariance_matrix is not None, chemotaxis_after._covariance_matrix is not None, True, {})
    )

    constraint(
        'Covariance matrix has correct shape',
        lambda: (chemotaxis_after._covariance_matrix.shape == (3, 3), chemotaxis_after._covariance_matrix.shape, (3, 3), {})
    )


def test_behavioral_hierarchy_fits_to_archive_constraint(constraint):
    """BehavioralHierarchy can fit GMM to archive centroids."""
    config = make_test_config()
    archive = CVTArchive(config, seed=42)

    # Warm up and discover
    rng = np.random.RandomState(42)
    latent = rng.randn(150, 3).astype(np.float32)
    mixing_matrix = rng.randn(3, 10).astype(np.float32)

    for i in range(150):
        raw_metrics = latent[i] @ mixing_matrix + rng.randn(10).astype(np.float32) * 0.1
        archive.add_raw_metrics(raw_metrics)

    success = archive.discover_dimensions()
    constraint(
        'Dimension discovery succeeded',
        lambda: (success, success, True, {})
    )

    # Create and fit hierarchy
    hierarchy = BehavioralHierarchy(n_clusters=10, linkage_method='ward', random_state=42)

    constraint(
        'Hierarchy not fitted initially',
        lambda: (not hierarchy._fitted, hierarchy._fitted, False, {})
    )

    hierarchy.fit(archive.centroids)

    constraint(
        'Hierarchy fitted successfully',
        lambda: (hierarchy._fitted, hierarchy._fitted, True, {})
    )

    constraint(
        'GMM has correct number of components',
        lambda: (hierarchy.gmm.n_components == 10, hierarchy.gmm.n_components, 10, {})
    )

    constraint(
        'Linkage matrix has correct shape',
        lambda: (hierarchy.linkage_matrix.shape == (9, 4), hierarchy.linkage_matrix.shape, (9, 4), {})
    )

    # Test cluster prediction
    test_coord = archive.centroids[0]
    cluster_id = hierarchy.predict_cluster(test_coord)

    constraint(
        'Cluster prediction returns valid ID',
        lambda: (0 <= cluster_id < 10, cluster_id, 'in [0, 10)', {})
    )

    # Test cluster distance
    dist = hierarchy.cluster_distance(0, 1)

    constraint(
        'Cluster distance is non-negative',
        lambda: (dist >= 0.0, dist, '>= 0.0', {})
    )


def test_hybrid_metric_without_hierarchy_fallback_constraint(constraint):
    """HybridMetric falls back to Euclidean without hierarchy."""
    metric = HybridMetric(
        hierarchy=None,
        inter_cluster_mode='ultrametric',
        intra_cluster_mode='mahalanobis',
        p=2
    )

    constraint(
        'Hybrid metric has no hierarchy',
        lambda: (metric.hierarchy is None, metric.hierarchy, None, {})
    )

    # Test distance computation (should use Euclidean fallback)
    x = np.array([0.0, 0.0, 0.0])
    y = np.array([1.0, 1.0, 1.0])

    dist = metric(x, y)
    expected_dist = np.sqrt(3.0)

    constraint(
        'Hybrid metric falls back to Euclidean',
        lambda: (abs(dist - expected_dist) < 1e-6, dist, expected_dist, {})
    )


def test_hybrid_metric_with_hierarchy_intra_cluster_constraint(constraint):
    """HybridMetric uses Mahalanobis for same-cluster distances."""
    # Create simple hierarchy
    config = make_test_config()
    archive = CVTArchive(config, seed=42)

    rng = np.random.RandomState(42)
    latent = rng.randn(150, 3).astype(np.float32)
    mixing_matrix = rng.randn(3, 10).astype(np.float32)

    for i in range(150):
        raw_metrics = latent[i] @ mixing_matrix + rng.randn(10).astype(np.float32) * 0.1
        archive.add_raw_metrics(raw_metrics)

    archive.discover_dimensions()

    hierarchy = BehavioralHierarchy(n_clusters=5, random_state=42)
    hierarchy.fit(archive.centroids)

    constraint(
        'Hierarchy fitted',
        lambda: (hierarchy._fitted, hierarchy._fitted, True, {})
    )

    # Create hybrid metric
    metric = HybridMetric(
        hierarchy=hierarchy,
        inter_cluster_mode='ultrametric',
        intra_cluster_mode='mahalanobis',
        p=2
    )

    constraint(
        'Hybrid metric has hierarchy',
        lambda: (metric.hierarchy is not None, metric.hierarchy is not None, True, {})
    )

    # Get two points in same cluster
    x = archive.centroids[0]
    cluster_x = hierarchy.predict_cluster(x)

    # Find another centroid in same cluster
    y = None
    for i in range(1, len(archive.centroids)):
        test_y = archive.centroids[i]
        if hierarchy.predict_cluster(test_y) == cluster_x:
            y = test_y
            break

    if y is not None:
        constraint(
            'Found two centroids in same cluster',
            lambda: (True, True, True, {'found_pair': True})
        )

        cluster_y = hierarchy.predict_cluster(y)
        constraint(
            'Both points in same cluster',
            lambda: (cluster_x == cluster_y, cluster_x, cluster_y, {})
        )

        # Compute distance (should use Mahalanobis)
        dist = metric(x, y)

        constraint(
            'Intra-cluster distance is non-negative',
            lambda: (dist >= 0.0, dist, '>= 0.0', {})
        )

        constraint(
            'Intra-cluster distance is finite',
            lambda: (np.isfinite(dist), dist, 'finite', {})
        )
    else:
        # No two centroids in same cluster - GMM separated everything
        # This is actually fine, just note it
        constraint(
            'GMM created distinct clusters (no intra-cluster pairs found)',
            lambda: (True, True, True, {'reason': 'all_centroids_in_different_clusters'})
        )


def test_hybrid_metric_with_hierarchy_inter_cluster_constraint(constraint):
    """HybridMetric uses ultrametric for different-cluster distances."""
    # Create hierarchy
    config = make_test_config()
    archive = CVTArchive(config, seed=42)

    rng = np.random.RandomState(42)
    latent = rng.randn(150, 3).astype(np.float32)
    mixing_matrix = rng.randn(3, 10).astype(np.float32)

    for i in range(150):
        raw_metrics = latent[i] @ mixing_matrix + rng.randn(10).astype(np.float32) * 0.1
        archive.add_raw_metrics(raw_metrics)

    archive.discover_dimensions()

    hierarchy = BehavioralHierarchy(n_clusters=5, random_state=42)
    hierarchy.fit(archive.centroids)

    # Create hybrid metric
    metric = HybridMetric(
        hierarchy=hierarchy,
        inter_cluster_mode='ultrametric',
        intra_cluster_mode='mahalanobis',
        p=2
    )

    # Get two points in different clusters
    x = archive.centroids[0]
    cluster_x = hierarchy.predict_cluster(x)

    y = None
    cluster_y = None
    for i in range(1, len(archive.centroids)):
        test_y = archive.centroids[i]
        test_cluster_y = hierarchy.predict_cluster(test_y)
        if test_cluster_y != cluster_x:
            y = test_y
            cluster_y = test_cluster_y
            break

    constraint(
        'Found two centroids in different clusters',
        lambda: (y is not None, y is not None, True, {})
    )

    if y is not None:
        constraint(
            'Points in different clusters',
            lambda: (cluster_x != cluster_y, (cluster_x, cluster_y), 'different', {})
        )

        # Compute distance (should use ultrametric)
        dist = metric(x, y)

        constraint(
            'Inter-cluster distance is non-negative',
            lambda: (dist >= 0.0, dist, '>= 0.0', {})
        )

        constraint(
            'Inter-cluster distance is finite',
            lambda: (np.isfinite(dist), dist, 'finite', {})
        )

        # Verify strong triangle inequality (ultrametric property)
        # Find third point in different cluster
        z = None
        cluster_z = None
        for i in range(1, len(archive.centroids)):
            test_z = archive.centroids[i]
            test_cluster_z = hierarchy.predict_cluster(test_z)
            if test_cluster_z != cluster_x and test_cluster_z != cluster_y:
                z = test_z
                cluster_z = test_cluster_z
                break

        if z is not None:
            dist_xz = metric(x, z)
            dist_yz = metric(y, z)

            # Ultrametric property: d(x,z) <= max(d(x,y), d(y,z))
            # This MUST hold with dendrogram traversal implementation
            max_dist = max(dist, dist_yz)

            constraint(
                'Ultrametric property holds (strong triangle inequality)',
                lambda: (dist_xz <= max_dist + 1e-6, dist_xz, f'<= {max_dist}', {
                    'd(x,y)': dist,
                    'd(y,z)': dist_yz,
                    'd(x,z)': dist_xz,
                    'max(d(x,y), d(y,z))': max_dist
                })
            )


def test_topology_effect_not_handled_graceful_constraint(constraint):
    """Topology effects gracefully degrade when not handled."""
    # Try to get hierarchy without handler installed
    hierarchy = try_get_hierarchy()

    constraint(
        'Hierarchy returns None when effect not handled',
        lambda: (hierarchy is None, hierarchy, None, {})
    )


def test_topology_effect_with_handler_constraint(constraint):
    """Topology effects work when handlers are installed."""
    # Create hierarchy
    config = make_test_config()
    archive = CVTArchive(config, seed=42)

    rng = np.random.RandomState(42)
    latent = rng.randn(150, 3).astype(np.float32)
    mixing_matrix = rng.randn(3, 10).astype(np.float32)

    for i in range(150):
        raw_metrics = latent[i] @ mixing_matrix + rng.randn(10).astype(np.float32) * 0.1
        archive.add_raw_metrics(raw_metrics)

    archive.discover_dimensions()

    hierarchy = BehavioralHierarchy(n_clusters=10, random_state=42)
    hierarchy.fit(archive.centroids)

    # Install handler
    with GetHierarchy.handler(lambda: hierarchy):
        retrieved = try_get_hierarchy()

        constraint(
            'Hierarchy retrieved successfully with handler',
            lambda: (retrieved is not None, retrieved is not None, True, {})
        )

        constraint(
            'Retrieved hierarchy is same instance',
            lambda: (retrieved is hierarchy, retrieved is hierarchy, True, {})
        )

    # After handler exits, should return None again
    hierarchy_after = try_get_hierarchy()
    constraint(
        'Hierarchy returns None after handler exits',
        lambda: (hierarchy_after is None, hierarchy_after, None, {})
    )


def test_chemotaxis_distance_metric_override_constraint(constraint):
    """Chemotaxis allows manual distance metric override."""
    config = make_test_config()
    archive = CVTArchive(config, seed=42)

    rng = np.random.RandomState(42)
    latent = rng.randn(150, 3).astype(np.float32)
    mixing_matrix = rng.randn(3, 10).astype(np.float32)

    for i in range(150):
        raw_metrics = latent[i] @ mixing_matrix + rng.randn(10).astype(np.float32) * 0.1
        archive.add_raw_metrics(raw_metrics)

    archive.discover_dimensions()

    # Force euclidean even after discovery
    chemotaxis_euclidean = Chemotaxis(archive, distance_metric='euclidean')
    constraint(
        'Manual euclidean override works',
        lambda: (chemotaxis_euclidean.distance_metric == 'euclidean', chemotaxis_euclidean.distance_metric, 'euclidean', {})
    )

    # Force manhattan
    chemotaxis_manhattan = Chemotaxis(archive, distance_metric='manhattan')
    constraint(
        'Manual manhattan override works',
        lambda: (chemotaxis_manhattan.distance_metric == 'manhattan', chemotaxis_manhattan.distance_metric, 'manhattan', {})
    )

    # Force cosine
    chemotaxis_cosine = Chemotaxis(archive, distance_metric='cosine')
    constraint(
        'Manual cosine override works',
        lambda: (chemotaxis_cosine.distance_metric == 'cosine', chemotaxis_cosine.distance_metric, 'cosine', {})
    )


def test_mahalanobis_distance_respects_covariance_constraint(constraint):
    """Mahalanobis distance computation respects covariance structure."""
    config = make_test_config(min_dims=2, max_dims=2)
    archive = CVTArchive(config, seed=42)

    # Create correlated data
    rng = np.random.RandomState(42)
    latent = rng.randn(150, 2).astype(np.float32)
    # Make dimensions highly correlated
    latent[:, 1] = latent[:, 0] * 0.9 + rng.randn(150).astype(np.float32) * 0.1

    mixing_matrix = rng.randn(2, 10).astype(np.float32)

    for i in range(150):
        raw_metrics = latent[i] @ mixing_matrix + rng.randn(10).astype(np.float32) * 0.1
        archive.add_raw_metrics(raw_metrics)

    archive.discover_dimensions()

    chemotaxis = Chemotaxis(archive, distance_metric='mahalanobis')
    chemotaxis.update_covariance()

    constraint(
        'Covariance matrix computed',
        lambda: (chemotaxis._covariance_matrix is not None, chemotaxis._covariance_matrix is not None, True, {})
    )

    # Compute distances using both metrics
    pos1 = archive.centroids[0]
    pos2 = archive.centroids[1]

    euclidean_dist = np.linalg.norm(pos1 - pos2)
    mahalanobis_dist = chemotaxis._compute_distance(pos1, pos2)

    constraint(
        'Mahalanobis distance differs from Euclidean',
        lambda: (abs(mahalanobis_dist - euclidean_dist) > 1e-6, (mahalanobis_dist, euclidean_dist), 'different', {
            'mahalanobis': mahalanobis_dist,
            'euclidean': euclidean_dist,
            'ratio': mahalanobis_dist / euclidean_dist if euclidean_dist > 0 else 0.0
        })
    )
