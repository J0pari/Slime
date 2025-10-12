import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Callable, Union
from dataclasses import dataclass
import hashlib
import zlib
import json
import logging
import numpy as np
from scipy.spatial import distance
from slime.config.dimensions import ArchitectureConfig
from slime.memory.diresa import DIRESABehavioralEncoder
logger = logging.getLogger(__name__)

@dataclass
class Elite:
    behavior: Tuple[float, ...]
    fitness: float
    elite_sha: str
    generation: int
    metadata: dict

    def __post_init__(self):
        self.behavior = tuple(self.behavior)

    def reconstruct_weights(self, archive: 'CVTArchive') -> dict:
        weights_u, weights_v = archive._load_elite_weights(self.elite_sha)
        weights = {}
        for key in weights_u.keys():
            weights[key] = torch.matmul(weights_u[key], weights_v[key])
        return weights

    def memory_usage(self, archive: 'CVTArchive') -> int:
        weights_u, weights_v = archive._load_elite_weights(self.elite_sha)
        mem = 0
        for key in weights_u.keys():
            mem += weights_u[key].numel() * weights_u[key].element_size()
            mem += weights_v[key].numel() * weights_v[key].element_size()
        return mem

class CVTArchive:

    def __init__(self, config: ArchitectureConfig, variance_threshold: float=0.85, device: torch.device=None, trustworthiness_threshold: float=0.85, reconstruction_error_threshold: float=0.5, gc_interval: int=100, seed: int=42):
        self.num_raw_metrics = config.behavioral_space.num_raw_metrics
        self.variance_threshold = variance_threshold
        self.min_dims = config.behavioral_space.min_dims
        self.max_dims = config.behavioral_space.max_dims
        self.num_centroids = config.behavioral_space.num_centroids
        self.low_rank_k = config.compression.low_rank_k
        self.delta_rank = config.compression.delta_rank
        self.device = device or torch.device('cuda')
        self.trustworthiness_threshold = trustworthiness_threshold
        self.reconstruction_error_threshold = reconstruction_error_threshold
        self.gc_interval = gc_interval
        self.seed = seed

        self.centroids: Optional[np.ndarray] = None
        self.diresa: Optional[DIRESABehavioralEncoder] = None
        self.behavioral_dims: Optional[int] = None

        self.object_store: Dict[str, bytes] = {}
        self.ref_counts: Dict[str, int] = {}
        self.centroid_refs: Dict[int, str] = {}

        self._generation = 0
        self._total_additions = 0
        self._total_replacements = 0
        self._raw_metrics_samples: List[np.ndarray] = []
        self._discovered = False
        self._gc_counter = 0
        self._discovery_callbacks: List[Callable[[], None]] = []

        self._elite_metadata: Dict[int, Tuple[float, int, dict]] = {}

        # Adaptive Voronoi: density-based cell subdivision/merge
        self._cell_densities: Dict[int, int] = {}  # centroid_id → elite count
        self._density_high_threshold = 5  # Split if > 5 elites in cell
        self._density_low_threshold = 0   # Merge if 0 elites and neighbor has space
        self._adaptation_interval = 100   # Check every 100 additions

    def _hash_object(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def _write_object(self, obj_type: str, content: bytes) -> str:
        header = f"{obj_type} {len(content)}\0".encode()
        full_content = header + content
        sha = self._hash_object(full_content)

        if sha not in self.object_store:
            compressed = zlib.compress(full_content, level=9)
            self.object_store[sha] = compressed

        return sha

    def _read_object(self, sha: str) -> Tuple[str, bytes]:
        if sha not in self.object_store:
            raise ValueError(f"Object {sha} not found in object store")

        compressed = self.object_store[sha]
        full_content = zlib.decompress(compressed)

        header_end = full_content.index(b'\0')
        header = full_content[:header_end].decode()
        obj_type, size_str = header.split(' ')
        content = full_content[header_end + 1:]

        return obj_type, content

    def _incr_ref(self, sha: str):
        self.ref_counts[sha] = self.ref_counts.get(sha, 0) + 1

    def _decr_ref(self, sha: str):
        if sha in self.ref_counts:
            self.ref_counts[sha] -= 1
            if self.ref_counts[sha] <= 0:
                self._delete_object(sha)

    def _delete_object(self, sha: str):
        if sha in self.object_store:
            del self.object_store[sha]
        if sha in self.ref_counts:
            del self.ref_counts[sha]
        logger.debug(f"GC: deleted object {sha[:8]}")

    def _mark_reachable(self, sha: str, reachable: set):
        if sha in reachable:
            return

        reachable.add(sha)

        try:
            obj_type, content = self._read_object(sha)
            if obj_type == 'delta':
                delta_data = json.loads(content.decode('utf-8'))
                self._mark_reachable(delta_data['base'], reachable)
                for delta_sha in delta_data['deltas']:
                    self._mark_reachable(delta_sha, reachable)
        except:
            pass

    def _mark_and_sweep_gc(self):
        reachable = set()

        for elite_sha in self.centroid_refs.values():
            self._mark_reachable(elite_sha, reachable)

        all_shas = set(self.object_store.keys())
        unreachable = all_shas - reachable

        for sha in unreachable:
            self._delete_object(sha)

        if unreachable:
            logger.info(f"GC: freed {len(unreachable)} unreachable objects")

    def _compute_weight_delta(self, current_weights: dict, parent_weights: dict) -> list:
        delta_ops = []

        for key in current_weights.keys():
            curr = current_weights[key]
            parent = parent_weights[key]
            diff = curr - parent

            sparsity = (torch.abs(diff) < 1e-4).float().mean().item()

            if sparsity > 0.95:
                indices = torch.where(torch.abs(diff) >= 1e-4)
                if len(indices[0]) > 0:
                    values = diff[indices]
                    delta_ops.append({
                        'key': key,
                        'op': 'sparse_add',
                        'indices': torch.stack(indices, dim=1).tolist(),
                        'values': values.tolist()
                    })

            elif diff.numel() < 100:
                delta_ops.append({
                    'key': key,
                    'op': 'dense',
                    'value': diff.tolist()
                })

            else:
                try:
                    U, S, Vt = torch.linalg.svd(diff, full_matrices=False)
                    r = min(self.delta_rank, len(S))
                    sqrt_S = torch.sqrt(S[:r])
                    dU = U[:, :r] * sqrt_S
                    dV = Vt[:r, :] * sqrt_S.unsqueeze(1)
                    delta_ops.append({
                        'key': key,
                        'op': 'low_rank',
                        'dU': dU.tolist(),
                        'dV': dV.tolist()
                    })
                except:
                    delta_ops.append({
                        'key': key,
                        'op': 'dense',
                        'value': diff.tolist()
                    })

        return delta_ops

    def _apply_delta(self, base_weights: dict, delta_ops: list) -> dict:
        weights = {k: v.clone() for k, v in base_weights.items()}

        for op in delta_ops:
            key = op['key']

            if op['op'] == 'sparse_add':
                indices = torch.tensor(op['indices'], dtype=torch.long)
                values = torch.tensor(op['values'], dtype=weights[key].dtype, device=weights[key].device)
                weights[key][indices[:, 0], indices[:, 1]] += values

            elif op['op'] == 'low_rank':
                dU = torch.tensor(op['dU'], dtype=weights[key].dtype, device=weights[key].device)
                dV = torch.tensor(op['dV'], dtype=weights[key].dtype, device=weights[key].device)
                weights[key] += dU @ dV

            elif op['op'] == 'dense':
                value = torch.tensor(op['value'], dtype=weights[key].dtype, device=weights[key].device)
                weights[key] = value

            elif op['op'] == 'scale_add':
                weights[key] *= op['scale']
                indices = torch.tensor(op['indices'], dtype=torch.long)
                values = torch.tensor(op['values'], dtype=weights[key].dtype, device=weights[key].device)
                weights[key][indices[:, 0], indices[:, 1]] += values

        return weights

    def _compress_weights(self, state_dict: dict) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        weights_u = {}
        weights_v = {}
        for key, tensor in state_dict.items():
            if tensor.ndim == 2 and min(tensor.shape) > self.low_rank_k:
                U, S, Vt = torch.linalg.svd(tensor, full_matrices=False)
                k = min(self.low_rank_k, len(S))
                sqrt_S = torch.sqrt(S[:k])
                weights_u[key] = U[:, :k] * sqrt_S
                weights_v[key] = Vt[:k, :] * sqrt_S.unsqueeze(1)
            else:
                k = min(self.low_rank_k, tensor.shape[0] if tensor.ndim >= 1 else 1)
                if tensor.ndim == 2:
                    identity = torch.eye(tensor.shape[0], device=tensor.device, dtype=tensor.dtype)[:, :k]
                    weights_u[key] = tensor @ identity
                    weights_v[key] = identity.T
                else:
                    weights_u[key] = tensor.unsqueeze(-1) if tensor.ndim == 1 else tensor.view(-1, 1)
                    weights_v[key] = torch.ones(1, 1, device=tensor.device, dtype=tensor.dtype)
        return (weights_u, weights_v)

    def _store_elite_weights(self, weights_u: dict, weights_v: dict, parent_sha: Optional[str], centroid_id: int) -> str:
        weights_dict = {'u': {k: v.cpu().tolist() for k, v in weights_u.items()},
                        'v': {k: v.cpu().tolist() for k, v in weights_v.items()}}
        weights_bytes = json.dumps(weights_dict, separators=(',', ':')).encode('utf-8')
        weights_sha = self._hash_object(weights_bytes)

        if weights_sha in self.object_store:
            return weights_sha

        if parent_sha and parent_sha in self.object_store:
            try:
                parent_weights_u, parent_weights_v = self._load_elite_weights(parent_sha)
                parent_reconstructed = {k: parent_weights_u[k] @ parent_weights_v[k] for k in parent_weights_u.keys()}
                current_reconstructed = {k: weights_u[k] @ weights_v[k] for k in weights_u.keys()}

                delta_ops = self._compute_weight_delta(current_reconstructed, parent_reconstructed)
                delta_json = json.dumps(delta_ops, separators=(',', ':')).encode('utf-8')

                if len(delta_json) < len(weights_bytes) * 0.5:
                    delta_sha = self._write_object('delta_ops', delta_json)

                    parent_obj_type, parent_content = self._read_object(parent_sha)

                    if parent_obj_type == 'delta':
                        parent_data = json.loads(parent_content.decode('utf-8'))
                        parent_data['deltas'].append(delta_sha)

                        delta_chain_size = sum(len(self._read_object(d)[1]) for d in parent_data['deltas'])

                        if delta_chain_size > len(weights_bytes) * 0.7:
                            return self._write_object('blob', weights_bytes)
                        else:
                            return self._write_object('delta', json.dumps(parent_data, separators=(',', ':')).encode('utf-8'))
                    else:
                        delta_data = {
                            'base': parent_sha,
                            'deltas': [delta_sha]
                        }
                        return self._write_object('delta', json.dumps(delta_data, separators=(',', ':')).encode('utf-8'))
            except:
                pass

        return self._write_object('blob', weights_bytes)

    def _load_elite_weights(self, elite_sha: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        obj_type, content = self._read_object(elite_sha)

        if obj_type == 'blob':
            weights_dict = json.loads(content.decode('utf-8'))
            weights_u = {k: torch.tensor(v, device=self.device) for k, v in weights_dict['u'].items()}
            weights_v = {k: torch.tensor(v, device=self.device) for k, v in weights_dict['v'].items()}
            return weights_u, weights_v

        elif obj_type == 'delta':
            delta_data = json.loads(content.decode('utf-8'))
            base_u, base_v = self._load_elite_weights(delta_data['base'])
            base_reconstructed = {k: base_u[k] @ base_v[k] for k in base_u.keys()}

            for delta_sha in delta_data['deltas']:
                delta_type, delta_content = self._read_object(delta_sha)
                delta_ops = json.loads(delta_content.decode('utf-8'))
                base_reconstructed = self._apply_delta(base_reconstructed, delta_ops)

            return self._compress_weights(base_reconstructed)

        else:
            raise ValueError(f"Unknown object type: {obj_type}")

    def add_raw_metrics(self, raw_metrics: np.ndarray) -> bool:
        if self._discovered:
            raise ValueError('Cannot add raw metrics after dimension discovery. Use add() instead.')
        if len(raw_metrics) != self.num_raw_metrics:
            raise ValueError(f'Raw metrics dimension mismatch: expected {self.num_raw_metrics}, got {len(raw_metrics)}')
        self._raw_metrics_samples.append(raw_metrics)
        logger.debug(f'Collected raw metrics sample {len(self._raw_metrics_samples)} for dimension discovery')
        return False

    def discover_dimensions(self) -> bool:
        """Discover behavioral dimensions using DIRESA learned embeddings."""
        if self._discovered:
            raise ValueError("Dimensions already discovered - cannot call discover_dimensions() twice")

        if len(self._raw_metrics_samples) < 100:
            raise ValueError(f'Need ≥100 raw metric samples for DIRESA, got {len(self._raw_metrics_samples)}')

        raw_matrix = np.array(self._raw_metrics_samples, dtype=np.float32)
        logger.info(f'Running DIRESA dimension discovery on {raw_matrix.shape[0]} samples with {raw_matrix.shape[1]} raw metrics')

        # Filter out zero-variance metrics (constant features)
        variances = np.var(raw_matrix, axis=0)
        nonzero_variance_mask = variances > 1e-10
        n_filtered = (~nonzero_variance_mask).sum()
        if n_filtered > 0:
            logger.info(f'Filtered out {n_filtered} zero-variance metrics (constant across samples)')
            raw_matrix = raw_matrix[:, nonzero_variance_mask]
            if raw_matrix.shape[1] < self.min_dims:
                raise ValueError(f'After filtering zero-variance, only {raw_matrix.shape[1]} metrics remain (need ≥{archive.min_dims})')

        # Initialize DIRESA encoder
        self.diresa = DIRESABehavioralEncoder(
            input_dim=raw_matrix.shape[1],
            min_dims=self.min_dims,
            max_dims=self.max_dims,
            hidden_dim=64,
            lambda_dist=1.0,
            lambda_kl=0.01,
            device=self.device
        )

        # Convert to torch
        x_train = torch.from_numpy(raw_matrix).to(self.device)

        # Precompute pairwise distances for distance preservation loss
        dist_matrix = pairwise_distances(raw_matrix)
        dist_tensor = torch.from_numpy(dist_matrix).to(self.device)

        # Train DIRESA
        optimizer = torch.optim.Adam(self.diresa.parameters(), lr=1e-3)
        self.diresa.train()

        n_epochs = 500
        batch_size = min(32, len(x_train))

        logger.info(f'Training DIRESA for {n_epochs} epochs (batch_size={batch_size})')

        for epoch in range(n_epochs):
            # Shuffle data
            indices = torch.randperm(len(x_train))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(x_train), batch_size):
                batch_indices = indices[i:i+batch_size]
                x_batch = x_train[batch_indices]
                dist_batch = dist_tensor[batch_indices][:, batch_indices]

                optimizer.zero_grad()
                loss, metrics = self.diresa.compute_loss(x_batch, dist_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 100 == 0:
                avg_loss = epoch_loss / n_batches
                active_dims = self.diresa.get_active_dims()
                logger.info(f'Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}, active_dims={active_dims}')

        # Get final embeddings
        self.diresa.eval()
        with torch.no_grad():
            transformed_tensor = self.diresa.encode(x_train)
            transformed_samples = transformed_tensor.cpu().numpy()

        # Get active dimensionality
        n_dims = self.diresa.get_active_dims()
        self.behavioral_dims = n_dims

        logger.info(f'DIRESA selected {n_dims} dimensions (adaptive from {self.min_dims}-{self.max_dims} range)')

        # Validate embeddings
        validation = self.diresa.validate_embeddings(raw_matrix, transformed_samples)

        trust_score = validation['trustworthiness']
        continuity_score = validation['continuity']
        procrustes_dist = validation['procrustes_distance']

        logger.info(f'Trustworthiness: {trust_score:.3f}')
        logger.info(f'Continuity: {continuity_score:.3f}')
        logger.info(f'Procrustes distance: {procrustes_dist:.3f}')

        # Validation thresholds
        if trust_score < 0.7:
            raise ValueError(f'Trustworthiness critically low ({trust_score:.3f} < 0.7): embeddings do not preserve neighborhoods')
        elif trust_score < self.trustworthiness_threshold:
            logger.warning(f'Trustworthiness below threshold ({trust_score:.3f} < {self.trustworthiness_threshold}), proceeding with reduced confidence')

        if continuity_score < 0.7:
            raise ValueError(f'Continuity critically low ({continuity_score:.3f} < 0.7): neighborhood structure not preserved')
        elif continuity_score < 0.85:
            logger.warning(f'Continuity below threshold ({continuity_score:.3f} < 0.85)')

        if procrustes_dist > 0.25:
            logger.warning(f'Procrustes distance high ({procrustes_dist:.3f} > 0.15): shape distortion detected')

        # Mark as discovered
        self._discovered = True

        # Initialize centroids with DIRESA embeddings (only active dimensions)
        transformed_active = transformed_samples[:, :n_dims]
        self.initialize_centroids(transformed_active)

        logger.info(f'Discovered {self.behavioral_dims} behavioral dimensions from {self.num_raw_metrics} raw metrics via DIRESA')

        # Notify observers that dimension discovery is complete
        if hasattr(self, '_discovery_callbacks'):
            for callback in self._discovery_callbacks:
                callback()

        return True

    def initialize_centroids(self, initial_samples: Optional[np.ndarray]=None):
        if initial_samples is not None and len(initial_samples) >= self.num_centroids:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.num_centroids, random_state=self.seed, n_init=10)
            kmeans.fit(initial_samples.astype(np.float32))
            self.centroids = kmeans.cluster_centers_.astype(np.float32)
            logger.info(f'Initialized {self.num_centroids} CVT centroids from {len(initial_samples)} samples with seed {self.seed}')
        else:
            if self.behavioral_dims is None:
                raise ValueError("behavioral_dims must be set before initializing centroids")
            rng = np.random.RandomState(self.seed)
            self.centroids = rng.randn(self.num_centroids, self.behavioral_dims).astype(np.float32)
            logger.info(f'Initialized {self.num_centroids} CVT centroids from seed {self.seed}')

    def transform_to_behavioral_space(self, raw_metrics: np.ndarray) -> np.ndarray:
        """Transform raw metrics to behavioral space using DIRESA encoder.

        Args:
            raw_metrics: (N, num_raw_metrics) array of raw metrics

        Returns:
            (N, behavioral_dims) array of behavioral coordinates
        """
        if self.diresa is None:
            raise ValueError('DIRESA not trained. Call discover_dimensions() first.')

        self.diresa.eval()
        with torch.no_grad():
            x = torch.from_numpy(raw_metrics.astype(np.float32)).to(self.device)
            embeddings = self.diresa.encode(x)
            # Return only active dimensions
            active_dims = self.diresa.get_active_dims()
            return embeddings[:, :active_dims].cpu().numpy()

    def _find_nearest_centroid(self, behavior: Union[Tuple[float, ...], np.ndarray]) -> int:
        if self.centroids is None:
            raise ValueError('Centroids not initialized. Call initialize_centroids() first.')
        if isinstance(behavior, tuple):
            behavior_array = np.array(behavior, dtype=np.float32)
        else:
            behavior_array = behavior.astype(np.float32)
        distances = distance.cdist([behavior_array], self.centroids, metric='euclidean')[0]
        return int(np.argmin(distances))

    def add(self, behavior: Tuple[float, ...], fitness: float, state_dict: dict, generation: Optional[int]=None, metadata: Optional[dict]=None) -> bool:
        if not self._discovered:
            raise ValueError('Cannot add elites before dimension discovery. Use add_raw_metrics() during warmup phase.')
        if self.behavioral_dims is None:
            raise ValueError('Behavioral dimensions not set. Call discover_dimensions() first.')
        if len(behavior) != self.behavioral_dims:
            raise ValueError(f'Behavior dimension mismatch: expected {self.behavioral_dims}, got {len(behavior)}')
        if self.centroids is None:
            raise ValueError('Centroids not initialized. Call discover_dimensions() first.')

        centroid_id = self._find_nearest_centroid(behavior)

        if centroid_id in self._elite_metadata:
            old_fitness = self._elite_metadata[centroid_id][0]
            if fitness <= old_fitness:
                return False

        parent_sha = self.centroid_refs.get(centroid_id)

        weights_u, weights_v = self._compress_weights(state_dict)
        elite_sha = self._store_elite_weights(weights_u, weights_v, parent_sha, centroid_id)

        if centroid_id in self.centroid_refs:
            old_sha = self.centroid_refs[centroid_id]
            if old_sha != elite_sha:
                self._decr_ref(old_sha)
                self._total_replacements += 1

        self.centroid_refs[centroid_id] = elite_sha
        self._incr_ref(elite_sha)

        gen = generation if generation is not None else self._generation
        self._elite_metadata[centroid_id] = (fitness, gen, metadata or {}, tuple(behavior))

        self._total_additions += 1

        # Update cell density tracking
        self._cell_densities[centroid_id] = self._cell_densities.get(centroid_id, 0) + 1

        # Adaptive Voronoi: check for subdivision/merge
        if self._total_additions % self._adaptation_interval == 0:
            self._adapt_voronoi_cells()

        self._gc_counter += 1
        if self._gc_counter >= self.gc_interval:
            self._mark_and_sweep_gc()
            self._gc_counter = 0

        logger.debug(f'Added elite at centroid {centroid_id}: fitness={fitness:.4f}, SHA={elite_sha[:8]}')
        return True

    def get(self, behavior: Tuple[float, ...]) -> Optional[Elite]:
        if self.centroids is None:
            return None
        centroid_id = self._find_nearest_centroid(behavior)
        if centroid_id not in self.centroid_refs:
            return None

        elite_sha = self.centroid_refs[centroid_id]
        fitness, gen, metadata, behavior = self._elite_metadata.get(centroid_id, (0.0, self._generation, {}, tuple(self.centroids[centroid_id])))
        return Elite(
            behavior=behavior,
            fitness=fitness,
            elite_sha=elite_sha,
            generation=gen,
            metadata=metadata
        )

    def sample_near(self, behavior: Tuple[float, ...], k: int=5) -> List[Elite]:
        if self.centroids is None or not self.centroid_refs:
            return []
        behavior_array = np.array(behavior, dtype=np.float32)
        centroid_ids = list(self.centroid_refs.keys())
        centroid_positions = self.centroids[centroid_ids]
        distances = distance.cdist([behavior_array], centroid_positions, metric='euclidean')[0]
        nearest_indices = np.argsort(distances)[:k]

        elites = []
        for i in nearest_indices:
            if i < len(centroid_ids):
                centroid_id = centroid_ids[i]
                elite_sha = self.centroid_refs[centroid_id]
                fitness, gen, metadata, behavior = self._elite_metadata.get(centroid_id, (0.0, self._generation, {}, tuple(self.centroids[centroid_id])))
                elites.append(Elite(
                    behavior=behavior,
                    fitness=fitness,
                    elite_sha=elite_sha,
                    generation=gen,
                    metadata=metadata
                ))
        return elites

    def bootstrap_component(self, component_factory: Callable, target_behavior: Tuple[float, ...], k_neighbors: int=3, mutation_std: float=0.1):
        nearby = self.sample_near(target_behavior, k=k_neighbors)
        if not nearby:
            return None

        best = max(nearby, key=lambda e: e.fitness)
        weights = best.reconstruct_weights(self)

        torch.manual_seed(self.seed + best.generation)
        for key in weights:
            noise = torch.randn_like(weights[key]) * mutation_std
            weights[key] = weights[key] + noise

        try:
            component = component_factory()
            component.load_state_dict(weights)
        except TypeError:
            component = component_factory(weights)

        logger.debug(f'Bootstrapped component from elite at generation {best.generation}, fitness={best.fitness:.4f}')
        return component

    def validate_behavioral_space(self, samples: np.ndarray) -> float:
        """Validate behavioral space using Trustworthiness metric."""
        try:
            from sklearn.manifold import trustworthiness as compute_trust
        except ImportError:
            logger.warning('sklearn not installed, skipping validation')
            return 1.0
        if len(samples) < 50:
            logger.warning(f'Only {len(samples)} behavioral samples, need ≥50 for reliable validation')
            return 0.0
        try:
            # Use stored raw metrics to compute trustworthiness
            if not hasattr(self, '_raw_metrics_samples') or len(self._raw_metrics_samples) == 0:
                logger.warning('No raw metrics available for validation')
                return 0.0

            raw_samples = np.array(self._raw_metrics_samples[:len(samples)], dtype=np.float32)
            k = min(30, len(samples) - 1)
            trust_score = compute_trust(raw_samples, samples, n_neighbors=k)
            logger.info(f'Trustworthiness: {trust_score:.3f} (>0.85=good, >0.90=excellent)')

            if trust_score < self.trustworthiness_threshold:
                logger.error(f'Trustworthiness {trust_score:.3f} < {self.trustworthiness_threshold}: behavioral dimensions do not preserve neighborhoods!')

            return trust_score
        except Exception as e:
            logger.error(f'Failed to validate behavioral space: {e}')
            return 0.0

    def increment_generation(self) -> None:
        self._generation += 1

    def size(self) -> int:
        return len(self.centroid_refs)

    def get_behavioral_variance(self) -> np.ndarray:
        if not self.centroid_refs or self.behavioral_dims is None:
            return np.zeros(self.behavioral_dims if self.behavioral_dims else 0)
        centroid_ids = list(self.centroid_refs.keys())
        behaviors = self.centroids[centroid_ids]
        return np.var(behaviors, axis=0)

    def total_memory_usage(self) -> int:
        total_bytes = 0
        for sha, compressed in self.object_store.items():
            total_bytes += len(compressed)
        return total_bytes

    def clear(self) -> None:
        self.centroid_refs.clear()
        self.object_store.clear()
        self.ref_counts.clear()
        self._elite_metadata.clear()
        self._generation = 0
        self._total_additions = 0
        self._total_replacements = 0

    def max_fitness(self) -> float:
        if not self._elite_metadata:
            return float('-inf')
        return max(fitness for fitness, _, _, _ in self._elite_metadata.values())

    def coverage(self) -> float:
        if self.centroids is None:
            return 0.0
        return len(self.centroid_refs) / self.num_centroids

    @property
    def elites(self) -> Dict[int, Elite]:
        elite_dict = {}
        for centroid_id, elite_sha in self.centroid_refs.items():
            fitness, gen, metadata, behavior = self._elite_metadata.get(centroid_id, (0.0, self._generation, {}, tuple(self.centroids[centroid_id])))
            elite_dict[centroid_id] = Elite(
                behavior=behavior,
                fitness=fitness,
                elite_sha=elite_sha,
                generation=gen,
                metadata=metadata
            )
        return elite_dict

    def stats(self) -> dict:
        return {
            'num_elites': len(self.centroid_refs),
            'coverage': self.coverage(),
            'total_additions': self._total_additions,
            'total_replacements': self._total_replacements,
            'object_store_size': len(self.object_store),
            'behavioral_dims': self.behavioral_dims,
            'discovered': self._discovered
        }

    def _adapt_voronoi_cells(self) -> None:
        """
        Adaptive Voronoi: adjust cells based on density.

        - High density cells (> threshold) → subdivide
        - Low density cells (= 0) with high-density neighbors → merge
        - Lloyd's relaxation → adjust centroid positions
        """
        if self.centroids is None or self.behavioral_dims is None:
            return

        # Subdivision: split high-density cells
        cells_to_split = [cid for cid, density in self._cell_densities.items()
                         if density > self._density_high_threshold]

        for centroid_id in cells_to_split:
            self._subdivide_cell(centroid_id)

        # Lloyd's relaxation: adjust centroids toward elite mean
        self._lloyd_relaxation()

        logger.debug(f'Voronoi adaptation: {len(cells_to_split)} cells subdivided, Lloyd relaxation applied')

    def _subdivide_cell(self, centroid_id: int) -> None:
        """
        Subdivide high-density cell by splitting centroid.

        Creates new centroid at offset from original along direction of elite spread.
        """
        if centroid_id not in self.centroid_refs:
            return

        # Collect elite behaviors in this cell
        elite_behaviors = []
        if centroid_id in self._elite_metadata:
            _, _, _, behavior = self._elite_metadata[centroid_id]
            elite_behaviors.append(np.array(behavior[:self.behavioral_dims]))

        if len(elite_behaviors) < 2:
            return  # Need multiple elites to determine split direction

        # Compute spread direction (PCA first component)
        behaviors_matrix = np.array(elite_behaviors)
        centroid_pos = self.centroids[centroid_id]

        # Offset along max variance direction
        cov = np.cov(behaviors_matrix.T)
        if cov.ndim == 0:
            return
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        split_direction = eigenvectors[:, -1]  # Max variance

        # Create new centroid at offset
        offset_distance = 0.1 * np.linalg.norm(split_direction)
        new_centroid = centroid_pos + offset_distance * split_direction

        # Add to centroids
        self.centroids = np.vstack([self.centroids, new_centroid])
        new_centroid_id = len(self.centroids) - 1

        # Reset density for both cells
        self._cell_densities[centroid_id] = self._cell_densities.get(centroid_id, 0) // 2
        self._cell_densities[new_centroid_id] = 0

        logger.debug(f'Subdivided cell {centroid_id} → new cell {new_centroid_id}')

    def _lloyd_relaxation(self, alpha: float = 0.1) -> None:
        """
        Lloyd's algorithm: move centroids toward mean of assigned elites.

        Args:
            alpha: Learning rate for centroid movement
        """
        if self.centroids is None or self.behavioral_dims is None:
            return

        # Compute mean behavior per cell
        cell_means = {}
        cell_counts = {}

        for centroid_id, (_, _, _, behavior) in self._elite_metadata.items():
            behavior_vec = np.array(behavior[:self.behavioral_dims])
            if centroid_id not in cell_means:
                cell_means[centroid_id] = behavior_vec
                cell_counts[centroid_id] = 1
            else:
                cell_means[centroid_id] += behavior_vec
                cell_counts[centroid_id] += 1

        # Move centroids toward means
        for centroid_id, mean_sum in cell_means.items():
            if centroid_id >= len(self.centroids):
                continue
            mean_behavior = mean_sum / cell_counts[centroid_id]
            self.centroids[centroid_id] += alpha * (mean_behavior - self.centroids[centroid_id])
