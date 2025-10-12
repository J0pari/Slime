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

    def __init__(self, num_raw_metrics: int=15, variance_threshold: float=0.85, min_dims: int=3, max_dims: int=7, num_centroids: int=1000, low_rank_k: int=64, delta_rank: int=8, device: torch.device=None, kmo_threshold: float=0.6, reconstruction_error_threshold: float=0.5, kernel_selection: str='auto', gc_interval: int=100, seed: int=42):
        self.num_raw_metrics = num_raw_metrics
        self.variance_threshold = variance_threshold
        self.min_dims = min_dims
        self.max_dims = max_dims
        self.num_centroids = num_centroids
        self.low_rank_k = low_rank_k
        self.delta_rank = delta_rank
        self.device = device or torch.device('cuda')
        self.kmo_threshold = kmo_threshold
        self.reconstruction_error_threshold = reconstruction_error_threshold
        self.kernel_selection = kernel_selection
        self.gc_interval = gc_interval
        self.seed = seed

        self.centroids: Optional[np.ndarray] = None
        self.kpca_transform = None
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

        self._elite_metadata: Dict[int, Tuple[float, int, dict]] = {}

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
        if self._discovered:
            logger.warning('Dimensions already discovered')
            return True
        if len(self._raw_metrics_samples) < 100:
            raise ValueError(f'Need ≥100 raw metric samples for Kernel PCA, got {len(self._raw_metrics_samples)}')

        raw_matrix = np.array(self._raw_metrics_samples, dtype=np.float32)
        logger.info(f'Running dimension discovery on {raw_matrix.shape[0]} samples with {raw_matrix.shape[1]} raw metrics')

        try:
            from sklearn.decomposition import PCA, KernelPCA
            from factor_analyzer.factor_analyzer import calculate_kmo
            from scipy.stats import bartlett
        except ImportError as e:
            raise ImportError(f'Missing required package for dimension discovery: {e}')

        pca_full = PCA(n_components=None)
        pca_full.fit(raw_matrix)
        variance_ratios = pca_full.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_ratios)

        n_dims = np.argmax(cumulative_variance > self.variance_threshold) + 1
        n_dims = np.clip(n_dims, self.min_dims, self.max_dims)

        logger.info(f"Scree plot elbow at {n_dims} dimensions (explains {cumulative_variance[n_dims-1]:.1%} variance)")

        if self.kernel_selection == 'auto':
            kernel_candidates = [
                ('rbf', {'gamma': 1.0}),
                ('rbf', {'gamma': 0.1}),
                ('poly', {'degree': 2}),
                ('poly', {'degree': 3}),
                ('cosine', {}),
            ]

            best_kernel, best_params, best_score = None, None, float('inf')

            for kernel_name, kernel_params in kernel_candidates:
                try:
                    kpca = KernelPCA(n_components=n_dims, kernel=kernel_name, random_state=self.seed,
                                     fit_inverse_transform=True, **kernel_params)
                    transformed = kpca.fit_transform(raw_matrix)
                    reconstructed = kpca.inverse_transform(transformed)

                    recon_error = np.mean((raw_matrix - reconstructed) ** 2)
                    _, kmo_model = calculate_kmo(transformed)

                    score = recon_error / (kmo_model + 1e-6)

                    logger.debug(f"Kernel {kernel_name}{kernel_params}: recon_error={recon_error:.3f}, KMO={kmo_model:.3f}, score={score:.3f}")

                    if score < best_score:
                        best_kernel, best_params, best_score = kernel_name, kernel_params, score
                        best_kpca = kpca
                except Exception as e:
                    logger.warning(f"Kernel {kernel_name}{kernel_params} failed: {e}")
                    continue

            if best_kernel is None:
                raise ValueError("All kernel candidates failed")

            logger.info(f"Selected kernel: {best_kernel} with params {best_params} (score={best_score:.3f})")
            kpca = best_kpca
        else:
            kpca = KernelPCA(n_components=n_dims, kernel=self.kernel_selection, random_state=self.seed,
                             fit_inverse_transform=True)
            kpca.fit(raw_matrix)

        transformed_samples = kpca.transform(raw_matrix)
        reconstructed = kpca.inverse_transform(transformed_samples)
        reconstruction_error = np.mean((raw_matrix - reconstructed) ** 2)
        logger.info(f'Kernel PCA reconstruction error: {reconstruction_error:.3f}')

        if reconstruction_error > self.reconstruction_error_threshold:
            raise ValueError(f'Kernel PCA reconstruction error {reconstruction_error:.3f} > {self.reconstruction_error_threshold}')

        kmo_all, kmo_model = calculate_kmo(raw_matrix)
        logger.info(f'KMO statistic: {kmo_model:.3f}')
        if kmo_model < self.kmo_threshold:
            raise ValueError(f'KMO {kmo_model:.3f} < {self.kmo_threshold}: raw metrics not factorable')

        stat, p_value = bartlett(*[raw_matrix[:, i] for i in range(raw_matrix.shape[1])])
        logger.info(f"Bartlett's test: statistic={stat:.2f}, p-value={p_value:.4f}")
        if p_value > 0.05:
            raise ValueError(f"Bartlett's test p={p_value:.3f} > 0.05: metrics are uncorrelated")

        self.kpca_transform = kpca
        self.behavioral_dims = n_dims
        self._discovered = True

        self.initialize_centroids(transformed_samples)

        logger.info(f'Discovered {self.behavioral_dims} behavioral dimensions from {self.num_raw_metrics} raw metrics')
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
        try:
            from factor_analyzer.factor_analyzer import calculate_kmo
        except ImportError:
            logger.warning('factor_analyzer not installed, skipping KMO validation')
            return 1.0
        if len(samples) < 50:
            logger.warning(f'Only {len(samples)} behavioral samples, need ≥50 for reliable KMO test')
            return 0.0
        try:
            from scipy.stats import bartlett
            kmo_all, kmo_model = calculate_kmo(samples)
            logger.info(f'KMO statistic: {kmo_model:.3f} (>0.6=acceptable, >0.8=good, >0.9=excellent)')
            if kmo_model < self.kmo_threshold:
                logger.error(f'KMO {kmo_model:.3f} < {self.kmo_threshold}: behavioral dimensions are not factorable!')
            corr_matrix = np.corrcoef(samples, rowvar=False)
            stat, p_value = bartlett(*[samples[:, i] for i in range(samples.shape[1])])
            logger.info(f"Bartlett's test: statistic={stat:.2f}, p-value={p_value:.4f}")
            return kmo_model
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
