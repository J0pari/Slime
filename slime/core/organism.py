import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging
import numpy as np
from slime.core.pseudopod import Pseudopod
from slime.core.state import FlowState
from slime.core.chemotaxis import Chemotaxis
from slime.memory.archive import CVTArchive
from slime.memory.pool import DynamicPool, PoolConfig
from slime.proto.kernel import Kernel
from slime.kernels.torch_fallback import TorchKernel
from slime.observability.metrics import MetricsCollector
from slime.config.dimensions import ArchitectureConfig
import os
from pathlib import Path
try:
    from slime.kernels.triton_impl import TritonKernel
    HAS_TRITON = True
    repo_root = Path(__file__).parent.parent.parent
    tcc_path = repo_root / '.local' / 'tcc' / 'tcc.exe'
    if tcc_path.exists():
        os.environ.setdefault('CC', str(tcc_path))
except ImportError:
    HAS_TRITON = False
logger = logging.getLogger(__name__)

class Organism(nn.Module):

    def __init__(self, sensory_dim: int, latent_dim: int, head_dim: int, arch_config: ArchitectureConfig, device: Optional[torch.device]=None, kernel: Optional[Kernel]=None, pool_config: Optional[PoolConfig]=None, metrics_collector: Optional[MetricsCollector]=None):
        super().__init__()
        self.sensory_dim = sensory_dim
        self.latent_dim = latent_dim
        self.head_dim = head_dim
        self.arch_config = arch_config
        self.device = device or torch.device('cuda')
        if kernel is not None:
            self.kernel = kernel
        elif HAS_TRITON:
            self.kernel = TritonKernel(self.device, arch_config.numerical)
            logger.info('Using Triton GPU kernels for maximum performance')
        else:
            self.kernel = TorchKernel(arch_config.numerical, self.device)
            logger.warning('Triton not available, using PyTorch fallback')
        self.metrics = metrics_collector
        self.encode = nn.Sequential(nn.Linear(sensory_dim, latent_dim), nn.LayerNorm(latent_dim), nn.Tanh()).to(self.device)
        self.decode = nn.Sequential(nn.Linear(latent_dim, sensory_dim), nn.Tanh()).to(self.device)
        self.predict_rank = nn.Linear(latent_dim, 1).to(self.device)
        self.predict_coherence = nn.Linear(latent_dim, 1).to(self.device)
        self.project_heads = nn.Linear(head_dim, latent_dim).to(self.device)
        self.archive = CVTArchive(config=arch_config, variance_threshold=0.85, device=self.device, trustworthiness_threshold=0.85, reconstruction_error_threshold=0.5, gc_interval=100, seed=42)
        self.chemotaxis = Chemotaxis(self.archive, self.device)

        # Register callback to update Mahalanobis covariance after dimension discovery
        def _on_discovery_complete():
            logger.info('Dimension discovery complete, updating chemotaxis to Mahalanobis distance')
            self.chemotaxis.distance_metric = 'mahalanobis'
            self.chemotaxis.update_covariance()

        self.archive._discovery_callbacks.append(_on_discovery_complete)
        if pool_config is None:
            pool_config = PoolConfig(min_size=4, max_size=32, birth_threshold=0.8, death_threshold=0.1, cull_interval=100)
        self.pseudopod_pool = DynamicPool(component_factory=lambda: Pseudopod(head_dim, self.kernel, arch_config.fitness, arch_config.numerical, self.device, latent_dim=head_dim, stimulus_dim=head_dim, num_heads=arch_config.dimensions.num_heads), config=pool_config, arch_config=arch_config, bootstrap_factory=lambda genome: Pseudopod.from_dict(genome, self.kernel, arch_config.fitness, arch_config.numerical, self.device), archive=self.archive, device=self.device)
        self._generation = 0

    def forward(self, stimulus: torch.Tensor, state: Optional[FlowState]=None) -> Tuple[torch.Tensor, FlowState]:
        if self.metrics:
            self.metrics.start_step()
        batch_size = stimulus.shape[0]
        fresh_body = self.encode(stimulus).unsqueeze(1)

        # Adaptive state blending based on predicted coherence
        # High coherence (learning) → trust fresh encoding more
        # Low coherence (plateaued) → rely on memory state more
        if state is not None:
            # Predict coherence from fresh encoding to decide blend
            body_for_coherence_pred = fresh_body.mean(dim=(0, 1))
            coherence_pred = torch.sigmoid(self.predict_coherence(body_for_coherence_pred.unsqueeze(0))).squeeze()
            # Blend: high coherence → α≈1 (mostly fresh), low coherence → α≈0 (mostly state)
            fresh_weight = coherence_pred.clamp(0.3, 0.9)  # Bounded to avoid extremes
            state_weight = 1.0 - fresh_weight
            body = fresh_weight * fresh_body + state_weight * state.body
        else:
            body = fresh_body
        body_for_prediction = body.mean(dim=(0, 1))
        rank = torch.sigmoid(self.predict_rank(body_for_prediction.unsqueeze(0))).squeeze().item()
        coherence = torch.sigmoid(self.predict_coherence(body_for_prediction.unsqueeze(0))).squeeze().item()
        behavior = (rank, coherence)
        pseudopods = self.pseudopod_pool.get_at(behavior, max_count=8)
        if not pseudopods:
            logger.warning('Empty pseudopod pool, spawning emergency pseudopod')
            pseudopods = [Pseudopod(self.head_dim, self.kernel, self.arch_config.fitness, self.arch_config.numerical, self.device, latent_dim=self.head_dim, stimulus_dim=self.head_dim, num_heads=self.arch_config.dimensions.num_heads)]
        outputs = []
        max_rank = torch.tensor(0.0, device=self.device)
        min_coherence = torch.tensor(1.0, device=self.device)
        for pod in pseudopods:
            pod_input = body[:, :, :self.head_dim]
            stim_input = stimulus[:, :self.head_dim].unsqueeze(1)
            output = pod(pod_input, stim_input)
            outputs.append(output)
            pod_rank = pod.effective_rank().mean()
            pod_coherence = pod.coherence().mean()
            max_rank = torch.maximum(max_rank, pod_rank)
            min_coherence = torch.minimum(min_coherence, pod_coherence)
        merged = torch.stack(outputs).mean(0)
        merged_sum_heads = merged.sum(dim=1)
        merged_latent = self.project_heads(merged_sum_heads)
        fitness = (max_rank * min_coherence).item()

        if not self.archive._discovered:
            # Collect 10D raw metrics from ALL active pseudopods for DIRESA
            pseudopod_raw_metrics = []
            for pod in pseudopods:
                if hasattr(pod, '_raw_metrics'):
                    pseudopod_raw_metrics.append(pod._raw_metrics)

            if pseudopod_raw_metrics:
                # Average raw metrics across all pseudopods
                avg_raw_metrics = torch.stack(pseudopod_raw_metrics).mean(0)
                self.archive.add_raw_metrics(avg_raw_metrics.detach().cpu().numpy().astype(np.float32))
            warmup_samples = self.arch_config.behavioral_space.num_centroids * 3
            if len(self.archive._raw_metrics_samples) >= warmup_samples:
                logger.info(f"Warmup complete ({warmup_samples} samples), discovering behavioral dimensions...")
                self.archive.discover_dimensions()
        else:
            for pod in pseudopods:
                pod_fitness = (pod.effective_rank() * pod.coherence()).mean().item()
                try:
                    self.archive.add(behavior, pod_fitness, pod.state_dict(), generation=self._generation)
                except ValueError:
                    pass

        self.pseudopod_pool.step(behavior)
        new_state = FlowState(body=merged_latent, behavior=behavior, generation=self._generation, fitness=fitness)
        output = self.decode(merged_latent)
        self._generation += 1
        self.archive.increment_generation()
        if self.metrics:
            self.metrics.end_step(batch_size=batch_size, pool_size=self.pseudopod_pool.size(), archive_size=self.archive.size(), archive_coverage=self.archive.coverage(), loss=None)
        return (output, new_state)

    def reset_state(self) -> None:
        self.archive.clear()
        self.pseudopod_pool.clear()
        self._generation = 0

    def stats(self) -> dict:
        return {'generation': self._generation, 'archive_size': self.archive.size(), 'archive_coverage': self.archive.coverage(), 'archive_max_fitness': self.archive.max_fitness(), 'pool_stats': self.pseudopod_pool.stats()}