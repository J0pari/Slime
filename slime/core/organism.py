import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging
from slime.core.pseudopod import Pseudopod
from slime.core.state import FlowState
from slime.core.chemotaxis import Chemotaxis
from slime.memory.archive import CVTArchive
from slime.memory.pool import DynamicPool, PoolConfig
from slime.proto.kernel import Kernel
from slime.kernels.torch_fallback import TorchKernel
from slime.observability.metrics import MetricsCollector
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

    def __init__(self, sensory_dim: int, latent_dim: int, head_dim: int, device: Optional[torch.device]=None, kernel: Optional[Kernel]=None, pool_config: Optional[PoolConfig]=None, metrics_collector: Optional[MetricsCollector]=None):
        super().__init__()
        self.sensory_dim = sensory_dim
        self.latent_dim = latent_dim
        self.head_dim = head_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if kernel is not None:
            self.kernel = kernel
        elif HAS_TRITON:
            self.kernel = TritonKernel(self.device)
            logger.info('Using Triton GPU kernels for maximum performance')
        else:
            self.kernel = TorchKernel(self.device)
            logger.warning('Triton not available, using PyTorch fallback')
        self.metrics = metrics_collector
        self.encode = nn.Sequential(nn.Linear(sensory_dim, latent_dim), nn.LayerNorm(latent_dim), nn.Tanh()).to(self.device)
        self.decode = nn.Sequential(nn.Linear(latent_dim, sensory_dim), nn.Tanh()).to(self.device)
        self.predict_rank = nn.Linear(latent_dim, 1).to(self.device)
        self.predict_coherence = nn.Linear(latent_dim, 1).to(self.device)
        self.project_heads = nn.Linear(head_dim, latent_dim).to(self.device)
        self.archive = CVTArchive(behavioral_dims=5, num_centroids=100, low_rank_k=32, kmo_threshold=0.6, seed=42)
        self.chemotaxis = Chemotaxis(self.archive, self.device)
        if pool_config is None:
            pool_config = PoolConfig(min_size=4, max_size=32, birth_threshold=0.8, death_threshold=0.1, cull_interval=100)
        self.pseudopod_pool = DynamicPool(component_factory=lambda: Pseudopod(head_dim, self.kernel, self.device), config=pool_config, bootstrap_factory=lambda genome: Pseudopod.from_dict(genome, self.kernel, self.device), archive=self.archive)
        self._generation = 0

    def forward(self, stimulus: torch.Tensor, state: Optional[FlowState]=None) -> Tuple[torch.Tensor, FlowState]:
        if self.metrics:
            self.metrics.start_step()
        batch_size = stimulus.shape[0]
        body = self.encode(stimulus)
        if state is not None:
            body = 0.7 * body + 0.3 * state.body
        rank = torch.sigmoid(self.predict_rank(body.mean(0, keepdim=True))).item()
        coherence = torch.sigmoid(self.predict_coherence(body.mean(0, keepdim=True))).item()
        behavior = (rank, coherence)
        pseudopods = self.pseudopod_pool.get_at(behavior, max_count=8)
        if not pseudopods:
            logger.warning('Empty pseudopod pool, spawning emergency pseudopod')
            pseudopods = [Pseudopod(self.head_dim, self.kernel, self.device)]
        outputs = []
        max_rank = torch.tensor(0.0, device=self.device)
        min_coherence = torch.tensor(1.0, device=self.device)
        for pod in pseudopods:
            pod_input = body[:, :self.head_dim]
            stim_input = stimulus[:, :self.head_dim]
            output = pod(pod_input, stim_input)
            outputs.append(output)
            pod_rank = pod.effective_rank()
            pod_coherence = pod.coherence()
            max_rank = torch.maximum(max_rank, pod_rank)
            min_coherence = torch.minimum(min_coherence, pod_coherence)
        merged = torch.stack(outputs).mean(0)
        merged_latent = self.project_heads(merged)
        fitness = (max_rank * min_coherence).item()
        for pod in pseudopods:
            pod_behavior = (torch.clamp(pod.effective_rank(), 0, 1).item(), torch.clamp(pod.coherence(), 0, 1).item())
            self.archive.add(pod, pod_behavior, pod.fitness)
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