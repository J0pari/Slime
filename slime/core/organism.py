"""Organism: main coordinating entity"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

from slime.core.pseudopod import Pseudopod
from slime.core.state import FlowState
from slime.memory.archive import BehavioralArchive
from slime.memory.pool import DynamicPool, PoolConfig

logger = logging.getLogger(__name__)


class Organism(nn.Module):
    """Plasmodium: self-organizing organism.

    Implements proto.model.Organism protocol.

    Key properties:
    - Dynamic pseudopod pool (birth/death based on fitness)
    - MAP-Elites archive for quality-diversity
    - Behavioral routing (location in rank-coherence space)
    - No static structure (everything scales with compute)
    """

    def __init__(
        self,
        sensory_dim: int,
        latent_dim: int,
        head_dim: int,
        device: torch.device = None,
        pool_config: Optional[PoolConfig] = None,
    ):
        super().__init__()

        self.sensory_dim = sensory_dim
        self.latent_dim = latent_dim
        self.head_dim = head_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Sensory encoding
        self.encode = nn.Sequential(
            nn.Linear(sensory_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
        ).to(self.device)

        # Decoding for reconstruction
        self.decode = nn.Sequential(
            nn.Linear(latent_dim, sensory_dim),
            nn.Tanh(),
        ).to(self.device)

        # Behavioral space predictors (learned)
        self.predict_rank = nn.Linear(latent_dim, 1).to(self.device)
        self.predict_coherence = nn.Linear(latent_dim, 1).to(self.device)

        # MAP-Elites archive
        self.archive = BehavioralArchive(
            dimensions=['rank', 'coherence'],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            resolution=50,
            device=self.device,
        )

        # Dynamic pseudopod pool
        if pool_config is None:
            pool_config = PoolConfig(
                min_size=4,
                max_size=32,
                birth_threshold=0.8,
                death_threshold=0.1,
                cull_interval=100,
            )

        self.pseudopod_pool = DynamicPool(
            component_factory=lambda: Pseudopod(head_dim, self.device),
            config=pool_config,
            archive=self.archive,
        )

        # State
        self._generation = 0

    def forward(
        self,
        stimulus: torch.Tensor,
        state: Optional[FlowState] = None,
    ) -> Tuple[torch.Tensor, FlowState]:
        """Single forward pass through organism.

        Args:
            stimulus: Input [batch, sensory_dim]
            state: Optional previous state

        Returns:
            (output, new_state): Output and updated state
        """
        batch_size = stimulus.shape[0]

        # Encode stimulus
        body = self.encode(stimulus)

        if state is not None:
            # Integrate with previous body state
            body = 0.7 * body + 0.3 * state.body

        # Predict behavioral coordinates
        rank = torch.sigmoid(self.predict_rank(body.mean(0, keepdim=True))).item()
        coherence = torch.sigmoid(self.predict_coherence(body.mean(0, keepdim=True))).item()
        behavior = (rank, coherence)

        # Get active pseudopods for this location
        pseudopods = self.pseudopod_pool.get_at(behavior, max_count=8)

        if not pseudopods:
            # Emergency: spawn if pool is empty
            logger.warning("Empty pseudopod pool, spawning emergency pseudopod")
            pseudopods = [Pseudopod(self.head_dim, self.device)]

        # Extend pseudopods
        outputs = []
        max_rank = torch.tensor(0.0, device=self.device)
        min_coherence = torch.tensor(1.0, device=self.device)

        for pod in pseudopods:
            # Slice body for this head
            pod_input = body[:, :self.head_dim]
            stim_input = stimulus[:, :self.head_dim]

            output = pod(pod_input, stim_input)
            outputs.append(output)

            # Track behavioral metrics
            pod_rank = pod.effective_rank()
            pod_coherence = pod.coherence()

            max_rank = torch.maximum(max_rank, pod_rank)
            min_coherence = torch.minimum(min_coherence, pod_coherence)

        # Merge pseudopod outputs
        merged = torch.stack(outputs).mean(0)

        # Compute fitness (information-theoretic)
        fitness = (max_rank * min_coherence).item()

        # Archive best pseudopods
        for pod in pseudopods:
            pod_behavior = (
                torch.clamp(pod.effective_rank(), 0, 1).item(),
                torch.clamp(pod.coherence(), 0, 1).item(),
            )
            self.archive.add(pod, pod_behavior, pod.fitness)

        # Pool lifecycle step
        self.pseudopod_pool.step(behavior)

        # Create new state
        new_state = FlowState(
            body=merged,
            behavior=behavior,
            generation=self._generation,
            fitness=fitness,
        )

        # Decode output
        output = self.decode(merged)

        self._generation += 1
        self.archive.increment_generation()

        return output, new_state

    def reset_state(self) -> None:
        """Clear all internal state"""
        self.archive.clear()
        self.pseudopod_pool.clear()
        self._generation = 0

    def stats(self) -> dict:
        """Get system statistics"""
        return {
            'generation': self._generation,
            'archive_size': self.archive.size(),
            'archive_coverage': self.archive.coverage(),
            'archive_max_fitness': self.archive.max_fitness(),
            'pool_stats': self.pseudopod_pool.stats(),
        }
