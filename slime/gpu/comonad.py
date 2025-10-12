"""
GPU Comonadic Orchestration.

Comonadic GPU context-aware spawn/retire decisions. Like Polynesian navigator
reading ocean/stars/birds as unified field, read warps/cache/tensor-cores as
unified substrate.

Interface:
    extract(warp_id) → LocalObservation (warp occupancy, cache hits, etc.)
    extend(decision_fn) → Apply context-aware decisions to whole field
"""

import torch
import torch.nn as nn
from typing import Callable, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LocalObservation:
    """
    GPU execution state visible at a single warp.

    Fields:
        warp_occupancy: Active threads / 32 (1.0 = full utilization)
        cache_hit_rate: L1/L2 cache hits (1.0 = all cached)
        tensor_core_util: Tensor core usage (1.0 = saturated)
        register_pressure: Register allocation (1.0 = at limit)
        memory_throughput: Fraction of peak HBM bandwidth
        compute_intensity: FLOPs per byte (arithmetic intensity)
    """
    warp_occupancy: float
    cache_hit_rate: float
    tensor_core_util: float
    register_pressure: float
    memory_throughput: float
    compute_intensity: float


class GPUContext(nn.Module):
    """
    Comonadic GPU execution context.

    Comonad structure:
        extract: Context → LocalObservation
        extend: (Context → Decision) → Context → Context

    GPU-aware resource allocation:
        - Spawn when occupancy < 0.7 (underutilized warps)
        - Retire when cache_hit_rate < 0.5 (thrashing)
        - Prefer tensor-core-heavy operations when available

    Polynesian navigator metaphor:
        Each warp sees local state (waves, wind), but decisions informed by
        whole field (stars, swells, birds across ocean).
    """

    def __init__(
        self,
        num_warps: int = 32,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.num_warps = num_warps
        self.device = device or torch.device('cuda')

        # GPU state (updated via profiling)
        self.register_buffer('_warp_occupancy', torch.ones(num_warps, device=self.device))
        self.register_buffer('_cache_hit_rate', torch.ones(num_warps, device=self.device))
        self.register_buffer('_tensor_core_util', torch.zeros(num_warps, device=self.device))
        self.register_buffer('_register_pressure', torch.zeros(num_warps, device=self.device))
        self.register_buffer('_memory_throughput', torch.ones(num_warps, device=self.device))
        self.register_buffer('_compute_intensity', torch.ones(num_warps, device=self.device))

        # EMA decay for smoothing metrics
        self.ema_decay = 0.9

    def extract(self, warp_id: int) -> LocalObservation:
        """
        Comonad extract: Get local observation at specific warp.

        Args:
            warp_id: Warp index [0, num_warps)

        Returns:
            LocalObservation with GPU state at this warp
        """
        return LocalObservation(
            warp_occupancy=self._warp_occupancy[warp_id].item(),
            cache_hit_rate=self._cache_hit_rate[warp_id].item(),
            tensor_core_util=self._tensor_core_util[warp_id].item(),
            register_pressure=self._register_pressure[warp_id].item(),
            memory_throughput=self._memory_throughput[warp_id].item(),
            compute_intensity=self._compute_intensity[warp_id].item()
        )

    def extend(
        self,
        decision_fn: Callable[[LocalObservation, torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Comonad extend: Apply context-aware decision function to whole field.

        Polynesian navigator: Each warp sees local state, but decision informed
        by entire GPU field (neighboring warps, global throughput, etc.).

        Args:
            decision_fn: (LocalObservation, global_context) → decision
                LocalObservation: Local warp state
                global_context: Tensor with aggregate GPU metrics
                Returns: Decision tensor (spawn/retire/continue)

        Returns:
            Tensor[num_warps] with decisions for each warp
        """
        # Global context: aggregate metrics across all warps
        global_context = torch.stack([
            self._warp_occupancy.mean(),
            self._cache_hit_rate.mean(),
            self._tensor_core_util.mean(),
            self._register_pressure.mean(),
            self._memory_throughput.mean(),
            self._compute_intensity.mean()
        ])

        # Apply decision function to each warp with global context
        decisions = []
        for warp_id in range(self.num_warps):
            local_obs = self.extract(warp_id)
            decision = decision_fn(local_obs, global_context)
            decisions.append(decision)

        return torch.stack(decisions)

    def update_metrics(
        self,
        warp_occupancy: Optional[torch.Tensor] = None,
        cache_hit_rate: Optional[torch.Tensor] = None,
        tensor_core_util: Optional[torch.Tensor] = None,
        register_pressure: Optional[torch.Tensor] = None,
        memory_throughput: Optional[torch.Tensor] = None,
        compute_intensity: Optional[torch.Tensor] = None
    ):
        """
        Update GPU execution metrics with EMA smoothing.

        Args:
            warp_occupancy: Tensor[num_warps] - active threads per warp
            cache_hit_rate: Tensor[num_warps] - L1/L2 cache hits
            tensor_core_util: Tensor[num_warps] - tensor core usage
            register_pressure: Tensor[num_warps] - register allocation
            memory_throughput: Tensor[num_warps] - HBM bandwidth usage
            compute_intensity: Tensor[num_warps] - FLOPs per byte
        """
        if warp_occupancy is not None:
            self._warp_occupancy = (self.ema_decay * self._warp_occupancy +
                                   (1 - self.ema_decay) * warp_occupancy)

        if cache_hit_rate is not None:
            self._cache_hit_rate = (self.ema_decay * self._cache_hit_rate +
                                   (1 - self.ema_decay) * cache_hit_rate)

        if tensor_core_util is not None:
            self._tensor_core_util = (self.ema_decay * self._tensor_core_util +
                                     (1 - self.ema_decay) * tensor_core_util)

        if register_pressure is not None:
            self._register_pressure = (self.ema_decay * self._register_pressure +
                                      (1 - self.ema_decay) * register_pressure)

        if memory_throughput is not None:
            self._memory_throughput = (self.ema_decay * self._memory_throughput +
                                      (1 - self.ema_decay) * memory_throughput)

        if compute_intensity is not None:
            self._compute_intensity = (self.ema_decay * self._compute_intensity +
                                      (1 - self.ema_decay) * compute_intensity)

    def should_spawn_decision(
        self,
        local_obs: LocalObservation,
        global_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Context-aware spawn decision function.

        Spawn when:
        - Low occupancy (underutilized warps)
        - High cache hit rate (good locality)
        - Low register pressure (room for more work)

        Args:
            local_obs: Local warp observation
            global_context: Aggregate GPU metrics

        Returns:
            Scalar spawn score [0, 1]
        """
        # Spawn when underutilized but cache-friendly
        occupancy_score = 1.0 - local_obs.warp_occupancy
        cache_score = local_obs.cache_hit_rate
        register_score = 1.0 - local_obs.register_pressure

        # Global context: only spawn if average occupancy low
        global_occupancy = global_context[0].item()
        global_gate = 1.0 if global_occupancy < 0.7 else 0.0

        spawn_score = (
            0.5 * occupancy_score +
            0.3 * cache_score +
            0.2 * register_score
        ) * global_gate

        return torch.tensor(spawn_score, device=self.device)

    def should_retire_decision(
        self,
        local_obs: LocalObservation,
        global_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Context-aware retire decision function.

        Retire when:
        - Low cache hit rate (thrashing)
        - High register pressure (resource contention)
        - Low compute intensity (memory-bound)

        Args:
            local_obs: Local warp observation
            global_context: Aggregate GPU metrics

        Returns:
            Scalar retire score [0, 1]
        """
        # Retire when thrashing or resource-starved
        cache_score = 1.0 - local_obs.cache_hit_rate
        register_score = local_obs.register_pressure
        compute_score = 1.0 - local_obs.compute_intensity

        # Global context: retire if average cache hit rate low (global thrashing)
        global_cache = global_context[1].item()
        global_gate = 1.0 if global_cache < 0.5 else 0.0

        retire_score = (
            0.5 * cache_score +
            0.3 * register_score +
            0.2 * compute_score
        ) * global_gate

        return torch.tensor(retire_score, device=self.device)


def make_spawn_retire_decisions(
    gpu_context: GPUContext,
    pool_size: int,
    max_pool_size: int
) -> Tuple[bool, int]:
    """
    GPU-aware spawn/retire decisions using comonadic context.

    Args:
        gpu_context: Comonadic GPU context
        pool_size: Current pool size
        max_pool_size: Maximum allowed pool size

    Returns:
        (should_spawn, retire_count) tuple
    """
    # Spawn decision: aggregate spawn scores across all warps
    spawn_scores = gpu_context.extend(gpu_context.should_spawn_decision)
    should_spawn = (spawn_scores.mean() > 0.5).item() and (pool_size < max_pool_size)

    # Retire decision: count warps with high retire scores
    retire_scores = gpu_context.extend(gpu_context.should_retire_decision)
    retire_count = (retire_scores > 0.7).sum().item()

    return should_spawn, retire_count
