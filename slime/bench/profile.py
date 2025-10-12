"""Performance profiling utilities for comparing architectures"""

import torch
import time
from typing import Dict, Optional, Callable, List, Tuple
from dataclasses import dataclass, field
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Results from a profiling run"""
    model_name: str
    batch_size: int
    sequence_length: int

    # Latency metrics
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float

    # Memory metrics
    peak_memory_mb: float
    allocated_memory_mb: float

    # Throughput
    samples_per_second: float
    tokens_per_second: float

    # Model-specific
    num_parameters: int
    flops: Optional[int] = None

    # Additional metrics
    extra_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Comparison between baseline and slime mold"""
    baseline: ProfileResult
    slime: ProfileResult

    # Speedup ratios (slime vs baseline)
    forward_speedup: float
    backward_speedup: float
    total_speedup: float
    memory_reduction: float
    throughput_improvement: float

    # Statistical significance
    num_trials: int
    forward_std: float
    backward_std: float


class Profiler:
    """Profile model performance with warm-up and multiple trials"""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        warmup_steps: int = 10,
        profile_steps: int = 100,
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.warmup_steps = warmup_steps
        self.profile_steps = profile_steps

    def profile_model(
        self,
        model: torch.nn.Module,
        batch_size: int,
        sequence_length: int,
        input_dim: int,
        model_name: str = "model",
        task_type: str = "regression",
    ) -> ProfileResult:
        """Profile a model's forward and backward pass.

        Args:
            model: Model to profile
            batch_size: Batch size for profiling
            sequence_length: Sequence length
            input_dim: Input dimension
            model_name: Name for result identification
            task_type: "regression" or "classification"

        Returns:
            ProfileResult with timing and memory metrics
        """
        model = model.to(self.device)
        model.train()

        # Generate synthetic input
        x = torch.randn(batch_size, sequence_length, input_dim, device=self.device)
        if task_type == "regression":
            target = torch.randn(batch_size, sequence_length, input_dim, device=self.device)
        else:
            target = torch.randint(0, input_dim, (batch_size, sequence_length), device=self.device)

        # Warmup
        logger.info(f"Warming up {model_name} for {self.warmup_steps} steps...")
        for _ in range(self.warmup_steps):
            output = model(x)
            if task_type == "regression":
                loss = torch.nn.functional.mse_loss(output, target)
            else:
                loss = torch.nn.functional.cross_entropy(
                    output.view(-1, output.size(-1)), target.view(-1)
                )
            loss.backward()
            model.zero_grad()

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # Profile forward pass
        logger.info(f"Profiling {model_name} forward pass...")
        forward_times = []
        for _ in range(self.profile_steps):
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()

            output = model(x)

            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            forward_times.append((end - start) * 1000)  # ms

        # Profile backward pass
        logger.info(f"Profiling {model_name} backward pass...")
        backward_times = []
        for _ in range(self.profile_steps):
            output = model(x)
            if task_type == "regression":
                loss = torch.nn.functional.mse_loss(output, target)
            else:
                loss = torch.nn.functional.cross_entropy(
                    output.view(-1, output.size(-1)), target.view(-1)
                )

            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()

            loss.backward()

            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            backward_times.append((end - start) * 1000)  # ms
            model.zero_grad()

        # Compute standard deviations
        import statistics
        forward_std = statistics.stdev(forward_times) if len(forward_times) > 1 else 0.0
        backward_std = statistics.stdev(backward_times) if len(backward_times) > 1 else 0.0

        # Memory metrics
        if self.device.type == 'cuda':
            peak_memory_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2
            allocated_memory_mb = torch.cuda.memory_allocated(self.device) / 1024**2
            torch.cuda.reset_peak_memory_stats(self.device)
        else:
            peak_memory_mb = 0.0
            allocated_memory_mb = 0.0

        # Compute statistics
        forward_time_ms = sum(forward_times) / len(forward_times)
        backward_time_ms = sum(backward_times) / len(backward_times)
        total_time_ms = forward_time_ms + backward_time_ms

        samples_per_second = 1000.0 * batch_size / total_time_ms
        tokens_per_second = samples_per_second * sequence_length

        # Count parameters
        num_parameters = sum(p.numel() for p in model.parameters())

        logger.info(f"{model_name} profile complete: "
                   f"forward={forward_time_ms:.2f}±{forward_std:.2f}ms, "
                   f"backward={backward_time_ms:.2f}±{backward_std:.2f}ms, "
                   f"memory={peak_memory_mb:.1f}MB")

        result = ProfileResult(
            model_name=model_name,
            batch_size=batch_size,
            sequence_length=sequence_length,
            forward_time_ms=forward_time_ms,
            backward_time_ms=backward_time_ms,
            total_time_ms=total_time_ms,
            peak_memory_mb=peak_memory_mb,
            allocated_memory_mb=allocated_memory_mb,
            samples_per_second=samples_per_second,
            tokens_per_second=tokens_per_second,
            num_parameters=num_parameters,
        )

        # Store std deviations as extra metrics
        result.extra_metrics['forward_std'] = forward_std
        result.extra_metrics['backward_std'] = backward_std

        return result

    def compare_models(
        self,
        baseline_model: torch.nn.Module,
        slime_model: torch.nn.Module,
        batch_size: int,
        sequence_length: int,
        input_dim: int,
        task_type: str = "regression",
    ) -> ComparisonResult:
        """Compare baseline transformer vs slime mold.

        Args:
            baseline_model: Baseline transformer model
            slime_model: Slime mold model
            batch_size: Batch size for profiling
            sequence_length: Sequence length
            input_dim: Input dimension
            task_type: "regression" or "classification"

        Returns:
            ComparisonResult with speedup ratios
        """
        logger.info("=" * 60)
        logger.info("PROFILING BASELINE")
        logger.info("=" * 60)
        baseline_result = self.profile_model(
            baseline_model,
            batch_size,
            sequence_length,
            input_dim,
            model_name="Baseline Transformer",
            task_type=task_type,
        )

        logger.info("=" * 60)
        logger.info("PROFILING SLIME MOLD")
        logger.info("=" * 60)
        slime_result = self.profile_model(
            slime_model,
            batch_size,
            sequence_length,
            input_dim,
            model_name="Slime Mold",
            task_type=task_type,
        )

        # Compute speedup ratios
        forward_speedup = baseline_result.forward_time_ms / slime_result.forward_time_ms
        backward_speedup = baseline_result.backward_time_ms / slime_result.backward_time_ms
        total_speedup = baseline_result.total_time_ms / slime_result.total_time_ms
        memory_reduction = baseline_result.peak_memory_mb / slime_result.peak_memory_mb
        throughput_improvement = slime_result.samples_per_second / baseline_result.samples_per_second

        comparison = ComparisonResult(
            baseline=baseline_result,
            slime=slime_result,
            forward_speedup=forward_speedup,
            backward_speedup=backward_speedup,
            total_speedup=total_speedup,
            memory_reduction=memory_reduction,
            throughput_improvement=throughput_improvement,
            num_trials=self.profile_steps,
            forward_std=baseline_result.extra_metrics.get('forward_std', 0.0),
            backward_std=baseline_result.extra_metrics.get('backward_std', 0.0),
        )

        self.print_comparison(comparison)
        return comparison

    def print_comparison(self, result: ComparisonResult) -> None:
        """Pretty-print comparison results"""
        print("\n" + "=" * 80)
        print("PERFORMANCE COMPARISON: Slime Mold vs Baseline Transformer")
        print("=" * 80)

        print(f"\n{'Metric':<30} {'Baseline':<20} {'Slime Mold':<20} {'Ratio':>10}")
        print("-" * 80)

        # Latency
        print(f"{'Forward time (ms)':<30} "
              f"{result.baseline.forward_time_ms:<20.2f} "
              f"{result.slime.forward_time_ms:<20.2f} "
              f"{result.forward_speedup:>9.2f}x")

        print(f"{'Backward time (ms)':<30} "
              f"{result.baseline.backward_time_ms:<20.2f} "
              f"{result.slime.backward_time_ms:<20.2f} "
              f"{result.backward_speedup:>9.2f}x")

        print(f"{'Total time (ms)':<30} "
              f"{result.baseline.total_time_ms:<20.2f} "
              f"{result.slime.total_time_ms:<20.2f} "
              f"{result.total_speedup:>9.2f}x")

        # Memory
        print(f"\n{'Peak memory (MB)':<30} "
              f"{result.baseline.peak_memory_mb:<20.1f} "
              f"{result.slime.peak_memory_mb:<20.1f} "
              f"{result.memory_reduction:>9.2f}x")

        # Throughput
        print(f"\n{'Samples/second':<30} "
              f"{result.baseline.samples_per_second:<20.1f} "
              f"{result.slime.samples_per_second:<20.1f} "
              f"{result.throughput_improvement:>9.2f}x")

        print(f"{'Tokens/second':<30} "
              f"{result.baseline.tokens_per_second:<20.1f} "
              f"{result.slime.tokens_per_second:<20.1f} "
              f"{result.throughput_improvement:>9.2f}x")

        # Parameters
        print(f"\n{'Parameters':<30} "
              f"{result.baseline.num_parameters:<20,} "
              f"{result.slime.num_parameters:<20,} "
              f"{result.slime.num_parameters / result.baseline.num_parameters:>9.2f}x")

        print("\n" + "=" * 80)

        # Verdict
        if result.total_speedup > 1.1:
            verdict = "✓ SLIME MOLD IS FASTER"
        elif result.total_speedup < 0.9:
            verdict = "✗ BASELINE IS FASTER"
        else:
            verdict = "≈ COMPARABLE PERFORMANCE"

        print(f"VERDICT: {verdict}")
        print("=" * 80 + "\n")


@contextmanager
def profile_section(name: str, device: Optional[torch.device] = None):
    """Context manager for profiling code sections.

    Usage:
        with profile_section("attention computation"):
            output = attention(q, k, v)
    """
    if device and device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()
    try:
        yield
    finally:
        if device and device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"{name}: {elapsed:.2f}ms")


def count_flops(
    model: torch.nn.Module,
    batch_size: int,
    sequence_length: int,
    input_dim: int,
) -> int:
    """Estimate FLOPs for a forward pass.

    Rough estimation based on parameter count and sequence length.
    For precise counting, use fvcore or torchinfo.

    Args:
        model: Model to count FLOPs for
        batch_size: Batch size
        sequence_length: Sequence length
        input_dim: Input dimension

    Returns:
        Estimated FLOPs (floating point operations)
    """
    num_params = sum(p.numel() for p in model.parameters())

    # Rough estimate: 2 FLOPs per parameter per token (forward + backward)
    # Attention: O(N^2 * D) for sequence length N, dimension D
    attention_flops = 2 * batch_size * sequence_length * sequence_length * input_dim

    # Linear layers: 2 * params (MAC = multiply + add)
    linear_flops = 2 * num_params * batch_size * sequence_length

    total_flops = attention_flops + linear_flops

    return int(total_flops)
