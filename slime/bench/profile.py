import torch
import time
from typing import Dict, Optional, Callable, List, Tuple
from dataclasses import dataclass, field
import logging
from contextlib import contextmanager
logger = logging.getLogger(__name__)

@dataclass
class ProfileResult:
    model_name: str
    batch_size: int
    sequence_length: int
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float
    peak_memory_mb: float
    allocated_memory_mb: float
    samples_per_second: float
    tokens_per_second: float
    num_parameters: int
    flops: Optional[int] = None
    extra_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class ComparisonResult:
    baseline: ProfileResult
    slime: ProfileResult
    forward_speedup: float
    backward_speedup: float
    total_speedup: float
    memory_reduction: float
    throughput_improvement: float
    num_trials: int
    forward_std: float
    backward_std: float

class Profiler:

    def __init__(self, device: Optional[torch.device]=None, warmup_steps: int=10, profile_steps: int=100):
        self.device = device or torch.device('cuda')
        self.warmup_steps = warmup_steps
        self.profile_steps = profile_steps

    def profile_model(self, model: torch.nn.Module, batch_size: int, sequence_length: int, input_dim: int, model_name: str='model', task_type: str='regression') -> ProfileResult:
        model = model.to(self.device)
        model.train()
        x = torch.randn(batch_size, sequence_length, input_dim, device=self.device)
        if task_type == 'regression':
            target = torch.randn(batch_size, sequence_length, input_dim, device=self.device)
        else:
            target = torch.randint(0, input_dim, (batch_size, sequence_length), device=self.device)
        logger.info(f'Warming up {model_name} for {self.warmup_steps} steps...')
        for _ in range(self.warmup_steps):
            output = model(x)
            if task_type == 'regression':
                loss = torch.nn.functional.mse_loss(output, target)
            else:
                loss = torch.nn.functional.cross_entropy(output.view(-1, output.size(-1)), target.view(-1))
            loss.backward()
            model.zero_grad()
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        logger.info(f'Profiling {model_name} forward pass...')
        forward_times = []
        for _ in range(self.profile_steps):
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            output = model(x)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            forward_times.append((end - start) * 1000)
        logger.info(f'Profiling {model_name} backward pass...')
        backward_times = []
        for _ in range(self.profile_steps):
            output = model(x)
            if task_type == 'regression':
                loss = torch.nn.functional.mse_loss(output, target)
            else:
                loss = torch.nn.functional.cross_entropy(output.view(-1, output.size(-1)), target.view(-1))
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            loss.backward()
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            backward_times.append((end - start) * 1000)
            model.zero_grad()
        import statistics
        forward_std = statistics.stdev(forward_times) if len(forward_times) > 1 else 0.0
        backward_std = statistics.stdev(backward_times) if len(backward_times) > 1 else 0.0
        if self.device.type == 'cuda':
            peak_memory_mb = torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
            allocated_memory_mb = torch.cuda.memory_allocated(self.device) / 1024 ** 2
            torch.cuda.reset_peak_memory_stats(self.device)
        else:
            peak_memory_mb = 0.0
            allocated_memory_mb = 0.0
        forward_time_ms = sum(forward_times) / len(forward_times)
        backward_time_ms = sum(backward_times) / len(backward_times)
        total_time_ms = forward_time_ms + backward_time_ms
        samples_per_second = 1000.0 * batch_size / total_time_ms
        tokens_per_second = samples_per_second * sequence_length
        num_parameters = sum((p.numel() for p in model.parameters()))
        logger.info(f'{model_name} profile complete: forward={forward_time_ms:.2f}±{forward_std:.2f}ms, backward={backward_time_ms:.2f}±{backward_std:.2f}ms, memory={peak_memory_mb:.1f}MB')
        result = ProfileResult(model_name=model_name, batch_size=batch_size, sequence_length=sequence_length, forward_time_ms=forward_time_ms, backward_time_ms=backward_time_ms, total_time_ms=total_time_ms, peak_memory_mb=peak_memory_mb, allocated_memory_mb=allocated_memory_mb, samples_per_second=samples_per_second, tokens_per_second=tokens_per_second, num_parameters=num_parameters)
        result.extra_metrics['forward_std'] = forward_std
        result.extra_metrics['backward_std'] = backward_std
        return result

    def compare_models(self, baseline_model: torch.nn.Module, slime_model: torch.nn.Module, batch_size: int, sequence_length: int, input_dim: int, task_type: str='regression') -> ComparisonResult:
        logger.info('=' * 60)
        logger.info('PROFILING BASELINE')
        logger.info('=' * 60)
        baseline_result = self.profile_model(baseline_model, batch_size, sequence_length, input_dim, model_name='Baseline Transformer', task_type=task_type)
        logger.info('=' * 60)
        logger.info('PROFILING SLIME MOLD')
        logger.info('=' * 60)
        slime_result = self.profile_model(slime_model, batch_size, sequence_length, input_dim, model_name='Slime Mold', task_type=task_type)
        forward_speedup = baseline_result.forward_time_ms / slime_result.forward_time_ms
        backward_speedup = baseline_result.backward_time_ms / slime_result.backward_time_ms
        total_speedup = baseline_result.total_time_ms / slime_result.total_time_ms
        memory_reduction = baseline_result.peak_memory_mb / slime_result.peak_memory_mb
        throughput_improvement = slime_result.samples_per_second / baseline_result.samples_per_second
        comparison = ComparisonResult(baseline=baseline_result, slime=slime_result, forward_speedup=forward_speedup, backward_speedup=backward_speedup, total_speedup=total_speedup, memory_reduction=memory_reduction, throughput_improvement=throughput_improvement, num_trials=self.profile_steps, forward_std=baseline_result.extra_metrics.get('forward_std', 0.0), backward_std=baseline_result.extra_metrics.get('backward_std', 0.0))
        self.print_comparison(comparison)
        return comparison

    def print_comparison(self, result: ComparisonResult) -> None:
        print('\n' + '=' * 80)
        print('PERFORMANCE COMPARISON: Slime Mold vs Baseline Transformer')
        print('=' * 80)
        print(f"\n{'Metric':<30} {'Baseline':<20} {'Slime Mold':<20} {'Ratio':>10}")
        print('-' * 80)
        print(f"{'Forward time (ms)':<30} {result.baseline.forward_time_ms:<20.2f} {result.slime.forward_time_ms:<20.2f} {result.forward_speedup:>9.2f}x")
        print(f"{'Backward time (ms)':<30} {result.baseline.backward_time_ms:<20.2f} {result.slime.backward_time_ms:<20.2f} {result.backward_speedup:>9.2f}x")
        print(f"{'Total time (ms)':<30} {result.baseline.total_time_ms:<20.2f} {result.slime.total_time_ms:<20.2f} {result.total_speedup:>9.2f}x")
        print(f"\n{'Peak memory (MB)':<30} {result.baseline.peak_memory_mb:<20.1f} {result.slime.peak_memory_mb:<20.1f} {result.memory_reduction:>9.2f}x")
        print(f"\n{'Samples/second':<30} {result.baseline.samples_per_second:<20.1f} {result.slime.samples_per_second:<20.1f} {result.throughput_improvement:>9.2f}x")
        print(f"{'Tokens/second':<30} {result.baseline.tokens_per_second:<20.1f} {result.slime.tokens_per_second:<20.1f} {result.throughput_improvement:>9.2f}x")
        print(f"\n{'Parameters':<30} {result.baseline.num_parameters:<20,} {result.slime.num_parameters:<20,} {result.slime.num_parameters / result.baseline.num_parameters:>9.2f}x")
        print('\n' + '=' * 80)
        if result.total_speedup > 1.1:
            verdict = '✓ SLIME MOLD IS FASTER'
        elif result.total_speedup < 0.9:
            verdict = '✗ BASELINE IS FASTER'
        else:
            verdict = '≈ COMPARABLE PERFORMANCE'
        print(f'VERDICT: {verdict}')
        print('=' * 80 + '\n')

@contextmanager
def profile_section(name: str, device: Optional[torch.device]=None):
    if device and device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        if device and device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f'{name}: {elapsed:.2f}ms')

def count_flops(model: torch.nn.Module, batch_size: int, sequence_length: int, input_dim: int) -> int:
    num_params = sum((p.numel() for p in model.parameters()))
    attention_flops = 2 * batch_size * sequence_length * sequence_length * input_dim
    linear_flops = 2 * num_params * batch_size * sequence_length
    total_flops = attention_flops + linear_flops
    return int(total_flops)