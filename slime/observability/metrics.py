import time
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import logging
logger = logging.getLogger(__name__)

@dataclass
class MetricsSnapshot:
    timestamp: float
    step: int
    inference_latency_ms: float
    memory_allocated_mb: float
    memory_reserved_mb: float
    gpu_utilization: Optional[float]
    batch_size: int
    throughput_samples_per_sec: float
    pool_size: int
    archive_size: int
    archive_coverage: float
    loss: Optional[float] = None

class MetricsCollector:

    def __init__(self, history_size: int=1000):
        self.history_size = history_size
        self._snapshots: deque = deque(maxlen=history_size)
        self._start_time: Optional[float] = None
        self._current_step: int = 0

    def start_step(self) -> None:
        self._start_time = time.perf_counter()

    def end_step(self, batch_size: int, pool_size: int, archive_size: int, archive_coverage: float, loss: Optional[float]=None) -> MetricsSnapshot:
        if self._start_time is None:
            raise RuntimeError('start_step() not called')
        end_time = time.perf_counter()
        latency_ms = (end_time - self._start_time) * 1000
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024 ** 2
            memory_reserved = torch.cuda.memory_reserved() / 1024 ** 2
            try:
                gpu_util = torch.cuda.utilization()
            except:
                gpu_util = None
        else:
            memory_allocated = 0.0
            memory_reserved = 0.0
            gpu_util = None
        throughput = batch_size / (latency_ms / 1000) if latency_ms > 0 else 0.0
        snapshot = MetricsSnapshot(timestamp=time.time(), step=self._current_step, inference_latency_ms=latency_ms, memory_allocated_mb=memory_allocated, memory_reserved_mb=memory_reserved, gpu_utilization=gpu_util, batch_size=batch_size, throughput_samples_per_sec=throughput, pool_size=pool_size, archive_size=archive_size, archive_coverage=archive_coverage, loss=loss)
        self._snapshots.append(snapshot)
        self._start_time = None
        self._current_step += 1
        return snapshot

    def get_recent(self, n: int=100) -> List[MetricsSnapshot]:
        return list(self._snapshots)[-n:]

    def compute_percentile(self, metric: str, percentile: float, window: Optional[int]=None) -> float:
        snapshots = list(self._snapshots)
        if window is not None:
            snapshots = snapshots[-window:]
        if not snapshots:
            return 0.0
        values = [getattr(s, metric) for s in snapshots]
        values_tensor = torch.tensor(values)
        return torch.quantile(values_tensor, percentile / 100.0).item()

    def compute_ema(self, metric: str, alpha: float=0.1, window: Optional[int]=None) -> float:
        snapshots = list(self._snapshots)
        if window is not None:
            snapshots = snapshots[-window:]
        if not snapshots:
            return 0.0
        ema = getattr(snapshots[0], metric)
        for snapshot in snapshots[1:]:
            value = getattr(snapshot, metric)
            ema = alpha * value + (1 - alpha) * ema
        return ema

    def get_summary(self) -> Dict:
        if not self._snapshots:
            return {'num_snapshots': 0, 'duration_sec': 0.0}
        first = self._snapshots[0]
        last = self._snapshots[-1]
        return {'num_snapshots': len(self._snapshots), 'duration_sec': last.timestamp - first.timestamp, 'latency_p50_ms': self.compute_percentile('inference_latency_ms', 50), 'latency_p95_ms': self.compute_percentile('inference_latency_ms', 95), 'latency_p99_ms': self.compute_percentile('inference_latency_ms', 99), 'memory_avg_mb': self.compute_ema('memory_allocated_mb'), 'memory_peak_mb': max((s.memory_allocated_mb for s in self._snapshots)), 'throughput_avg': self.compute_ema('throughput_samples_per_sec'), 'pool_size_avg': self.compute_ema('pool_size'), 'archive_size_final': last.archive_size, 'archive_coverage_final': last.archive_coverage}

    def clear(self) -> None:
        self._snapshots.clear()
        self._start_time = None
        self._current_step = 0