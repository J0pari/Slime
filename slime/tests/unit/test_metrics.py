import pytest
import torch
from slime.observability.metrics import MetricsCollector, MetricsSnapshot

class TestMetricsCollector:
    @pytest.fixture
    def collector(self):
        return MetricsCollector(window_size=100)

    def test_initialization(self, collector):
        assert collector.window_size == 100
        assert len(collector._history) == 0

    def test_record_metrics(self, collector):
        snapshot = MetricsSnapshot(
            timestamp=1.0,
            step=0,
            inference_latency_ms=10.0,
            memory_allocated_mb=100.0,
            memory_reserved_mb=150.0,
            gpu_utilization=0.8,
            batch_size=32,
            throughput_samples_per_sec=200.0,
            pool_size=16,
            archive_size=50,
            archive_coverage=0.3,
            loss=0.5,
        )
        collector.record(snapshot)
        assert len(collector._history) == 1

    def test_window_size_enforcement(self, collector):
        for i in range(150):
            snapshot = MetricsSnapshot(
                timestamp=float(i),
                step=i,
                inference_latency_ms=10.0,
                memory_allocated_mb=100.0,
                memory_reserved_mb=150.0,
                gpu_utilization=0.8,
                batch_size=32,
                throughput_samples_per_sec=200.0,
                pool_size=16,
                archive_size=50,
                archive_coverage=0.3,
                loss=0.5,
            )
            collector.record(snapshot)
        assert len(collector._history) <= collector.window_size

    def test_get_latest(self, collector):
        snapshot1 = MetricsSnapshot(
            timestamp=1.0, step=0, inference_latency_ms=10.0,
            memory_allocated_mb=100.0, memory_reserved_mb=150.0,
            gpu_utilization=0.8, batch_size=32, throughput_samples_per_sec=200.0,
            pool_size=16, archive_size=50, archive_coverage=0.3, loss=0.5,
        )
        snapshot2 = MetricsSnapshot(
            timestamp=2.0, step=1, inference_latency_ms=12.0,
            memory_allocated_mb=110.0, memory_reserved_mb=160.0,
            gpu_utilization=0.85, batch_size=32, throughput_samples_per_sec=210.0,
            pool_size=17, archive_size=52, archive_coverage=0.32, loss=0.48,
        )
        collector.record(snapshot1)
        collector.record(snapshot2)
        latest = collector.get_latest()
        assert latest.timestamp == 2.0
        assert latest.step == 1

    def test_get_statistics(self, collector):
        for i in range(10):
            snapshot = MetricsSnapshot(
                timestamp=float(i), step=i, inference_latency_ms=10.0 + i,
                memory_allocated_mb=100.0, memory_reserved_mb=150.0,
                gpu_utilization=0.8, batch_size=32, throughput_samples_per_sec=200.0,
                pool_size=16, archive_size=50, archive_coverage=0.3, loss=0.5,
            )
            collector.record(snapshot)
        stats = collector.get_statistics()
        assert 'mean_latency' in stats
        assert 'p50_latency' in stats
        assert 'p99_latency' in stats

    def test_clear_history(self, collector):
        for i in range(10):
            snapshot = MetricsSnapshot(
                timestamp=float(i), step=i, inference_latency_ms=10.0,
                memory_allocated_mb=100.0, memory_reserved_mb=150.0,
                gpu_utilization=0.8, batch_size=32, throughput_samples_per_sec=200.0,
                pool_size=16, archive_size=50, archive_coverage=0.3, loss=0.5,
            )
            collector.record(snapshot)
        assert len(collector._history) == 10
        collector.clear()
        assert len(collector._history) == 0

    def test_empty_collector_stats(self, collector):
        stats = collector.get_statistics()
        assert stats['mean_latency'] == 0.0

    def test_get_history_range(self, collector):
        for i in range(20):
            snapshot = MetricsSnapshot(
                timestamp=float(i), step=i, inference_latency_ms=10.0,
                memory_allocated_mb=100.0, memory_reserved_mb=150.0,
                gpu_utilization=0.8, batch_size=32, throughput_samples_per_sec=200.0,
                pool_size=16, archive_size=50, archive_coverage=0.3, loss=0.5,
            )
            collector.record(snapshot)
        history = collector.get_history(last_n=5)
        assert len(history) == 5
        assert history[-1].step == 19
