"""SLO definitions and checking for production readiness"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SLOStatus(Enum):
    """SLO compliance status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    VIOLATED = "violated"


@dataclass
class SLO:
    """Service Level Objective definition"""
    name: str
    metric_name: str
    threshold: float
    comparison: str
    window_size: int
    error_budget: float

    def __post_init__(self):
        if self.comparison not in ['<', '<=', '>', '>=', '==']:
            raise ValueError(f"Invalid comparison operator: {self.comparison}")
        if self.error_budget < 0 or self.error_budget > 1:
            raise ValueError(f"Error budget must be in [0, 1], got {self.error_budget}")
        if self.window_size <= 0:
            raise ValueError(f"Window size must be positive, got {self.window_size}")


class SLOChecker:
    """Check SLOs against collected metrics"""

    def __init__(self):
        self.slos: List[SLO] = []
        self._violations: Dict[str, List[float]] = {}
        self._last_check: Dict[str, float] = {}

    def register_slo(self, slo: SLO) -> None:
        """Register an SLO for checking"""
        if any(s.name == slo.name for s in self.slos):
            raise ValueError(f"SLO {slo.name} already registered")
        self.slos.append(slo)
        self._violations[slo.name] = []
        self._last_check[slo.name] = time.time()
        logger.info(f"Registered SLO: {slo.name}")

    def check_slo(self, slo: SLO, metric_value: float) -> SLOStatus:
        """Check if metric satisfies SLO"""
        violation = False

        if slo.comparison == '<':
            violation = metric_value >= slo.threshold
        elif slo.comparison == '<=':
            violation = metric_value > slo.threshold
        elif slo.comparison == '>':
            violation = metric_value <= slo.threshold
        elif slo.comparison == '>=':
            violation = metric_value < slo.threshold
        elif slo.comparison == '==':
            violation = metric_value != slo.threshold

        current_time = time.time()

        if violation:
            self._violations[slo.name].append(current_time)
            logger.warning(
                f"SLO violation: {slo.name} - "
                f"metric={metric_value:.4f}, threshold={slo.threshold}"
            )

        window_start = current_time - slo.window_size
        self._violations[slo.name] = [
            t for t in self._violations[slo.name] if t >= window_start
        ]

        violation_rate = len(self._violations[slo.name]) / max(1, slo.window_size)

        if violation_rate > slo.error_budget:
            return SLOStatus.VIOLATED
        elif violation_rate > slo.error_budget * 0.5:
            return SLOStatus.DEGRADED
        else:
            return SLOStatus.HEALTHY

    def check_all(self, metrics: Dict[str, float]) -> Dict[str, SLOStatus]:
        """Check all registered SLOs against current metrics"""
        results = {}

        for slo in self.slos:
            if slo.metric_name not in metrics:
                logger.warning(f"Metric {slo.metric_name} not found for SLO {slo.name}")
                continue

            metric_value = metrics[slo.metric_name]
            status = self.check_slo(slo, metric_value)
            results[slo.name] = status
            self._last_check[slo.name] = time.time()

        return results

    def get_error_budget_burn_rate(self, slo_name: str) -> float:
        """Get current error budget burn rate for an SLO"""
        slo = next((s for s in self.slos if s.name == slo_name), None)
        if not slo:
            raise ValueError(f"Unknown SLO: {slo_name}")

        current_time = time.time()
        window_start = current_time - slo.window_size

        recent_violations = [
            t for t in self._violations[slo_name] if t >= window_start
        ]

        if not recent_violations:
            return 0.0

        violation_rate = len(recent_violations) / max(1, slo.window_size)
        burn_rate = violation_rate / slo.error_budget if slo.error_budget > 0 else float('inf')

        return burn_rate

    def get_remaining_error_budget(self, slo_name: str) -> float:
        """Get remaining error budget percentage for an SLO"""
        slo = next((s for s in self.slos if s.name == slo_name), None)
        if not slo:
            raise ValueError(f"Unknown SLO: {slo_name}")

        current_time = time.time()
        window_start = current_time - slo.window_size

        recent_violations = [
            t for t in self._violations[slo_name] if t >= window_start
        ]

        violation_rate = len(recent_violations) / max(1, slo.window_size)
        remaining = max(0.0, slo.error_budget - violation_rate)

        return remaining / slo.error_budget if slo.error_budget > 0 else 0.0

    def stats(self) -> Dict:
        """Get statistics about SLO compliance"""
        results = {}

        for slo in self.slos:
            current_time = time.time()
            window_start = current_time - slo.window_size

            recent_violations = [
                t for t in self._violations[slo.name] if t >= window_start
            ]

            violation_rate = len(recent_violations) / max(1, slo.window_size)
            burn_rate = violation_rate / slo.error_budget if slo.error_budget > 0 else 0.0
            remaining_budget = max(0.0, slo.error_budget - violation_rate)

            results[slo.name] = {
                'violations_in_window': len(recent_violations),
                'violation_rate': violation_rate,
                'error_budget_burn_rate': burn_rate,
                'remaining_error_budget': remaining_budget,
                'last_check': self._last_check[slo.name],
                'threshold': slo.threshold,
                'comparison': slo.comparison,
            }

        return results

    def clear(self) -> None:
        """Clear all violation history"""
        for name in self._violations:
            self._violations[name].clear()


def create_default_slos() -> List[SLO]:
    """Create default SLOs for slime mold system"""
    return [
        SLO(
            name="latency_p95",
            metric_name="inference_latency_ms",
            threshold=100.0,
            comparison="<",
            window_size=3600,
            error_budget=0.05,
        ),
        SLO(
            name="latency_p99",
            metric_name="inference_latency_ms",
            threshold=200.0,
            comparison="<",
            window_size=3600,
            error_budget=0.01,
        ),
        SLO(
            name="memory_limit",
            metric_name="memory_allocated_mb",
            threshold=8000.0,
            comparison="<",
            window_size=1800,
            error_budget=0.1,
        ),
        SLO(
            name="pool_size_min",
            metric_name="pool_size",
            threshold=1.0,
            comparison=">=",
            window_size=600,
            error_budget=0.0,
        ),
        SLO(
            name="archive_coverage",
            metric_name="archive_coverage",
            threshold=0.1,
            comparison=">=",
            window_size=3600,
            error_budget=0.2,
        ),
    ]
