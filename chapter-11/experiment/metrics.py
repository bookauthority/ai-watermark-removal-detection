from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

from .util import percentile


@dataclass
class RunMetricsCollector:
    """Collects latency and outcome metrics for a single run.

    This collector is intentionally simple and uses only in-memory storage.
    """

    latencies_ms: List[float] = field(default_factory=list)
    errors: int = 0
    total: int = 0

    def record(self, latency_ms: float, success: bool) -> None:
        self.latencies_ms.append(float(latency_ms))
        self.total += 1
        if not success:
            self.errors += 1

    def summarize(self) -> dict:
        if self.total == 0:
            mean_latency = 0.0
            p95 = 0.0
            p99 = 0.0
            error_rate = 0.0
        else:
            mean_latency = float(sum(self.latencies_ms) / len(self.latencies_ms))
            p95 = float(percentile(self.latencies_ms, 95))
            p99 = float(percentile(self.latencies_ms, 99))
            error_rate = float(self.errors / self.total)

        # Throughput here is approximated as total requests per total observed
        # time across all requests. In this simulator, we consider the latency
        # list as representative of request durations and compute a simple
        # requests-per-second metric.
        if self.total == 0:
            throughput = 0.0
        else:
            total_time_s = sum(l / 1000.0 for l in self.latencies_ms)
            if total_time_s <= 0:
                throughput = float(self.total)
            else:
                throughput = float(self.total / total_time_s)

        return {
            "mean_latency_ms": mean_latency,
            "p95_latency_ms": p95,
            "p99_latency_ms": p99,
            "throughput_rps": throughput,
            "error_rate": error_rate,
        }
