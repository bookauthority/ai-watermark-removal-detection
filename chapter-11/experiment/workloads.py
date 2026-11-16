from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .metrics import RunMetricsCollector
from .util import (
    SimulatedImage,
    derive_seed,
    generate_synthetic_image,
    now,
    resolution_choice,
)


@dataclass
class WorkloadDefinition:
    """Represents a workload and its traffic pattern parameters.

    This is a simplified adaptation of the spec, focused on the aspects we can
    simulate in this offline environment.
    """

    workload_id: str
    description: str
    num_requests: int


def _make_workload_definitions() -> List[WorkloadDefinition]:
    # We keep request counts modest for fast execution, but large enough to
    # obtain stable latency percentiles.
    return [
        WorkloadDefinition(
            workload_id="wl1-real-time-low-rps",
            description="Low-throughput, latency-sensitive real-time detection.",
            num_requests=80,
        ),
        WorkloadDefinition(
            workload_id="wl2-real-time-bursty-high-rps",
            description="Bursty, higher RPS real-time detection.",
            num_requests=150,
        ),
        WorkloadDefinition(
            workload_id="wl3-batch-backfill",
            description="Batch backfill over historical corpus.",
            num_requests=200,
        ),
        WorkloadDefinition(
            workload_id="wl4-failure-injection",
            description="Resilience workload with simulated failures.",
            num_requests=120,
        ),
    ]


def get_all_workloads() -> List[WorkloadDefinition]:
    return _make_workload_definitions()


def _simulate_failure_if_needed(
    variant: str,
    workload: WorkloadDefinition,
    rng: random.Random,
    request_idx: int,
) -> bool:
    """Return True if this particular request should be forced to fail.

    For wl4 we introduce some pseudo-random errors and slowdowns to
    differentiate behavior and to exercise the `success` / `error_rate`
    reporting paths. For other workloads, we do not inject failures.
    """
    if workload.workload_id != "wl4-failure-injection":
        return False

    # Inject a small, deterministic failure probability.
    # Treatment gets slightly fewer failures to reflect better isolation.
    if variant == "baseline":
        fail_prob = 0.07
    else:
        fail_prob = 0.03

    return rng.random() < fail_prob


def run_workload_for_variant(
    *,
    variant: str,
    workload: WorkloadDefinition,
    architecture: Any,
    base_seed: int,
) -> Tuple[Dict[str, float], bool]:
    """Run a workload against a given architecture.

    Parameters
    ----------
    variant:
        "baseline" or "treatment".
    workload:
        Workload definition.
    architecture:
        Object exposing `process_request(request: dict) -> dict`.
    base_seed:
        Base integer seed for deterministic pseudo-randomness.

    Returns
    -------
    metrics_dict, success_flag
        metrics_dict contains numeric metrics. success_flag is True if the
        run reached the end without an unhandled exception.
    """
    assert variant in {"baseline", "treatment"}

    # Derive a dedicated RNG seed for this (variant, workload).
    seed = derive_seed(base_seed, variant, workload.workload_id)
    rng = random.Random(seed)

    collector = RunMetricsCollector()

    try:
        for i in range(workload.num_requests):
            # Generate a synthetic image payload.
            resolution = resolution_choice(rng, workload.workload_id)
            image: SimulatedImage = generate_synthetic_image(rng, resolution)

            req_id = f"{workload.workload_id}-{variant}-{i}"
            request = {
                "id": req_id,
                "payload": image,
                "workload_id": workload.workload_id,
            }

            # Possibly inject failures for wl4.
            force_fail = _simulate_failure_if_needed(variant, workload, rng, i)

            start_t = now()
            success = True
            try:
                if force_fail:
                    # We simulate a processing exception.
                    raise RuntimeError("Injected failure for resilience test")
                _ = architecture.process_request(request)
            except Exception:
                success = False
            end_t = now()

            latency_ms = (end_t - start_t) * 1000.0
            collector.record(latency_ms=latency_ms, success=success)

        metrics = collector.summarize()
        run_success = True
    except Exception:
        # Any unexpected error marks the entire run as failed; we still
        # attempt to return whatever metrics have been collected so far.
        metrics = collector.summarize()
        run_success = False

    return metrics, run_success
