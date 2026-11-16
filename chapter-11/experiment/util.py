from __future__ import annotations

import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple


def get_environment_info() -> dict:
    """Return environment details for the evidence file.

    Uses only standard library calls.
    """
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
    }


def now() -> float:
    """Monotonic time for measuring durations (seconds)."""
    return time.perf_counter()


@dataclass
class SimulatedImage:
    """Simple representation of an image payload for simulation.

    We represent an image as a list of integers (0-255). The exact values are
    not important; they are used only to drive deterministic CPU work.
    """

    width: int
    height: int
    data: List[int]


def generate_synthetic_image(rng: random.Random, resolution: Tuple[int, int]) -> SimulatedImage:
    """Generate a synthetic image with the given resolution.

    The contents are pseudo-random but deterministic given the rng.
    """
    w, h = resolution
    size = w * h // 8  # keep bounded so tests run quickly
    data = [rng.randint(0, 255) for _ in range(size)]
    return SimulatedImage(width=w, height=h, data=data)


def busy_work(rng: random.Random, iterations: int, payload: Iterable[int] | None = None) -> float:
    """Deterministic CPU work loop.

    Returns a value derived from the work to avoid Python optimizing it away.
    """
    acc = 0.0
    if payload is not None:
        payload_list = list(payload)
    else:
        payload_list = [rng.randint(0, 255) for _ in range(16)]
    n = len(payload_list)

    for i in range(iterations):
        v = payload_list[i % n]
        acc += math.sin(v) * math.cos(v + 0.1) * 0.0001
    return acc


def percentile(values: List[float], pct: float) -> float:
    """Compute percentile for a non-empty list of values.

    Uses the nearest-rank method with linear interpolation for simplicity.
    """
    if not values:
        raise ValueError("Cannot compute percentile of empty list")
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def bounded_sleep(seconds: float) -> None:
    """Sleep helper that ensures non-negative and modest durations.

    We clamp sleep time to [0, 0.5] seconds to keep test runs fast.
    """
    if seconds <= 0:
        return
    time.sleep(min(seconds, 0.5))


def derive_seed(base_seed: int, *components: str) -> int:
    """Derive a deterministic integer seed from a base seed and components."""
    s = str(base_seed) + "::" + "::".join(components)
    value = 0
    for ch in s:
        value = (value * 31 + ord(ch)) & 0xFFFFFFFF
    return value


def resolution_choice(rng: random.Random, workload_id: str) -> Tuple[int, int]:
    """Return a resolution tuple based on workload id.

    This is a simplified mapping derived from the spec's resolution
    distributions. It ensures we exercise different payload sizes.
    """
    if workload_id == "wl1-real-time-low-rps":
        choices = [(640, 480), (1920, 1080), (3840, 2160)]
    elif workload_id == "wl2-real-time-bursty-high-rps":
        choices = [(1280, 720), (1920, 1080), (3840, 2160)]
    elif workload_id == "wl3-batch-backfill":
        choices = [(640, 480), (1280, 720), (1920, 1080), (3840, 2160)]
    else:
        # wl4 or unknown, use moderate resolutions
        choices = [(1280, 720), (1920, 1080)]
    return rng.choice(choices)
