from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict

from .util import SimulatedImage, busy_work


@dataclass
class BaselineDetector:
    """Simulated baseline monolithic detector service.

    The baseline combines image preprocessing, feature extraction, model
    inference, logging, and audit into a single synchronous call path.
    """

    def __post_init__(self) -> None:
        # Seed is per-request; we keep a base RNG only for structural choices
        self._base_rng = random.Random(12345)

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a detection request synchronously.

        The request is expected to contain a "payload" field with a
        SimulatedImage and a "workload_id" for tailoring work size.
        """
        workload_id = request.get("workload_id", "unknown")
        image: SimulatedImage = request["payload"]

        # Derive a per-request RNG seed to make the simulated processing
        # deterministic while varying by request id.
        req_id = str(request.get("id", "0"))
        seed_val = 0
        for ch in req_id + workload_id:
            seed_val = (seed_val * 31 + ord(ch)) & 0xFFFFFFFF
        rng = random.Random(seed_val)

        # 1. Decode step (simulated)
        _ = busy_work(rng, iterations=100, payload=image.data)

        # 2. Preprocessing / feature extraction
        _ = busy_work(rng, iterations=150, payload=image.data)

        # 3. Model inference (CPU-bound synthetic work)
        _ = busy_work(rng, iterations=300, payload=image.data)

        # 4. Logging & audit (lightweight work)
        _ = busy_work(rng, iterations=30)

        # 5. Decision thresholding (trivial arithmetic)
        score = rng.random()
        label = "watermark_removed" if score > 0.5 else "no_issue"

        return {
            "id": req_id,
            "score": float(score),
            "label": label,
            "workload_id": workload_id,
        }
