import os
import sys
import time
import math
import random
import platform
import uuid
from typing import Dict, Any


def get_environment_info() -> Dict[str, str]:
    """Return a minimal environment description (OS + Python version)."""
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
    }


def now_monotonic() -> float:
    """Monotonic time for runtime measurement."""
    return time.perf_counter()


def elapsed_seconds(start: float, end: float) -> float:
    return float(end - start)


def make_rng(seed: int) -> random.Random:
    """Create a dedicated RNG instance so baseline and treatment don't interfere."""
    return random.Random(seed)


def deterministic_uuid(namespace: str, name: str) -> str:
    """Derive a deterministic UUID5 based on a namespace and name string."""
    ns_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, namespace)
    return str(uuid.uuid5(ns_uuid, name))


def safe_ratio(numerator: int, denominator: int) -> float:
    """Return numerator/denominator as float, or 0.0 if denominator is 0."""
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def is_finite_number(x: Any) -> bool:
    """Return True if x is int/float and finite."""
    if not isinstance(x, (int, float)):
        return False
    return math.isfinite(float(x))


def evidence_dir() -> str:
    """Resolve the evidence directory from env or default to a local ./evidence.

    Using a relative directory by default avoids permission issues that can
    occur with absolute paths like /evidence on some systems.
    """
    return os.environ.get("BOOKMAS_EVIDENCE_DIR", os.path.join(os.getcwd(), "evidence"))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def generate_run_id(prefix: str, rng: random.Random) -> str:
    """Generate a short pseudo-random run identifier using the provided RNG."""
    val = rng.getrandbits(32)
    return f"{prefix}-{val:08x}"
