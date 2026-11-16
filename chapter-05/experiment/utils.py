import platform
import random
import sys
from typing import Dict


def set_global_seed(seed: int) -> None:
    """Set global random seed for reproducibility.

    We only use the standard library `random` module in this project.
    """

    random.seed(seed)


def get_environment_metadata() -> Dict[str, str]:
    """Return minimal environment metadata for the evidence file."""

    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
    }
