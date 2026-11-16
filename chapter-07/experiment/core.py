import math
import random
from dataclasses import dataclass
from typing import List


GLOBAL_SEED = 17


def set_global_seed(seed: int) -> None:
    """Set the global random seed for deterministic behavior.

    Only Python's `random` module is used in this project.
    """
    global GLOBAL_SEED
    GLOBAL_SEED = seed
    random.seed(seed)


@dataclass
class VideoSample:
    """Represents a single synthetic video instance.

    Attributes
    ----------
    frames: List[List[float]]
        Sequence of frames; each frame is a list of numeric features.
    label: int
        Binary label: 0 (non_removed) or 1 (removed).
    """

    frames: List[List[float]]
    label: int


@dataclass
class DatasetSplits:
    """Train/val/test splits for a workload."""

    train: List[VideoSample]
    val: List[VideoSample]
    test: List[VideoSample]


def sigmoid(x: float) -> float:
    # Numerically stable sigmoid for small magnitudes (our case).
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def mse_loss(preds: List[float], targets: List[float]) -> float:
    n = len(preds)
    if n == 0:
        return 0.0
    return sum((p - t) ** 2 for p, t in zip(preds, targets)) / float(n)


def accuracy_at_threshold(preds: List[float], targets: List[int], threshold: float = 0.5) -> float:
    if not preds:
        return 0.0
    correct = 0
    total = len(preds)
    for p, t in zip(preds, targets):
        pred_label = 1 if p >= threshold else 0
        if pred_label == t:
            correct += 1
    return correct / float(total)
