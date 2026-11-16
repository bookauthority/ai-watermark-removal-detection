from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

GLOBAL_SEED = 1337


class RandomState:
    """Simple deterministic RNG wrapper based on Python's random.

    Separate instance from global random module to avoid pollution.
    """

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def rand(self) -> float:
        return self._rng.random()

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def choice(self, seq: List[int]) -> int:
        return self._rng.choice(seq)

    def shuffle(self, seq: List[int]) -> None:
        self._rng.shuffle(seq)

    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        # Box-Muller
        u1 = max(self.rand(), 1e-12)
        u2 = self.rand()
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return mu + sigma * z0


@dataclass
class DatasetSplit:
    features: List[List[float]]  # list of vectors
    labels: List[int]  # 0: clean, 1: watermarked, 2: removed


@dataclass
class FullDataset:
    train_clean: DatasetSplit
    train_watermarked: DatasetSplit
    train_removed: DatasetSplit
    test_clean: DatasetSplit
    test_watermarked: DatasetSplit
    test_removed: DatasetSplit


def _make_class_cluster(
    rng: RandomState, center: List[float], n: int, noise: float
) -> List[List[float]]:
    d = len(center)
    samples: List[List[float]] = []
    for _ in range(n):
        vec = []
        for c in center:
            v = c + rng.normal(0.0, noise)
            v = min(1.0, max(0.0, v))
            vec.append(v)
        samples.append(vec)
    return samples


def generate_synthetic_dataset(rng: RandomState) -> FullDataset:
    """Generate small synthetic dataset with three classes.

    Uses 8-D vectors with distinct centers per class.
    """

    dim = 8
    # Distinct centers for each semantic class
    center_clean = [0.2] * dim
    center_watermarked = [0.6] * dim
    center_removed = [0.8] * dim

    n_train = 80
    n_test = 40
    noise = 0.08

    train_clean_feat = _make_class_cluster(rng, center_clean, n_train, noise)
    train_watermarked_feat = _make_class_cluster(rng, center_watermarked, n_train, noise)
    train_removed_feat = _make_class_cluster(rng, center_removed, n_train, noise)

    test_clean_feat = _make_class_cluster(rng, center_clean, n_test, noise)
    test_watermarked_feat = _make_class_cluster(rng, center_watermarked, n_test, noise)
    test_removed_feat = _make_class_cluster(rng, center_removed, n_test, noise)

    return FullDataset(
        train_clean=DatasetSplit(train_clean_feat, [0] * n_train),
        train_watermarked=DatasetSplit(train_watermarked_feat, [1] * n_train),
        train_removed=DatasetSplit(train_removed_feat, [2] * n_train),
        test_clean=DatasetSplit(test_clean_feat, [0] * n_test),
        test_watermarked=DatasetSplit(test_watermarked_feat, [1] * n_test),
        test_removed=DatasetSplit(test_removed_feat, [2] * n_test),
    )


def softmax(logits: List[float]) -> List[float]:
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]


def argmax(v: List[float]) -> int:
    best_i = 0
    best_v = v[0]
    for i in range(1, len(v)):
        if v[i] > best_v:
            best_v = v[i]
            best_i = i
    return best_i


def accuracy(preds: List[int], labels: List[int]) -> float:
    assert len(preds) == len(labels)
    if not preds:
        return 0.0
    correct = sum(1 for p, t in zip(preds, labels) if p == t)
    return correct / float(len(preds))


def attack_evasion_rate(
    baseline_preds_before: List[int],
    baseline_preds_after: List[int],
    true_labels: List[int],
    positive_label: int,
) -> float:
    """Compute fraction of correctly-detected positives that are flipped.

    For each example:
      - If true_label == positive_label and baseline_preds_before == positive_label
        and baseline_preds_after != positive_label, then it is an evasion.
    """

    assert (
        len(baseline_preds_before)
        == len(baseline_preds_after)
        == len(true_labels)
    )

    num_candidates = 0
    num_evaded = 0
    for pb, pa, t in zip(baseline_preds_before, baseline_preds_after, true_labels):
        if t == positive_label and pb == positive_label:
            num_candidates += 1
            if pa != positive_label:
                num_evaded += 1
    if num_candidates == 0:
        return 0.0
    return num_evaded / float(num_candidates)
