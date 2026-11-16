from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .core import softmax, argmax


@dataclass
class LinearDetector:
    """Simple linear classifier with 3 output classes.

    y = softmax(ReLU(Wx + b))
    """

    weights: List[List[float]]  # shape (3, d)
    bias: List[float]  # shape (3,)

    def predict_proba(self, x: List[float]) -> List[float]:
        logits: List[float] = []
        for k in range(3):
            w = self.weights[k]
            s = self.bias[k]
            s += sum(wi * xi for wi, xi in zip(w, x))
            s = max(0.0, s)  # ReLU
            logits.append(s)
        return softmax(logits)

    def predict(self, x: List[float]) -> int:
        probs = self.predict_proba(x)
        return argmax(probs)


class BaselineDetector(LinearDetector):
    def __init__(self) -> None:
        # Deterministic, hand-coded parameters that roughly match the
        # synthetic data clusters defined in core.generate_synthetic_dataset.
        dim = 8
        # Class 0 (clean) center ~ 0.2
        # Class 1 (watermarked) center ~ 0.6
        # Class 2 (removed) center ~ 0.8
        w0 = [-0.5] * dim
        w1 = [0.3] * dim
        w2 = [0.6] * dim
        b0 = 0.3
        b1 = -0.2
        b2 = -0.8
        super().__init__(weights=[w0, w1, w2], bias=[b0, b1, b2])


class TreatmentDetector(LinearDetector):
    @classmethod
    def from_baseline(cls, baseline: BaselineDetector) -> "TreatmentDetector":
        weights = [w.copy() for w in baseline.weights]
        bias = baseline.bias.copy()
        return cls(weights=weights, bias=bias)


def train_treatment_detector(
    treatment: TreatmentDetector,
    hard_negative_features: List[List[float]],
    hard_negative_labels: List[int],
    lr: float = 0.05,
    epochs: int = 5,
) -> None:
    """Simple gradient-free update: move weights to increase margin on hard negatives.

    For each hard negative (true_label = 2 but predicted != 2), encourage
    the logit for class 2 to increase relative to others.
    """

    if not hard_negative_features:
        return

    for _ in range(epochs):
        for x, y_true in zip(hard_negative_features, hard_negative_labels):
            if y_true != 2:
                continue
            probs = treatment.predict_proba(x)
            pred = max(range(3), key=lambda k: probs[k])
            if pred == 2:
                continue
            # Increase weights for class 2 in direction of x, decrease others slightly.
            for j in range(len(x)):
                treatment.weights[2][j] += lr * x[j]
                for k in range(2):
                    treatment.weights[k][j] -= lr * 0.5 * x[j]
            treatment.bias[2] += lr * 0.1
            treatment.bias[0] -= lr * 0.05
            treatment.bias[1] -= lr * 0.05
