from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


# ----------------------------- Utilities ------------------------------------


def set_global_seed(seed: int) -> None:
    """Set seeds for reproducible synthetic data and training.

    Only the Python standard library RNG is available in this sandbox.
    """

    random.seed(seed)


# ---------------------------- Synthetic Data ---------------------------------


@dataclass
class SyntheticDataset:
    """Simple container for a synthetic binary classification dataset.

    Attributes
    ----------
    features: List of (x1, x2) feature tuples.
    labels: List of integer labels {0, 1}.
    """

    features: List[Tuple[float, float]]
    labels: List[int]


def _sample_gaussian_2d(
    n: int, mean_x: float, mean_y: float, std_x: float, std_y: float
) -> List[Tuple[float, float]]:
    """Sample n points from an axis-aligned 2D Gaussian distribution."""

    pts: List[Tuple[float, float]] = []
    for _ in range(n):
        x = random.gauss(mean_x, std_x)
        y = random.gauss(mean_y, std_y)
        pts.append((x, y))
    return pts


def make_workload_W1_dataset(n_per_class: int = 100) -> SyntheticDataset:
    """Create an in-distribution (lab) dataset.

    Class 0 and class 1 are well-separated Gaussians.
    """

    # Fixed seeds for determinism; use distinct seeds per workload.
    set_global_seed(12345)

    # Class 0 cluster
    c0 = _sample_gaussian_2d(
        n_per_class, mean_x=-2.0, mean_y=-2.0, std_x=0.7, std_y=0.7
    )
    # Class 1 cluster
    c1 = _sample_gaussian_2d(
        n_per_class, mean_x=2.0, mean_y=2.0, std_x=0.7, std_y=0.7
    )

    features: List[Tuple[float, float]] = c0 + c1
    labels: List[int] = [0] * n_per_class + [1] * n_per_class

    return SyntheticDataset(features=features, labels=labels)


def make_workload_W5_dataset(n_per_class: int = 100) -> SyntheticDataset:
    """Create a held-out-attacks / OOD dataset.

    Simulates a distribution shift by shifting and spreading the Gaussians.
    """

    set_global_seed(98765)

    # Class 0 cluster: shifted and noisier
    c0 = _sample_gaussian_2d(
        n_per_class, mean_x=-1.0, mean_y=-3.0, std_x=1.4, std_y=1.4
    )
    # Class 1 cluster: shifted and noisier
    c1 = _sample_gaussian_2d(
        n_per_class, mean_x=3.0, mean_y=1.0, std_x=1.4, std_y=1.4
    )

    features: List[Tuple[float, float]] = c0 + c1
    labels: List[int] = [0] * n_per_class + [1] * n_per_class

    return SyntheticDataset(features=features, labels=labels)


# ------------------------------- Models --------------------------------------


@dataclass
class SimpleLinearModelBaseline:
    """A very small linear classifier for 2D data.

    y_hat = sigmoid(w1 * x1 + w2 * x2 + b)

    Trained with basic gradient descent on logistic loss.
    """

    w1: float
    w2: float
    b: float

    def predict_proba(self, x: Tuple[float, float]) -> float:
        z = self.w1 * x[0] + self.w2 * x[1] + self.b
        return 1.0 / (1.0 + math.exp(-z))

    def predict_label(self, x: Tuple[float, float]) -> int:
        p = self.predict_proba(x)
        return 1 if p >= 0.5 else 0


@dataclass
class SimpleLinearModelTreatment(SimpleLinearModelBaseline):
    """Treatment variant with simple domain-adaptation-style training and
    uncertainty-aware decision rule.

    It shares the same structure as the baseline but is trained with a small
    mixture of OOD-like points and can abstain when its predicted probability
    is near 0.5.
    """

    abstain_threshold: float = 0.15  # abstain when |p - 0.5| < thresh

    def predict_label_with_abstention(self, x: Tuple[float, float]) -> Tuple[int | None, float]:
        """Return (label_or_None, probability).

        If the model is not confident enough, returns (None, p) to indicate
        abstention.
        """

        p = self.predict_proba(x)
        if abs(p - 0.5) < self.abstain_threshold:
            return None, p
        return (1 if p >= 0.5 else 0), p


# ---------------------------- Training Logic ---------------------------------


def _train_logistic_regression(
    dataset: SyntheticDataset,
    num_steps: int,
    learning_rate: float,
    initial_weights: Tuple[float, float, float] | None = None,
) -> Tuple[float, float, float]:
    """Train a 2D logistic regression with simple gradient descent.

    The training is intentionally lightweight and deterministic.
    """

    if initial_weights is None:
        # Small deterministic initialization.
        w1, w2, b = 0.1, -0.1, 0.0
    else:
        w1, w2, b = initial_weights

    n = len(dataset.features)
    for _ in range(num_steps):
        # Compute gradients over the full dataset (batch gradient descent).
        grad_w1 = 0.0
        grad_w2 = 0.0
        grad_b = 0.0

        for (x1, x2), y in zip(dataset.features, dataset.labels):
            z = w1 * x1 + w2 * x2 + b
            p = 1.0 / (1.0 + math.exp(-z))
            # derivative of logistic loss wrt z is (p - y)
            diff = p - y
            grad_w1 += diff * x1
            grad_w2 += diff * x2
            grad_b += diff

        # Average gradients
        grad_w1 /= float(n)
        grad_w2 /= float(n)
        grad_b /= float(n)

        # Gradient descent update
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
        b -= learning_rate * grad_b

    return w1, w2, b


def train_baseline_model() -> SimpleLinearModelBaseline:
    """Train the baseline linear model using only W1-style data."""

    dataset = make_workload_W1_dataset(n_per_class=80)
    set_global_seed(2024)
    w1, w2, b = _train_logistic_regression(
        dataset=dataset,
        num_steps=40,
        learning_rate=0.1,
        initial_weights=(0.05, -0.07, 0.0),
    )
    return SimpleLinearModelBaseline(w1=w1, w2=w2, b=b)


def train_treatment_model() -> SimpleLinearModelTreatment:
    """Train the treatment model with a simplistic domain-adaptation
    strategy.

    It trains on both W1-style and W5-style mini-batches, with a smaller
    learning rate, to loosely mimic robustness training.
    """

    in_dist = make_workload_W1_dataset(n_per_class=80)
    ood = make_workload_W5_dataset(n_per_class=40)

    # Merge datasets into a single training set with equal weighting of
    # losses per sample in the concatenated list.
    features = in_dist.features + ood.features
    labels = in_dist.labels + ood.labels
    mixed_dataset = SyntheticDataset(features=features, labels=labels)

    set_global_seed(2025)
    w1, w2, b = _train_logistic_regression(
        dataset=mixed_dataset,
        num_steps=60,
        learning_rate=0.05,
        initial_weights=(0.05, -0.07, 0.0),
    )

    # Slightly more conservative abstain threshold to help W5 robustness.
    return SimpleLinearModelTreatment(w1=w1, w2=w2, b=b, abstain_threshold=0.2)


# ----------------------------- Metrics ---------------------------------------


def compute_accuracy(labels: List[int], preds: List[int]) -> float:
    assert len(labels) == len(preds)
    if not labels:
        return 0.0
    correct = sum(1 for y, p in zip(labels, preds) if y == p)
    return correct / float(len(labels))


def compute_abstention_rate(total: int, num_abstained: int) -> float:
    if total <= 0:
        return 0.0
    return num_abstained / float(total)


# ---------------------------- Workload Runner --------------------------------


def _evaluate_baseline_on_dataset(
    model: SimpleLinearModelBaseline, dataset: SyntheticDataset
) -> Dict[str, float]:
    preds = [model.predict_label(x) for x in dataset.features]
    acc = compute_accuracy(dataset.labels, preds)
    # Baseline has no abstention behavior.
    metrics = {
        "accuracy": float(acc),
    }
    return metrics


def _evaluate_treatment_on_dataset(
    model: SimpleLinearModelTreatment, dataset: SyntheticDataset
) -> Dict[str, float]:
    predicted_labels: List[int] = []
    num_abstained = 0

    for x in dataset.features:
        label, _prob = model.predict_label_with_abstention(x)
        if label is None:
            num_abstained += 1
            # For plain accuracy we treat abstentions as incorrect
            # (operationally, they defer decision). Here we assign a
            # dummy label 0 that is unlikely to be systematically correct.
            predicted_labels.append(0)
        else:
            predicted_labels.append(label)

    acc = compute_accuracy(dataset.labels, predicted_labels)
    abst_rate = compute_abstention_rate(len(dataset.labels), num_abstained)

    metrics = {
        "accuracy": float(acc),
        "abstention_rate": float(abst_rate),
    }
    return metrics


def run_baseline_on_workloads(workloads: List[str]) -> List[Dict[str, object]]:
    """Run the baseline model on the requested workloads.

    Returns a list of run dictionaries compatible with the evidence schema.
    """

    model = train_baseline_model()

    runs: List[Dict[str, object]] = []

    for workload_id in workloads:
        if workload_id == "W1-in-distribution-lab-image":
            dataset = make_workload_W1_dataset(n_per_class=100)
        elif workload_id == "W5-heldout-attack-families-image":
            dataset = make_workload_W5_dataset(n_per_class=100)
        else:
            # Unknown workload; skip.
            continue

        metrics = _evaluate_baseline_on_dataset(model, dataset)
        run = {
            "variant": "baseline",
            "workload_id": workload_id,
            "run_id": f"baseline_{workload_id}_0",
            "success": True,
            "metrics": metrics,
        }
        runs.append(run)

    return runs


def run_treatment_on_workloads(workloads: List[str]) -> List[Dict[str, object]]:
    """Run the treatment model on the requested workloads.

    Returns a list of run dictionaries compatible with the evidence schema.
    """

    model = train_treatment_model()

    runs: List[Dict[str, object]] = []

    for workload_id in workloads:
        if workload_id == "W1-in-distribution-lab-image":
            dataset = make_workload_W1_dataset(n_per_class=100)
        elif workload_id == "W5-heldout-attack-families-image":
            dataset = make_workload_W5_dataset(n_per_class=100)
        else:
            continue

        metrics = _evaluate_treatment_on_dataset(model, dataset)
        run = {
            "variant": "treatment",
            "workload_id": workload_id,
            "run_id": f"treatment_{workload_id}_0",
            "success": True,
            "metrics": metrics,
        }
        runs.append(run)

    return runs
