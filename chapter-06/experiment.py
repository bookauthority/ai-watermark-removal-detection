import math
import os
import random
import platform
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


# -------------------------------
# Synthetic data and model logic
# -------------------------------


@dataclass
class SyntheticDataset:
    features: List[List[float]]
    labels: List[int]


def set_global_seed(seed: int) -> None:
    """Seed Python's random module for deterministic behavior."""
    random.seed(seed)


def generate_synthetic_dataset(
    n_samples: int,
    n_features: int,
    difficulty: float,
    seed: int,
) -> SyntheticDataset:
    """Generate a simple binary classification dataset.

    Args:
        n_samples: number of samples.
        n_features: number of features per sample.
        difficulty: controls class separability; higher is more difficult.
        seed: random seed for reproducibility.
    """
    set_global_seed(seed)

    features: List[List[float]] = []
    labels: List[int] = []

    # True underlying weight vector for label generation
    true_w = [random.uniform(-1.0, 1.0) for _ in range(n_features)]

    for _ in range(n_samples):
        x = [random.gauss(0.0, 1.0) for _ in range(n_features)]
        # Linear score plus noise
        score = sum(w * xi for w, xi in zip(true_w, x))
        noise = random.gauss(0.0, difficulty)
        y = 1 if score + noise > 0.0 else 0
        features.append(x)
        labels.append(y)

    return SyntheticDataset(features=features, labels=labels)


@dataclass
class SyntheticModel:
    weights: List[float]
    bias: float
    variant_name: str


def initialize_model(n_features: int, variant_name: str, seed: int) -> SyntheticModel:
    """Initialize a simple linear model with small random weights.

    Baseline and treatment use slightly different initializations
    to reflect architectural differences while keeping fairness.
    """
    set_global_seed(seed)

    if variant_name == "baseline":
        scale = 0.1
    else:
        # Treatment uses a different initialization scale
        scale = 0.15

    weights = [random.uniform(-scale, scale) for _ in range(n_features)]
    bias = random.uniform(-scale, scale)
    return SyntheticModel(weights=weights, bias=bias, variant_name=variant_name)


def sigmoid(x: float) -> float:
    # Numerically safe-ish sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def train_model(
    model: SyntheticModel,
    dataset: SyntheticDataset,
    lr: float,
    epochs: int,
    loss_type: str,
) -> SyntheticModel:
    """Simple gradient-based training loop.

    Args:
        model: initial model.
        dataset: synthetic data.
        lr: learning rate.
        epochs: number of passes over data.
        loss_type: "bce" for baseline or "focal" for treatment.
    """
    w = list(model.weights)
    b = model.bias

    gamma = 2.0 if loss_type == "focal" else 0.0

    for _ in range(epochs):
        # One pass over all examples (full-batch for simplicity)
        grad_w = [0.0 for _ in w]
        grad_b = 0.0

        for x, y in zip(dataset.features, dataset.labels):
            # Linear model
            logit = sum(wi * xi for wi, xi in zip(w, x)) + b
            p = sigmoid(logit)

            # Binary focal loss gradient approximation
            # For baseline (gamma=0), this reduces to BCE gradient.
            # loss ~ -alpha * (1 - pt)^gamma * log(pt)
            # We approximate gradient w.r.t logit as (p - y) * weight
            pt = p if y == 1 else (1.0 - p)
            if pt <= 0.0:
                pt = 1e-8
            if pt >= 1.0:
                pt = 1.0 - 1e-8

            modulation = (1.0 - pt) ** gamma if gamma > 0.0 else 1.0
            # Standard BCE grad: (p - y)
            base_grad = p - float(y)
            grad_factor = modulation * base_grad

            for i in range(len(w)):
                grad_w[i] += grad_factor * x[i]
            grad_b += grad_factor

        # Average gradients
        n = float(len(dataset.labels)) if dataset.labels else 1.0
        for i in range(len(w)):
            w[i] -= lr * (grad_w[i] / n)
        b -= lr * (grad_b / n)

    model.weights = w
    model.bias = b
    return model


def evaluate_model(model: SyntheticModel, dataset: SyntheticDataset) -> float:
    """Compute a simple accuracy metric on the dataset."""
    if not dataset.labels:
        return 0.0

    correct = 0
    for x, y in zip(dataset.features, dataset.labels):
        logit = sum(wi * xi for wi, xi in zip(model.weights, x)) + model.bias
        p = sigmoid(logit)
        pred = 1 if p >= 0.5 else 0
        if pred == y:
            correct += 1
    return float(correct) / float(len(dataset.labels))


# -------------------------------
# Workload orchestration
# -------------------------------


def run_workload_wl1(seed: int) -> List[Dict[str, Any]]:
    """WL1-CleanSplit-Standard synthetic workload.

    Generates one baseline and one treatment run.
    """
    workload_id = "WL1-CleanSplit-Standard"
    n_features = 16

    # Generate a single dataset used by both baseline and treatment
    dataset = generate_synthetic_dataset(
        n_samples=300,
        n_features=n_features,
        difficulty=0.7,
        seed=seed,
    )

    runs: List[Dict[str, Any]] = []

    # Baseline run
    baseline_model = initialize_model(n_features=n_features, variant_name="baseline", seed=seed + 1)
    baseline_model = train_model(
        baseline_model,
        dataset,
        lr=0.05,
        epochs=10,
        loss_type="bce",
    )
    baseline_metric = evaluate_model(baseline_model, dataset)

    runs.append(
        {
            "variant": "baseline",
            "workload_id": workload_id,
            "run_id": f"{workload_id}-baseline-seed{seed}",
            "success": True,
            "metrics": {"synthetic_accuracy": float(baseline_metric)},
        }
    )

    # Treatment run
    treatment_model = initialize_model(n_features=n_features, variant_name="treatment", seed=seed + 2)
    treatment_model = train_model(
        treatment_model,
        dataset,
        lr=0.04,
        epochs=12,
        loss_type="focal",
    )
    treatment_metric = evaluate_model(treatment_model, dataset)

    runs.append(
        {
            "variant": "treatment",
            "workload_id": workload_id,
            "run_id": f"{workload_id}-treatment-seed{seed}",
            "success": True,
            "metrics": {"synthetic_accuracy": float(treatment_metric)},
        }
    )

    return runs


def run_workload_wl2(seed: int) -> List[Dict[str, Any]]:
    """WL2-Stress-HeldOutAttacks synthetic workload.

    Simulate distribution shift by increasing difficulty for a held-out set.
    """
    workload_id = "WL2-Stress-HeldOutAttacks"
    n_features = 16

    # Training dataset: easier
    train_dataset = generate_synthetic_dataset(
        n_samples=300,
        n_features=n_features,
        difficulty=0.6,
        seed=seed + 10,
    )
    # Held-out dataset: harder (higher difficulty)
    heldout_dataset = generate_synthetic_dataset(
        n_samples=300,
        n_features=n_features,
        difficulty=1.2,
        seed=seed + 11,
    )

    runs: List[Dict[str, Any]] = []

    # Baseline
    baseline_model = initialize_model(n_features=n_features, variant_name="baseline", seed=seed + 12)
    baseline_model = train_model(
        baseline_model,
        train_dataset,
        lr=0.05,
        epochs=10,
        loss_type="bce",
    )
    baseline_metric = evaluate_model(baseline_model, heldout_dataset)

    runs.append(
        {
            "variant": "baseline",
            "workload_id": workload_id,
            "run_id": f"{workload_id}-baseline-seed{seed}",
            "success": True,
            "metrics": {"synthetic_accuracy": float(baseline_metric)},
        }
    )

    # Treatment
    treatment_model = initialize_model(n_features=n_features, variant_name="treatment", seed=seed + 13)
    treatment_model = train_model(
        treatment_model,
        train_dataset,
        lr=0.04,
        epochs=12,
        loss_type="focal",
    )
    treatment_metric = evaluate_model(treatment_model, heldout_dataset)

    runs.append(
        {
            "variant": "treatment",
            "workload_id": workload_id,
            "run_id": f"{workload_id}-treatment-seed{seed}",
            "success": True,
            "metrics": {"synthetic_accuracy": float(treatment_metric)},
        }
    )

    return runs


def run_workload_wl3(seed: int) -> List[Dict[str, Any]]:
    """WL3-CrossValidation-ContentStratified synthetic workload.

    Implements a small K-fold cross-validation (K=3 for bounded runtime).
    Produces at least one baseline and one treatment run overall.
    """
    workload_id = "WL3-CrossValidation-ContentStratified"
    n_features = 12
    n_samples = 240
    k_folds = 3

    # Single dataset to be split into folds
    dataset = generate_synthetic_dataset(
        n_samples=n_samples,
        n_features=n_features,
        difficulty=0.8,
        seed=seed + 20,
    )

    indices = list(range(n_samples))
    set_global_seed(seed + 21)
    random.shuffle(indices)
    fold_size = n_samples // k_folds

    runs: List[Dict[str, Any]] = []

    for fold in range(k_folds):
        start = fold * fold_size
        end = start + fold_size
        test_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]

        train_features = [dataset.features[i] for i in train_idx]
        train_labels = [dataset.labels[i] for i in train_idx]
        test_features = [dataset.features[i] for i in test_idx]
        test_labels = [dataset.labels[i] for i in test_idx]

        train_ds = SyntheticDataset(features=train_features, labels=train_labels)
        test_ds = SyntheticDataset(features=test_features, labels=test_labels)

        # Baseline for this fold
        baseline_model = initialize_model(
            n_features=n_features,
            variant_name="baseline",
            seed=seed + 22 + fold,
        )
        baseline_model = train_model(
            baseline_model,
            train_ds,
            lr=0.05,
            epochs=8,
            loss_type="bce",
        )
        baseline_metric = evaluate_model(baseline_model, test_ds)

        runs.append(
            {
                "variant": "baseline",
                "workload_id": workload_id,
                "run_id": f"{workload_id}-baseline-fold{fold}-seed{seed}",
                "success": True,
                "metrics": {"synthetic_accuracy": float(baseline_metric)},
            }
        )

        # Treatment for this fold
        treatment_model = initialize_model(
            n_features=n_features,
            variant_name="treatment",
            seed=seed + 25 + fold,
        )
        treatment_model = train_model(
            treatment_model,
            train_ds,
            lr=0.04,
            epochs=10,
            loss_type="focal",
        )
        treatment_metric = evaluate_model(treatment_model, test_ds)

        runs.append(
            {
                "variant": "treatment",
                "workload_id": workload_id,
                "run_id": f"{workload_id}-treatment-fold{fold}-seed{seed}",
                "success": True,
                "metrics": {"synthetic_accuracy": float(treatment_metric)},
            }
        )

    return runs


def run_all_workloads() -> List[Dict[str, Any]]:
    """Run all workloads (WL1, WL2, WL3) and return a flat list of runs."""
    base_seed = 42
    all_runs: List[Dict[str, Any]] = []
    all_runs.extend(run_workload_wl1(seed=base_seed))
    all_runs.extend(run_workload_wl2(seed=base_seed + 100))
    all_runs.extend(run_workload_wl3(seed=base_seed + 200))
    return all_runs
