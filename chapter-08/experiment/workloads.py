from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .core import DatasetSplit, FullDataset, RandomState, accuracy, attack_evasion_rate
from .model import BaselineDetector, TreatmentDetector, train_treatment_detector
from .attacks import (
    HeuristicAttackSuite,
    OptimizationAttackSuite,
    GenerativeAttackSuite,
    attack_pgd_like,
    attack_square_like,
)


def _concat_splits(*splits: DatasetSplit) -> DatasetSplit:
    feats: List[List[float]] = []
    labels: List[int] = []
    for s in splits:
        feats.extend(s.features)
        labels.extend(s.labels)
    return DatasetSplit(feats, labels)


def run_clean_reference_eval(
    detector, datasets: FullDataset, rng: RandomState
) -> Dict[str, float]:
    """Evaluate detector accuracy on clean test data.

    Returns metrics dict with at least one numeric metric.
    """

    test = _concat_splits(datasets.test_clean, datasets.test_watermarked, datasets.test_removed)
    preds = [detector.predict(x) for x in test.features]
    acc = accuracy(preds, test.labels)
    return {"accuracy": float(acc)}


def run_attack_eval(
    detector,
    datasets: FullDataset,
    rng: RandomState,
    attack_suite,
    workload_id: str,
) -> Dict[str, float]:
    """Run attacks on test_removed split and measure attack evasion rate.

    We treat class 2 (watermark-removed) as positive.
    """

    # Baseline predictions before attack (same detector).
    original_features = datasets.test_removed.features
    original_labels = datasets.test_removed.labels
    preds_before = [detector.predict(x) for x in original_features]

    # Apply all attacks in the suite and aggregate evasion.
    all_evaded = 0
    all_candidates = 0

    if isinstance(attack_suite, HeuristicAttackSuite):
        for attack_id, fn in attack_suite.attacks:
            attacked_features = [fn(x, rng) for x in original_features]
            preds_after = [detector.predict(x) for x in attacked_features]
            num_candidates = 0
            num_evaded = 0
            for pb, pa, t in zip(preds_before, preds_after, original_labels):
                if t == 2 and pb == 2:
                    num_candidates += 1
                    if pa != 2:
                        num_evaded += 1
            all_candidates += num_candidates
            all_evaded += num_evaded
    elif isinstance(attack_suite, OptimizationAttackSuite):
        # Need loss function: encourage detector to reject class 2.
        def make_loss(x):
            probs = detector.predict_proba(x)
            return probs[2]  # higher is worse for attacker

        for attack_id, kind in attack_suite.attacks:
            if kind == "pgd":
                attacked_features = [
                    attack_pgd_like(x, rng, make_loss) for x in original_features
                ]
            else:
                attacked_features = [
                    attack_square_like(x, rng, make_loss) for x in original_features
                ]
            preds_after = [detector.predict(x) for x in attacked_features]
            num_candidates = 0
            num_evaded = 0
            for pb, pa, t in zip(preds_before, preds_after, original_labels):
                if t == 2 and pb == 2:
                    num_candidates += 1
                    if pa != 2:
                        num_evaded += 1
            all_candidates += num_candidates
            all_evaded += num_evaded
    elif isinstance(attack_suite, GenerativeAttackSuite):
        for attack_id, fn in attack_suite.attacks:
            attacked_features = [fn(x, rng) for x in original_features]
            preds_after = [detector.predict(x) for x in attacked_features]
            num_candidates = 0
            num_evaded = 0
            for pb, pa, t in zip(preds_before, preds_after, original_labels):
                if t == 2 and pb == 2:
                    num_candidates += 1
                    if pa != 2:
                        num_evaded += 1
            all_candidates += num_candidates
            all_evaded += num_evaded
    else:
        raise ValueError("Unknown attack suite type")

    if all_candidates == 0:
        evasion_rate = 0.0
    else:
        evasion_rate = all_evaded / float(all_candidates)

    return {"attack_evasion_rate": float(evasion_rate)}


@dataclass
class HardNegativeSet:
    features: List[List[float]]
    labels: List[int]


def _mine_hard_negatives(
    baseline_detector: BaselineDetector,
    datasets: FullDataset,
    rng: RandomState,
    heuristic_attacks: HeuristicAttackSuite,
    optimization_attacks: OptimizationAttackSuite,
    generative_attacks: GenerativeAttackSuite,
) -> HardNegativeSet:
    """Generate simple hard negatives from train_removed set.

    We apply only heuristic attacks here for simplicity and
    computational efficiency.
    """

    src_features = datasets.train_removed.features
    src_labels = datasets.train_removed.labels
    features: List[List[float]] = []
    labels: List[int] = []

    # Only heuristic attacks to keep computation low.
    for attack_id, fn in heuristic_attacks.attacks:
        for x, y in zip(src_features, src_labels):
            assert y == 2
            attacked = fn(x, rng)
            pred = baseline_detector.predict(attacked)
            if pred != 2:
                features.append(attacked)
                labels.append(2)
    return HardNegativeSet(features, labels)


def run_adversarial_training_loop(
    baseline_detector: BaselineDetector,
    treatment_detector: TreatmentDetector,
    datasets: FullDataset,
    rng: RandomState,
    heuristic_attacks: HeuristicAttackSuite,
    optimization_attacks: OptimizationAttackSuite,
    generative_attacks: GenerativeAttackSuite,
) -> (Dict[str, float], Dict[str, float]):
    """Toy adversarial training loop with comparable metrics.

    1. Mine hard negatives from training removed set.
    2. Measure baseline accuracy on train_removed.
    3. Train treatment detector on these hard negatives.
    4. Measure treatment accuracy on train_removed with the same metric
       name so baseline and treatment can be compared.
    """

    hard_negatives = _mine_hard_negatives(
        baseline_detector,
        datasets,
        rng,
        heuristic_attacks,
        optimization_attacks,
        generative_attacks,
    )

    # Accuracy for baseline before any adversarial training.
    feats = datasets.train_removed.features
    labels = datasets.train_removed.labels
    baseline_preds = [baseline_detector.predict(x) for x in feats]
    baseline_acc = accuracy(baseline_preds, labels)

    # Train treatment detector on hard negatives.
    train_treatment_detector(
        treatment_detector,
        hard_negative_features=hard_negatives.features,
        hard_negative_labels=hard_negatives.labels,
    )

    # Accuracy for treatment after training, measured on the same data.
    treatment_preds = [treatment_detector.predict(x) for x in feats]
    treatment_acc = accuracy(treatment_preds, labels)

    metrics_baseline = {"removed_train_accuracy": float(baseline_acc)}
    metrics_treatment = {"removed_train_accuracy": float(treatment_acc)}

    return metrics_baseline, metrics_treatment
