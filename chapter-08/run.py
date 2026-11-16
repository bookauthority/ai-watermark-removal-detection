import json
import os
import sys
import platform
from typing import List, Dict, Any, Tuple

from experiment.core import (
    GLOBAL_SEED,
    RandomState,
    generate_synthetic_dataset,
)
from experiment.model import (
    BaselineDetector,
    TreatmentDetector,
    train_treatment_detector,
)
from experiment.attacks import (
    HeuristicAttackSuite,
    OptimizationAttackSuite,
    GenerativeAttackSuite,
)
from experiment.workloads import (
    run_clean_reference_eval,
    run_attack_eval,
    run_adversarial_training_loop,
)


EXPERIMENT_ID = "exp-ch8-attack-defense-redteam-v1"
CHAPTER = 8


def ensure_evidence_dir() -> str:
    path = os.environ.get("BOOKMAS_EVIDENCE_DIR", "/evidence")
    os.makedirs(path, exist_ok=True)
    return path


def build_environment_metadata() -> Dict[str, str]:
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
    }


def main() -> None:
    rng = RandomState(GLOBAL_SEED)

    # 1. Generate synthetic datasets
    datasets = generate_synthetic_dataset(rng)

    # 2. Instantiate baseline and treatment detectors
    baseline_detector = BaselineDetector()

    # Treatment starts as a copy of baseline, then gets updated during
    # adversarial training workload.
    treatment_detector = TreatmentDetector.from_baseline(baseline_detector)

    heuristic_attacks = HeuristicAttackSuite()
    optimization_attacks = OptimizationAttackSuite()
    generative_attacks = GenerativeAttackSuite()

    runs: List[Dict[str, Any]] = []

    # Workload: clean_reference_eval
    for variant, detector in (("baseline", baseline_detector), ("treatment", treatment_detector)):
        metrics = run_clean_reference_eval(detector, datasets, rng)
        runs.append(
            {
                "variant": variant,
                "workload_id": "clean_reference_eval",
                "run_id": f"clean_{variant}_v1",
                "success": True,
                "metrics": metrics,
            }
        )

    # Workload: heuristic_attack_eval
    for variant, detector in (("baseline", baseline_detector), ("treatment", treatment_detector)):
        metrics = run_attack_eval(
            detector=detector,
            datasets=datasets,
            rng=rng,
            attack_suite=heuristic_attacks,
            workload_id="heuristic_attack_eval",
        )
        runs.append(
            {
                "variant": variant,
                "workload_id": "heuristic_attack_eval",
                "run_id": f"heuristic_{variant}_v1",
                "success": True,
                "metrics": metrics,
            }
        )

    # Workload: optimization_based_attack_eval
    for variant, detector in (("baseline", baseline_detector), ("treatment", treatment_detector)):
        metrics = run_attack_eval(
            detector=detector,
            datasets=datasets,
            rng=rng,
            attack_suite=optimization_attacks,
            workload_id="optimization_based_attack_eval",
        )
        runs.append(
            {
                "variant": variant,
                "workload_id": "optimization_based_attack_eval",
                "run_id": f"optimization_{variant}_v1",
                "success": True,
                "metrics": metrics,
            }
        )

    # Workload: generative_attack_eval
    for variant, detector in (("baseline", baseline_detector), ("treatment", treatment_detector)):
        metrics = run_attack_eval(
            detector=detector,
            datasets=datasets,
            rng=rng,
            attack_suite=generative_attacks,
            workload_id="generative_attack_eval",
        )
        runs.append(
            {
                "variant": variant,
                "workload_id": "generative_attack_eval",
                "run_id": f"generative_{variant}_v1",
                "success": True,
                "metrics": metrics,
            }
        )

    # Workload: adversarial_training_loop
    # This workload updates the treatment detector using adverse examples
    # and yields a shared numeric metric comparable between baseline and
    # treatment.
    (
        adv_training_metrics_baseline,
        adv_training_metrics_treatment,
    ) = run_adversarial_training_loop(
        baseline_detector=baseline_detector,
        treatment_detector=treatment_detector,
        datasets=datasets,
        rng=rng,
        heuristic_attacks=heuristic_attacks,
        optimization_attacks=optimization_attacks,
        generative_attacks=generative_attacks,
    )

    # For adversarial_training_loop we now record *comparable* metrics
    # for both baseline and treatment, preserving the intended
    # baseline/treatment comparison.
    runs.append(
        {
            "variant": "baseline",
            "workload_id": "adversarial_training_loop",
            "run_id": "advloop_baseline_v1",
            "success": True,
            "metrics": adv_training_metrics_baseline,
        }
    )
    runs.append(
        {
            "variant": "treatment",
            "workload_id": "adversarial_training_loop",
            "run_id": "advloop_treatment_v1",
            "success": True,
            "metrics": adv_training_metrics_treatment,
        }
    )

    evidence: Dict[str, Any] = {
        "chapter": CHAPTER,
        "experiment_id": EXPERIMENT_ID,
        "environment": build_environment_metadata(),
        "runs": runs,
        "notes": (
            "Toy offline implementation of exp-ch8-attack-defense-redteam-v1 "
            "using synthetic data and linear detectors. Metrics are limited "
            "to quantities computable without external libraries."
        ),
    }

    evidence_dir = ensure_evidence_dir()
    out_path = os.path.join(evidence_dir, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)


if __name__ == "__main__":
    main()
