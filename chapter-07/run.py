import json
import os
import platform
from typing import Any, Dict, List

from experiment.core import set_global_seed
from experiment.data import generate_all_workloads
from experiment.models import (
    train_and_evaluate_baseline,
    train_and_evaluate_treatment,
)


EXPERIMENT_ID = "exp-ch7-video-temporal-consistency-v1"
CHAPTER_NUMBER = 7


def ensure_evidence_dir() -> str:
    """Return the evidence directory path, creating it if necessary.

    Priority: BOOKMAS_EVIDENCE_DIR env var, else /evidence.
    """
    base = os.environ.get("BOOKMAS_EVIDENCE_DIR", "/evidence")
    os.makedirs(base, exist_ok=True)
    return base


def main() -> None:
    # Reproducible experiments
    set_global_seed(17)

    # Generate synthetic workloads
    workloads = generate_all_workloads()

    runs: List[Dict[str, Any]] = []

    # For each workload, train and evaluate baseline and treatment
    for workload_id, dataset in workloads.items():
        # Baseline run
        baseline_result = train_and_evaluate_baseline(workload_id, dataset)
        runs.append(baseline_result)

        # Treatment run
        treatment_result = train_and_evaluate_treatment(workload_id, dataset)
        runs.append(treatment_result)

    evidence: Dict[str, Any] = {
        "chapter": CHAPTER_NUMBER,
        "experiment_id": EXPERIMENT_ID,
        "environment": {
            "os": platform.platform(),
            "python": platform.python_version(),
        },
        "runs": runs,
        "notes": (
            "Synthetic, offline-friendly approximation of Chapter 7 video temporal "
            "consistency experiment. Uses small in-memory datasets and simple "
            "baseline/treatment models implemented with Python standard library only."
        ),
    }

    evidence_dir = ensure_evidence_dir()
    out_path = os.path.join(evidence_dir, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
