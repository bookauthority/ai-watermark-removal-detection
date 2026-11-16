import json
import os
import platform
from pathlib import Path

from experiment.core import (
    run_baseline_on_workloads,
    run_treatment_on_workloads,
)


CHAPTER = 9
EXPERIMENT_ID = "exp-ch9-robustness-generalization-distshift-v1"


def get_environment_info() -> dict:
    """Collect minimal environment metadata using the standard library only."""
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
    }


def main() -> None:
    # Define workloads. We simulate two workloads from the full spec:
    # - W1-in-distribution-lab-image
    # - W5-heldout-attack-families-image
    workloads = [
        "W1-in-distribution-lab-image",
        "W5-heldout-attack-families-image",
    ]

    # Run baseline and treatment variants.
    baseline_runs = run_baseline_on_workloads(workloads)
    treatment_runs = run_treatment_on_workloads(workloads)

    all_runs = baseline_runs + treatment_runs

    evidence = {
        "chapter": CHAPTER,
        "experiment_id": EXPERIMENT_ID,
        "environment": get_environment_info(),
        "runs": all_runs,
        "notes": (
            "Synthetic, self-contained robustness experiment skeleton. "
            "Metrics are computed from simple linear models on synthetic "
            "datasets simulating in-distribution and held-out-attack "
            "workloads."
        ),
    }

    evidence_dir = os.environ.get("BOOKMAS_EVIDENCE_DIR", "/evidence")
    evidence_path = Path(evidence_dir) / "results.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)

    with evidence_path.open("w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
