import json
import os
import sys

from experiment.pipeline import run_full_experiment
from experiment.utils import get_environment_metadata


CHAPTER_NUMBER = 5
EXPERIMENT_ID = "exp-ch5-baseline-detection-classical-features-v1"


def main() -> None:
    # Determine evidence directory
    evidence_dir = os.environ.get("BOOKMAS_EVIDENCE_DIR", "/evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    evidence_path = os.path.join(evidence_dir, "results.json")

    # Run experiment
    runs, notes = run_full_experiment()

    env_meta = get_environment_metadata()

    evidence = {
        "chapter": CHAPTER_NUMBER,
        "experiment_id": EXPERIMENT_ID,
        "environment": env_meta,
        "runs": runs,
        "notes": notes,
    }

    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, sort_keys=True)

    # Also print a short message so humans can see something if running manually
    sys.stdout.write(f"Evidence written to {evidence_path}\n")


if __name__ == "__main__":
    main()
