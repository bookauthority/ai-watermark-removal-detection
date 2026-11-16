import json
import os
import platform
import sys
from typing import Any, Dict, List

from experiment.config import CHAPTER_NUMBER, EXPERIMENT_ID, WORKLOAD_IDS
from experiment.pipeline import run_all_workloads


def main() -> None:
    evidence_dir = os.environ.get("BOOKMAS_EVIDENCE_DIR", "/evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    results_path = os.path.join(evidence_dir, "results.json")

    runs: List[Dict[str, Any]] = []

    # Orchestrate all workloads; this returns a list of run dicts
    for workload_id in WORKLOAD_IDS:
        workload_runs = run_all_workloads(workload_id)
        runs.extend(workload_runs)

    evidence: Dict[str, Any] = {
        "chapter": CHAPTER_NUMBER,
        "experiment_id": EXPERIMENT_ID,
        "environment": {
            "os": platform.platform(),
            "python": sys.version.split()[0],
        },
        "runs": runs,
        "notes": (
            "Synthetic, deterministic experiment using standard-library-only "
            "models and data. Metrics are computed on simulated multimodal "
            "features and labels; latency and throughput are intentionally "
            "omitted to avoid fabricating timing information."
        ),
    }

    # Write evidence JSON atomically where possible
    tmp_path = results_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, sort_keys=True)
    os.replace(tmp_path, results_path)


if __name__ == "__main__":
    main()
