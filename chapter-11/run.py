import json
import os
import sys
from pathlib import Path

from experiment.arch_baseline import BaselineDetector
from experiment.arch_treatment import TreatmentPipeline
from experiment.workloads import (
    WorkloadDefinition,
    get_all_workloads,
    run_workload_for_variant,
)
from experiment.util import get_environment_info


EXPERIMENT_ID = "exp-ch11-arch-deploy-watermark-detection-v1"
CHAPTER = 11


def get_evidence_dir() -> Path:
    """Resolve evidence directory, defaulting to /evidence.

    The directory is created if it does not exist.
    """
    path_str = os.environ.get("BOOKMAS_EVIDENCE_DIR", "/evidence")
    path = Path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    evidence_dir = get_evidence_dir()
    results_path = evidence_dir / "results.json"

    environment = get_environment_info()

    # Instantiate architectures once to amortize any setup cost and keep the
    # experiment structure simple and deterministic.
    baseline = BaselineDetector()
    treatment = TreatmentPipeline()

    runs = []

    workloads: list[WorkloadDefinition] = get_all_workloads()

    # For reproducibility, we use a deterministic seed base and derive
    # variant-specific seeds from it in the workload runner.
    base_seed = 1337

    for workload in workloads:
        # Baseline run(s)
        run_id_baseline = f"baseline_{workload.workload_id}_run1"
        baseline_metrics, baseline_success = run_workload_for_variant(
            variant="baseline",
            workload=workload,
            architecture=baseline,
            base_seed=base_seed,
        )
        runs.append(
            {
                "variant": "baseline",
                "workload_id": workload.workload_id,
                "run_id": run_id_baseline,
                "success": bool(baseline_success),
                "metrics": baseline_metrics,
            }
        )

        # Treatment run(s)
        run_id_treatment = f"treatment_{workload.workload_id}_run1"
        treatment_metrics, treatment_success = run_workload_for_variant(
            variant="treatment",
            workload=workload,
            architecture=treatment,
            base_seed=base_seed,
        )
        runs.append(
            {
                "variant": "treatment",
                "workload_id": workload.workload_id,
                "run_id": run_id_treatment,
                "success": bool(treatment_success),
                "metrics": treatment_metrics,
            }
        )

    notes = (
        "This evidence file was generated in an offline sandbox. "
        "Baseline vs treatment architectures are simulated in-process using "
        "Python threads and queues. Only directly measurable metrics, such as "
        "per-request latency and throughput, are reported. More complex "
        "system-level metrics from the chapter specification (e.g., real "
        "queue depth in external brokers, GPU utilization, Prometheus-based "
        "SLOs) are not observable in this constrained environment and are "
        "therefore omitted."
    )

    evidence = {
        "chapter": CHAPTER,
        "experiment_id": EXPERIMENT_ID,
        "environment": environment,
        "runs": runs,
        "notes": notes,
    }

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, sort_keys=True)

    return 0


if __name__ == "__main__":  # pragma: no cover - direct CLI entry point
    raise SystemExit(main())
