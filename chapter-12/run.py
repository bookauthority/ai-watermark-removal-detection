import json
import os
import sys
import time
from typing import Any, Dict, List

from experiments.ch12_metrics_reporting.shared_utils import (
    ensure_dir,
    generate_synthetic_detection_logs,
    generate_synthetic_incident_logs,
    load_simple_yaml,
)
from experiments.ch12_metrics_reporting.baseline.compute_technical_metrics import (
    run_baseline_offline_metrics,
    run_baseline_incident_metrics,
)
from experiments.ch12_metrics_reporting.treatment.run_full_reporting_pipeline import (
    run_treatment_offline_pipeline,
    run_treatment_incident_pipeline,
)


CHAPTER_NUMBER = 12
EXPERIMENT_ID = "ch12-metrics-reporting-bridge-tech-risk-v1"


def get_evidence_dir() -> str:
    path = os.environ.get("BOOKMAS_EVIDENCE_DIR", "/evidence")
    ensure_dir(path)
    return path


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = os.path.join(base_dir, "_work")
    ensure_dir(work_dir)

    # Config directories (synthetic but present in repo)
    config_dir = os.path.join(base_dir, "experiments", "ch12_metrics_reporting", "config")
    templates_dir = os.path.join(base_dir, "experiments", "ch12_metrics_reporting", "templates")

    # Generate synthetic datasets deterministically
    rng_seed = 12345
    detection_logs_path = os.path.join(work_dir, "detection_logs.jsonl")
    incidents_logs_path = os.path.join(work_dir, "incident_logs.jsonl")

    if not os.path.exists(detection_logs_path):
        generate_synthetic_detection_logs(detection_logs_path, num_items=5000, seed=rng_seed)
    if not os.path.exists(incidents_logs_path):
        generate_synthetic_incident_logs(incidents_logs_path, num_items=1000, seed=rng_seed)

    # Baseline and treatment output dirs
    baseline_out = os.path.join(work_dir, "baseline_outputs")
    treatment_out = os.path.join(work_dir, "treatment_outputs")
    ensure_dir(baseline_out)
    ensure_dir(treatment_out)

    runs: List[Dict[str, Any]] = []

    # Workload 1: offline_model_evaluation_logs
    workload_offline = "offline_model_evaluation_logs"

    # Baseline offline metrics
    start = time.perf_counter()
    base_off_metrics = run_baseline_offline_metrics(
        logs_path=detection_logs_path,
        output_dir=baseline_out,
    )
    elapsed = time.perf_counter() - start
    base_off_metrics["execution_runtime_seconds"] = float(elapsed)
    runs.append(
        {
            "variant": "baseline",
            "workload_id": workload_offline,
            "run_id": "baseline-offline-1",
            "success": True,
            "metrics": {k: float(v) for k, v in base_off_metrics.items()},
        }
    )

    # Treatment offline pipeline
    start = time.perf_counter()
    treat_off_metrics = run_treatment_offline_pipeline(
        logs_path=detection_logs_path,
        output_dir=treatment_out,
        config_dir=config_dir,
        templates_dir=templates_dir,
    )
    elapsed = time.perf_counter() - start
    treat_off_metrics["execution_runtime_seconds"] = float(elapsed)
    runs.append(
        {
            "variant": "treatment",
            "workload_id": workload_offline,
            "run_id": "treatment-offline-1",
            "success": True,
            "metrics": {k: float(v) for k, v in treat_off_metrics.items()},
        }
    )

    # Workload 2: incident_case_study_logs
    workload_incidents = "incident_case_study_logs"

    # Baseline incidents: simple FN latency measure
    start = time.perf_counter()
    base_inc_metrics = run_baseline_incident_metrics(
        incidents_path=incidents_logs_path,
        output_dir=baseline_out,
    )
    elapsed = time.perf_counter() - start
    base_inc_metrics["execution_runtime_seconds"] = float(elapsed)
    runs.append(
        {
            "variant": "baseline",
            "workload_id": workload_incidents,
            "run_id": "baseline-incidents-1",
            "success": True,
            "metrics": {k: float(v) for k, v in base_inc_metrics.items()},
        }
    )

    # Treatment incidents: richer incident reporting
    start = time.perf_counter()
    treat_inc_metrics = run_treatment_incident_pipeline(
        incidents_path=incidents_logs_path,
        output_dir=treatment_out,
        config_dir=config_dir,
        templates_dir=templates_dir,
    )
    elapsed = time.perf_counter() - start
    treat_inc_metrics["execution_runtime_seconds"] = float(elapsed)
    runs.append(
        {
            "variant": "treatment",
            "workload_id": workload_incidents,
            "run_id": "treatment-incidents-1",
            "success": True,
            "metrics": {k: float(v) for k, v in treat_inc_metrics.items()},
        }
    )

    evidence = {
        "chapter": CHAPTER_NUMBER,
        "experiment_id": EXPERIMENT_ID,
        "environment": {
            "os": os.name,
            "python": sys.version.split()[0],
        },
        "runs": runs,
        "notes": "Synthetic offline execution for Chapter 12 baseline vs. treatment reporting experiment.",
    }

    evidence_dir = get_evidence_dir()
    out_path = os.path.join(evidence_dir, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, sort_keys=True)

    print(f"Wrote evidence to {out_path}")


if __name__ == "main__":  # pragma: no cover
    # Intentional bug in name would break execution; correct it below.
    main()

if __name__ == "__main__":
    main()
