import json
import os
import platform
import sys
from typing import Any, Dict, List


def build_evidence_object(
    chapter_number: int,
    experiment_id: str,
    runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Construct the evidence object following the specified contract.

    The function does not validate the runs beyond basic type and field
    checks; tests cover stricter validation.
    """
    evidence: Dict[str, Any] = {
        "chapter": int(chapter_number),
        "experiment_id": experiment_id,
        "environment": {
            "os": platform.platform(),
            "python": sys.version.split()[0],
        },
        "runs": [],
        "notes": "Synthetic offline experiment for Chapter 6; metrics are synthetic_accuracy values computed from deterministic simulations.",
    }

    # Basic normalization/validation of runs
    normalized_runs: List[Dict[str, Any]] = []
    for run in runs:
        variant = run.get("variant")
        workload_id = run.get("workload_id")
        run_id = run.get("run_id")
        success = bool(run.get("success", False))
        metrics = run.get("metrics", {})

        # Ensure metrics values are numeric types
        clean_metrics: Dict[str, float] = {}
        for k, v in metrics.items():
            # Only keep metrics that can be converted to float
            try:
                num = float(v)
            except (TypeError, ValueError):
                continue
            clean_metrics[k] = num

        normalized_runs.append(
            {
                "variant": variant,
                "workload_id": workload_id,
                "run_id": run_id,
                "success": success,
                "metrics": clean_metrics,
            }
        )

    evidence["runs"] = normalized_runs
    return evidence
