import json
import os
from typing import Any, Dict

from experiments.ch12_metrics_reporting.shared_utils import ensure_dir
from experiments.ch12_metrics_reporting.treatment.compute_risk_metrics import (
    compute_risk_and_cohort_metrics,
)
from experiments.ch12_metrics_reporting.treatment.generate_cohort_reports import (
    generate_cohort_report,
)
from experiments.ch12_metrics_reporting.treatment.generate_incident_retrospectives import (
    generate_incident_retrospectives,
)
from experiments.ch12_metrics_reporting.treatment.generate_executive_summary import (
    generate_executive_summary,
)
from experiments.ch12_metrics_reporting.baseline.compute_technical_metrics import (
    run_baseline_offline_metrics,
)


def run_treatment_offline_pipeline(
    logs_path: str,
    output_dir: str,
    config_dir: str,
    templates_dir: str,  # unused but kept for interface similarity
) -> Dict[str, float]:
    ensure_dir(output_dir)
    # First compute baseline-like technical metrics (shared layer)
    base_metrics = run_baseline_offline_metrics(logs_path, output_dir)
    risk_metrics = compute_risk_and_cohort_metrics(logs_path, output_dir, config_dir)

    cohort_metrics_path = os.path.join(output_dir, "cohort_metrics.json")
    cohort_report_metrics = generate_cohort_report(cohort_metrics_path, output_dir)

    # executive summary
    exec_metrics = generate_executive_summary(
        metrics_overall={**base_metrics, **risk_metrics},
        cohort_metrics_path=cohort_metrics_path,
        output_dir=output_dir,
    )

    # Compose a subset of metrics that are most relevant
    treatment_metrics: Dict[str, float] = {}
    treatment_metrics.update(base_metrics)
    treatment_metrics.update(risk_metrics)
    treatment_metrics.update(cohort_report_metrics)
    treatment_metrics.update(exec_metrics)

    # Persist as a risk metrics artifact
    out_json = os.path.join(output_dir, "risk_metrics_overall.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(treatment_metrics, f, indent=2, sort_keys=True)

    return treatment_metrics


def run_treatment_incident_pipeline(
    incidents_path: str,
    output_dir: str,
    config_dir: str,
    templates_dir: str,  # unused but kept for symmetry
) -> Dict[str, float]:
    ensure_dir(output_dir)
    incidents_dir = os.path.join(output_dir, "incident_reports")
    ensure_dir(incidents_dir)
    metrics = generate_incident_retrospectives(incidents_path, incidents_dir)

    out_json = os.path.join(output_dir, "treatment_incident_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    return metrics
