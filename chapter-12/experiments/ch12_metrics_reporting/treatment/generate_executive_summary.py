import json
import os
from typing import Any, Dict

from experiments.ch12_metrics_reporting.shared_utils import ensure_dir


def generate_executive_summary(
    metrics_overall: Dict[str, float],
    cohort_metrics_path: str,
    output_dir: str,
) -> Dict[str, float]:
    ensure_dir(output_dir)

    with open(cohort_metrics_path, "r", encoding="utf-8") as f:
        cohorts = json.load(f)

    num_cohorts = len(cohorts)
    avg_precision = 0.0
    avg_recall = 0.0
    if num_cohorts:
        avg_precision = sum(float(c["precision"]) for c in cohorts) / float(num_cohorts)
        avg_recall = sum(float(c["recall"]) for c in cohorts) / float(num_cohorts)

    out_path = os.path.join(output_dir, "executive_summary.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Executive Summary\n\n")
        f.write("This is a synthetic executive-style summary for the treatment pipeline.\n\n")
        f.write("## Key Metrics\n\n")
        for k, v in sorted(metrics_overall.items()):
            f.write(f"- {k}: {v:.4f}\n")
        f.write("\n## Cohort Coverage\n\n")
        f.write(f"- Number of cohorts with KPIs: {num_cohorts}\n")
        f.write(f"- Average cohort precision: {avg_precision:.3f}\n")
        f.write(f"- Average cohort recall: {avg_recall:.3f}\n")
        f.write("\n## Limitations and Uncertainty\n\n")
        f.write("- Synthetic data only; values do not correspond to real-world performance.\n")
        f.write("- Incident patterns and risk weights are simplified for didactic purposes.\n")
        f.write("- No temporal drift or retraining effects are modeled.\n")

    # Uncertainty documentation presence is encoded as 1 when section is written
    return {
        "uncertainty_documentation_presence": 1.0,
        "avg_cohort_precision": float(avg_precision),
        "avg_cohort_recall": float(avg_recall),
    }
