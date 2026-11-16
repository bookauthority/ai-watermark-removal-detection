import json
import os
from typing import Any, Dict, List

from experiments.ch12_metrics_reporting.shared_utils import ensure_dir


def generate_cohort_report(cohort_metrics_path: str, output_dir: str) -> Dict[str, float]:
    """Generate a simple cohort report artifact and return a metric.

    We compute here the same num_cohorts_with_kpis metric, but this function
    mainly demonstrates generating an artifact.
    """

    ensure_dir(output_dir)
    with open(cohort_metrics_path, "r", encoding="utf-8") as f:
        rows: List[Dict[str, Any]] = json.load(f)

    # Dump a markdown-ish table summarizing a few cohorts
    out_path = os.path.join(output_dir, "cohort_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Cohort Metrics Report\n\n")
        f.write("cohort_id | support | precision | recall | risk_weighted_fn_rate\n")
        f.write("---|---|---|---|---\n")
        for row in rows[:20]:
            f.write(
                f"{row['cohort_id']} | {row['support']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['risk_weighted_false_negative_rate']:.3f}\n"
            )

    return {"num_cohorts_with_kpis": float(len(rows))}
