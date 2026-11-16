import json
import os
from typing import Any, Dict, List

from experiments.ch12_metrics_reporting.shared_utils import (
    compute_confusion,
    ensure_dir,
    load_jsonl,
    parse_iso8601,
    safe_div,
)


def run_baseline_offline_metrics(logs_path: str, output_dir: str) -> Dict[str, float]:
    """Compute overall technical metrics from detection logs.

    Returns metrics dict with numeric values.
    """

    ensure_dir(output_dir)
    records = load_jsonl(logs_path)
    y_true = [int(r.get("y_true", 0)) for r in records]
    y_pred = [int(r.get("y_pred", 0)) for r in records]
    tp, fp, tn, fn = compute_confusion(y_true, y_pred)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)

    metrics = {
        "overall_technical_precision": float(precision),
        "overall_technical_recall": float(recall),
        "num_items": float(len(records)),
    }

    # Persist a simple JSON summary as if it were the baseline artifact.
    out_json = os.path.join(output_dir, "metrics_overall.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    return metrics


def run_baseline_incident_metrics(incidents_path: str, output_dir: str) -> Dict[str, float]:
    """Compute a minimal incident FN latency metric for baseline.

    Baseline doesn't do full risk-centric analysis, but we can still
    approximate incident_detection_latency_median from the incident logs.
    """

    ensure_dir(output_dir)
    records = load_jsonl(incidents_path)
    latencies: List[float] = []
    for r in records:
        if int(r.get("y_true", 0)) == 1 and int(r.get("y_pred", 0)) == 0:
            # FN item
            ts_event = parse_iso8601(r["event_timestamp"])
            ts_discovered = parse_iso8601(r["ground_truth_discovered_at"])
            lat = max(0.0, ts_discovered - ts_event)
            latencies.append(lat)

    if not latencies:
        median_latency = 0.0
    else:
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        mid = n // 2
        if n % 2 == 1:
            median_latency = float(latencies_sorted[mid])
        else:
            median_latency = float((latencies_sorted[mid - 1] + latencies_sorted[mid]) / 2.0)

    metrics = {
        "incident_detection_latency_median": float(median_latency),
        "num_incident_items": float(len(records)),
    }

    out_json = os.path.join(output_dir, "baseline_incident_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    return metrics
