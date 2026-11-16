import json
import os
from collections import defaultdict
from typing import Any, Dict, List

from experiments.ch12_metrics_reporting.shared_utils import (
    ensure_dir,
    load_jsonl,
    parse_iso8601,
)


def generate_incident_retrospectives(incidents_path: str, output_dir: str) -> Dict[str, float]:
    ensure_dir(output_dir)
    records = load_jsonl(incidents_path)

    # Group by incident_id
    by_incident: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_incident[r["incident_id"]].append(r)

    median_latencies: List[float] = []

    for inc_id, items in by_incident.items():
        fns: List[float] = []
        for r in items:
            if int(r.get("y_true", 0)) == 1 and int(r.get("y_pred", 0)) == 0:
                t_event = parse_iso8601(r["event_timestamp"])
                t_disc = parse_iso8601(r["ground_truth_discovered_at"])
                fns.append(max(0.0, t_disc - t_event))
        if not fns:
            continue
        fns_sorted = sorted(fns)
        n = len(fns_sorted)
        mid = n // 2
        if n % 2 == 1:
            med = float(fns_sorted[mid])
        else:
            med = float((fns_sorted[mid - 1] + fns_sorted[mid]) / 2.0)
        median_latencies.append(med)
        # Write a simple markdown incident summary file
        out_path = os.path.join(output_dir, f"incident_{inc_id}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# Incident {inc_id}\n\n")
            f.write(f"Median FN latency (s): {med:.1f}\n\n")
            f.write(f"Items in incident: {len(items)}\n")

    if not median_latencies:
        overall_median = 0.0
    else:
        median_latencies.sort()
        n = len(median_latencies)
        mid = n // 2
        if n % 2 == 1:
            overall_median = float(median_latencies[mid])
        else:
            overall_median = float((median_latencies[mid - 1] + median_latencies[mid]) / 2.0)

    return {"incident_detection_latency_median": float(overall_median)}
