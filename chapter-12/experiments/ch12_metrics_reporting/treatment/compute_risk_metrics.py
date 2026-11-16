import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from experiments.ch12_metrics_reporting.shared_utils import (
    compute_confusion,
    ensure_dir,
    load_jsonl,
    load_simple_yaml,
    safe_div,
)


def _load_risk_weights(config_dir: str) -> Dict[Tuple[str, str, str], float]:
    path = os.path.join(config_dir, "risk_weights.yaml")
    cfg = load_simple_yaml(path)
    weights: Dict[Tuple[str, str, str], float] = {}
    table = cfg.get("weights", {})
    for key, val in table.items():
        if not isinstance(val, (int, float)):
            continue
        # keys like "US:news:high_reach"
        parts = key.split(":")
        if len(parts) != 3:
            continue
        weights[(parts[0], parts[1], parts[2])] = float(val)
    default = float(cfg.get("default", 1.0))
    weights[("*", "*", "*")] = default
    return weights


def _load_high_risk_def(config_dir: str) -> Dict[str, Any]:
    path = os.path.join(config_dir, "cohorts.yaml")
    return load_simple_yaml(path)


def _cohort_id(geo: str, cat: str, seg: str) -> str:
    return f"{geo}|{cat}|{seg}"


def compute_risk_and_cohort_metrics(
    logs_path: str,
    output_dir: str,
    config_dir: str,
) -> Dict[str, float]:
    ensure_dir(output_dir)
    records = load_jsonl(logs_path)
    risk_weights = _load_risk_weights(config_dir)
    cohorts_cfg = _load_high_risk_def(config_dir)
    high_geos = set(cohorts_cfg.get("high_risk_geographies", "").split(",")) if isinstance(cohorts_cfg.get("high_risk_geographies"), str) else set()
    high_cats = set(cohorts_cfg.get("high_risk_categories", "").split(",")) if isinstance(cohorts_cfg.get("high_risk_categories"), str) else set()

    def get_weight(geo: str, cat: str, seg: str) -> float:
        return risk_weights.get((geo, cat, seg)) or risk_weights.get(("*", "*", "*"), 1.0)

    # Overall risk-weighted FN rate
    total_weight = 0.0
    weighted_fn = 0.0

    # Per-cohort metrics
    cohort_items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for r in records:
        geo = r.get("geography", "NA")
        cat = r.get("content_category", "NA")
        seg = r.get("creator_segment", "NA")
        cid = _cohort_id(geo, cat, seg)
        cohort_items[cid].append(r)
        wt = get_weight(geo, cat, seg)
        total_weight += wt
        if int(r.get("y_true", 0)) == 1 and int(r.get("y_pred", 0)) == 0:
            weighted_fn += wt

    risk_weighted_fn_rate = safe_div(weighted_fn, total_weight)

    # High-risk segment recall
    high_y_true = []
    high_y_pred = []
    for r in records:
        geo = r.get("geography", "NA")
        cat = r.get("content_category", "NA")
        if geo in high_geos and cat in high_cats:
            high_y_true.append(int(r.get("y_true", 0)))
            high_y_pred.append(int(r.get("y_pred", 0)))
    if high_y_true:
        tp, fp, tn, fn = compute_confusion(high_y_true, high_y_pred)
        high_risk_recall = safe_div(tp, tp + fn)
    else:
        high_risk_recall = 0.0

    # Per-cohort KPIs
    cohort_metrics_rows: List[Dict[str, Any]] = []
    for cid, items in cohort_items.items():
        ys = [int(r.get("y_true", 0)) for r in items]
        yp = [int(r.get("y_pred", 0)) for r in items]
        tp, fp, tn, fn = compute_confusion(ys, yp)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        # risk-weighted FN rate within cohort
        c_geo, c_cat, c_seg = cid.split("|")
        wt = get_weight(c_geo, c_cat, c_seg)
        denom = wt * float(len(items)) if len(items) > 0 else 1.0
        c_fn_rate = safe_div(float(fn) * wt, denom)
        cohort_metrics_rows.append(
            {
                "cohort_id": cid,
                "geography": c_geo,
                "content_category": c_cat,
                "creator_segment": c_seg,
                "support": len(items),
                "precision": float(precision),
                "recall": float(recall),
                "risk_weighted_false_negative_rate": float(c_fn_rate),
            }
        )

    # Save cohort metrics CSV-like JSON for dashboarding
    out_cohort = os.path.join(output_dir, "cohort_metrics.json")
    with open(out_cohort, "w", encoding="utf-8") as f:
        json.dump(cohort_metrics_rows, f, indent=2, sort_keys=True)

    metrics = {
        "risk_weighted_false_negative_rate": float(risk_weighted_fn_rate),
        "high_risk_segment_recall": float(high_risk_recall),
        "num_cohorts_with_kpis": float(len(cohort_metrics_rows)),
    }
    return metrics
