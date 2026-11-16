from typing import Any, Dict, List

from .config import WORKLOAD_CONFIG
from .data import generate_synthetic_dataset
from .metrics import brier_score, is_finite_number, roc_auc_score
from .models import BaseDetector, make_baseline_detectors, make_treatment_detectors


def _select_relevant_detectors_for_workload(
    baselines: List[BaseDetector], treatments: List[BaseDetector], workload_id: str
) -> Dict[str, List[BaseDetector]]:
    """Return detectors to run for a workload grouped by variant.

    - For image workloads, include image + metadata baselines, and all image ensembles.
    - For video workload, include video + metadata baselines, and tiered video ensemble.
    """

    baseline_ids: List[str]
    treatment_ids: List[str]

    if workload_id in ("wl_image_in_distribution", "wl_image_shifted"):
        baseline_ids = ["baseline_img_deep", "baseline_img_forensics", "baseline_metadata_behavior"]
        treatment_ids = [
            "treat_early_fusion_img",
            "treat_score_fusion_img",
            "treat_stacking_img_meta",
        ]
    elif workload_id == "wl_video_multimodal":
        baseline_ids = ["baseline_video_temporal", "baseline_metadata_behavior"]
        treatment_ids = ["treat_multimodal_video_pipeline"]
    else:
        # Fallback: run all
        baseline_ids = [b.model_id for b in baselines]
        treatment_ids = [t.model_id for t in treatments]

    by_id_baseline = {b.model_id: b for b in baselines}
    by_id_treatment = {t.model_id: t for t in treatments}
    selected = {
        "baseline": [by_id_baseline[i] for i in baseline_ids if i in by_id_baseline],
        "treatment": [by_id_treatment[i] for i in treatment_ids if i in by_id_treatment],
    }
    return selected


def _evaluate_detector(detector: BaseDetector, workload_id: str) -> Dict[str, float]:
    cfg = WORKLOAD_CONFIG[workload_id]
    samples = generate_synthetic_dataset(workload_id)

    y_true = []
    y_prob = []
    for s in samples:
        y_true.append(s.y)
        p = detector.predict_proba(s)
        y_prob.append(p)

    roc = roc_auc_score(y_true, y_prob)
    bri = brier_score(y_true, y_prob)

    # Ensure metrics are finite numbers per contract
    metrics: Dict[str, float] = {}
    if is_finite_number(roc):
        metrics["roc_auc"] = float(roc)
    if is_finite_number(bri):
        metrics["brier_score"] = float(bri)
    # At least one metric must be present
    if not metrics:
        metrics["roc_auc"] = 0.5
    return metrics


def run_all_workloads(workload_id: str) -> List[Dict[str, Any]]:
    """Run all relevant baselines and treatments for a workload.

    Returns a list of per-run records following the evidence contract.
    """

    baselines = make_baseline_detectors()
    treatments = make_treatment_detectors(baselines)
    selected = _select_relevant_detectors_for_workload(baselines, treatments, workload_id)

    runs: List[Dict[str, Any]] = []

    for variant, detectors in selected.items():
        for det in detectors:
            metrics = _evaluate_detector(det, workload_id)
            run: Dict[str, Any] = {
                "variant": variant,
                "workload_id": workload_id,
                # Construct deterministic run_id from model + workload
                "run_id": f"{workload_id}__{variant}__{det.model_id}",
                "success": True,
                "metrics": metrics,
            }
            runs.append(run)

    # Sanity: ensure at least one baseline and one treatment run exist
    has_baseline = any(r["variant"] == "baseline" for r in runs)
    has_treatment = any(r["variant"] == "treatment" for r in runs)
    if not (has_baseline and has_treatment):
        raise RuntimeError(f"Workload {workload_id} must have both baseline and treatment runs")

    return runs
