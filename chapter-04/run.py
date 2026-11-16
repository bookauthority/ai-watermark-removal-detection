import json
import os
from typing import List, Dict, Any

from config_schemas import (
    SyntheticImagesConfig,
    MixedRealSyntheticConfig,
    VideoSegmentsConfig,
    GovernanceAuditConfig,
)
from pipeline_baseline import (
    run_baseline_synthetic_images,
    run_baseline_mixed_real_synthetic,
    run_baseline_video_segments,
    run_baseline_governance_audit,
)
from pipeline_treatment import (
    run_treatment_synthetic_images,
    run_treatment_mixed_real_synthetic,
    run_treatment_video_segments,
    run_treatment_governance_audit,
)
from metrics import compute_basic_stats
from util import (
    get_environment_info,
    now_monotonic,
    elapsed_seconds,
    evidence_dir,
    ensure_dir,
    generate_run_id,
    make_rng,
)


CHAPTER_NUMBER = 4
EXPERIMENT_ID = "ch04-watermark-dataset-pipeline-v1"


def _run_baseline(cfg_obj, seed: int, dependent_manifests: Dict[str, Dict[str, List[Dict[str, Any]]]] = None) -> Dict[str, Any]:
    workload_id = cfg_obj.workload_id
    start = now_monotonic()

    if isinstance(cfg_obj, SyntheticImagesConfig):
        manifest = run_baseline_synthetic_images(cfg_obj, seed)
    elif isinstance(cfg_obj, MixedRealSyntheticConfig):
        manifest = run_baseline_mixed_real_synthetic(cfg_obj, seed)
    elif isinstance(cfg_obj, VideoSegmentsConfig):
        manifest = run_baseline_video_segments(cfg_obj, seed)
    elif isinstance(cfg_obj, GovernanceAuditConfig):
        if dependent_manifests is None:
            dependent_manifests = {}
        manifest = run_baseline_governance_audit(cfg_obj, seed, dependent_manifests)
    else:
        raise ValueError("Unsupported config type for baseline")

    end = now_monotonic()
    runtime = elapsed_seconds(start, end)

    metrics = compute_basic_stats(manifest)
    metrics["pipeline_runtime"] = float(runtime)

    notes = f"Baseline run for {workload_id} with {len(manifest)} records."

    return {
        "manifest": manifest,
        "metrics": metrics,
        "success": True,
        "notes": notes,
    }


def _run_treatment(cfg_obj, seed: int, dependent_manifests: Dict[str, Dict[str, List[Dict[str, Any]]]] = None) -> Dict[str, Any]:
    workload_id = cfg_obj.workload_id
    start = now_monotonic()

    if isinstance(cfg_obj, SyntheticImagesConfig):
        manifest = run_treatment_synthetic_images(cfg_obj, seed)
    elif isinstance(cfg_obj, MixedRealSyntheticConfig):
        manifest = run_treatment_mixed_real_synthetic(cfg_obj, seed)
    elif isinstance(cfg_obj, VideoSegmentsConfig):
        manifest = run_treatment_video_segments(cfg_obj, seed)
    elif isinstance(cfg_obj, GovernanceAuditConfig):
        if dependent_manifests is None:
            dependent_manifests = {}
        manifest = run_treatment_governance_audit(cfg_obj, seed, dependent_manifests)
    else:
        raise ValueError("Unsupported config type for treatment")

    end = now_monotonic()
    runtime = elapsed_seconds(start, end)

    metrics = compute_basic_stats(manifest)
    metrics["pipeline_runtime"] = float(runtime)

    notes = f"Treatment run for {workload_id} with {len(manifest)} records."

    return {
        "manifest": manifest,
        "metrics": metrics,
        "success": True,
        "notes": notes,
    }


def main() -> None:
    env_info = get_environment_info()

    # Define workload configurations (small numbers for safe execution time).
    synthetic_cfg = SyntheticImagesConfig(num_base_images=50)
    mixed_cfg = MixedRealSyntheticConfig(num_base_real=30, num_base_synthetic=30)
    video_cfg = VideoSegmentsConfig(num_base_videos=20, max_frames_per_clip=16)
    governance_cfg = GovernanceAuditConfig()

    workloads = [synthetic_cfg, mixed_cfg, video_cfg, governance_cfg]

    evidence_runs: List[Dict[str, Any]] = []

    # Store manifests keyed by workload_id and variant for governance audit.
    manifests_for_governance: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for cfg in workloads:
        workload_id = cfg.workload_id

        # Baseline
        baseline_seed = 12345
        baseline_result = _run_baseline(cfg, baseline_seed, manifests_for_governance if isinstance(cfg, GovernanceAuditConfig) else None)
        baseline_rng = make_rng(baseline_seed)
        baseline_run_id = generate_run_id(f"{workload_id}-baseline", baseline_rng)

        # Store baseline manifests for non-governance workloads
        if not isinstance(cfg, GovernanceAuditConfig):
            manifests_for_governance.setdefault(workload_id, {})["baseline"] = baseline_result["manifest"]

        evidence_runs.append({
            "variant": "baseline",
            "workload_id": workload_id,
            "run_id": baseline_run_id,
            "success": bool(baseline_result["success"]),
            "metrics": baseline_result["metrics"],
            "notes": baseline_result["notes"],
        })

        # Treatment
        treatment_seed = 67890
        if isinstance(cfg, GovernanceAuditConfig):
            treatment_result = _run_treatment(cfg, treatment_seed, manifests_for_governance)
        else:
            treatment_result = _run_treatment(cfg, treatment_seed)
            manifests_for_governance.setdefault(workload_id, {})["treatment"] = treatment_result["manifest"]

        treatment_rng = make_rng(treatment_seed)
        treatment_run_id = generate_run_id(f"{workload_id}-treatment", treatment_rng)

        evidence_runs.append({
            "variant": "treatment",
            "workload_id": workload_id,
            "run_id": treatment_run_id,
            "success": bool(treatment_result["success"]),
            "metrics": treatment_result["metrics"],
            "notes": treatment_result["notes"],
        })

    # Build final evidence object.
    evidence_obj: Dict[str, Any] = {
        "chapter": CHAPTER_NUMBER,
        "experiment_id": EXPERIMENT_ID,
        "environment": env_info,
        "runs": evidence_runs,
        "notes": "Simplified offline simulation of watermark dataset pipeline baseline vs treatment. Metrics are record counts, ratios, and runtime only.",
    }

    # Write evidence file.
    out_dir = evidence_dir()
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence_obj, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
