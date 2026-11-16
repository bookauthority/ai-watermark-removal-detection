from typing import List, Dict, Any

from config_schemas import (
    SyntheticImagesConfig,
    MixedRealSyntheticConfig,
    VideoSegmentsConfig,
    GovernanceAuditConfig,
)
from util import make_rng, deterministic_uuid


PIPELINE_VERSION_TREATMENT = "treatment-1.0.0"


def _make_record(
    workload_id: str,
    base_index: int,
    variant_type: str,
    source_type: str,
    rng,
) -> Dict[str, Any]:
    """Construct a richer manifest-style record for the treatment pipeline.

    This simulates the unified, configuration-driven, well-documented pipeline.
    """
    base_id = deterministic_uuid("treat-base:" + workload_id, str(base_index))
    variant_key = f"{base_id}:{variant_type}:{base_index}"
    asset_id = deterministic_uuid("treat-asset:" + workload_id, variant_key)

    # Deterministic pseudo file hash using UUID (representing cryptographic hash)
    file_hash = deterministic_uuid("hash:" + workload_id, variant_key).replace("-", "")

    if variant_type == "clean":
        watermark_presence = "clean"
        attack_type = None
    elif variant_type == "watermarked":
        watermark_presence = "watermarked"
        attack_type = None
    else:
        watermark_presence = "removed"
        attack_type = "simulated_attack_generic"

    record = {
        "id": asset_id,
        "base_id": base_id,
        "variant_type": variant_type,
        "split": "train" if (base_index % 10) < 8 else ("val" if (base_index % 10) == 8 else "test"),
        "file_path": f"artifacts/{workload_id}/{variant_type}/{asset_id}.dat",
        "file_hash": file_hash,
        "modality": "image" if "video" not in workload_id else "video",
        "source_type": source_type,
        "source_dataset": f"simulated_{workload_id}",
        "license": {
            "type": "CC-BY-4.0" if source_type == "real" else "synthetic-generated",
            "attribution_required": bool(source_type == "real"),
            "notes": "Simulated license metadata for offline experiment.",
        },
        "labels": {
            "watermark_presence": watermark_presence,
            "attack_type": attack_type,
            "watermark_scheme": "visible_logo_bottom_right" if variant_type != "clean" else None,
        },
        "parameters": {
            "watermark_params": {
                "opacity": 0.5,
                "location": "bottom_right",
            } if variant_type != "clean" else {},
            "attack_params": {
                "severity": 0.7,
            } if attack_type is not None else {},
            "generation_params": {
                "generator": "simulated",
                "seed": rng.random(),  # numeric; used only for differentiation
            } if source_type == "synthetic" else {},
        },
        "quality_metrics": {
            # Simulate structural quality metrics with bounded numeric values.
            "psnr_to_base": 40.0 if variant_type == "clean" else 35.0,
            "ssim_to_base": 0.99 if variant_type == "clean" else 0.92,
            "perceptual_distance": 0.01 if variant_type == "clean" else 0.1,
        },
        "run_metadata": {
            "pipeline_version": PIPELINE_VERSION_TREATMENT,
            "dataset_version": "v1.0.0",
            "config_path": f"configs/ch4/{workload_id}.yaml",
            "random_seed": 0,  # will be updated by caller if desired
            "timestamp_utc": "1970-01-01T00:00:00Z",  # constant for determinism
            "environment": {
                "python_version": "3.11",
                "os": "simulated",
                "gpu_info": None,
            },
        },
    }
    return record


def run_treatment_synthetic_images(cfg: SyntheticImagesConfig, seed: int) -> List[Dict[str, Any]]:
    rng = make_rng(seed)
    manifest: List[Dict[str, Any]] = []

    # Treatment carefully enforces approximate ratios by deterministic pattern.
    per_base_variants = ["clean", "watermarked", "attacked", "attacked"]

    for i in range(cfg.num_base_images):
        source_type = "synthetic"
        for v in per_base_variants:
            rec = _make_record(cfg.workload_id, i, v, source_type, rng)
            manifest.append(rec)

    return manifest


def run_treatment_mixed_real_synthetic(cfg: MixedRealSyntheticConfig, seed: int) -> List[Dict[str, Any]]:
    rng = make_rng(seed)
    manifest: List[Dict[str, Any]] = []

    per_base_variants = ["clean", "watermarked", "attacked", "attacked"]

    # Real portion
    for i in range(cfg.num_base_real):
        for v in per_base_variants:
            manifest.append(_make_record(cfg.workload_id, i, v, "real", rng))

    # Synthetic portion
    for j in range(cfg.num_base_synthetic):
        base_index = cfg.num_base_real + j
        for v in per_base_variants:
            manifest.append(_make_record(cfg.workload_id, base_index, v, "synthetic", rng))

    return manifest


def run_treatment_video_segments(cfg: VideoSegmentsConfig, seed: int) -> List[Dict[str, Any]]:
    rng = make_rng(seed)
    manifest: List[Dict[str, Any]] = []

    # Each base video is represented as a single record per variant; frame-level
    # metadata is abstracted away but could be added here.
    per_base_variants = ["clean", "watermarked", "attacked"]

    for i in range(cfg.num_base_videos):
        source_type = "real"
        for v in per_base_variants:
            rec = _make_record(cfg.workload_id, i, v, source_type, rng)
            manifest.append(rec)

    return manifest


def run_treatment_governance_audit(cfg: GovernanceAuditConfig, seed: int,
                                   dependent_manifests: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
    """Simulate a governance and bias audit over existing manifests.

    Aggregates counts across dependent workloads and emits a single summary
    record containing counts in `parameters`. The summary explicitly
    distinguishes baseline and treatment inputs.
    """
    rng = make_rng(seed)
    workload_id = cfg.workload_id

    total_items_baseline = 0
    total_items_treatment = 0
    total_bases_baseline = 0
    total_bases_treatment = 0

    for w_id in cfg.dependent_workloads:
        dep = dependent_manifests.get(w_id, {})
        man_baseline = dep.get("baseline", [])
        man_treatment = dep.get("treatment", [])

        total_items_baseline += len(man_baseline)
        total_items_treatment += len(man_treatment)

        total_bases_baseline += len({r.get("base_id") for r in man_baseline})
        total_bases_treatment += len({r.get("base_id") for r in man_treatment})

    base_index = 0
    source_type = "synthetic"
    rec = _make_record(workload_id, base_index, "clean", source_type, rng)
    rec["parameters"] = {
        "governance_summary": {
            "dependent_workloads": list(cfg.dependent_workloads),
            "baseline": {
                "total_items": total_items_baseline,
                "total_base_items": total_bases_baseline,
            },
            "treatment": {
                "total_items": total_items_treatment,
                "total_base_items": total_bases_treatment,
            },
        }
    }

    return [rec]
