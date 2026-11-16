from typing import List, Dict, Any

from config_schemas import (
    SyntheticImagesConfig,
    MixedRealSyntheticConfig,
    VideoSegmentsConfig,
    GovernanceAuditConfig,
)
from util import make_rng, deterministic_uuid


PIPELINE_VERSION_BASELINE = "baseline-0.1.0"


def _make_record(
    workload_id: str,
    base_index: int,
    variant_type: str,
    source_type: str,
    rng,
) -> Dict[str, Any]:
    """Construct a minimal manifest-like record for the baseline pipeline.

    The baseline intentionally omits many governance and metadata fields to
    emulate an ad hoc process.
    """
    base_id = f"{workload_id}-base-{base_index}"
    asset_id = deterministic_uuid("baseline:" + workload_id, f"{base_id}:{variant_type}:{rng.random()}")
    labels = {}
    if variant_type == "clean":
        labels["watermark_presence"] = "clean"
    elif variant_type == "watermarked":
        labels["watermark_presence"] = "watermarked"
    else:
        # attacked variant implicitly assumed to have had watermark removal attempt
        labels["watermark_presence"] = "removed"

    rec = {
        "id": asset_id,
        "base_id": base_id,
        "variant_type": variant_type,
        "split": "train",  # baseline uses a single split
        "file_path": f"artifacts/{workload_id}/{variant_type}/{asset_id}.dat",
        "file_hash": "",  # missing in baseline
        "modality": "image" if "video" not in workload_id else "video",
        "source_type": source_type,
        "license": {},  # incomplete licensing info
        "labels": labels,
        "parameters": {},
        "quality_metrics": {},
        "run_metadata": {
            "pipeline_version": PIPELINE_VERSION_BASELINE,
        },
    }
    return rec


def run_baseline_synthetic_images(cfg: SyntheticImagesConfig, seed: int) -> List[Dict[str, Any]]:
    rng = make_rng(seed)
    manifest: List[Dict[str, Any]] = []

    for i in range(cfg.num_base_images):
        # Baseline loosely approximates configured ratios with random draws
        # but is not precise nor logged.
        # For each base, randomly decide how many variants exist.
        variants = ["clean", "watermarked", "attacked"]
        for v in variants:
            # 80% chance each variant exists
            if rng.random() < 0.8:
                rec = _make_record(cfg.workload_id, i, v, "synthetic", rng)
                manifest.append(rec)

    return manifest


def run_baseline_mixed_real_synthetic(cfg: MixedRealSyntheticConfig, seed: int) -> List[Dict[str, Any]]:
    rng = make_rng(seed)
    manifest: List[Dict[str, Any]] = []

    # Real images
    for i in range(cfg.num_base_real):
        for v in ("clean", "watermarked", "attacked"):
            if rng.random() < 0.7:
                manifest.append(_make_record(cfg.workload_id, i, v, "real", rng))

    # Synthetic images
    for i in range(cfg.num_base_synthetic):
        for v in ("clean", "watermarked", "attacked"):
            if rng.random() < 0.9:
                manifest.append(_make_record(cfg.workload_id, cfg.num_base_real + i, v, "synthetic", rng))

    return manifest


def run_baseline_video_segments(cfg: VideoSegmentsConfig, seed: int) -> List[Dict[str, Any]]:
    rng = make_rng(seed)
    manifest: List[Dict[str, Any]] = []

    for i in range(cfg.num_base_videos):
        # Each video contributes variants but baseline ignores frame-level details.
        for v in ("clean", "watermarked", "attacked"):
            if rng.random() < 0.85:
                manifest.append(_make_record(cfg.workload_id, i, v, "real", rng))

    return manifest


def run_baseline_governance_audit(cfg: GovernanceAuditConfig, seed: int,
                                   dependent_manifests: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Baseline governance audit over dependent manifests.

    This simulates an ad hoc process that emits one simple summary record per
    dependent workload, capturing only coarse counts and omitting rich
    governance metadata.
    """
    rng = make_rng(seed)
    workload_id = cfg.workload_id
    manifest: List[Dict[str, Any]] = []

    for idx, dep_id in enumerate(cfg.dependent_workloads):
        dep_manifest = dependent_manifests.get(dep_id, [])
        base_id = f"{workload_id}-summary-{idx}"
        asset_id = deterministic_uuid("baseline:" + workload_id, base_id)
        rec = {
            "id": asset_id,
            "base_id": base_id,
            "variant_type": "clean",
            "split": "train",
            "file_path": f"artifacts/{workload_id}/summary/{asset_id}.dat",
            "file_hash": "",
            "modality": "text",
            "source_type": "synthetic",
            "license": {},
            "labels": {"watermark_presence": "clean"},
            "parameters": {
                "note": "baseline governance audit does not use structured governance schemas",
                "dependent_workload": dep_id,
                "num_items_observed": len(dep_manifest),
            },
            "quality_metrics": {},
            "run_metadata": {
                "pipeline_version": PIPELINE_VERSION_BASELINE,
                "seed": seed,
            },
        }
        manifest.append(rec)
        _ = rng.random()

    return manifest
