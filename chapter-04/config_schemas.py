from dataclasses import dataclass
from typing import List, Dict, Union


@dataclass
class SyntheticImagesConfig:
    workload_id: str = "workload_synthetic_images_only"
    num_base_images: int = 1000
    target_variant_ratio: Dict[str, float] = None

    def __post_init__(self):
        if self.target_variant_ratio is None:
            # Default 1:1:2 ratio for clean:watermarked:attacked
            self.target_variant_ratio = {"clean": 0.25, "watermarked": 0.25, "attacked": 0.5}


@dataclass
class MixedRealSyntheticConfig:
    workload_id: str = "workload_mixed_real_and_synthetic"
    num_base_real: int = 1000
    num_base_synthetic: int = 1000
    target_variant_ratio: Dict[str, float] = None

    def __post_init__(self):
        if self.target_variant_ratio is None:
            self.target_variant_ratio = {"clean": 0.25, "watermarked": 0.25, "attacked": 0.5}


@dataclass
class VideoSegmentsConfig:
    workload_id: str = "workload_video_segments"
    num_base_videos: int = 100
    max_frames_per_clip: int = 64


@dataclass
class GovernanceAuditConfig:
    workload_id: str = "workload_governance_and_bias_audit"
    # In a real system, this would reference existing manifests.
    # Here we simulate that manifests from other workloads are given.
    dependent_workloads: List[str] = None

    def __post_init__(self):
        if self.dependent_workloads is None:
            self.dependent_workloads = [
                "workload_synthetic_images_only",
                "workload_mixed_real_and_synthetic",
            ]


WorkloadConfig = Union[
    SyntheticImagesConfig,
    MixedRealSyntheticConfig,
    VideoSegmentsConfig,
    GovernanceAuditConfig,
]
