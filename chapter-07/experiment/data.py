import random
from typing import Dict, List

from .core import DatasetSplits, VideoSample


def _generate_video(num_frames: int, num_features: int, label: int) -> VideoSample:
    """Generate a single synthetic video.

    The data distribution is slightly different for label 0 vs 1 so that models
    can learn non-trivial decision boundaries using simple statistics.

    - label 0: frames have features with mean around 0.25.
    - label 1: frames have features with mean around 0.75.

    Some per-frame noise is added to avoid degenerate patterns.
    """
    frames: List[List[float]] = []
    base_mean = 0.25 if label == 0 else 0.75
    for _ in range(num_frames):
        # Add mild temporal drift by varying the mean a bit for each frame.
        frame_mean = base_mean + random.uniform(-0.05, 0.05)
        frame = [max(0.0, min(1.0, random.gauss(frame_mean, 0.08))) for _ in range(num_features)]
        frames.append(frame)
    return VideoSample(frames=frames, label=label)


def _generate_split(num_videos: int, num_frames: int, num_features: int) -> List[VideoSample]:
    samples: List[VideoSample] = []
    for _ in range(num_videos):
        label = 1 if random.random() < 0.5 else 0
        samples.append(_generate_video(num_frames, num_features, label))
    return samples


def generate_workload_splits_short() -> DatasetSplits:
    """Generate synthetic splits for `wd_vid_synthetic_short`.

    - 16 frames per video
    - 6 features per frame
    """
    num_frames = 16
    num_features = 6

    train = _generate_split(20, num_frames, num_features)
    val = _generate_split(8, num_frames, num_features)
    test = _generate_split(8, num_frames, num_features)

    return DatasetSplits(train=train, val=val, test=test)


def generate_workload_splits_long() -> DatasetSplits:
    """Generate synthetic splits for `wd_vid_synthetic_long`.

    - 32 frames per video
    - 6 features per frame
    """
    num_frames = 32
    num_features = 6

    train = _generate_split(20, num_frames, num_features)
    val = _generate_split(8, num_frames, num_features)
    test = _generate_split(8, num_frames, num_features)

    return DatasetSplits(train=train, val=val, test=test)


def generate_workload_splits_realistic() -> DatasetSplits:
    """Generate synthetic splits for `wd_vid_realistic_eval`.

    This workload is evaluation-only in the original spec, but here we still
    generate train/val splits for completeness. Only the test split is used
    for reported metrics.

    - 24 frames per video
    - 6 features per frame
    """
    num_frames = 24
    num_features = 6

    train = _generate_split(10, num_frames, num_features)
    val = _generate_split(4, num_frames, num_features)
    test = _generate_split(10, num_frames, num_features)

    return DatasetSplits(train=train, val=val, test=test)


def generate_all_workloads() -> Dict[str, DatasetSplits]:
    """Generate splits for all workloads.

    Returns a mapping from workload_id to DatasetSplits.
    """
    return {
        "wd_vid_synthetic_short": generate_workload_splits_short(),
        "wd_vid_synthetic_long": generate_workload_splits_long(),
        "wd_vid_realistic_eval": generate_workload_splits_realistic(),
    }
