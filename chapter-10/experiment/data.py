import math
import random
from dataclasses import dataclass
from typing import List, Tuple

from .config import GLOBAL_RANDOM_SEED, WORKLOAD_CONFIG


@dataclass
class Sample:
    """Synthetic multimodal sample.

    Attributes
    ----------
    x_img : List[float]
        Pseudo image-level features (e.g., pixel stats).
    x_forensics : List[float]
        Pseudo forensic features (noise, DCT stats, etc.).
    x_meta : List[float]
        Metadata / behavioral features.
    x_video : List[float]
        Temporal / video features.
    y : int
        Binary label: 0 = clean_or_watermarked_intact, 1 = watermark_removed.
    """

    x_img: List[float]
    x_forensics: List[float]
    x_meta: List[float]
    x_video: List[float]
    y: int


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def generate_synthetic_dataset(workload_id: str) -> List[Sample]:
    """Generate a deterministic synthetic dataset for a workload.

    The generator uses a fixed global seed plus the workload id to ensure
    reproducibility while giving slightly different distributions.
    """

    cfg = WORKLOAD_CONFIG[workload_id]
    num_samples = int(cfg["num_samples"])
    shift = float(cfg["shift"])

    # Derive workload-specific seed
    seed = GLOBAL_RANDOM_SEED
    for ch in workload_id:
        seed = (seed * 131 + ord(ch)) % (2**32)

    rnd = random.Random(seed)

    samples: List[Sample] = []
    for i in range(num_samples):
        # Latent difficulty: some points are easy, some hard
        base = rnd.uniform(-1.0, 1.0)
        # Simulate attack strength / watermark removal likelihood
        attack_signal = rnd.gauss(0.0 + shift, 1.0)

        # True underlying score for watermark removal
        true_score = 0.6 * attack_signal + 0.4 * base
        p_removed = _sigmoid(true_score)

        # Label with Bernoulli sampling
        # To keep determinism, use rnd.random()
        y = 1 if rnd.random() < p_removed else 0

        # Create correlated feature sets with different views of true_score
        def noisy_feature(scale: float, noise: float) -> float:
            return scale * true_score + rnd.gauss(0.0, noise)

        # Image features (pixels)
        x_img = [
            noisy_feature(1.0, 0.8),
            noisy_feature(0.5, 0.8),
            rnd.gauss(0.0, 1.0),
        ]

        # Forensic features
        x_forensics = [
            noisy_feature(1.2, 0.5),
            noisy_feature(-0.8, 0.5),
            noisy_feature(0.7, 0.7),
        ]

        # Metadata / behavior (sometimes more informative under shift)
        meta_strength = 1.0 if workload_id != "wl_image_in_distribution" else 0.7
        x_meta = [
            noisy_feature(meta_strength, 0.6),
            rnd.gauss(0.0, 1.0),
        ]

        # Video / temporal features
        video_strength = 1.1 if workload_id == "wl_video_multimodal" else 0.6
        x_video = [
            noisy_feature(video_strength, 0.7),
            noisy_feature(-0.4, 0.9),
            rnd.gauss(0.0, 1.2),
        ]

        samples.append(Sample(x_img=x_img, x_forensics=x_forensics, x_meta=x_meta, x_video=x_video, y=y))

    return samples


def train_val_split(samples: List[Sample], val_fraction: float = 0.25) -> Tuple[List[Sample], List[Sample]]:
    """Deterministic train/validation split.

    We perform a simple prefix/suffix split to avoid randomness.
    """

    n = len(samples)
    n_val = int(round(n * val_fraction))
    n_train = n - n_val
    train = samples[:n_train]
    val = samples[n_train:]
    return train, val
