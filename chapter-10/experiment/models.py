import math
from dataclasses import dataclass
from typing import List

from .data import Sample


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class BaseDetector:
    """Abstract base for all detectors.

    Subclasses implement `predict_proba(sample)` returning float in [0, 1].
    """

    model_id: str

    def predict_proba(self, sample: Sample) -> float:  # pragma: no cover - abstract
        raise NotImplementedError


# ---------------------- Baseline detectors ----------------------


@dataclass
class BaselineImgDeep(BaseDetector):
    """Simulated deep image detector using linear projection of image features."""

    weights: List[float]
    bias: float

    def predict_proba(self, sample: Sample) -> float:
        z = self.bias
        for w, x in zip(self.weights, sample.x_img):
            z += w * x
        return _sigmoid(z)


@dataclass
class BaselineImgForensics(BaseDetector):
    """Simulated classical forensic detector on handcrafted features."""

    weights: List[float]
    bias: float

    def predict_proba(self, sample: Sample) -> float:
        z = self.bias
        for w, x in zip(self.weights, sample.x_forensics):
            z += w * x
        return _sigmoid(z)


@dataclass
class BaselineVideoTemporal(BaseDetector):
    """Simulated temporal video detector."""

    weights: List[float]
    bias: float

    def predict_proba(self, sample: Sample) -> float:
        z = self.bias
        for w, x in zip(self.weights, sample.x_video):
            z += w * x
        return _sigmoid(z)


@dataclass
class BaselineMetadataBehavior(BaseDetector):
    """Simulated metadata/behavior-only detector."""

    weights: List[float]
    bias: float

    def predict_proba(self, sample: Sample) -> float:
        z = self.bias
        for w, x in zip(self.weights, sample.x_meta):
            z += w * x
        return _sigmoid(z)


# ---------------------- Ensemble detectors ----------------------


@dataclass
class EarlyFusionImageEnsemble(BaseDetector):
    """Early fusion ensemble: concatenate img + forensic features.

    Implements a tiny 2-layer MLP-like model with a hidden layer of size 4.
    We do not train this model; parameters are chosen to leverage both
    feature groups and create different behavior from baselines.
    """

    # Input dimension is len(x_img) + len(x_forensics)
    w1: List[List[float]]  # shape (hidden_dim, input_dim)
    b1: List[float]
    w2: List[float]  # shape (hidden_dim,)
    b2: float

    def _relu(self, x: float) -> float:
        return x if x > 0.0 else 0.0

    def predict_proba(self, sample: Sample) -> float:
        x = sample.x_img + sample.x_forensics
        # Hidden layer
        h: List[float] = []
        for row, b in zip(self.w1, self.b1):
            z = b
            for w, xi in zip(row, x):
                z += w * xi
            h.append(self._relu(z))
        # Output layer
        z2 = self.b2
        for w, hi in zip(self.w2, h):
            z2 += w * hi
        return _sigmoid(z2)


@dataclass
class ScoreFusionImageEnsemble(BaseDetector):
    """Score-level fusion of two image detectors with fixed weights."""

    deep: BaselineImgDeep
    forensic: BaselineImgForensics
    w_deep: float
    w_forensic: float

    def predict_proba(self, sample: Sample) -> float:
        p_deep = self.deep.predict_proba(sample)
        p_for = self.forensic.predict_proba(sample)
        total = self.w_deep + self.w_forensic
        if total <= 0.0:
            alpha = 0.5
        else:
            alpha = self.w_deep / total
        return alpha * p_deep + (1.0 - alpha) * p_for


@dataclass
class StackingImageMetadataEnsemble(BaseDetector):
    """Stacking ensemble using image detectors + metadata features.

    Uses a simple logistic regression over concatenated level-1 probabilities
    and raw metadata features.
    """

    deep: BaselineImgDeep
    forensic: BaselineImgForensics
    meta: BaselineMetadataBehavior
    weights: List[float]
    bias: float

    def predict_proba(self, sample: Sample) -> float:
        # Level-1 features: probabilities and metadata
        p_deep = self.deep.predict_proba(sample)
        p_for = self.forensic.predict_proba(sample)
        p_meta = self.meta.predict_proba(sample)
        x = [p_deep, p_for, p_meta] + sample.x_meta
        z = self.bias
        for w, xi in zip(self.weights, x):
            z += w * xi
        return _sigmoid(z)


@dataclass
class TieredVideoPipeline(BaseDetector):
    """Tiered multimodal video pipeline.

    Tier 1: metadata + cheap image deep score based on image features only.
    Tier 2: if ambiguous, adds temporal and forensic signals.
    """

    deep: BaselineImgDeep
    forensic: BaselineImgForensics
    video: BaselineVideoTemporal
    meta: BaselineMetadataBehavior
    low_threshold: float = 0.25
    high_threshold: float = 0.75

    def predict_proba(self, sample: Sample) -> float:
        # Tier 1: combine metadata and deep image lightly
        p_meta = self.meta.predict_proba(sample)
        p_deep = self.deep.predict_proba(sample)
        p_tier1 = 0.6 * p_deep + 0.4 * p_meta

        if p_tier1 <= self.low_threshold or p_tier1 >= self.high_threshold:
            # Confident enough; stop here
            return p_tier1

        # Tier 2: incorporate forensic + temporal models
        p_for = self.forensic.predict_proba(sample)
        p_vid = self.video.predict_proba(sample)
        # Weighted average giving more weight to temporal on video workload
        return 0.3 * p_tier1 + 0.3 * p_for + 0.4 * p_vid


# ---------------------- Factory functions ----------------------


def make_baseline_detectors() -> List[BaseDetector]:
    # Chosen weights differ so detectors complement each other.
    baseline_img_deep = BaselineImgDeep(
        model_id="baseline_img_deep",
        weights=[1.0, 0.5, -0.3],
        bias=-0.1,
    )
    baseline_img_forensics = BaselineImgForensics(
        model_id="baseline_img_forensics",
        weights=[0.9, -0.7, 0.4],
        bias=0.05,
    )
    baseline_video_temporal = BaselineVideoTemporal(
        model_id="baseline_video_temporal",
        weights=[1.1, -0.2, 0.3],
        bias=-0.05,
    )
    baseline_metadata_behavior = BaselineMetadataBehavior(
        model_id="baseline_metadata_behavior",
        weights=[1.0, 0.3],
        bias=0.0,
    )
    return [
        baseline_img_deep,
        baseline_img_forensics,
        baseline_video_temporal,
        baseline_metadata_behavior,
    ]


def make_treatment_detectors(baselines: List[BaseDetector]) -> List[BaseDetector]:
    # Map by id for convenience
    by_id = {b.model_id: b for b in baselines}

    # Early fusion image ensemble
    # Input dim = 3 (img) + 3 (forensics) = 6, hidden_dim = 4
    w1 = [
        [0.7, 0.5, -0.2, 0.6, -0.4, 0.3],
        [-0.5, 0.3, 0.4, 0.2, 0.1, -0.3],
        [0.2, -0.6, 0.5, -0.1, 0.4, 0.2],
        [0.1, 0.2, 0.1, 0.3, 0.3, -0.2],
    ]
    b1 = [0.0, 0.1, -0.1, 0.05]
    w2 = [0.8, -0.4, 0.6, 0.3]
    b2 = -0.05
    treat_early_fusion_img = EarlyFusionImageEnsemble(
        model_id="treat_early_fusion_img",
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
    )

    # Score fusion ensemble
    treat_score_fusion_img = ScoreFusionImageEnsemble(
        model_id="treat_score_fusion_img",
        deep=by_id["baseline_img_deep"],
        forensic=by_id["baseline_img_forensics"],
        w_deep=0.6,
        w_forensic=0.4,
    )

    # Stacking image + metadata
    stacking_weights = [0.9, 0.6, 0.7, 0.5, 0.2]
    stacking_bias = -0.2
    treat_stacking_img_meta = StackingImageMetadataEnsemble(
        model_id="treat_stacking_img_meta",
        deep=by_id["baseline_img_deep"],
        forensic=by_id["baseline_img_forensics"],
        meta=by_id["baseline_metadata_behavior"],
        weights=stacking_weights,
        bias=stacking_bias,
    )

    # Tiered video pipeline
    treat_multimodal_video_pipeline = TieredVideoPipeline(
        model_id="treat_multimodal_video_pipeline",
        deep=by_id["baseline_img_deep"],
        forensic=by_id["baseline_img_forensics"],
        video=by_id["baseline_video_temporal"],
        meta=by_id["baseline_metadata_behavior"],
        low_threshold=0.3,
        high_threshold=0.7,
    )

    return [
        treat_early_fusion_img,
        treat_score_fusion_img,
        treat_stacking_img_meta,
        treat_multimodal_video_pipeline,
    ]
