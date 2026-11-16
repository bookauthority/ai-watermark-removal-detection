from dataclasses import dataclass
from typing import Dict, List, Tuple

from .core import VideoSample, DatasetSplits, mse_loss, accuracy_at_threshold


@dataclass
class BaselineModel:
    """Simple per-frame linear model aggregated by mean over frames.

    score(frame) = dot(w, frame) + b
    video_score(video) = mean_frame(score(frame)).
    """

    weights: List[float]
    bias: float

    def predict_video(self, video: VideoSample) -> float:
        scores = []
        for frame in video.frames:
            s = sum(w * x for w, x in zip(self.weights, frame)) + self.bias
            scores.append(s)
        if not scores:
            return 0.5
        avg_score = sum(scores) / float(len(scores))
        # Convert to probability-like score using logistic squashing.
        from .core import sigmoid

        return sigmoid(avg_score)


@dataclass
class TreatmentTemporalModel:
    """Temporal model that uses both per-frame and sequence-level features.

    It reuses a per-frame linear scoring but adds sequence statistics:

    - per-frame scores mean and std
    - lag-1 difference mean and std

    Then maps this 4D temporal feature vector to a final probability-like
    score via a small linear layer + sigmoid.
    """

    frame_weights: List[float]
    frame_bias: float
    temporal_weights: List[float]  # length 4
    temporal_bias: float

    def _frame_scores(self, video: VideoSample) -> List[float]:
        scores: List[float] = []
        for frame in video.frames:
            s = sum(w * x for w, x in zip(self.frame_weights, frame)) + self.frame_bias
            scores.append(s)
        return scores

    def predict_video(self, video: VideoSample) -> float:
        from .core import sigmoid

        scores = self._frame_scores(video)
        n = len(scores)
        if n == 0:
            return 0.5
        mean_s = sum(scores) / float(n)
        var_s = sum((s - mean_s) ** 2 for s in scores) / float(n)
        std_s = var_s ** 0.5

        # lag-1 differences
        if n > 1:
            diffs = [scores[i + 1] - scores[i] for i in range(n - 1)]
            mean_d = sum(diffs) / float(len(diffs))
            var_d = sum((d - mean_d) ** 2 for d in diffs) / float(len(diffs))
            std_d = var_d ** 0.5
        else:
            mean_d = 0.0
            std_d = 0.0

        feats = [mean_s, std_s, mean_d, std_d]
        lin = sum(w * f for w, f in zip(self.temporal_weights, feats)) + self.temporal_bias
        return sigmoid(lin)


def _initialize_baseline(num_features: int) -> BaselineModel:
    # Simple deterministic initialization around small values.
    weights = [0.01 * (i + 1) for i in range(num_features)]
    bias = 0.0
    return BaselineModel(weights=weights, bias=bias)


def _initialize_treatment(num_features: int) -> TreatmentTemporalModel:
    frame_weights = [0.01 * (i + 1) for i in range(num_features)]
    frame_bias = 0.0
    temporal_weights = [0.1, -0.05, 0.05, -0.02]
    temporal_bias = 0.0
    return TreatmentTemporalModel(
        frame_weights=frame_weights,
        frame_bias=frame_bias,
        temporal_weights=temporal_weights,
        temporal_bias=temporal_bias,
    )


def _collect_predictions(model, videos: List[VideoSample]) -> Tuple[List[float], List[int]]:
    preds: List[float] = []
    labels: List[int] = []
    for v in videos:
        preds.append(model.predict_video(v))
        labels.append(v.label)
    return preds, labels


def _simple_train_baseline(model: BaselineModel, train_videos: List[VideoSample], steps: int = 20, lr: float = 0.1) -> None:
    """Naive gradient-free training loop.

    We update each parameter slightly in the direction that reduces MSE
    using a simple coordinate search with finite-difference estimation.
    This is not efficient but is deterministic and quick for tiny data.
    """
    if not train_videos:
        return

    num_features = len(model.weights)
    for _ in range(steps):
        preds, labels = _collect_predictions(model, train_videos)
        base_loss = mse_loss(preds, [float(l) for l in labels])

        # Update weights one by one
        for j in range(num_features):
            orig = model.weights[j]
            delta = 0.01
            model.weights[j] = orig + delta
            preds_pos, _ = _collect_predictions(model, train_videos)
            loss_pos = mse_loss(preds_pos, [float(l) for l in labels])
            model.weights[j] = orig - delta
            preds_neg, _ = _collect_predictions(model, train_videos)
            loss_neg = mse_loss(preds_neg, [float(l) for l in labels])
            # Restore
            model.weights[j] = orig

            # Estimate gradient-like signal
            if loss_pos < base_loss or loss_neg < base_loss:
                if loss_pos < loss_neg:
                    model.weights[j] = orig + lr * delta
                else:
                    model.weights[j] = orig - lr * delta

        # Update bias
        orig_b = model.bias
        delta_b = 0.01
        model.bias = orig_b + delta_b
        preds_pos_b, _ = _collect_predictions(model, train_videos)
        loss_pos_b = mse_loss(preds_pos_b, [float(l) for l in labels])
        model.bias = orig_b - delta_b
        preds_neg_b, _ = _collect_predictions(model, train_videos)
        loss_neg_b = mse_loss(preds_neg_b, [float(l) for l in labels])
        model.bias = orig_b

        if loss_pos_b < base_loss or loss_neg_b < base_loss:
            if loss_pos_b < loss_neg_b:
                model.bias = orig_b + lr * delta_b
            else:
                model.bias = orig_b - lr * delta_b


def _simple_train_treatment(model: TreatmentTemporalModel, train_videos: List[VideoSample], steps: int = 20, lr: float = 0.1) -> None:
    """Naive training for the temporal model.

    First train the frame-level parameters similar to the baseline
    (reusing the baseline training approach), then tune the temporal layer
    with a similar coordinate search.
    """
    if not train_videos:
        return

    # Phase 1: tune frame_weights and frame_bias as in baseline
    baseline_like = BaselineModel(weights=list(model.frame_weights), bias=model.frame_bias)
    _simple_train_baseline(baseline_like, train_videos, steps=steps, lr=lr)
    model.frame_weights = list(baseline_like.weights)
    model.frame_bias = baseline_like.bias

    # Phase 2: tune temporal_weights and temporal_bias
    def preds_and_labels() -> Tuple[List[float], List[int]]:
        preds_local: List[float] = []
        labels_local: List[int] = []
        for v in train_videos:
            preds_local.append(model.predict_video(v))
            labels_local.append(v.label)
        return preds_local, labels_local

    for _ in range(steps):
        preds, labels = preds_and_labels()
        base_loss = mse_loss(preds, [float(l) for l in labels])
        # temporal_weights
        for j in range(len(model.temporal_weights)):
            orig = model.temporal_weights[j]
            delta = 0.01
            model.temporal_weights[j] = orig + delta
            preds_pos, _ = preds_and_labels()
            loss_pos = mse_loss(preds_pos, [float(l) for l in labels])
            model.temporal_weights[j] = orig - delta
            preds_neg, _ = preds_and_labels()
            loss_neg = mse_loss(preds_neg, [float(l) for l in labels])
            model.temporal_weights[j] = orig

            if loss_pos < base_loss or loss_neg < base_loss:
                if loss_pos < loss_neg:
                    model.temporal_weights[j] = orig + lr * delta
                else:
                    model.temporal_weights[j] = orig - lr * delta

        # temporal_bias
        orig_b = model.temporal_bias
        delta_b = 0.01
        model.temporal_bias = orig_b + delta_b
        preds_pos_b, _ = preds_and_labels()
        loss_pos_b = mse_loss(preds_pos_b, [float(l) for l in labels])
        model.temporal_bias = orig_b - delta_b
        preds_neg_b, _ = preds_and_labels()
        loss_neg_b = mse_loss(preds_neg_b, [float(l) for l in labels])
        model.temporal_bias = orig_b

        if loss_pos_b < base_loss or loss_neg_b < base_loss:
            if loss_pos_b < loss_neg_b:
                model.temporal_bias = orig_b + lr * delta_b
            else:
                model.temporal_bias = orig_b - lr * delta_b


def train_and_evaluate_baseline(workload_id: str, dataset: DatasetSplits) -> Dict:
    """Train and evaluate the baseline model on a given workload.

    Returns a run record compatible with the evidence schema.
    """
    # Assume at least one video and one frame for dimensions.
    example_video = dataset.train[0] if dataset.train else dataset.test[0]
    num_features = len(example_video.frames[0]) if example_video.frames else 1

    model = _initialize_baseline(num_features)
    _simple_train_baseline(model, dataset.train)

    # Evaluate on test set
    preds, labels = _collect_predictions(model, dataset.test)
    loss = mse_loss(preds, [float(l) for l in labels])
    acc = accuracy_at_threshold(preds, labels, 0.5)

    run = {
        "variant": "baseline",
        "workload_id": workload_id,
        "run_id": f"baseline_{workload_id}",
        "success": True,
        "metrics": {
            "loss": float(loss),
            "accuracy": float(acc),
        },
    }
    return run


def train_and_evaluate_treatment(workload_id: str, dataset: DatasetSplits) -> Dict:
    """Train and evaluate the treatment (temporal) model on a workload."""
    example_video = dataset.train[0] if dataset.train else dataset.test[0]
    num_features = len(example_video.frames[0]) if example_video.frames else 1

    model = _initialize_treatment(num_features)
    _simple_train_treatment(model, dataset.train)

    preds, labels = _collect_predictions(model, dataset.test)
    loss = mse_loss(preds, [float(l) for l in labels])
    acc = accuracy_at_threshold(preds, labels, 0.5)

    run = {
        "variant": "treatment",
        "workload_id": workload_id,
        "run_id": f"treatment_{workload_id}",
        "success": True,
        "metrics": {
            "loss": float(loss),
            "accuracy": float(acc),
        },
    }
    return run
