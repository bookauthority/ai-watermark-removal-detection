import math
from typing import Iterable, List, Tuple


def _sorted_by_score(y_true: List[int], y_score: List[float]) -> List[Tuple[float, int]]:
    paired = list(zip(y_score, y_true))
    paired.sort(key=lambda t: t[0], reverse=True)
    return paired


def roc_auc_score(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    """Compute ROC-AUC for binary labels.

    Implemented from first principles to avoid external dependencies.
    If there are no positive or no negative examples, returns 0.5.
    """

    y_true_list = list(y_true)
    y_score_list = list(y_score)
    assert len(y_true_list) == len(y_score_list)

    n_pos = sum(1 for y in y_true_list if y == 1)
    n_neg = len(y_true_list) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    sorted_pairs = _sorted_by_score(y_true_list, y_score_list)

    # Compute TPR/FPR as we sweep threshold; use trapezoidal rule
    tp = 0
    fp = 0
    prev_fpr = 0.0
    prev_tpr = 0.0
    auc = 0.0

    for score, label in sorted_pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        # Trapezoid between previous and current
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr = fpr
        prev_tpr = tpr

    return float(auc)


def brier_score(y_true: Iterable[int], y_prob: Iterable[float]) -> float:
    y_true_list = list(y_true)
    y_prob_list = list(y_prob)
    assert len(y_true_list) == len(y_prob_list)
    if not y_true_list:
        return 0.0
    s = 0.0
    for yt, yp in zip(y_true_list, y_prob_list):
        yp_clamped = min(1.0, max(0.0, yp))
        s += (yp_clamped - yt) ** 2
    return s / len(y_true_list)


def is_finite_number(x: float) -> bool:
    return not (math.isinf(x) or math.isnan(x))
