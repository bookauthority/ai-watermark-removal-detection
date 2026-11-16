from typing import Dict, List, Tuple

from .data import Example, generate_synthetic_dataset, train_val_test_split
from .features import pixel_baseline_features, forensic_features
from .models import LogisticRegressionGD, RandomForestLike
from .utils import set_global_seed


def _compute_confusion(y_true: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 1 and yp == 0:
            fn += 1
    return tp, fp, tn, fn


def _compute_basic_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    tp, fp, tn, fn = _compute_confusion(y_true, y_pred)
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {
        "accuracy": accuracy,
        "precision_positive_class": precision,
        "recall_positive_class": recall,
        "f1_positive_class": f1,
    }


def _prepare_dataset(seed: int = 42) -> Tuple[List[Example], List[Example], List[Example], List[Example], List[Example], List[Example]]:
    main, cross_clean, cross_heavy = generate_synthetic_dataset(seed=seed)
    main_train, main_val, main_test = train_val_test_split(main, seed=seed)
    cc_train, cc_val, cc_test = train_val_test_split(cross_clean, seed=seed + 1)
    ch_train, ch_val, ch_test = train_val_test_split(cross_heavy, seed=seed + 2)

    # For wl-static-image-detection we use main_{train,val,test}
    # For wl-cross-distribution-eval we train on main_train and test on cc_test and ch_test combined.
    return main_train, main_val, main_test, cc_test, ch_test, []


def run_workload_static_image_detection(seed: int = 123) -> List[Dict]:
    """Run wl-static-image-detection for baseline and treatment.

    Returns list of run dicts for the evidence file.
    """

    set_global_seed(seed)
    main_train, main_val, main_test, _, _, _ = _prepare_dataset(seed=seed)

    # Baseline: pixel-only logistic regression
    X_train = pixel_baseline_features(main_train)
    y_train = [ex.label for ex in main_train]
    X_test = pixel_baseline_features(main_test)
    y_test = [ex.label for ex in main_test]

    baseline_model = LogisticRegressionGD(learning_rate=0.3, n_iter=120, l2=0.001)
    baseline_model.fit(X_train, y_train)
    y_pred_base = baseline_model.predict(X_test)

    base_metrics = _compute_basic_metrics(y_test, y_pred_base)

    baseline_run = {
        "variant": "baseline",
        "workload_id": "wl-static-image-detection",
        "run_id": "baseline_pixel_logreg_static",
        "success": True,
        "metrics": base_metrics,
    }

    # Treatment: forensic features + random-forest-like ensemble
    X_train_f = forensic_features(main_train)
    X_test_f = forensic_features(main_test)

    treatment_model = RandomForestLike(n_estimators=9, n_candidates_per_stump=25)
    treatment_model.fit(X_train_f, y_train)
    y_pred_treat = treatment_model.predict(X_test_f)

    treat_metrics = _compute_basic_metrics(y_test, y_pred_treat)

    treatment_run = {
        "variant": "treatment",
        "workload_id": "wl-static-image-detection",
        "run_id": "treatment_forensic_rf_static",
        "success": True,
        "metrics": treat_metrics,
    }

    return [baseline_run, treatment_run]


def run_workload_cross_distribution_eval(seed: int = 456) -> List[Dict]:
    """Run wl-cross-distribution-eval for baseline and treatment.

    Train on main_train and evaluate on a held-out cross-distribution
    combination of cross_clean_test and cross_heavy_test.
    """

    set_global_seed(seed)
    main_train, _, _, cc_test, ch_test, _ = _prepare_dataset(seed=seed)

    cross_test = cc_test + ch_test

    # Baseline
    X_train = pixel_baseline_features(main_train)
    y_train = [ex.label for ex in main_train]
    X_test = pixel_baseline_features(cross_test)
    y_test = [ex.label for ex in cross_test]

    baseline_model = LogisticRegressionGD(learning_rate=0.3, n_iter=120, l2=0.001)
    baseline_model.fit(X_train, y_train)
    y_pred_base = baseline_model.predict(X_test)

    base_metrics = _compute_basic_metrics(y_test, y_pred_base)

    baseline_run = {
        "variant": "baseline",
        "workload_id": "wl-cross-distribution-eval",
        "run_id": "baseline_pixel_logreg_cross",
        "success": True,
        "metrics": base_metrics,
    }

    # Treatment
    X_train_f = forensic_features(main_train)
    X_test_f = forensic_features(cross_test)

    treatment_model = RandomForestLike(n_estimators=9, n_candidates_per_stump=25)
    treatment_model.fit(X_train_f, y_train)
    y_pred_treat = treatment_model.predict(X_test_f)

    treat_metrics = _compute_basic_metrics(y_test, y_pred_treat)

    treatment_run = {
        "variant": "treatment",
        "workload_id": "wl-cross-distribution-eval",
        "run_id": "treatment_forensic_rf_cross",
        "success": True,
        "metrics": treat_metrics,
    }

    return [baseline_run, treatment_run]


def run_full_experiment() -> Tuple[List[Dict], str]:
    """Run all workloads and return (runs, notes).

    Notes contain an explanation of which metrics are implemented and
    which ones are omitted compared to the original specification.
    """

    runs: List[Dict] = []
    runs.extend(run_workload_static_image_detection(seed=123))
    runs.extend(run_workload_cross_distribution_eval(seed=456))

    notes = (
        "This self-contained experiment approximates Chapter 5 baselines "
        "using a synthetic dataset and minimal models implemented with the "
        "Python standard library only. Metrics are limited to accuracy, "
        "precision, recall, and F1 for the positive class; other metrics "
        "from the original spec (ROC-AUC, PR-AUC, calibration, timing, "
        "model size, etc.) are omitted because they cannot be measured "
        "robustly in this constrained sandbox."
    )

    return runs, notes
