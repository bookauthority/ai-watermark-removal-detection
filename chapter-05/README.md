# Chapter 5 Baseline Detection: Classical Features and Shallow Models (Self-Contained Experiment)

This mini-project is a **self-contained, dependency-free** approximation of the Chapter 5 experiment:

> "Baseline Detection: Classical Features and Shallow Models"

Because the execution environment is offline and restricted to the Python standard library, this project:

- Implements a **synthetic dataset** that mimics key aspects of watermark and watermark-removal artifacts.
- Implements:
  - A **pixel-only logistic regression baseline** (downsampled greyscale image, raw pixels only).
  - A **handcrafted-feature treatment model** using simple noise/edge/texture-like statistics and a small decision-tree ensemble.
- Runs both **baseline** and **treatment** in two workloads:
  - `wl-static-image-detection`
  - `wl-cross-distribution-eval`
- Produces a single evidence JSON file at `${BOOKMAS_EVIDENCE_DIR:-/evidence}/results.json` that conforms to the contract given in the prompt.

This code does **not** depend on any external libraries such as NumPy, scikit-learn, or OpenCV. All logic is implemented using the Python standard library only (e.g. `random`, `math`, `statistics`).

The goal is to demonstrate a *reproducible*, *bounded*, and *fair* experiment between a pixel baseline and a handcrafted-feature treatment, following the spirit of the specification while respecting sandbox constraints.

## Layout

- `run.py` – Orchestrates the entire experiment and writes `results.json` evidence.
- `experiment/
  - __init__.py
  - data.py` – Synthetic dataset generation and simple cross-distribution split.
  - `features.py` – Pixel baseline representation and handcrafted feature extractor.
  - `models.py` – Minimal implementations of:
    - Logistic regression (gradient descent classifier).
    - Tiny decision tree and random forest–style ensemble classifier.
  - `pipeline.py` – Training, evaluation, metrics, and workload wiring.
  - `utils.py` – Helpers for randomness control and environment metadata.
- `tests/test_experiment.py` – Tests that:
  - Run the experiment in a temporary evidence directory.
  - Validate the evidence JSON **schema** and basic properties.

## Running the Experiment

1. Ensure you are in the project directory (where `run.py` lives).
2. Run the experiment:

   ```bash
   python run.py
   ```

   This will:

   - Create a synthetic dataset.
   - Train a baseline logistic regression and a treatment ensemble model for each workload.
   - Evaluate them and compute metrics (accuracy and a few others that can be robustly measured in this environment).
   - Write `/evidence/results.json` or `${BOOKMAS_EVIDENCE_DIR}/results.json` (if that env var is set).

## Running Tests

Tests are based only on the Python standard library (`unittest`, `json`, `os`, `tempfile`, etc.):

```bash
python -m unittest discover -s tests -p "test_*.py"
``

The tests:

- Run the full experiment using a temporary evidence directory.
- Load the generated `results.json`.
- Assert that the schema specified in the instructions is satisfied:
  - `chapter` (integer)
  - `experiment_id` (string)
  - `environment` with `os` and `python`
  - `runs` list with entries containing `variant`, `workload_id`, `run_id`, `success`, and numeric `metrics` entries.
- Ensure **at least one baseline and one treatment** per workload are present.

## Notes and Limitations

- No external packages are used, so the models and feature extractors are simplified approximations of the specification:
  - Images are represented as 2D integer arrays (lists of lists).
  - The pixel baseline flattens a downsampled greyscale version into a vector.
  - Handcrafted features include simple statistics to mimic:
    - Noise (per-pixel high-frequency residual statistics)
    - Edges (finite-difference gradient counts)
    - Texture-like measures (local variance)
    - Simple frequency-like proxies (coarse block means and differences)
- Many rich metrics from the original spec (e.g. ROC-AUC, PR-AUC, Brier score, DeLong test) are **not implemented**, because they require more elaborate numerical integration or external libs. Instead we reliably report:
  - `accuracy`
  - `precision_positive_class`
  - `recall_positive_class`
  - `f1_positive_class`

  All of these are computed directly from the confusion matrix.

- Since we cannot measure true wall-clock times or model size in a portable, deterministic way across sandboxes, those metrics are omitted. This is documented in the `notes` field of the evidence file.

- Every metric in `metrics` is computed from actual model predictions on actual synthetic data inside this run; no fabricated numbers are used.

## Reproducibility

- Randomness is controlled via the Python `random` module with fixed seeds.
- The synthetic dataset generation and model training procedures are deterministic for a given seed.
- `run.py` uses a fixed seed for each workload/variant pair, so repeated runs in the same environment will produce identical results.

## Extensibility

Although this project is intentionally minimal for the sandbox, the structure mirrors a real experiment:

- `data.py` could be swapped out for real image loading and split logic.
- `features.py` could integrate NumPy, scikit-image, OpenCV, etc.
- `models.py` could be replaced with scikit-learn classifiers (logistic regression, SVM, random forests, gradient boosting).
- `pipeline.py` could be expanded to include cross-validation, hyperparameter search, ROC/PR curves, and model interpretability.

This makes it a drop-in template for a richer, fully-featured Chapter 5 implementation in a normal environment.
