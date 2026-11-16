# Experiment: Multimodal and Ensemble Detection Strategies (Synthetic Sandbox)

This is a **self-contained, offline-safe** Python 3.11+ experiment project
implementing a miniature version of the Chapter 10 experiment:

> **Multimodal and Ensemble Detection Strategies**

The real chapter integrates heterogeneous detectors (classical forensics,
deep models, metadata/behavior, video) and ensembles them. In this sandbox
project we **simulate** such detectors with deterministic synthetic data and
simple models implemented with the Python standard library only.

The focus here is **reproducible experiment plumbing**:

- multiple workloads
- baseline vs. treatment (ensemble) variants
- simple multimodal features ("pixels" + "metadata" + "temporal")
- early fusion, score fusion, stacking, and tiered decision rules
- structured evidence written in a fixed JSON schema to `/evidence/results.json`

No external ML frameworks (e.g., PyTorch, scikit-learn) are used.

## Contents

- `run.py` – main entry point. Runs the experiment and writes evidence.
- `experiment/__init__.py` – package marker.
- `experiment/config.py` – constants for chapter, experiment id, workloads.
- `experiment/data.py` – deterministic synthetic dataset generators.
- `experiment/models.py` – simple baseline and ensemble detector classes.
- `experiment/metrics.py` – metric implementations (ROC-AUC, etc.).
- `experiment/pipeline.py` – orchestration of runs for each workload.
- `tests/test_experiment.py` – unit tests for evidence schema and behavior.
- `requirements.txt` – intentionally empty of runtime dependencies.

## Running the experiment

From the project root:

```bash
python run.py
```

This will:

1. Generate deterministic synthetic data for three workloads:
   - `wl_image_in_distribution`
   - `wl_image_shifted`
   - `wl_video_multimodal`
2. Instantiate baselines:
   - `baseline_img_deep`
   - `baseline_img_forensics`
   - `baseline_video_temporal`
   - `baseline_metadata_behavior`
3. Instantiate treatments (ensembles):
   - `treat_early_fusion_img`
   - `treat_score_fusion_img`
   - `treat_stacking_img_meta`
   - `treat_multimodal_video_pipeline`
4. Evaluate them on each workload (small synthetic size for speed).
5. Write a single evidence file at:

   ```
   ${BOOKMAS_EVIDENCE_DIR:-/evidence}/results.json
   ```

   following the exact contract specified in the prompt.

If `/evidence` does not exist it will be created. You can override the
location with:

```bash
BOOKMAS_EVIDENCE_DIR=/tmp/evidence python run.py
```

## Evidence schema (runtime contract)

`run.py` writes JSON of the form:

```json
{
  "chapter": 10,
  "experiment_id": "exp-10-multimodal-ensemble-detection-v1",
  "environment": {"os": "...", "python": "..."},
  "runs": [
    {
      "variant": "baseline" | "treatment",
      "workload_id": "wl_image_in_distribution" | "wl_image_shifted" | "wl_video_multimodal",
      "run_id": "<unique string>",
      "success": true,
      "metrics": {
        "roc_auc": 0.87,
        "brier_score": 0.21
      }
    }
  ],
  "notes": "Optional free-form text."
}
```

Requirements enforced by code and tests:

- At least **one baseline** and **one treatment** run per workload.
- Every run includes at least one numeric metric; metric values are
  **finite numbers** (no `null`, strings, objects).
- Additional non-metric information (stdout, rationales, etc.) must **not**
  be placed inside `metrics`.

## What is actually measured?

Because the sandbox has no external ML libraries or datasets, the project
uses **synthetic features and labels** with simple statistical models:

- Features:
  - `image_frame` → vectors of floats representing pseudo-pixel statistics.
  - `metadata` → low-dimensional numeric summary features.
  - `video_clip` → temporal sequences aggregated into statistics.
- Baseline detectors:
  - `baseline_img_deep` – linear model on pseudo-pixel features.
  - `baseline_img_forensics` – linear model on handcrafted features.
  - `baseline_video_temporal` – linear model on temporal features.
  - `baseline_metadata_behavior` – logistic-style model on metadata.
- Ensembles:
  - Early fusion: concatenated features + small MLP-like scoring.
  - Score fusion: weighted average of calibrated baseline scores.
  - Stacking: meta-model combining baseline outputs + metadata.
  - Tiered video: cheap first tier + expensive second tier.

All models are implemented manually; there is no training phase reading
from disk. Instead, we choose deterministic parameters that result in
non-trivial but bounded performance differences, allowing us to compute
valid metrics such as ROC-AUC and a simple Brier score analogue.

## Metrics implemented

For each run we compute a subset of metrics from the chapter spec,
subject to what can be meaningfully measured in this sandbox:

- `roc_auc` – area under ROC curve for binary labels.
- `brier_score` – mean squared error between predicted probabilities and
  binary labels, in [0, 1].

Other chapter metrics (e.g., PR-AUC, latency percentiles, throughput,
ECE, per-attack breakdown) are **not** measured here to keep the
implementation simple and avoid fabricating timing numbers. They are
therefore omitted from `metrics`.

## Running tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The tests verify that:

- `run.py` creates an evidence file at the expected location.
- The JSON structure matches the required contract:
  - correct top-level keys
  - per-run keys and types
  - numeric metrics only
- Each workload has both baseline and treatment variants recorded.

## Design notes

- **Determinism**: We fix the random seed inside generators so repeated
  runs yield identical results.
- **Safety**: No network, no shell commands, no access outside the
  current working directory and `/evidence`.
- **Simplicity**: All code uses only the Python standard library, and is
  written to be readable and easy to adapt for real experiments.

## Extending this project

If you want to adapt this skeleton to a real research project:

1. Replace synthetic data in `experiment/data.py` with real dataset
   loaders.
2. Replace simple models in `experiment/models.py` with genuine ML
   models (e.g., from PyTorch or scikit-learn).
3. Extend `experiment/metrics.py` with additional metrics from the full
   chapter spec.
4. Keep the evidence-writing behavior of `run.py` stable so automated
   tooling can ingest it without modification.
