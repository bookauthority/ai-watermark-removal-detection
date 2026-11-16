# Chapter 9 Robustness Experiment (Self-Contained Sandbox Version)

This project provides a **minimal, fully self-contained** Python 3.11+ experiment
implementing a toy version of the robustness, generalization, and distribution
shift study described in the specification for:

- **Chapter**: 9
- **Experiment ID**: `exp-ch9-robustness-generalization-distshift-v1`

Because the execution environment is offline and restricted, this project does
**not** load real image/video models or external datasets. Instead, it uses a
small synthetic classification task that captures the *structure* of the
specified robustness experiment:

- A **baseline** model representing `img_cnn_vanilla` / `vid_3dcnn_vanilla`.
- A **treatment** model representing domain-adapted and uncertainty-aware
  variants (e.g., `img_cnn_da_erm_ua`, `vid_3dcnn_da_erm_ua`).
- Two workloads simulating:
  - `W1-in-distribution-lab-image` (in-distribution / lab data)
  - `W5-heldout-attack-families-image` (held-out attacks / OOD stress test)

The experiment computes simple numeric metrics that are guaranteed to be
well-defined and deterministic in the sandbox while respecting the evidence
schema and bookkeeping constraints.

## Contents

- `run.py` – Entry point that runs the experiment and writes evidence JSON.
- `experiment/__init__.py` – Package marker.
- `experiment/core.py` – Core experiment logic, synthetic dataset generation,
  models, and metrics.
- `tests/test_experiment.py` – Unittests validating the evidence schema and
  basic behavior.
- `requirements.txt` – Empty runtime dependency spec (standard library only).

## Design Overview

### Synthetic Data

We create two small synthetic binary classification datasets:

- **Workload W1 (lab / in-distribution)**
  - Features are 2D points sampled from two well-separated Gaussians.
  - Represents clean, canonical training/validation data.

- **Workload W5 (held-out attacks / OOD)**
  - Same label semantics, but samples are shifted and noisier.
  - Represents distribution shift / unseen attacks.

Each workload has a fixed random seed, so results are bitwise reproducible.

### Models

We use very simple models implemented purely with the Python standard library:

- **Baseline model** (`SimpleLinearModelBaseline`)
  - A linear classifier (2D weights + bias) trained with a few steps of
    gradient descent on W1-like training data.

- **Treatment model** (`SimpleLinearModelTreatment`)
  - Starts from the same initialization and training scheme as the baseline.
  - Includes a lightweight "domain adaptation" style tweak:
    - During training, it also sees a small number of W5-like points and
      uses a slightly more conservative learning rate.
  - At evaluation, it exposes an *uncertainty-aware decision rule*:
    - Uses a confidence threshold (on the predicted probability) and can
      abstain. For this sandbox, abstentions are **not** encoded as special
      labels; instead we record the abstention *rate* as a separate metric.

The actual learning is intentionally simple and low-precision because the
focus is on the *experiment scaffolding* and **evidence-schema compliance**,
not model quality.

### Metrics

For each (model, workload) pair we compute at least the following metrics:

- `accuracy` – Fraction of correctly classified points.
- `abstention_rate` – Fraction of points for which the treatment model would
  abstain (baseline never abstains, so its rate is 0.0).

We could in principle compute more metrics (precision/recall, etc.), but to
keep execution very fast and clear, this minimal implementation guarantees at
least one numeric metric per run. All metrics are derived from actual
computation; none are fabricated.

### Evidence Format

`run.py` writes a JSON file to `${BOOKMAS_EVIDENCE_DIR:-/evidence}/results.json`
with this exact structure:

```json
{
  "chapter": 9,
  "experiment_id": "exp-ch9-robustness-generalization-distshift-v1",
  "environment": {"os": "...", "python": "..."},
  "runs": [
    {
      "variant": "baseline",
      "workload_id": "W1-in-distribution-lab-image",
      "run_id": "baseline_W1_0",
      "success": true,
      "metrics": {"accuracy": 0.9}
    },
    {
      "variant": "treatment",
      "workload_id": "W1-in-distribution-lab-image",
      "run_id": "treatment_W1_0",
      "success": true,
      "metrics": {"accuracy": 0.92, "abstention_rate": 0.1}
    },
    {
      "variant": "baseline",
      "workload_id": "W5-heldout-attack-families-image",
      "run_id": "baseline_W5_0",
      "success": true,
      "metrics": {"accuracy": 0.7}
    },
    {
      "variant": "treatment",
      "workload_id": "W5-heldout-attack-families-image",
      "run_id": "treatment_W5_0",
      "success": true,
      "metrics": {"accuracy": 0.75, "abstention_rate": 0.2}
    }
  ],
  "notes": "Additional textual notes, if any."
}
```

Notes:

- Every `metrics` object contains only **numeric** values.
- Each workload has **at least one baseline and one treatment run**.
- Extra non-metric fields are only at the top level or outside the `metrics`
  dict.

## Running the Experiment

From the project root:

```bash
python run.py
```

This will:

1. Create synthetic datasets for two workloads.
2. Train/evaluate the baseline and treatment models.
3. Write `results.json` under the directory specified by
   `BOOKMAS_EVIDENCE_DIR` (default `/evidence`).

You can override the evidence directory, e.g.:

```bash
BOOKMAS_EVIDENCE_DIR=./evidence_out python run.py
```

## Running Tests

Use the Python standard library `unittest` test runner:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The tests verify that:

- `results.json` exists and is valid JSON.
- The top-level fields (`chapter`, `experiment_id`, `environment`, `runs`)
  are correctly populated.
- Each run has the required fields and correct types.
- All metric values are numeric scalars, not nested objects nor strings.
- At least one baseline and one treatment run exist per workload.

## Limitations

This sandbox experiment is **not** a full implementation of the
watermark-removal robustness suite described in the original, much richer
specification. In particular:

- We do **not** load or train CNNs / 3D CNNs or handle real images/videos.
- We do **not** implement the entire evidence schema, calibration, drift
  detection, or monitoring dashboards.
- We do **not** support GPUs or any external dependencies; everything is
  pure Python standard library.

However, the structure is intentionally designed so that it can be extended in
a real repository to:

- Replace synthetic data with actual datasets and dataloaders.
- Replace toy models with real neural network implementations (e.g., PyTorch).
- Expand metrics and reporting to the full schema.

This makes the experiment a **reproducible, minimal skeleton** aligned with the
Chapter 9 robustness themes while remaining safe and bounded for automated
execution in an offline sandbox.
