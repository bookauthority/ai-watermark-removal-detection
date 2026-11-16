# Chapter 7 Video Temporal Consistency Experiment (Offline Synthetic Version)

This project is a **self-contained, offline-safe** Python 3.11+ experiment that approximates the structure of the Chapter 7 experiment:

> **"Extending Detection to Video and Temporal Consistency"**

It is adapted for the execution sandbox:

- No external packages (standard library only)
- No network, shell calls, or external tools (no ffmpeg, no GPUs)
- Fully deterministic synthetic data generation
- Lightweight, fast models implemented in NumPy-like style using pure Python lists

The goal is to:

- Provide a **baseline** frame-wise model extended to video by uniform frame sampling and simple aggregation.
- Provide a **treatment** temporal model using sequence features across frames.
- Run them across multiple synthetic "workloads" and produce an evidence file with actual measured metrics.

The structure mirrors the high-level intent of the original spec while remaining safe and runnable within tight constraints.

## Project Layout

- `run.py` — Entry point. Generates synthetic data, runs baseline and treatment models, and writes evidence JSON.
- `experiment/core.py` — Core types and utility functions for random seeding, metric computation, and simple models.
- `experiment/data.py` — Synthetic dataset generator and loaders for three workloads.
- `experiment/models.py` — Baseline and treatment model implementations and inference wrappers.
- `experiment/metrics.py` — Simple numeric metric computations used in this offline experiment.
- `tests/test_experiment.py` — Unit tests validating the evidence schema and basic consistency.
- `requirements.txt` — Empty of third-party dependencies; standard library only.

## Workloads in This Offline Approximation

We approximate the three workloads in the specification with tiny synthetic datasets:

1. `wd_vid_synthetic_short`
   - 20 synthetic short videos
   - Each video is represented as a list of 16 "frames", where each frame is a list of float features.
2. `wd_vid_synthetic_long`
   - 20 synthetic long videos with a larger number of frames (e.g., 32).
3. `wd_vid_realistic_eval`
   - 10 synthetic evaluation-only videos.

Each video has a binary label (0 = non_removed, 1 = removed), and the feature-generating process differs between classes so that models can achieve better-than-random performance.

## Models

### Baseline: Frame-wise Model

- Per-frame linear score: `score = dot(frame_features, w) + b`
- Video-level score: mean of frame scores across all frames.
- Parameters `w` and `b` are "trained" by a few steps of gradient-free coordinate updates on a small training set (still very fast).

### Treatment: Temporal Sequence Model

- Computes per-frame scores like the baseline but augments them with crude temporal features:
  - Mean and standard deviation across frames
  - Lag-1 difference statistics
- Uses a small linear mapping from an aggregated temporal feature vector to a final video-level score.
- Also trained using a simple gradient-free procedure.

Both models share the same synthetic training/validation/test splits per workload for a fair comparison.

## Metrics

To satisfy the requirement that *every reported metric must come from actual instrumented execution*, we compute simple, deterministic metrics that are cheap to compute:

- `loss` — Mean squared error between predicted scores and ground-truth labels on the test partition.
- `accuracy` — Fraction of correctly classified videos on the test partition using a fixed threshold of 0.5.

These metrics are stored under `metrics` for each run. No fabricated or guessed numbers are used.

## Evidence File Contract

`run.py` writes JSON evidence to:

- Path: `${BOOKMAS_EVIDENCE_DIR:-/evidence}/results.json`

Schema (required parts):

```json
{
  "chapter": 7,
  "experiment_id": "exp-ch7-video-temporal-consistency-v1",
  "environment": {
    "os": "...",
    "python": "..."
  },
  "runs": [
    {
      "variant": "baseline" or "treatment",
      "workload_id": "wd_vid_synthetic_short" | "wd_vid_synthetic_long" | "wd_vid_realistic_eval",
      "run_id": "unique string",
      "success": true or false,
      "metrics": {
        "loss": <number>,
        "accuracy": <number>
      }
    }
  ],
  "notes": "... optional explanatory notes ..."
}
```

Constraints enforced by tests:

- `chapter` is integer 7.
- `experiment_id` is exactly `exp-ch7-video-temporal-consistency-v1`.
- `runs` includes **at least one baseline and one treatment** run for **each** workload.
- Each `metrics` object contains only numeric values (no strings, dicts, arrays, or nulls).

Extra metadata fields may be added outside of `metrics`, but they must not be used for numeric measurements.

## Running the Experiment

From the project root:

```bash
python run.py
```

Optional: choose a custom evidence directory (must exist and be writable):

```bash
BOOKMAS_EVIDENCE_DIR=/tmp/evidence python run.py
```

This will:

1. Generate synthetic data for each workload (with a fixed random seed for reproducibility).
2. Train the baseline and treatment models on each workload.
3. Evaluate both models on the test partition.
4. Write the evidence JSON to `<BOOKMAS_EVIDENCE_DIR>/results.json`.

## Running Tests

Use Python's built-in unittest discovery:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The tests will:

- Ensure `results.json` exists and is valid JSON.
- Validate that it follows the schema described above.
- Confirm that there is at least one baseline and one treatment run per workload.
- Verify that all metric values are numeric.

## Reproducibility Guarantees

- Random seeds are set deterministically (fixed seed = 17) for Python's `random` module.
- Data generation and simple training loops are purely in-memory and deterministic.
- No external dependencies beyond the Python standard library.

## Limitations

- The actual chapter spec involves real video data, 2D and 3D CNNs, RNNs, and complex metrics like AUROC and throughput. Those are **not** implemented here because of sandbox and dependency constraints.
- Instead, we test the *structure* of the experiment (baseline vs. temporal treatment, three workloads, consistent evaluation) using synthetic data and extremely lightweight models.

This setup is intended as an executable, verifiable placeholder that respects the evidence contract and lays out a clear, reproducible pattern for more realistic implementations outside of this constrained environment.
