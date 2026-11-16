# Chapter 8 – Adversarial Attacks and Red-Teaming Your Detector (Offline Toy Experiment)

This self-contained project implements a **toy but fully runnable** version of the
Chapter 8 experiment:

> **Adversarial Attacks and Red-Teaming Your Detector**
>
> Experiment ID: `exp-ch8-attack-defense-redteam-v1`

Because this sandbox is **offline** and restricted to the Python standard
library, the implementation simulates the structure of the full
attack–defense pipeline using small, deterministic arrays instead of real
images, neural networks, or diffusion models.

The code is designed to:

- Respect the specified evidence schema and runtime contract.
- Distinguish **baseline** vs **treatment** detectors.
- Implement **attack families** (heuristic, optimization, generative) as
  simple, deterministic transformations on numeric vectors.
- Provide **actual, computed metrics** (not fabricated) that are
  comparable between baseline and treatment.
- Run quickly and safely in a constrained environment.

## Project Layout

- `run.py` – Entry point for the experiment. Runs all workloads for
  both baseline and treatment models and writes evidence.
- `experiment/__init__.py` – Package marker.
- `experiment/core.py` – Core data structures and utilities (dataset
  creation, deterministic RNG, simple metrics).
- `experiment/model.py` – Baseline and treatment "detectors"
  implemented as linear classifiers over synthetic features.
- `experiment/attacks.py` – Toy heuristic, optimization-based, and
  generative attack pipelines implemented as numeric transforms.
- `experiment/workloads.py` – Workload runners corresponding to the
  clean and attack evaluation protocols and the adversarial training
  loop.
- `tests/test_experiment.py` – Unit tests ensuring that evidence
  written by `run.py` matches the required schema and that
  baseline/treatment have shared numeric metrics per workload.
- `requirements.txt` – Empty of third-party runtime dependencies as
  required (standard library only).

## How the Toy Experiment Maps to the Specification

The full spec mentions real images, CNNs, LPIPS, PSNR, and diffusion
models. In this constrained environment we mimic these concepts:

- **Images** → 8-dimensional float vectors in `[0, 1]`.
- **Labels** → Three-class task:
  - `0`: clean
  - `1`: watermarked
  - `2`: watermark-removed
- **Detector** → Simple linear model with ReLU and softmax-like scores.
  - Baseline parameters are fixed.
  - Treatment parameters are produced by a deterministic "adversarial
    training" step that nudges weights using hard negatives.
- **Attacks**:
  - *Heuristic*: Small random-looking but deterministic affine changes
    to vector elements (e.g., scale, shift, swap entries).
  - *Optimization-based*: Iterative perturbations that greedily reduce
    the detector's confidence on watermark-removed class.
  - *Generative*: Larger but still constrained transforms that mimic
    "resynthesis" by mixing the sample with class centroids.

We cannot implement real LPIPS or PSNR, so those metrics are **omitted
from `metrics`** in the evidence JSON. Instead we provide numeric metrics
we can actually compute:

- `accuracy` for clean/reference workloads and post-training checks.
- `attack_evasion_rate` for attack workloads.

All other narrative elements (e.g., AUROC, F1, GPU hours) are not
reported as numeric fields, as that would violate the requirement to
avoid fabricated metrics.

## Evidence Schema and Contract

`run.py` must write a single JSON file at:

```text
$BOOKMAS_EVIDENCE_DIR/results.json
```

with structure:

```json
{
  "chapter": 8,
  "experiment_id": "exp-ch8-attack-defense-redteam-v1",
  "environment": {"os": "...", "python": "..."},
  "runs": [
    {
      "variant": "baseline" | "treatment",
      "workload_id": "clean_reference_eval" | "heuristic_attack_eval" |
                      "optimization_based_attack_eval" | "generative_attack_eval" |
                      "adversarial_training_loop",
      "run_id": "...",   // unique per workload+variant
      "success": true,    // did the run complete as intended
      "metrics": {
        "some_metric": 0.123  // numeric values only
      }
    },
    ...
  ],
  "notes": "optional descriptive text"
}
```

Constraints enforced by the code and tests:

- `chapter` is the integer `8`.
- `experiment_id` is exactly
  `"exp-ch8-attack-defense-redteam-v1"`.
- `environment.os` and `environment.python` are strings.
- At least one **baseline** and one **treatment** run per workload.
- Every `metrics` value is a **finite JSON number** (no strings,
  `null`, nested dicts, etc.).
- **Every declared workload has at least one numeric metric name shared
  between its baseline and treatment runs**, enabling direct comparison.
- The schema can include extra top-level fields and run-level fields,
  but tests do not treat them as measurements.

## How to Run

From the project root:

```bash
export BOOKMAS_EVIDENCE_DIR=/evidence  # or another writable path
python run.py
```

This will:

1. Generate synthetic datasets and models.
2. Run all five workloads for both baseline and treatment variants.
3. Write evidence JSON to `$BOOKMAS_EVIDENCE_DIR/results.json`.

To run tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The tests will:

- Import and execute `run.py` in a temporary evidence directory.
- Load and validate `/evidence/results.json` (or the temporary path it
  sets) against the required minimal schema.
- Check that every workload has shared numeric metric keys across
  baseline and treatment.

## Notes on Determinism

- A simple `RandomState` wrapper (built on Python's `random` module) is
  used with a fixed global seed (1337) to generate synthetic data and
  attack perturbations.
- All transformations and training steps are deterministic given the
  fixed seed.
- There is no multi-processing or external dependency that could
  introduce nondeterminism.

## Limitations vs. Full Chapter

This toy experiment is **structurally faithful** to the chapter's
attack–defense setup but does **not**:

- Use real images or watermarking schemes.
- Use deep learning frameworks (e.g., PyTorch, diffusion models).
- Measure image-based metrics like LPIPS or PSNR.

It is intended solely as a **reproducible scaffolding** that mirrors the
experiment workflow under tight sandbox constraints.
