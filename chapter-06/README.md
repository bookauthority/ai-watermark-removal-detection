# Chapter 6 Synthetic Deep Learning Experiment (Offline Sandbox)

This project provides a **self-contained, offline-safe** simulation of the Chapter 6 experiment:

> **"Deep Learning Architectures for Image-Based Detection"**

It does **not** download data, use GPUs, or depend on any deep learning framework. Instead, it:

- Implements **baseline** and **treatment** variants using simple NumPy-like logic from the Python standard library.
- Simulates **three workloads** corresponding to WL1, WL2, and WL3 via deterministic pseudo-random data.
- Produces an evidence file at `$BOOKMAS_EVIDENCE_DIR/results.json` with the exact schema requested by the runtime contract.
- Includes unit tests that validate this schema and basic sanity of produced metrics.

This structure mirrors the intended real deep-learning setup while remaining lightweight and executable in an offline sandbox.

## Contents

- `run.py` – Main entry point that runs the synthetic experiments and writes evidence.
- `experiment.py` – Core logic for the synthetic baseline and treatment models and workloads.
- `schema.py` – Helpers for building and validating the evidence JSON object.
- `requirements.txt` – Empty runtime dependency list (standard library only).
- `tests/test_experiment.py` – Unit tests verifying that evidence matches the required schema and that runs are present.

## How It Works

### Baseline vs Treatment

In a real Chapter 6 experiment, baseline and treatment would be:

- **Baseline**: ResNet-50 on RGB only with standard BCE loss.
- **Treatments**: Hybrid CNNs/ViTs with forensic channels and robust losses.

In this sandbox, we simulate those ideas as:

- **Baseline variant**: A simple linear classifier applied to a fixed-size synthetic feature vector, trained with a basic gradient-descent style loop.
- **Treatment variant**: A slightly different architecture and training regime (e.g., different loss form / hyperparameters) applied to the *same synthetic data* for fairness.

The code ensures:

- **Same workloads** (WL1, WL2, WL3) for both baseline and treatment.
- **Deterministic behavior** by seeding Python's `random` module.
- At least **one numeric metric** per run (here, a simple accuracy-style score) generated from actual computations.

### Workloads

The three workloads are implemented as follows:

- **WL1-CleanSplit-Standard**
  - Generates a synthetic dataset with a fixed train/validation/test split.
  - Both baseline and treatment are trained/evaluated on the same generated data.

- **WL2-Stress-HeldOutAttacks**
  - Simulates a domain shift by modifying the distribution for a "held-out" test partition.

- **WL3-CrossValidation-ContentStratified**
  - Simulates a simple K-fold cross-validation with fixed folds (K=3 here for brevity and bounded runtime).

Each workload produces at least **one baseline run and one treatment run**, with separate `run_id`s and a shared `workload_id` string.

### Metrics

Due to sandbox constraints, we cannot measure all real Chapter 6 metrics (AUC, ECE, latency, etc.). Instead, we:

- Compute a **single synthetic metric**, `synthetic_accuracy`, for each run.
- This metric is constructed from the synthetic predictions and labels.
- All metric values are genuine results of computation — never fabricated constants.
- If you extend this to a real environment, you can add more numeric metrics under the `metrics` dict.

## Running the Experiment

1. Ensure you are in the project root (where `run.py` lives).

2. (Optional) Set the evidence directory:

   ```bash
   export BOOKMAS_EVIDENCE_DIR=/evidence
   ```

   If not set, the code defaults to `/evidence`.

3. Run the experiment:

   ```bash
   python run.py
   ```

   This will:

   - Run baseline and treatment variants for each of the three workloads.
   - Construct an evidence object following the required contract.
   - Write it to `$BOOKMAS_EVIDENCE_DIR/results.json`.

4. Inspect the evidence (example):

   ```bash
   cat /evidence/results.json
   ```

## Running Tests

To run the tests using the Python standard library test runner:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The tests verify:

- The evidence file exists after running `run.py`.
- The JSON structure conforms to the specified schema.
- There is at least one baseline and one treatment run **per workload**.
- All metrics values are numeric.

## Extending This Project

If you adapt this skeleton to a real environment with PyTorch or similar:

- Replace the synthetic data and models in `experiment.py` with real datasets and deep networks.
- Preserve the evidence-writing contract in `run.py` so that Chapter 6 tooling and governance layers can remain unchanged.
- Add richer metrics (AUC, ECE, latency, etc.) to the `metrics` dict, ensuring they remain numeric scalars.

## Reproducibility Notes

- All randomness flows through Python's `random` module with explicit seeds.
- Workloads and runs are constructed deterministically based on those seeds.
- No external I/O or network calls are performed except writing the evidence file.

This makes the project suitable as a minimal but faithful stand-in for the full Chapter 6 experiment in constrained environments.
