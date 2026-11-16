# Chapter 4 – Watermark Dataset Pipeline (Reproducible Experiment)

This is a **self-contained, simplified experiment project** for:

- **Chapter:** 4
- **Title:** *Building Datasets for Watermark Removal Detection*
- **Experiment ID:** `ch04-watermark-dataset-pipeline-v1`

The goal is to demonstrate a **reproducible, configuration-driven dataset pipeline** with:

- Baseline: ad hoc, non-versioned pipeline logic.
- Treatment: unified, manifest-based, configuration-driven pipeline.

This miniature project does **not** implement real watermarking or image/video I/O. Instead, it:

- Simulates **base media items**, **watermarked** and **attacked** variants as structured records.
- Emulates **workloads** described in the specification using deterministic PRNG.
- Produces **manifests** and **summary statistics** in memory.
- Measures simple, real runtime metrics (e.g., counts and elapsed time).
- Writes an evidence file `results.json` in an evidence directory (see below).

It is intentionally restricted to the Python standard library to ensure offline reproducibility.

## Contents

- `run.py` – Main entry point. Runs baseline and treatment variants for all workloads and writes evidence.
- `pipeline_baseline.py` – Ad hoc, non-configuration-driven dataset simulation.
- `pipeline_treatment.py` – Configuration-driven, manifest-based dataset pipeline simulation.
- `config_schemas.py` – Simple configuration structures for workloads.
- `metrics.py` – Helper functions to compute counts and basic ratios.
- `util.py` – Common utilities (RNG seeding, timing, OS/Python info, IDs, evidence directory resolution).
- `requirements.txt` – Empty runtime dependency list (standard library only).
- `tests/test_experiment.py` – Unit tests that verify the evidence schema and basic behavior.

## Workloads Modeled

The experiment implements the following logical workloads (all simulated):

1. `workload_synthetic_images_only`
2. `workload_mixed_real_and_synthetic`
3. `workload_video_segments`
4. `workload_governance_and_bias_audit`

For each workload, the runner executes **both**:

- a **baseline** run, and
- a **treatment** run.

Each run records at least one numeric metric (e.g., number of items, runtime seconds) derived from actual execution.

## Evidence Format

`run.py` writes `${BOOKMAS_EVIDENCE_DIR}/results.json` (or a safe default) with this structure:

```json
{
  "chapter": 4,
  "experiment_id": "ch04-watermark-dataset-pipeline-v1",
  "environment": {"os": "...", "python": "..."},
  "runs": [
    {
      "variant": "baseline" | "treatment",
      "workload_id": "workload_synthetic_images_only" | ...,
      "run_id": "unique-per-workload-and-variant",
      "success": true,
      "metrics": {
        "num_items_total": 123,
        "num_base_items": 10,
        "pipeline_runtime": 0.0123
      },
      "notes": "Textual notes (not interpreted as metrics)."
    }
  ],
  "notes": "Global notes and limitations."
}
```

All metric values are **numbers**. Textual notes and explanations are kept outside `metrics`.

## Evidence directory

The evidence directory is resolved as follows:

1. If the environment variable `BOOKMAS_EVIDENCE_DIR` is set, it is used.
2. Otherwise, the default is a local `./evidence` directory under the current working directory.

This avoids permission issues that can arise when attempting to write to a hard-coded absolute path like `/evidence` on systems where the user is unprivileged.

## Running the Experiment

From the project root (where `run.py` is located):

```bash
# (Optional) Set evidence directory; default is ./evidence
export BOOKMAS_EVIDENCE_DIR=/evidence

# Run the experiment
python run.py

# Inspect results
cat "${BOOKMAS_EVIDENCE_DIR:-./evidence}/results.json"
```

## Running Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The tests:

- Validate the **schema** of `results.json`.
- Ensure there is at least one **baseline** and **treatment** run per workload.
- Check that metric values are numeric and finite.

## Baseline vs. Treatment Governance Workload

For the governance workload (`workload_governance_and_bias_audit`):

- The **baseline** governance audit simulates an ad hoc process that emits a single summary record per dependent workload without inspecting structured manifests.
- The **treatment** governance audit aggregates over dependent manifests from both baseline and treatment pipelines for upstream workloads. It records separate counts for baseline and treatment items and bases, illustrating richer, structured governance behavior.

This design maintains the intended contrast while ensuring that both variants operate over the same underlying (simulated) data.

## Limitations

- No external libraries (e.g., NumPy, Pillow, OpenCV) are used.
- No real image/video files are processed.
- Many complex aspects from the full specification are **simulated**, including:
  - Watermark schemes
  - Attack recipes
  - File hashing and storage footprints

Despite the simplifications, the core **reproducibility** and **governance** patterns are represented:

- Deterministic RNG usage per workload and variant.
- Configuration-driven vs. ad hoc pipeline structure.
- Manifest-style records and summary metrics.

This makes it a focused, offline-friendly companion experiment.
