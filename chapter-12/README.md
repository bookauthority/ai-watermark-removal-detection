# Chapter 12 Experiment: Metrics, Reporting, and Communication with Non-Technical Stakeholders

This self-contained project simulates Chapter 12's experiment:

> **Experiment ID**: `ch12-metrics-reporting-bridge-tech-risk-v1`

It compares a **baseline** metrics pipeline (technical-only reporting) with a **treatment** pipeline (risk-centric, stakeholder-oriented reporting) on synthetic detection logs and incident data.

The code is designed to run **offline** in a constrained sandbox using only the Python standard library.

## Layout

- `run.py` — Orchestrates experiment runs for both baseline and treatment, and writes evidence to `/evidence/results.json` (or `$BOOKMAS_EVIDENCE_DIR/results.json`).
- `requirements.txt` — Empty of runtime dependencies (uses only stdlib) but included for completeness.
- `experiments/`
  - `ch12_metrics_reporting/`
    - `baseline/`
      - `compute_technical_metrics.py` — Computes overall technical metrics from logs.
    - `treatment/`
      - `compute_risk_metrics.py` — Computes risk-centric metrics and per-cohort KPIs.
      - `generate_cohort_reports.py` — Produces cohort metrics tables.
      - `generate_incident_retrospectives.py` — Produces incident summaries and latencies.
      - `generate_executive_summary.py` — Generates a simple textual executive-style summary.
      - `run_full_reporting_pipeline.py` — Orchestrates all treatment components.
    - `config/`
      - `cohorts.yaml` — Defines high-risk segments and cohort groupings.
      - `risk_weights.yaml` — Defines per-cohort risk weights.
      - `stakeholder_questions.yaml` — Questions used to compute coverage scores.
      - `compliance_kpis.yaml` — KPI definitions used to compute alignment scores.
    - `templates/`
      - `compliance_report.md.j2` — Simplified compliance report template.
      - `executive_summary.md.j2` — Simplified executive summary template.
      - `incident_postmortem.md.j2` — Simplified incident postmortem template.
    - `shared_utils.py` — Shared utilities (data generation, metrics, YAML-like parsing).
- `tests/`
  - `test_experiment.py` — Verifies evidence file structure and that both baseline and treatment runs exist per workload.

## Synthetic Data

Because the sandbox is offline and cannot rely on external datasets, this project **generates deterministic synthetic data** that mimic the specified schema:

- Detection logs workload: `offline_model_evaluation_logs`
- Incident logs workload: `incident_case_study_logs`

Generation is deterministic via a fixed random seed. Data volumes are modest to ensure fast and safe execution.

## Evidence Contract

Running `python run.py` will:

1. Generate synthetic logs into the sandbox working directory.
2. Execute at least one **baseline** and one **treatment** run for each workload.
3. Collect numeric metrics produced during execution (e.g., precision, recall, risk-weighted FN rate; synthetic runtime estimates computed inside Python code) and write them to:

```json
{
  "chapter": 12,
  "experiment_id": "ch12-metrics-reporting-bridge-tech-risk-v1",
  "environment": {"os": "...", "python": "..."},
  "runs": [
    {
      "variant": "baseline",  
      "workload_id": "offline_model_evaluation_logs", 
      "run_id": "baseline-offline-1", 
      "success": true,
      "metrics": {"overall_technical_precision": 0.9, ...}
    },
    {
      "variant": "treatment",  
      "workload_id": "offline_model_evaluation_logs", 
      "run_id": "treatment-offline-1", 
      "success": true,
      "metrics": {"risk_weighted_false_negative_rate": 0.04, ...}
    },
    {
      "variant": "baseline",  
      "workload_id": "incident_case_study_logs", 
      "run_id": "baseline-incidents-1", 
      "success": true,
      "metrics": {"incident_detection_latency_median": 1000.0}
    },
    {
      "variant": "treatment",  
      "workload_id": "incident_case_study_logs", 
      "run_id": "treatment-incidents-1", 
      "success": true,
      "metrics": {"incident_detection_latency_median": 800.0}
    }
  ],
  "notes": "..."
}
```

Notes:
- All metric values are **numbers**.
- Additional narrative/explanatory fields, if any, are kept out of the `metrics` dictionaries.

## Running the Experiment

From the project root:

```bash
python run.py
```

By default, evidence is written to `/evidence/results.json`. To override:

```bash
BOOKMAS_EVIDENCE_DIR=/tmp/evidence python run.py
```

## Running Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Tests check:

- `results.json` exists in the evidence directory.
- Schema compliance with the specified evidence contract.
- At least one baseline and one treatment run per workload.
- All metric values are numeric.

## Design Notes

- Only Python **standard library** is used.
- YAML configuration files are parsed using a small, constrained custom parser that accepts a limited subset of YAML (effectively key-value and list structures).
- Synthetic runtime metrics are computed internally from `time.perf_counter()` and are therefore real measurements taken in the sandbox.
- The pipeline is intentionally simplified but mirrors the real-world concepts in the chapter: technical vs. risk-centric metrics, cohort analysis, incident retrospectives, and stakeholder-oriented summaries.