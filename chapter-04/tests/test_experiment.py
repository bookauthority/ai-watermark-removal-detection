import json
import os
import unittest

from util import evidence_dir, is_finite_number
import run as experiment_run


class TestExperimentEvidence(unittest.TestCase):
    def setUp(self) -> None:
        # Ensure experiment has been run; this will create results.json
        experiment_run.main()
        self.evidence_path = os.path.join(evidence_dir(), "results.json")
        self.assertTrue(os.path.exists(self.evidence_path), "results.json must exist after running main().")
        with open(self.evidence_path, "r", encoding="utf-8") as f:
            self.evidence = json.load(f)

    def test_top_level_schema(self):
        ev = self.evidence
        self.assertIsInstance(ev, dict)
        self.assertIn("chapter", ev)
        self.assertIn("experiment_id", ev)
        self.assertIn("environment", ev)
        self.assertIn("runs", ev)

        self.assertEqual(ev["chapter"], 4)
        self.assertEqual(ev["experiment_id"], "ch04-watermark-dataset-pipeline-v1")
        self.assertIsInstance(ev["environment"], dict)
        self.assertIn("os", ev["environment"])
        self.assertIn("python", ev["environment"])
        self.assertIsInstance(ev["runs"], list)
        self.assertGreater(len(ev["runs"]), 0)

    def test_runs_schema_and_metrics(self):
        runs = self.evidence["runs"]
        workloads = set()
        variants_by_workload = {}

        for r in runs:
            self.assertIsInstance(r, dict)
            self.assertIn("variant", r)
            self.assertIn("workload_id", r)
            self.assertIn("run_id", r)
            self.assertIn("success", r)
            self.assertIn("metrics", r)

            variant = r["variant"]
            workload_id = r["workload_id"]
            workloads.add(workload_id)

            self.assertIn(variant, ("baseline", "treatment"))
            self.assertIsInstance(r["run_id"], str)
            self.assertIsInstance(r["success"], bool)

            variants_by_workload.setdefault(workload_id, set()).add(variant)

            metrics = r["metrics"]
            self.assertIsInstance(metrics, dict)
            self.assertGreater(len(metrics), 0)

            # All metric values must be finite numbers
            for key, value in metrics.items():
                self.assertTrue(
                    is_finite_number(value),
                    msg=f"Metric {key} must be finite numeric, got {value!r}",
                )

            # Ensure a few expected core metrics appear
            self.assertIn("num_items_total", metrics)
            self.assertIn("num_base_items", metrics)
            self.assertIn("pipeline_runtime", metrics)

        # We expect four workloads from the spec
        expected_workloads = {
            "workload_synthetic_images_only",
            "workload_mixed_real_and_synthetic",
            "workload_video_segments",
            "workload_governance_and_bias_audit",
        }
        self.assertTrue(expected_workloads.issubset(workloads))

        # For each workload, at least one baseline and one treatment run
        for w in expected_workloads:
            self.assertIn(w, variants_by_workload, msg=f"Missing runs for workload {w}")
            self.assertIn("baseline", variants_by_workload[w], msg=f"Missing baseline run for {w}")
            self.assertIn("treatment", variants_by_workload[w], msg=f"Missing treatment run for {w}")


if __name__ == "__main__":
    unittest.main()
