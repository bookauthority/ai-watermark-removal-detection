import json
import os
import shutil
import tempfile
import unittest


class TestExperimentEvidenceSchema(unittest.TestCase):
    def setUp(self) -> None:
        # Prepare a temporary evidence directory
        self.tmpdir = tempfile.mkdtemp(prefix="evidence_test_")
        os.environ["BOOKMAS_EVIDENCE_DIR"] = self.tmpdir

        # Run the experiment
        import run as experiment_run  # local module

        experiment_run.main()

        self.evidence_path = os.path.join(self.tmpdir, "results.json")
        self.assertTrue(os.path.exists(self.evidence_path), "Evidence file not created")

        with open(self.evidence_path, "r", encoding="utf-8") as f:
            self.evidence = json.load(f)

    def tearDown(self) -> None:
        try:
            shutil.rmtree(self.tmpdir)
        except Exception:
            pass

    def test_top_level_schema(self) -> None:
        ev = self.evidence
        self.assertIn("chapter", ev)
        self.assertIsInstance(ev["chapter"], int)

        self.assertIn("experiment_id", ev)
        self.assertIsInstance(ev["experiment_id"], str)

        self.assertIn("environment", ev)
        self.assertIsInstance(ev["environment"], dict)
        self.assertIn("os", ev["environment"])
        self.assertIn("python", ev["environment"])

        self.assertIn("runs", ev)
        self.assertIsInstance(ev["runs"], list)

    def test_runs_schema_and_metrics(self) -> None:
        runs = self.evidence["runs"]
        self.assertGreaterEqual(len(runs), 2)

        # Track presence of baseline and treatment per workload
        workloads = {}

        for run in runs:
            # Required fields
            self.assertIn("variant", run)
            self.assertIn("workload_id", run)
            self.assertIn("run_id", run)
            self.assertIn("success", run)
            self.assertIn("metrics", run)

            self.assertIn(run["variant"], ["baseline", "treatment"])
            self.assertIsInstance(run["workload_id"], str)
            self.assertIsInstance(run["run_id"], str)
            self.assertIsInstance(run["success"], bool)

            # Metrics must be a dict of numeric values
            metrics = run["metrics"]
            self.assertIsInstance(metrics, dict)
            self.assertGreater(len(metrics), 0)
            for k, v in metrics.items():
                self.assertIsInstance(v, (int, float), msg=f"Metric {k} must be numeric")

            wid = run["workload_id"]
            workloads.setdefault(wid, {"baseline": 0, "treatment": 0})
            workloads[wid][run["variant"]] += 1

        # Ensure at least one baseline and one treatment per workload
        for wid, counts in workloads.items():
            self.assertGreaterEqual(counts["baseline"], 1, f"No baseline for workload {wid}")
            self.assertGreaterEqual(counts["treatment"], 1, f"No treatment for workload {wid}")


if __name__ == "__main__":
    unittest.main()
