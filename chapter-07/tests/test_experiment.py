import json
import os
import unittest


class TestChapter7VideoTemporalExperiment(unittest.TestCase):
    """Validate that the experiment produces correct evidence JSON.

    These tests assume that `python run.py` has been executed before running
    the test suite. The evidence file is expected at
    `${BOOKMAS_EVIDENCE_DIR:-/evidence}/results.json`.
    """

    def _load_evidence(self):
        base = os.environ.get("BOOKMAS_EVIDENCE_DIR", "/evidence")
        path = os.path.join(base, "results.json")
        self.assertTrue(os.path.exists(path), f"Evidence file not found at {path}. Run `python run.py` first.")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def test_top_level_schema(self):
        evidence = self._load_evidence()

        # Check basic keys
        self.assertIn("chapter", evidence)
        self.assertIn("experiment_id", evidence)
        self.assertIn("environment", evidence)
        self.assertIn("runs", evidence)

        # Chapter number and experiment id must match spec
        self.assertEqual(evidence["chapter"], 7)
        self.assertEqual(evidence["experiment_id"], "exp-ch7-video-temporal-consistency-v1")

        # Environment fields must be strings
        env = evidence["environment"]
        self.assertIsInstance(env, dict)
        self.assertIn("os", env)
        self.assertIn("python", env)
        self.assertIsInstance(env["os"], str)
        self.assertIsInstance(env["python"], str)

        # Runs list must not be empty
        runs = evidence["runs"]
        self.assertIsInstance(runs, list)
        self.assertGreater(len(runs), 0)

    def test_runs_and_metrics_schema(self):
        evidence = self._load_evidence()
        runs = evidence["runs"]

        workload_ids = {"wd_vid_synthetic_short", "wd_vid_synthetic_long", "wd_vid_realistic_eval"}
        seen = {wid: {"baseline": 0, "treatment": 0} for wid in workload_ids}

        for run in runs:
            # Required run fields
            self.assertIn("variant", run)
            self.assertIn("workload_id", run)
            self.assertIn("run_id", run)
            self.assertIn("success", run)
            self.assertIn("metrics", run)

            variant = run["variant"]
            workload_id = run["workload_id"]
            run_id = run["run_id"]
            success = run["success"]
            metrics = run["metrics"]

            self.assertIsInstance(variant, str)
            self.assertIsInstance(workload_id, str)
            self.assertIsInstance(run_id, str)
            self.assertIsInstance(success, bool)
            self.assertIsInstance(metrics, dict)

            # Workload id should be one of the known workloads
            self.assertIn(workload_id, workload_ids)

            # Variant should be either baseline or treatment
            self.assertIn(variant, ("baseline", "treatment"))
            seen[workload_id][variant] += 1

            # Metrics must contain at least one numeric value
            self.assertGreater(len(metrics), 0)
            for key, value in metrics.items():
                # Metric values must be real numbers (int or float), not bool or other types
                self.assertIsInstance(key, str)
                # Ensure not bool (which is a subclass of int)
                self.assertNotIsInstance(value, bool, msg=f"Metric {key} must not be bool")
                self.assertIsInstance(value, (int, float), msg=f"Metric {key} must be numeric, got {type(value)}")

        # Ensure at least one baseline and one treatment run per workload
        for wid in workload_ids:
            self.assertGreaterEqual(seen[wid]["baseline"], 1, f"No baseline run for workload {wid}")
            self.assertGreaterEqual(seen[wid]["treatment"], 1, f"No treatment run for workload {wid}")


if __name__ == "__main__":
    unittest.main()
