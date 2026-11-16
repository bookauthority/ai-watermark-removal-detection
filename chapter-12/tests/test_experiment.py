import json
import os
import unittest


class TestExperimentEvidence(unittest.TestCase):
    def setUp(self) -> None:
        # The evidence directory is /evidence by contract, but tests
        # should not assume it exists unless run.py was executed.
        self.evidence_dir = os.environ.get("BOOKMAS_EVIDENCE_DIR", "/evidence")
        self.results_path = os.path.join(self.evidence_dir, "results.json")

    def test_results_schema(self) -> None:
        self.assertTrue(os.path.exists(self.results_path), msg="results.json not found; run run.py first")
        with open(self.results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Top-level keys
        self.assertIn("chapter", data)
        self.assertIn("experiment_id", data)
        self.assertIn("environment", data)
        self.assertIn("runs", data)

        self.assertIsInstance(data["chapter"], int)
        self.assertEqual(data["experiment_id"], "ch12-metrics-reporting-bridge-tech-risk-v1")

        env = data["environment"]
        self.assertIsInstance(env, dict)
        self.assertIn("os", env)
        self.assertIn("python", env)
        self.assertIsInstance(env["os"], str)
        self.assertIsInstance(env["python"], str)

        runs = data["runs"]
        self.assertIsInstance(runs, list)
        self.assertGreaterEqual(len(runs), 4)  # at least 4 runs expected

        # Track presence of baseline/treatment per workload
        seen = {}

        for run in runs:
            self.assertIn("variant", run)
            self.assertIn("workload_id", run)
            self.assertIn("run_id", run)
            self.assertIn("success", run)
            self.assertIn("metrics", run)

            variant = run["variant"]
            workload_id = run["workload_id"]
            self.assertIn(variant, ("baseline", "treatment"))
            self.assertIsInstance(run["run_id"], str)
            self.assertIsInstance(run["success"], bool)

            key = (workload_id, variant)
            seen[key] = True

            metrics = run["metrics"]
            self.assertIsInstance(metrics, dict)
            # All metric values must be numeric
            for mk, mv in metrics.items():
                self.assertIsInstance(
                    mv,
                    (int, float),
                    msg=f"Metric {mk} in run {run['run_id']} is not numeric: {type(mv)}",
                )

        # Ensure at least one baseline and one treatment run per workload
        workloads = set(w for (w, _) in seen.keys())
        for workload in workloads:
            self.assertIn((workload, "baseline"), seen, msg=f"No baseline run for workload {workload}")
            self.assertIn((workload, "treatment"), seen, msg=f"No treatment run for workload {workload}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
