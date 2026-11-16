import json
import os
import shutil
import tempfile
import unittest


class TestExperimentEvidenceSchema(unittest.TestCase):
    def setUp(self) -> None:
        # Create a temporary evidence directory and run the experiment writing there.
        self.tmpdir = tempfile.mkdtemp()
        os.environ["BOOKMAS_EVIDENCE_DIR"] = self.tmpdir

        # Import run after setting env var to ensure it writes to tmpdir.
        import run  # noqa: F401

        # Execute main once for tests.
        run.main()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_results_json_schema(self) -> None:
        path = os.path.join(self.tmpdir, "results.json")
        self.assertTrue(os.path.exists(path), "results.json must exist")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Top-level keys
        self.assertIn("chapter", data)
        self.assertIn("experiment_id", data)
        self.assertIn("environment", data)
        self.assertIn("runs", data)

        self.assertEqual(data["chapter"], 8)
        self.assertEqual(data["experiment_id"], "exp-ch8-attack-defense-redteam-v1")

        env = data["environment"]
        self.assertIsInstance(env, dict)
        self.assertIn("os", env)
        self.assertIn("python", env)
        self.assertIsInstance(env["os"], str)
        self.assertIsInstance(env["python"], str)

        runs = data["runs"]
        self.assertIsInstance(runs, list)
        self.assertGreaterEqual(len(runs), 1)

        # Track variants per workload and their metric keys.
        workload_variants = {}
        workload_metric_keys = {}

        for run in runs:
            self.assertIn("variant", run)
            self.assertIn("workload_id", run)
            self.assertIn("run_id", run)
            self.assertIn("success", run)
            self.assertIn("metrics", run)

            variant = run["variant"]
            workload_id = run["workload_id"]

            self.assertIn(variant, ["baseline", "treatment"])
            self.assertIsInstance(run["run_id"], str)
            self.assertIsInstance(run["success"], bool)

            # Metrics must be a dict of numeric values only.
            metrics = run["metrics"]
            self.assertIsInstance(metrics, dict)
            self.assertGreaterEqual(len(metrics), 1)
            for k, v in metrics.items():
                self.assertIsInstance(k, str)
                # JSON numbers map to int or float in Python.
                self.assertIsInstance(
                    v,
                    (int, float),
                    msg=f"Metric {k} must be numeric, got {type(v)}",
                )

            workload_variants.setdefault(workload_id, set()).add(variant)
            workload_metric_keys.setdefault(workload_id, {})[variant] = set(metrics.keys())

        # Ensure each workload has at least one baseline and one treatment run.
        for workload_id, variants in workload_variants.items():
            self.assertIn(
                "baseline", variants, f"Workload {workload_id} missing baseline run"
            )
            self.assertIn(
                "treatment", variants, f"Workload {workload_id} missing treatment run"
            )

        # Ensure every workload has at least one shared numeric metric name
        # between its baseline and treatment runs.
        for workload_id, by_variant in workload_metric_keys.items():
            if "baseline" in by_variant and "treatment" in by_variant:
                shared = by_variant["baseline"].intersection(by_variant["treatment"])
                self.assertGreater(
                    len(shared),
                    0,
                    msg=(
                        f"Workload {workload_id} has no shared numeric metrics "
                        f"between baseline and treatment"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
