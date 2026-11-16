import json
import os
import unittest
from pathlib import Path


class TestExperimentEvidence(unittest.TestCase):
    """Validate that running the experiment produces correctly-shaped evidence.

    These tests are intentionally strict about types and required fields,
    but they only rely on the Python standard library and the presence of
    results.json written by run.py.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Ensure the experiment has been run. We call run.py via import and
        # main() to avoid subprocesses and keep everything in-process.
        evidence_dir = os.environ.get("BOOKMAS_EVIDENCE_DIR", "/evidence")
        cls.evidence_path = Path(evidence_dir) / "results.json"

        # Remove any pre-existing evidence to avoid stale data.
        if cls.evidence_path.exists():
            cls.evidence_path.unlink()

        # Import and run the experiment main().
        import run as experiment_run

        experiment_run.main()

    def test_results_file_exists_and_is_json(self) -> None:
        self.assertTrue(self.evidence_path.exists(), "results.json must exist")
        with self.evidence_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIsInstance(data, dict)

    def test_top_level_schema(self) -> None:
        with self.evidence_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Required top-level keys
        self.assertIn("chapter", data)
        self.assertIn("experiment_id", data)
        self.assertIn("environment", data)
        self.assertIn("runs", data)

        self.assertEqual(data["chapter"], 9)
        self.assertEqual(
            data["experiment_id"],
            "exp-ch9-robustness-generalization-distshift-v1",
        )

        env = data["environment"]
        self.assertIsInstance(env, dict)
        self.assertIn("os", env)
        self.assertIn("python", env)
        self.assertIsInstance(env["os"], str)
        self.assertIsInstance(env["python"], str)

    def test_runs_schema_and_metrics(self) -> None:
        with self.evidence_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        runs = data["runs"]
        self.assertIsInstance(runs, list)
        self.assertGreaterEqual(len(runs), 2)

        # Track workloads seen per variant.
        workloads_by_variant = {}

        for run in runs:
            self.assertIsInstance(run, dict)
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

            self.assertIn(variant, ("baseline", "treatment"))
            self.assertIsInstance(workload_id, str)
            self.assertIsInstance(run_id, str)
            self.assertIsInstance(success, bool)
            self.assertIsInstance(metrics, dict)

            # Accumulate workload coverage.
            workloads_by_variant.setdefault(variant, set()).add(workload_id)

            # All metric values must be plain numbers (int or float), not
            # nested objects, lists, or None.
            for key, value in metrics.items():
                with self.subTest(metric=key):
                    self.assertIsInstance(
                        value,
                        (int, float),
                        msg=f"Metric {key} must be numeric, got {type(value)}",
                    )

        # We require at least one baseline and one treatment run per workload.
        # Extract all workload_ids that appear in any run.
        all_workloads = set()
        for variant, wset in workloads_by_variant.items():
            all_workloads.update(wset)

        for workload_id in all_workloads:
            with self.subTest(workload=workload_id):
                self.assertIn("baseline", workloads_by_variant)
                self.assertIn("treatment", workloads_by_variant)
                self.assertIn(workload_id, workloads_by_variant["baseline"])
                self.assertIn(workload_id, workloads_by_variant["treatment"])

    def test_at_least_one_shared_metric_name(self) -> None:
        """Ensure that baseline and treatment share at least one metric name
        per workload, to enable fair comparison.
        """
        with self.evidence_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        runs = data["runs"]

        # Organize by workload then variant.
        by_workload = {}
        for run in runs:
            w = run["workload_id"]
            v = run["variant"]
            by_workload.setdefault(w, {}).setdefault(v, []).append(run)

        for workload_id, variants in by_workload.items():
            with self.subTest(workload=workload_id):
                self.assertIn("baseline", variants)
                self.assertIn("treatment", variants)

                # Take the first run for each variant.
                base_metrics = variants["baseline"][0]["metrics"]
                treat_metrics = variants["treatment"][0]["metrics"]

                shared_keys = set(base_metrics.keys()) & set(
                    treat_metrics.keys()
                )
                self.assertTrue(
                    len(shared_keys) >= 1,
                    msg=(
                        f"Expected at least one shared metric between baseline "
                        f"and treatment for workload {workload_id}"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
