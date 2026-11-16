import json
import os
import sys
import tempfile
import unittest


class TestExperimentEvidence(unittest.TestCase):
    def setUp(self) -> None:
        # Ensure project root on path
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)

    def _run_experiment(self, evidence_dir: str) -> str:
        os.environ["BOOKMAS_EVIDENCE_DIR"] = evidence_dir
        import run as run_module

        # Reload run module in case of repeated tests
        import importlib

        importlib.reload(run_module)
        run_module.main()
        results_path = os.path.join(evidence_dir, "results.json")
        self.assertTrue(os.path.exists(results_path), "results.json not created")
        return results_path

    def test_evidence_schema_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = self._run_experiment(tmpdir)
            with open(results_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        # Top-level keys
        self.assertIn("chapter", data)
        self.assertIn("experiment_id", data)
        self.assertIn("environment", data)
        self.assertIn("runs", data)

        self.assertIsInstance(data["chapter"], int)
        self.assertEqual(data["experiment_id"], "exp-10-multimodal-ensemble-detection-v1")
        self.assertIsInstance(data["environment"], dict)
        self.assertIsInstance(data["runs"], list)

        runs = data["runs"]
        self.assertGreater(len(runs), 0)

        # Collect workloads and variants
        workloads = set()
        workload_variant_pairs = set()

        for run in runs:
            # Required per-run keys
            self.assertIn("variant", run)
            self.assertIn("workload_id", run)
            self.assertIn("run_id", run)
            self.assertIn("success", run)
            self.assertIn("metrics", run)

            self.assertIn(run["variant"], ("baseline", "treatment"))
            self.assertIsInstance(run["success"], bool)
            self.assertIsInstance(run["metrics"], dict)

            workload_id = run["workload_id"]
            workloads.add(workload_id)
            workload_variant_pairs.add((workload_id, run["variant"]))

            # Metrics must be numeric
            for key, value in run["metrics"].items():
                self.assertIsInstance(
                    value,
                    (int, float),
                    msg=f"Metric {key} must be numeric, got {type(value)}",
                )

        # Ensure each known workload has baseline and treatment runs
        expected_workloads = {
            "wl_image_in_distribution",
            "wl_image_shifted",
            "wl_video_multimodal",
        }
        self.assertTrue(expected_workloads.issubset(workloads))
        for wid in expected_workloads:
            self.assertIn((wid, "baseline"), workload_variant_pairs)
            self.assertIn((wid, "treatment"), workload_variant_pairs)

    def test_run_is_idempotent(self) -> None:
        """Running main twice should overwrite results deterministically."""

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["BOOKMAS_EVIDENCE_DIR"] = tmpdir
            import run as run_module

            import importlib

            importlib.reload(run_module)
            run_module.main()
            first_path = os.path.join(tmpdir, "results.json")
            with open(first_path, "r", encoding="utf-8") as f:
                first_data = json.load(f)

            # Run again
            importlib.reload(run_module)
            run_module.main()
            with open(first_path, "r", encoding="utf-8") as f:
                second_data = json.load(f)

        self.assertEqual(first_data, second_data)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
