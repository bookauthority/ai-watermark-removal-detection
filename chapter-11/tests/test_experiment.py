import json
import os
import tempfile
import unittest
from pathlib import Path


class ExperimentEvidenceSchemaTest(unittest.TestCase):
    """Tests that `run.py` produces evidence matching the required schema.

    These tests execute `run.py` as an imported module to ensure we can
    inspect the resulting evidence without spawning subprocesses.
    """

    def setUp(self) -> None:
        # Create a temporary directory to act as BOOKMAS_EVIDENCE_DIR.
        self.tmpdir = tempfile.TemporaryDirectory()
        self.evidence_dir = Path(self.tmpdir.name)
        os.environ["BOOKMAS_EVIDENCE_DIR"] = str(self.evidence_dir)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()
        os.environ.pop("BOOKMAS_EVIDENCE_DIR", None)

    def _run_experiment(self) -> Path:
        # Import the run module and invoke main(). This will write the
        # results.json file into BOOKMAS_EVIDENCE_DIR.
        import run as experiment_run  # type: ignore

        exit_code = experiment_run.main()
        self.assertEqual(exit_code, 0, "run.main() should return 0 on success")

        results_path = self.evidence_dir / "results.json"
        self.assertTrue(results_path.exists(), "results.json must be created")
        return results_path

    def test_results_schema_and_metrics(self) -> None:
        results_path = self._run_experiment()

        with results_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Top-level keys
        self.assertIn("chapter", data)
        self.assertIn("experiment_id", data)
        self.assertIn("environment", data)
        self.assertIn("runs", data)

        self.assertEqual(data["chapter"], 11)
        self.assertEqual(
            data["experiment_id"],
            "exp-ch11-arch-deploy-watermark-detection-v1",
        )

        # Environment
        env = data["environment"]
        self.assertIsInstance(env, dict)
        self.assertIn("os", env)
        self.assertIn("python", env)
        self.assertIsInstance(env["os"], str)
        self.assertIsInstance(env["python"], str)

        # Runs
        runs = data["runs"]
        self.assertIsInstance(runs, list)
        self.assertGreaterEqual(len(runs), 8, "At least 8 runs (4 workloads x 2 variants)")

        # Track coverage of workloads and variants
        seen = {("baseline", "wl1-real-time-low-rps"): False,
                ("baseline", "wl2-real-time-bursty-high-rps"): False,
                ("baseline", "wl3-batch-backfill"): False,
                ("baseline", "wl4-failure-injection"): False,
                ("treatment", "wl1-real-time-low-rps"): False,
                ("treatment", "wl2-real-time-bursty-high-rps"): False,
                ("treatment", "wl3-batch-backfill"): False,
                ("treatment", "wl4-failure-injection"): False}

        for run in runs:
            self.assertIn("variant", run)
            self.assertIn("workload_id", run)
            self.assertIn("run_id", run)
            self.assertIn("success", run)
            self.assertIn("metrics", run)

            variant = run["variant"]
            workload_id = run["workload_id"]
            self.assertIn(variant, {"baseline", "treatment"})
            self.assertIn(
                workload_id,
                {
                    "wl1-real-time-low-rps",
                    "wl2-real-time-bursty-high-rps",
                    "wl3-batch-backfill",
                    "wl4-failure-injection",
                },
            )

            key = (variant, workload_id)
            if key in seen:
                seen[key] = True

            run_id = run["run_id"]
            self.assertIsInstance(run_id, str)

            success = run["success"]
            self.assertIsInstance(success, bool)

            metrics = run["metrics"]
            self.assertIsInstance(metrics, dict)

            # Required metrics must be present and numeric
            required_metric_keys = {
                "mean_latency_ms",
                "p95_latency_ms",
                "p99_latency_ms",
                "throughput_rps",
                "error_rate",
            }
            for key_metric in required_metric_keys:
                self.assertIn(key_metric, metrics)
                value = metrics[key_metric]
                self.assertIsInstance(
                    value,
                    (int, float),
                    msg=f"Metric {key_metric} must be numeric",
                )

        # Ensure coverage: at least one baseline and one treatment run per workload
        for combo, seen_flag in seen.items():
            self.assertTrue(
                seen_flag,
                msg=f"Missing run for variant={combo[0]}, workload={combo[1]}",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
