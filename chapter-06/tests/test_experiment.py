import json
import os
import tempfile
import unittest
from typing import Any, Dict, List


class TestExperimentEvidence(unittest.TestCase):
    def setUp(self) -> None:
        # Run the experiment before each test in a temporary evidence dir
        self.tmpdir = tempfile.TemporaryDirectory()
        self.evidence_dir = self.tmpdir.name
        os.environ["BOOKMAS_EVIDENCE_DIR"] = self.evidence_dir

        # Import run after setting env var
        import run  # type: ignore

        run.main()

        self.evidence_path = os.path.join(self.evidence_dir, "results.json")
        self.assertTrue(os.path.exists(self.evidence_path), "Evidence file was not created")

        with open(self.evidence_path, "r", encoding="utf-8") as f:
            self.evidence = json.load(f)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_schema_top_level(self) -> None:
        ev = self.evidence
        self.assertIsInstance(ev, dict)
        self.assertIn("chapter", ev)
        self.assertIn("experiment_id", ev)
        self.assertIn("environment", ev)
        self.assertIn("runs", ev)

        self.assertEqual(ev["chapter"], 6)
        self.assertEqual(ev["experiment_id"], "ch6-image-detectors-cnns-vit-hybrids-v1")

        env = ev["environment"]
        self.assertIsInstance(env, dict)
        self.assertIn("os", env)
        self.assertIn("python", env)
        self.assertIsInstance(env["os"], str)
        self.assertIsInstance(env["python"], str)

    def test_runs_structure_and_metrics(self) -> None:
        runs = self.evidence["runs"]
        self.assertIsInstance(runs, list)
        self.assertGreater(len(runs), 0)

        for run in runs:
            self.assertIsInstance(run, dict)
            self.assertIn("variant", run)
            self.assertIn("workload_id", run)
            self.assertIn("run_id", run)
            self.assertIn("success", run)
            self.assertIn("metrics", run)

            self.assertIn(run["variant"], ["baseline", "treatment"])
            self.assertIsInstance(run["workload_id"], str)
            self.assertIsInstance(run["run_id"], str)
            self.assertIsInstance(run["success"], bool)

            metrics = run["metrics"]
            self.assertIsInstance(metrics, dict)
            # At least one numeric metric per run
            self.assertGreater(len(metrics), 0)
            for mk, mv in metrics.items():
                self.assertIsInstance(mk, str)
                # Must be a finite JSON number (int or float)
                self.assertIsInstance(mv, (int, float))

    def test_at_least_one_baseline_and_treatment_per_workload(self) -> None:
        runs = self.evidence["runs"]
        by_workload: Dict[str, Dict[str, int]] = {}
        for run in runs:
            wl = run["workload_id"]
            var = run["variant"]
            by_workload.setdefault(wl, {"baseline": 0, "treatment": 0})
            if var in by_workload[wl]:
                by_workload[wl][var] += 1

        # Expect three workloads with at least one baseline and one treatment each
        expected_workloads = {
            "WL1-CleanSplit-Standard",
            "WL2-Stress-HeldOutAttacks",
            "WL3-CrossValidation-ContentStratified",
        }
        self.assertTrue(expected_workloads.issubset(set(by_workload.keys())))

        for wl in expected_workloads:
            counts = by_workload[wl]
            self.assertGreaterEqual(counts["baseline"], 1, f"No baseline runs for {wl}")
            self.assertGreaterEqual(counts["treatment"], 1, f"No treatment runs for {wl}")


if __name__ == "__main__":
    unittest.main()
