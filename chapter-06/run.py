import json
import os
import platform
import sys
from typing import Any, Dict, List

from experiment import run_all_workloads
from schema import build_evidence_object


def main() -> None:
    # Determine evidence directory
    evidence_dir = os.environ.get("BOOKMAS_EVIDENCE_DIR", "/evidence")
    os.makedirs(evidence_dir, exist_ok=True)
    evidence_path = os.path.join(evidence_dir, "results.json")

    # Run synthetic workloads and collect raw run results
    runs: List[Dict[str, Any]] = run_all_workloads()

    # Build evidence object according to the specified contract
    evidence = build_evidence_object(
        chapter_number=6,
        experiment_id="ch6-image-detectors-cnns-vit-hybrids-v1",
        runs=runs,
    )

    # Write evidence JSON to disk
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
