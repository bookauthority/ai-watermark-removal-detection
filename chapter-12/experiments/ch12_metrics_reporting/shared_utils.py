import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_iso8601(ts: str) -> float:
    """Very small iso8601 parser for synthetic data (YYYY-MM-DDTHH:MM:SSZ)."""
    # For synthetic data we always emit seconds precision and 'Z'.
    from datetime import datetime, timezone

    if ts.endswith("Z"):
        ts = ts[:-1]
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def format_iso8601(ts: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def generate_synthetic_detection_logs(path: str, num_items: int, seed: int = 12345) -> None:
    """Generate deterministic synthetic detection log lines (JSONL).

    Schema (subset):
    - content_id, creator_id
    - upload_timestamp
    - geography, content_category, creator_segment
    - model_version, attack_type
    - y_true, y_pred, score
    - in_high_risk_workflow, policy_outcome
    """

    random.seed(seed)
    from datetime import datetime, timedelta, timezone

    geos = ["US", "DE", "FR", "IN", "BR"]
    categories = ["news", "entertainment", "politics", "user_generated"]
    segments = ["high_reach", "long_tail", "new_creator"]
    attacks = ["compression", "diffusion_resynthesis", "crop_blur", "none"]
    policies = ["allowed", "labelled", "blocked", "escalated"]
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

    with open(path, "w", encoding="utf-8") as f:
        for i in range(num_items):
            geo = random.choice(geos)
            cat = random.choice(categories)
            seg = random.choice(segments)
            atk = random.choice(attacks)
            t = base_time + timedelta(minutes=i % 1000)
            y_true = 1 if random.random() < 0.3 else 0
            # Model predictive behavior biased by attack and category
            base_score = 0.7 if y_true == 1 else 0.2
            if atk in ("diffusion_resynthesis", "crop_blur"):
                base_score -= 0.2
            if cat == "news" and geo in ("DE", "FR"):
                base_score += 0.1
            score = min(max(base_score + random.uniform(-0.1, 0.1), 0.0), 1.0)
            y_pred = 1 if score >= 0.5 else 0
            in_high = bool(geo in ("DE", "FR") and cat in ("news", "politics"))
            policy = random.choice(policies)

            rec = {
                "content_id": f"c_{i}",
                "creator_id": f"u_{i % 100}",
                "upload_timestamp": t.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "geography": geo,
                "content_category": cat,
                "creator_segment": seg,
                "model_version": "1.0.0",
                "attack_type": atk,
                "y_true": y_true,
                "y_pred": y_pred,
                "score": float(score),
                "in_high_risk_workflow": in_high,
                "policy_outcome": policy,
            }
            f.write(json.dumps(rec) + "\n")


def generate_synthetic_incident_logs(path: str, num_items: int, seed: int = 12345) -> None:
    """Generate deterministic synthetic incident-focused logs (JSONL).

    This is a small subset of the full schema; we include timestamps to
    derive incident_detection_latency_median.
    """

    random.seed(seed + 1)
    from datetime import datetime, timedelta, timezone

    base_time = datetime(2024, 2, 1, tzinfo=timezone.utc)
    severities = ["sev1", "sev2", "sev3"]

    with open(path, "w", encoding="utf-8") as f:
        for i in range(num_items):
            inc_id = f"inc_{i % 3}"
            sev = severities[i % 3]
            window_start = base_time + timedelta(hours=i % 24)
            window_end = window_start + timedelta(hours=2)
            gt_discovered = window_end + timedelta(hours=2)
            manual = bool(i % 5 == 0)
            y_true = 1 if random.random() < 0.5 else 0
            # Simulate worse performance at beginning of incident
            minutes_since_start = i % 60
            base_score = 0.4 if minutes_since_start < 30 else 0.7
            if y_true == 0:
                base_score = 1.0 - base_score
            score = min(max(base_score + random.uniform(-0.1, 0.1), 0.0), 1.0)
            y_pred = 1 if score >= 0.5 else 0

            rec = {
                "incident_id": inc_id,
                "incident_severity": sev,
                "incident_window_start": window_start.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "incident_window_end": window_end.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "ground_truth_discovered_at": gt_discovered.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "manual_review_flag": manual,
                "event_timestamp": (window_start + timedelta(minutes=minutes_since_start)).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "y_true": y_true,
                "y_pred": y_pred,
            }
            f.write(json.dumps(rec) + "\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def compute_confusion(y_true: Iterable[int], y_pred: Iterable[int]) -> Tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 1 and yp == 0:
            fn += 1
    return tp, fp, tn, fn


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den not in (0, 0.0) else 0.0


# Minimalistic YAML-like loader

def load_simple_yaml(path: str) -> Any:
    """Load a very small YAML subset.

    Supports only:
    - top-level dict of key: value
    - nested dicts via indentation (2 spaces)
    - lists with '- ' prefix
    - scalar values: strings, ints, floats, booleans

    This is sufficient for the small config files included.
    """

    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    def parse_scalar(val: str) -> Any:
        val = val.strip()
        if val.lower() in ("true", "false"):
            return val.lower() == "true"
        try:
            if "." in val:
                return float(val)
            return int(val)
        except ValueError:
            return val

    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Any]] = [(-1, root)]

    for raw in lines:
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if line.startswith("- "):
            item_val = line[2:].strip()
            if not isinstance(parent, list):
                # Convert parent to list if it's an empty dict placeholder
                if isinstance(parent, dict) and not parent:
                    # nothing to do here in simple parser
                    pass
            if isinstance(parent, dict):
                # create implicit list under special key 'items'
                if "items" not in parent:
                    parent["items"] = []
                parent["items"].append(parse_scalar(item_val))
            elif isinstance(parent, list):
                parent.append(parse_scalar(item_val))
            continue

        if ":" in line:
            key, rest = line.split(":", 1)
            key = key.strip()
            rest = rest.strip()
            if rest == "":
                new_obj: Dict[str, Any] = {}
                if isinstance(parent, dict):
                    parent[key] = new_obj
                stack.append((indent, new_obj))
            else:
                val = parse_scalar(rest)
                if isinstance(parent, dict):
                    parent[key] = val
        else:
            # bare scalar line ignored for our configs
            continue

    return root


@dataclass
class RuntimeMetric:
    name: str
    start: float

    def stop(self) -> float:
        elapsed = time.perf_counter() - self.start
        return float(elapsed)
