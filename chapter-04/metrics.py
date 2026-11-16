from typing import List, Dict, Any
from util import safe_ratio


def compute_basic_stats(manifest: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute basic numeric metrics from a manifest-like list.

    The manifest is a list of dicts where keys include:
    - base_id
    -, variant_type in {"clean", "watermarked", "attacked"}
    - labels["watermark_presence"]
    - source_type

    Returns a dict with numeric metrics only.
    """
    num_items_total = len(manifest)
    base_ids = set()
    counts_by_variant: Dict[str, int] = {"clean": 0, "watermarked": 0, "attacked": 0}
    counts_presence: Dict[str, int] = {}
    counts_source: Dict[str, int] = {}

    for rec in manifest:
        base_ids.add(rec.get("base_id"))
        v_type = rec.get("variant_type")
        if v_type in counts_by_variant:
            counts_by_variant[v_type] += 1
        labels = rec.get("labels", {})
        wp = labels.get("watermark_presence", "unknown")
        counts_presence[wp] = counts_presence.get(wp, 0) + 1
        st = rec.get("source_type", "unknown")
        counts_source[st] = counts_source.get(st, 0) + 1

    num_base_items = len(base_ids)

    # Ratios: all fractions in [0, 1]
    variant_ratio = {
        k: safe_ratio(v, num_items_total)
        for k, v in counts_by_variant.items()
    }

    presence_ratio = {
        k: safe_ratio(v, num_items_total)
        for k, v in counts_presence.items()
    }

    source_ratio = {
        k: safe_ratio(v, num_items_total)
        for k, v in counts_source.items()
    }

    # Flatten variant ratios into numeric metrics with explicit names.
    metrics: Dict[str, float] = {
        "num_items_total": float(num_items_total),
        "num_base_items": float(num_base_items),
    }

    for k, v in variant_ratio.items():
        metrics[f"variant_ratio_{k}"] = float(v)
    for k, v in presence_ratio.items():
        metrics[f"class_balance_{k}"] = float(v)
    for k, v in source_ratio.items():
        metrics[f"source_type_ratio_{k}"] = float(v)

    return metrics
