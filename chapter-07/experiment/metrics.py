from typing import List


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))
