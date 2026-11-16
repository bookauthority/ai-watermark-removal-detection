from typing import List

from .data import Example, Image


def _downsample_image(img: Image, new_w: int, new_h: int) -> Image:
    """Simple nearest-neighbor downsampling for small images."""

    h = len(img)
    w = len(img[0]) if h > 0 else 0
    if w == 0 or h == 0:
        return [[0 for _ in range(new_w)] for _ in range(new_h)]

    out: Image = []
    for y in range(new_h):
        row: List[int] = []
        src_y = int(y * h / new_h)
        for x in range(new_w):
            src_x = int(x * w / new_w)
            row.append(img[src_y][src_x])
        out.append(row)
    return out


def pixel_baseline_features(examples: List[Example], new_w: int = 16, new_h: int = 16) -> List[List[float]]:
    """Flattened, normalized grayscale pixel features for baseline.

    Each image is downsampled to new_w x new_h and normalized to [0, 1].
    """

    feats: List[List[float]] = []
    for ex in examples:
        ds = _downsample_image(ex.image, new_w, new_h)
        flat: List[float] = []
        for row in ds:
            for v in row:
                flat.append(v / 255.0)
        feats.append(flat)
    return feats


def _gradient_magnitude_features(img: Image) -> List[float]:
    h = len(img)
    w = len(img[0]) if h > 0 else 0
    if w < 2 or h < 2:
        return [0.0, 0.0, 0.0]

    mags: List[float] = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            gx = img[y][x + 1] - img[y][x - 1]
            gy = img[y + 1][x] - img[y - 1][x]
            m = (gx * gx + gy * gy) ** 0.5
            mags.append(m)

    if not mags:
        return [0.0, 0.0, 0.0]

    mags_sorted = sorted(mags)
    n = len(mags_sorted)
    mean_m = sum(mags_sorted) / n
    p90 = mags_sorted[int(0.9 * (n - 1))]
    max_m = mags_sorted[-1]
    return [mean_m / 255.0, p90 / 255.0, max_m / 255.0]


def _noise_residual_features(img: Image) -> List[float]:
    h = len(img)
    w = len(img[0]) if h > 0 else 0
    if w < 2 or h < 2:
        return [0.0, 0.0]

    residuals: List[float] = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            neighbors = [
                img[y - 1][x],
                img[y + 1][x],
                img[y][x - 1],
                img[y][x + 1],
            ]
            avg = sum(neighbors) / 4.0
            residuals.append(img[y][x] - avg)

    if not residuals:
        return [0.0, 0.0]

    mean_r = sum(residuals) / len(residuals)
    var_r = sum((r - mean_r) ** 2 for r in residuals) / len(residuals)
    return [mean_r / 255.0, (var_r ** 0.5) / 255.0]


def _block_mean_diff_features(img: Image, block: int = 4) -> List[float]:
    h = len(img)
    w = len(img[0]) if h > 0 else 0
    if w == 0 or h == 0:
        return [0.0, 0.0]

    means: List[float] = []
    for by in range(0, h, block):
        for bx in range(0, w, block):
            vals: List[int] = []
            for y in range(by, min(by + block, h)):
                for x in range(bx, min(bx + block, w)):
                    vals.append(img[y][x])
            if vals:
                means.append(sum(vals) / len(vals))
    if len(means) < 2:
        return [0.0, 0.0]

    diffs: List[float] = []
    for i in range(1, len(means)):
        diffs.append(abs(means[i] - means[i - 1]))

    mean_d = sum(diffs) / len(diffs)
    max_d = max(diffs)
    return [mean_d / 255.0, max_d / 255.0]


def _local_variance_feature(img: Image, window: int = 3) -> List[float]:
    h = len(img)
    w = len(img[0]) if h > 0 else 0
    if w < window or h < window:
        return [0.0]

    half = window // 2
    vars_: List[float] = []
    for y in range(half, h - half):
        for x in range(half, w - half):
            vals: List[int] = []
            for dy in range(-half, half + 1):
                for dx in range(-half, half + 1):
                    vals.append(img[y + dy][x + dx])
            if vals:
                m = sum(vals) / len(vals)
                v = sum((v - m) ** 2 for v in vals) / len(vals)
                vars_.append(v)

    if not vars_:
        return [0.0]

    mean_v = sum(vars_) / len(vars_)
    return [mean_v / (255.0 ** 2)]


def forensic_features(examples: List[Example]) -> List[List[float]]:
    """Handcrafted feature vector for each image.

    Combines:
    - gradient magnitude statistics (edge-like)
    - noise residual statistics (noise-like)
    - block mean differences (compression/frequency-like)
    - local variance (texture-like)
    """

    all_feats: List[List[float]] = []
    for ex in examples:
        img = ex.image
        grad = _gradient_magnitude_features(img)
        noise = _noise_residual_features(img)
        block = _block_mean_diff_features(img)
        tex = _local_variance_feature(img)
        vec = grad + noise + block + tex
        all_feats.append(vec)
    return all_feats
