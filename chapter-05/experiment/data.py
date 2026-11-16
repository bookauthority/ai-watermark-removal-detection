import random
from dataclasses import dataclass
from typing import List, Tuple


Label = int  # 0 = negative (intact / never watermarked), 1 = positive (removed / tampered)
Image = List[List[int]]  # simple grayscale image, values 0-255


@dataclass
class Example:
    image: Image
    label: Label
    domain: str  # e.g. 'main', 'cross_clean', 'cross_heavy'


def _generate_base_image(width: int, height: int) -> Image:
    """Generate a simple synthetic base image with smooth content.

    This is a 2D list of ints in [0, 255]. We create simple gradients
    with added low-amplitude noise to resemble natural-ish images.
    """

    img: Image = []
    for y in range(height):
        row: List[int] = []
        for x in range(width):
            base = int(255 * (x / max(1, width - 1)) * 0.7 + 255 * (y / max(1, height - 1)) * 0.3)
            noise = random.randint(-10, 10)
            val = max(0, min(255, base + noise))
            row.append(val)
        img.append(row)
    return img


def _apply_positive_artifact(img: Image, strength: int) -> Image:
    """Apply synthetic 'watermark-removed' artifact.

    We mimic removal artifacts by:
    - Adding blocky noise to a central region.
    - Over-smoothing surrounding area.
    """

    height = len(img)
    width = len(img[0]) if height > 0 else 0
    out: Image = [row[:] for row in img]

    cx0 = width // 4
    cx1 = 3 * width // 4
    cy0 = height // 4
    cy1 = 3 * height // 4

    # Central block: add random noise and block artifacts
    block_size = max(2, strength // 2)
    for y in range(cy0, cy1):
        for x in range(cx0, cx1):
            by = (y - cy0) // block_size
            bx = (x - cx0) // block_size
            base_noise = ((bx + by + strength) * 13) % 40 - 20
            orig = out[y][x]
            val = max(0, min(255, orig + base_noise))
            out[y][x] = val

    # Surrounding area: slight smoothing
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if cy0 <= y < cy1 and cx0 <= x < cx1:
                continue
            neighbors = [
                out[y][x],
                out[y - 1][x],
                out[y + 1][x],
                out[y][x - 1],
                out[y][x + 1],
            ]
            avg = sum(neighbors) // len(neighbors)
            out[y][x] = (out[y][x] + avg) // 2
    return out


def _apply_negative_artifact(img: Image, strength: int) -> Image:
    """Apply synthetic 'non-removed' (intact/never-watermarked) artifact.

    We introduce milder global noise or blur so the classifier cannot
    trivially separate classes by low-level statistics alone.
    """

    height = len(img)
    width = len(img[0]) if height > 0 else 0
    out: Image = [row[:] for row in img]

    # Mild Gaussian-like blur approximation
    for _ in range(max(1, strength // 3)):
        new_out: Image = [row[:] for row in out]
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                neighbors = [
                    out[y][x],
                    out[y - 1][x],
                    out[y + 1][x],
                    out[y][x - 1],
                    out[y][x + 1],
                ]
                avg = sum(neighbors) // len(neighbors)
                new_out[y][x] = avg
        out = new_out

    return out


def generate_synthetic_dataset(
    n_main: int = 200,
    n_cross_clean: int = 80,
    n_cross_heavy: int = 80,
    width: int = 32,
    height: int = 32,
    seed: int = 42,
) -> Tuple[List[Example], List[Example], List[Example]]:
    """Generate synthetic datasets for main and cross-distribution workloads.

    Returns (main_examples, cross_clean_examples, cross_heavy_examples).
    """

    random.seed(seed)

    def make_examples(n: int, domain: str, pos_strength: int, neg_strength: int) -> List[Example]:
        exs: List[Example] = []
        for i in range(n):
            base = _generate_base_image(width, height)
            label = 1 if i % 2 == 0 else 0  # balanced
            if label == 1:
                img = _apply_positive_artifact(base, pos_strength)
            else:
                img = _apply_negative_artifact(base, neg_strength)
            exs.append(Example(image=img, label=label, domain=domain))
        random.shuffle(exs)
        return exs

    main_examples = make_examples(n_main, "main", pos_strength=8, neg_strength=3)
    cross_clean_examples = make_examples(n_cross_clean, "cross_clean", pos_strength=5, neg_strength=2)
    cross_heavy_examples = make_examples(n_cross_heavy, "cross_heavy", pos_strength=12, neg_strength=6)

    return main_examples, cross_clean_examples, cross_heavy_examples


def train_val_test_split(examples: List[Example], seed: int = 123) -> Tuple[List[Example], List[Example], List[Example]]:
    """Deterministic train/val/test split with 70/10/20 proportion.

    Stratification by label is approximated by shuffling with a fixed seed
    and slicing.
    """

    rng = random.Random(seed)
    idxs = list(range(len(examples)))
    rng.shuffle(idxs)

    n = len(idxs)
    n_train = int(0.7 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val

    train_idx = idxs[:n_train]
    val_idx = idxs[n_train : n_train + n_val]
    test_idx = idxs[n_train + n_val :]

    train = [examples[i] for i in train_idx]
    val = [examples[i] for i in val_idx]
    test = [examples[i] for i in test_idx]
    return train, val, test
