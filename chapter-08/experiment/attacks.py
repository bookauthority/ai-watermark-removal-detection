from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

from .core import RandomState


Vector = List[float]


def clip_vec(v: Vector, lo: float = 0.0, hi: float = 1.0) -> Vector:
    return [min(hi, max(lo, x)) for x in v]


# Heuristic attacks ---------------------------------------------------------


def attack_crop_like(v: Vector, rng: RandomState) -> Vector:
    # Simulate crop-resize by zeroing edge dimensions slightly.
    out = v.copy()
    if len(out) >= 2:
        out[0] *= 0.8
        out[-1] *= 0.8
    return clip_vec(out)


def attack_jpeg_like(v: Vector, rng: RandomState) -> Vector:
    # Simulate JPEG by adding small noise and quantization.
    out = []
    for x in v:
        noise = (rng.rand() - 0.5) * 0.08
        y = round((x + noise) * 10.0) / 10.0
        out.append(y)
    return clip_vec(out)


def attack_blur_like(v: Vector, rng: RandomState) -> Vector:
    # Simulate blur via local averaging.
    out = v.copy()
    n = len(v)
    for i in range(1, n - 1):
        out[i] = (v[i - 1] + v[i] + v[i + 1]) / 3.0
    return clip_vec(out)


def attack_color_jitter_like(v: Vector, rng: RandomState) -> Vector:
    # Simulate color jitter by small per-dimension scale and shift.
    out = []
    for x in v:
        scale = 1.0 + (rng.rand() - 0.5) * 0.1
        shift = (rng.rand() - 0.5) * 0.05
        out.append(x * scale + shift)
    return clip_vec(out)


@dataclass
class HeuristicAttackSuite:
    attacks: List[Tuple[str, Callable[[Vector, RandomState], Vector]]] = None

    def __post_init__(self) -> None:
        if self.attacks is None:
            self.attacks = [
                ("crop-resize-16-9", attack_crop_like),
                ("jpeg-50", attack_jpeg_like),
                ("gaussian-blur-3", attack_blur_like),
                ("color-jitter-light", attack_color_jitter_like),
            ]


# Optimization-based attacks -----------------------------------------------


def attack_pgd_like(
    v: Vector,
    rng: RandomState,
    loss_fn: Callable[[Vector], float],
    steps: int = 10,
    step_size: float = 0.05,
) -> Vector:
    # Approximate gradient by coordinate-wise finite differences.
    x = v.copy()
    for _ in range(steps):
        base_loss = loss_fn(x)
        grad = []
        eps = 1e-3
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += eps
            lp = loss_fn(xp)
            g = (lp - base_loss) / eps
            grad.append(g)
        # Move in -sign(grad) direction to reduce loss.
        for i in range(len(x)):
            direction = -1.0 if grad[i] > 0 else 1.0
            x[i] += step_size * direction
        x = clip_vec(x)
    return x


def attack_square_like(
    v: Vector,
    rng: RandomState,
    loss_fn: Callable[[Vector], float],
    max_queries: int = 30,
) -> Vector:
    x = v.copy()
    current_loss = loss_fn(x)
    n = len(x)
    for _ in range(max_queries):
        i = rng.randint(0, n - 1)
        delta = (rng.rand() - 0.5) * 0.4
        xp = x.copy()
        xp[i] = clip_vec([xp[i] + delta])[0]
        lp = loss_fn(xp)
        if lp < current_loss:
            x = xp
            current_loss = lp
    return clip_vec(x)


@dataclass
class OptimizationAttackSuite:
    attacks: List[Tuple[str, str]] = None  # (id, type)

    def __post_init__(self) -> None:
        if self.attacks is None:
            self.attacks = [
                ("pgd-l2-perceptual", "pgd"),
                ("gradient-free-square-attack", "square"),
            ]


# Generative attacks -------------------------------------------------------


def attack_sr_resynthesis(v: Vector, rng: RandomState) -> Vector:
    # Simulate downsample + SR by smoothing then adding small noise.
    blurred = attack_blur_like(v, rng)
    out = []
    for x in blurred:
        noise = (rng.rand() - 0.5) * 0.05
        out.append(x + noise)
    return clip_vec(out)


def attack_latent_diffusion(v: Vector, rng: RandomState) -> Vector:
    # Simulate diffusion edit by mixing with a high-intensity template.
    template = [0.75] * len(v)
    alpha = 0.6
    out = []
    for xi, ti in zip(v, template):
        noise = (rng.rand() - 0.5) * 0.1
        out.append(alpha * ti + (1 - alpha) * xi + noise)
    return clip_vec(out)


@dataclass
class GenerativeAttackSuite:
    attacks: List[Tuple[str, callable]] = None

    def __post_init__(self) -> None:
        if self.attacks is None:
            self.attacks = [
                ("sr-resynthesis", attack_sr_resynthesis),
                ("latent-diffusion-edit", attack_latent_diffusion),
            ]
