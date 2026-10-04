"""Repeat a photographed motif at a new count without changing its radii.

Angles use the same origin as metrics.to_polar: sample zero is at the right,
with positive angles towards the bottom of the image. Both the optimiser
and the preview use this mapping; neither substitutes a stock gem drawing.
"""

from __future__ import annotations

import numpy as np

__all__ = ["detect_fold", "remap_polar_fold", "remap_image_fold", "align_view_to_n"]

FOLD_CANDIDATES = tuple(range(3, 25))
# A real repeating stone still matches itself after one sector turn. A single
# motif can have a loud frequency and still fall apart under that turn.
REPEAT_AGREEMENT = 0.75


def _fold(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value or not 3 <= value <= 24:
        raise ValueError("fold count must be an integer between 3 and 24")
    return int(value)


def _fft_fold(polar: np.ndarray, candidates: tuple[int, ...] = FOLD_CANDIDATES) -> int:
    """Loudest repeating count, even when the picture does not survive a turn.

    Opposite bright/dark rings can cancel in an angular mean, even in a
    perfectly periodic photo. Summing their powers preserves that evidence.
    Featureless images keep the conventional count of eight. That count sizes
    a symmetric search; it is not permission to copy one sector around the circle.
    """
    arr = np.asarray(polar, dtype=np.float64)
    if arr.ndim != 2 or min(arr.shape) < 2 or not np.isfinite(arr).all():
        raise ValueError("expected a finite 2D polar image")
    rows = arr[max(0, arr.shape[0] // 8):max(1, 7 * arr.shape[0] // 8)]
    rows = rows - rows.mean(axis=1, keepdims=True)
    power = (np.abs(np.fft.rfft(rows, axis=1)) ** 2).sum(axis=0)
    choices = [int(n) for n in candidates if 3 <= n < len(power)]
    if not choices:
        raise ValueError("no fold candidates fit the angular resolution")
    peak = max(choices, key=lambda n: power[n])
    if power[peak] <= 1e-12:
        return 8 if 8 in choices else choices[0]
    # A strong fundamental plus a sharper second harmonic is still one motif.
    divisors = [n for n in choices if peak % n == 0 and power[n] >= 0.20 * power[peak]]
    return min(divisors)


def _rotated_agreement(polar: np.ndarray, n: int) -> float:
    """How well the picture matches itself after one sector turn."""
    arr = np.asarray(polar, dtype=np.float64)
    rows = arr[max(0, arr.shape[0] // 8):max(1, 7 * arr.shape[0] // 8)]
    if rows.size == 0 or rows.shape[1] < 2 or n < 1:
        return 0.0
    cols = rows.shape[1]
    pos = (np.arange(cols) + cols / float(n)) % cols
    left = np.floor(pos).astype(int) % cols
    weight = pos - np.floor(pos)
    rolled = rows[:, left] * (1.0 - weight) + rows[:, (left + 1) % cols] * weight
    scores = []
    for row, turned in zip(rows, rolled):
        a = row - row.mean()
        b = turned - turned.mean()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-8:
            continue
        scores.append(float(np.dot(a, b) / denom))
    if not scores:
        return 0.0
    # Median, so a thin symmetric ring cannot outvote the rest of the drawing.
    return float(np.median(scores))


def detect_fold(polar: np.ndarray, candidates: tuple[int, ...] = FOLD_CANDIDATES) -> int | None:
    """The count of a picture that really repeats, or nothing for one motif.

    A single drawing can still have a loud frequency. It is a repeating stone
    only when turning it by one sector leaves the same picture. Otherwise
    there is no fold to copy around the circle.
    """
    n = _fft_fold(polar, candidates)
    if _rotated_agreement(polar, n) < REPEAT_AGREEMENT:
        return None
    return n


def _phase(polar: np.ndarray, n: int) -> float:
    """A motif axis, in turns; retain its orientation when changing count."""
    spec = np.fft.rfft(polar, axis=1)
    if n >= spec.shape[1]:
        return 0.0
    coeff = spec[:, n]
    row = int(np.argmax(np.abs(coeff)))
    if abs(coeff[row]) < 1e-12:
        return 0.0
    return float((-np.angle(coeff[row]) / (2 * np.pi * n)) % (1.0 / n))


def _sector_coordinate(theta: np.ndarray, src_n: int, dst_n: int, phase: float):
    return phase + np.mod((theta - phase) * dst_n, 1.0) / src_n


def remap_polar_fold(polar: np.ndarray, src_n: int, dst_n: int,
                     *, phase: float | None = None) -> np.ndarray:
    """Average corresponding sectors, compress their angles and repeat them.

    No radial resampling: shaft length, central opening and ring positions
    stay fixed. Equal counts are an exact identity, including photograph noise.
    """
    arr = np.asarray(polar, dtype=np.float64)
    src_n, dst_n = _fold(src_n), _fold(dst_n)
    if arr.ndim != 2 or min(arr.shape) < 2 or not np.isfinite(arr).all():
        raise ValueError("expected a finite 2D polar image")
    if src_n == dst_n:
        return arr.copy()
    if phase is None:
        phase = _phase(arr, src_n)
    n_theta = arr.shape[1]
    theta = np.arange(n_theta) / n_theta
    source = _sector_coordinate(theta, src_n, dst_n, phase)
    acc = np.zeros_like(arr)
    for k in range(src_n):
        col = np.mod(source + k / src_n, 1.0) * n_theta
        left = np.floor(col).astype(int)
        weight = col - left
        acc += (arr[:, left % n_theta] * (1 - weight)
                + arr[:, (left + 1) % n_theta] * weight)
    return acc / src_n


def _sample_rgb(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    return (img[y0, x0] * (1 - fx) * (1 - fy)
            + img[y0, x1] * fx * (1 - fy)
            + img[y1, x0] * (1 - fx) * fy
            + img[y1, x1] * fx * fy)


def remap_image_fold(image: np.ndarray, src_n: int, dst_n: int,
                     *, phase: float | None = None) -> np.ndarray:
    """RGB counterpart of remap_polar_fold, preserving the outside canvas."""
    from .metrics import to_polar

    img = np.asarray(image, dtype=np.float64)
    src_n, dst_n = _fold(src_n), _fold(dst_n)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    if img.ndim != 3 or img.shape[2] != 3 or min(img.shape[:2]) < 2 or not np.isfinite(img).all():
        raise ValueError("expected a finite RGB image of at least 2 by 2 pixels")
    if src_n == dst_n:
        return img.copy()
    if phase is None:
        phase = _phase(to_polar(img), src_n)
    h, w = img.shape[:2]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[:h, :w]
    radius = np.hypot(xx - cx, yy - cy)
    theta = np.arctan2(yy - cy, xx - cx) / (2 * np.pi)
    source = _sector_coordinate(theta, src_n, dst_n, phase)
    acc = np.zeros_like(img)
    for k in range(src_n):
        angle = (source + k / src_n) * 2 * np.pi
        acc += _sample_rgb(img, cx + radius * np.cos(angle), cy + radius * np.sin(angle))
    inside = radius <= (min(h, w) - 1) / 2.0
    return np.where(inside[..., None], acc / src_n, img)


def align_view_to_n(polar: np.ndarray, rgb: np.ndarray, dest_n: int,
                    source_n: int | None = None):
    """Use one detected count and phase for scoring and display.

    A picture that is one motif is returned unchanged. Changing the column
    count must not tile it around the circle.
    """
    src_n = detect_fold(polar) if source_n is None else _fold(source_n)
    if src_n is None:
        return (np.asarray(polar, dtype=np.float64).copy(),
                np.asarray(rgb, dtype=np.float64).copy(), None)
    dest_n = _fold(dest_n)
    phase = _phase(np.asarray(polar), src_n)
    return (remap_polar_fold(polar, src_n, dest_n, phase=phase),
            remap_image_fold(rgb, src_n, dest_n, phase=phase), src_n)
