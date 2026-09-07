"""Map a photograph of one fold count onto another.

Hearts-and-arrows pictures are almost always 8-fold.  Scoring a 16-main stone
against that picture pixel-for-pixel asks the search to *unmake* eight of the
arrows, which is how N=16 came out as a cluttered red disc with a dark ring.
The fix is to take one source sector, scale it angularly to 360/N, and copy
it around the circle so the target the optimiser sees has the same fold as
the stone.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "detect_fold",
    "remap_polar_fold",
    "remap_image_fold",
    "align_view_to_n",
]


FOLD_CANDIDATES = (6, 7, 8, 9, 10, 12, 16, 18, 20, 24)


def detect_fold(polar: np.ndarray, candidates: tuple[int, ...] = FOLD_CANDIDATES) -> int:
    """Angular fundamental of a polar image, collapsing even harmonics.

    An 8-arrow pattern is also 16-fold (and 4-fold).  The FFT peak may sit on
    a harmonic; walking down by two while the half-frequency is still strong
    recovers the fold a cutter would name.
    """
    arr = np.asarray(polar, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 16:
        return 8
    profile = arr.mean(axis=0)
    profile = profile - profile.mean()
    spec = np.abs(np.fft.rfft(profile)) ** 2
    spec[0] = 0.0
    lo, hi = min(candidates), max(candidates)
    if hi >= len(spec):
        hi = len(spec) - 1
    if hi < lo:
        return 8
    peak = int(np.argmax(spec[lo:hi + 1]) + lo)
    while peak >= 8 and peak % 2 == 0 and peak // 2 >= lo:
        if spec[peak // 2] >= 0.55 * spec[peak]:
            peak //= 2
        else:
            break
    # snap to the nearest candidate so we never return a nonsense fold
    return int(min(candidates, key=lambda n: abs(n - peak)))


def remap_polar_fold(polar: np.ndarray, src_n: int, dst_n: int) -> np.ndarray:
    """Resample ``polar`` so one source sector becomes one destination sector.

    Every source sector is averaged first, so photographic lighting tilt does
    not get copied sixteen times.  ``src_n == dst_n`` still averages, which
    is the symmetric target a simulated stone can actually reach.
    """
    arr = np.asarray(polar, dtype=np.float64)
    src_n = max(3, int(src_n))
    dst_n = max(3, int(dst_n))
    n_r, n_theta = arr.shape
    theta = (np.arange(n_theta) + 0.5) / n_theta
    u = (theta * dst_n) % 1.0
    acc = np.zeros_like(arr)
    cols = np.arange(n_theta)
    for k in range(src_n):
        src = ((k + u) / src_n) * n_theta
        c0 = np.floor(src).astype(int) % n_theta
        f = (src - np.floor(src))[None, :]
        c1 = (c0 + 1) % n_theta
        acc += arr[:, c0] * (1.0 - f) + arr[:, c1] * f
    return acc / src_n


def _sample_rgb(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    x0c = np.clip(x0, 0, w - 2)
    y0c = np.clip(y0, 0, h - 2)
    out = (
        img[y0c, x0c] * (1 - fx) * (1 - fy)
        + img[y0c, x0c + 1] * fx * (1 - fy)
        + img[y0c + 1, x0c] * (1 - fx) * fy
        + img[y0c + 1, x0c + 1] * fx * fy
    )
    inside = (x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1)
    return np.where(inside[..., None], out, 0.0)


def remap_image_fold(image: np.ndarray, src_n: int, dst_n: int) -> np.ndarray:
    """Cartesian twin of :func:`remap_polar_fold`, for the preview square."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    src_n = max(3, int(src_n))
    dst_n = max(3, int(dst_n))
    h, w = img.shape[:2]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    dx = xx - cx
    dy = yy - cy
    radius = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    theta01 = np.mod(theta, 2.0 * np.pi) / (2.0 * np.pi)
    u = (theta01 * dst_n) % 1.0
    acc = np.zeros_like(img)
    for k in range(src_n):
        src_theta = (k + u) / src_n * 2.0 * np.pi
        acc += _sample_rgb(img, cx + radius * np.cos(src_theta),
                           cy + radius * np.sin(src_theta))
    return np.clip(acc / src_n, 0.0, 1.0)


def align_view_to_n(polar: np.ndarray, rgb: np.ndarray, dest_n: int):
    """Detect the photo's fold, remap both polar and RGB onto ``dest_n``.

    Returns ``(polar, rgb, source_fold)``.  RGB is only rewritten when the
    fold actually changes, so an 8-fold photograph still looks like itself
    when the stone is also 8-fold.
    """
    dest_n = max(3, int(dest_n))
    src_n = detect_fold(polar)
    polar_out = remap_polar_fold(polar, src_n, dest_n)
    rgb_out = remap_image_fold(rgb, src_n, dest_n) if src_n != dest_n else np.asarray(rgb, dtype=np.float64)
    return polar_out, rgb_out, src_n
