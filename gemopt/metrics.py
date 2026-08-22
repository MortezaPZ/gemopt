"""Turning a picture of a stone into a number.

The client's brief is "get as close as possible to this reference image", so
the objective function has to compare two pictures.  Doing that naively --
subtracting pixel from pixel -- fails for two reasons: the stone can be
rotated by any angle inside the frame, and a pattern that is slightly the
wrong size still counts as a good pattern.

So every image is first put into a canonical form: find the stone, centre it,
scale it to a fixed radius, and resample it into polar coordinates.  In polar
form a rotation of the stone is a *shift* along one axis, which means the best
rotation can be found exactly with one correlation instead of searched for.

On top of the picture comparison there are the physical numbers that come
straight out of the tracer -- how much light comes back, how much leaks --
and a measure of how well the pattern holds the stone's own symmetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Target",
    "ScoreWeights",
    "to_polar",
    "align_rotation",
    "pattern_similarity",
    "achievable_ceiling",
    "relative_match",
    "uniform_target",
    "mirror_target",
    "is_uniform",
    "symmetry_error",
    "contrast",
    "load_target",
    "target_from_image",
]

POLAR_R = 64
POLAR_THETA = 256


# --------------------------------------------------------------------------
# canonical framing
# --------------------------------------------------------------------------


def _find_disk(gray: np.ndarray, background: float | None = None) -> tuple[float, float, float]:
    """Locate the stone in a photograph: centre and radius in pixels.

    Gem viewer photographs put the stone on a plain dark surround, so the
    stone is simply "everything that is not the border colour".  Sampling the
    border rather than assuming black keeps it working on white backgrounds
    too.
    """
    h, w = gray.shape
    if background is None:
        border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
        background = float(np.median(border))

    fg = np.abs(gray - background) > 0.12
    if fg.sum() < 16:
        return w / 2.0, h / 2.0, min(w, h) / 2.0

    ys, xs = np.nonzero(fg)
    cx, cy = float(xs.mean()), float(ys.mean())
    # A robust radius: the stone is round, so use a high percentile of the
    # distance rather than the maximum, which any stray speck would inflate.
    r = float(np.percentile(np.hypot(xs - cx, ys - cy), 99.0))
    return cx, cy, max(r, 1.0)


def to_polar(
    image: np.ndarray,
    centre: tuple[float, float] | None = None,
    radius: float | None = None,
    n_r: int = POLAR_R,
    n_theta: int = POLAR_THETA,
) -> np.ndarray:
    """Resample an image onto a ``(n_r, n_theta)`` polar grid.

    Bilinear, written out by hand so the package needs nothing beyond numpy.
    """
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr @ np.array([0.2126, 0.7152, 0.0722])
    h, w = arr.shape
    cx, cy = centre if centre is not None else (w / 2.0 - 0.5, h / 2.0 - 0.5)
    rad = radius if radius is not None else min(w, h) / 2.0

    rr = (np.arange(n_r) + 0.5) / n_r * rad
    tt = np.arange(n_theta) / n_theta * 2.0 * np.pi
    R, T = np.meshgrid(rr, tt, indexing="ij")
    x = cx + R * np.cos(T)
    y = cy + R * np.sin(T)

    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx = x - x0
    fy = y - y0
    x0c = np.clip(x0, 0, w - 2)
    y0c = np.clip(y0, 0, h - 2)

    out = (
        arr[y0c, x0c] * (1 - fx) * (1 - fy)
        + arr[y0c, x0c + 1] * fx * (1 - fy)
        + arr[y0c + 1, x0c] * (1 - fx) * fy
        + arr[y0c + 1, x0c + 1] * fx * fy
    )
    inside = (x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1)
    return np.where(inside, out, 0.0)


@dataclass
class Target:
    """A reference pattern, already in canonical polar form."""

    polar: np.ndarray
    name: str = "target"
    n_fold: int = 8
    preview: np.ndarray | None = None
    """The square image it came from, kept only for reports."""

    uniform: bool = False
    """Set when the goal is "as bright as possible" rather than a pattern.
    See :func:`uniform_target`."""


def target_from_image(image: np.ndarray, n_fold: int = 8, name: str = "target") -> Target:
    """Build a target from an array that is already framed on the stone."""
    arr = np.asarray(image, dtype=np.float64)
    gray = arr @ np.array([0.2126, 0.7152, 0.0722]) if arr.ndim == 3 else arr
    h, w = gray.shape
    pol = to_polar(gray, centre=(w / 2.0 - 0.5, h / 2.0 - 0.5), radius=min(w, h) / 2.0 * 0.98)
    return Target(polar=_normalise(pol), name=name, n_fold=n_fold, preview=arr)


def load_target(path, n_fold: int = 8) -> Target:
    """Read a photograph of a stone and canonicalise it.

    Handles the common case of a side-by-side pair (hearts next to arrows, as
    in the client's reference) by keeping whichever half is squarer once the
    stone has been located.
    """
    from PIL import Image

    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    gray = img @ np.array([0.2126, 0.7152, 0.0722])

    h, w = gray.shape
    if w > 1.5 * h:
        # two views pasted together: split and take the left one
        half = w // 2
        img, gray = img[:, :half], gray[:, :half]

    cx, cy, r = _find_disk(gray)
    pol = to_polar(img, centre=(cx, cy), radius=r * 0.98)
    return Target(polar=_normalise(pol), name=str(path), n_fold=n_fold, preview=img)


def _normalise(pol: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance -- so brightness and contrast stop mattering.

    We are matching the *shape* of the pattern.  A picture taken with a
    brighter lamp, or our simulation using slightly different band colours,
    should not change the score.
    """
    p = pol - pol.mean()
    s = p.std()
    return p / s if s > 1e-9 else p


# --------------------------------------------------------------------------
# comparisons
# --------------------------------------------------------------------------


def align_rotation(a: np.ndarray, b: np.ndarray) -> tuple[int, float]:
    """Best rotational alignment of two polar images.

    Correlation along the angular axis via FFT, which finds the optimum over
    every possible shift at once instead of trying them one by one.
    Returns the shift in angular samples and the correlation there.
    """
    fa = np.fft.rfft(a, axis=1)
    fb = np.fft.rfft(b, axis=1)
    corr = np.fft.irfft(fa * np.conj(fb), n=a.shape[1], axis=1).sum(axis=0)
    shift = int(np.argmax(corr))
    return shift, float(corr[shift] / (a.size))


def pattern_similarity(rendered_polar: np.ndarray, target: Target) -> float:
    """How much the rendered pattern looks like the target, in ``0..1``.

    The value is a normalised cross-correlation at the best rotation, mapped
    from ``-1..1`` onto ``0..1`` so it can be mixed with the other scores.
    """
    a = _normalise(rendered_polar)
    b = target.polar
    if a.shape != b.shape:
        raise ValueError(f"polar grids differ: {a.shape} vs {b.shape}")
    _, corr = align_rotation(a, b)
    return float(np.clip(0.5 * (corr + 1.0), 0.0, 1.0))


def symmetry_error(polar: np.ndarray, n_fold: int) -> float:
    """How far the pattern is from being ``n_fold`` symmetric, ``0`` = perfect.

    Measured in the angular frequency domain: a pattern that repeats ``n_fold``
    times around the stone can only contain harmonics that are multiples of
    ``n_fold``, so the share of the energy sitting anywhere else is exactly
    the asymmetry.

    Doing it this way rather than by rolling the image and subtracting means
    it works for any fold, including ones that do not divide the number of
    angular samples.  Rolling by a rounded number of columns would have
    quietly reported perfect symmetry in those cases -- and reporting
    "flawless" when you cannot measure is the worst possible failure for a
    term that is meant to catch misaligned facets.
    """
    if n_fold < 2:
        return 0.0
    spectrum = np.abs(np.fft.rfft(_normalise(polar), axis=1)) ** 2
    power = spectrum.sum(axis=0)
    power[0] = 0.0                       # the flat background carries no symmetry
    total = power.sum()
    if total <= 1e-12:
        return 0.0
    aligned = power[np.arange(len(power)) % n_fold == 0].sum()
    return float(np.clip(1.0 - aligned / total, 0.0, 1.0))


def contrast(polar: np.ndarray) -> float:
    """Spread of brightness across the stone, ``0..1``.

    A pattern only reads as hearts or arrows if the bright and dark regions
    are clearly separated; a flat grey disc scores zero however "bright" it
    is.  Standard deviation is a good enough proxy and costs nothing.
    """
    return float(np.clip(polar.std() * 2.0, 0.0, 1.0))


# --------------------------------------------------------------------------
# combining
# --------------------------------------------------------------------------


@dataclass
class ScoreWeights:
    """How much each ingredient counts towards the final number.

    Defaults lean on the pattern, because that is what the client actually
    asked for; the physical terms keep the search from finding a shape that
    draws a pretty picture while leaking most of its light.
    """

    pattern: float = 1.0
    light_return: float = 0.35
    leakage: float = 0.25
    symmetry: float = 0.20
    contrast: float = 0.20

    contrast_floor: float = 0.25
    """Contrast at which the pattern term is trusted in full.

    Correlation is computed on contrast-normalised images, so a nearly flat
    grey disc gets its faint noise amplified and can correlate with anything.
    Left ungated the search discovers that washing the stone out is an easy
    way to score well, which is the opposite of what we want.  Below this
    contrast the pattern term is faded out proportionally.
    """

    def combine(self, parts: dict[str, float]) -> float:
        """Weighted sum, normalised so a perfect stone scores 1.0.

        ``leakage`` and ``symmetry`` are penalties, so they enter as
        ``1 - value``.
        """
        gate = 1.0
        if self.contrast_floor > 0.0:
            gate = float(np.clip(parts.get("contrast", 0.0) / self.contrast_floor, 0.0, 1.0))
        terms = {
            "pattern": parts.get("pattern", 0.0) * gate,
            "light_return": parts.get("light_return", 0.0),
            "leakage": 1.0 - parts.get("leakage", 0.0),
            "symmetry": 1.0 - parts.get("symmetry", 0.0),
            "contrast": parts.get("contrast", 0.0),
        }
        weights = {
            "pattern": self.pattern,
            "light_return": self.light_return,
            "leakage": self.leakage,
            "symmetry": self.symmetry,
            "contrast": self.contrast,
        }
        total = sum(weights.values())
        if total <= 0.0:
            return 0.0
        return float(sum(weights[k] * terms[k] for k in terms) / total)


def achievable_ceiling(target: Target) -> float:
    """The best score any symmetric simulation could get against ``target``.

    A simulated stone is perfectly ``n_fold`` symmetric by construction.  A
    photograph never is: the lighting is a little uneven, the camera is a
    little off axis, the real stone was cut by a person, and JPEG has been
    over it.  None of that is reproducible, and none of it *should* be --
    chasing it would mean fitting noise.

    Averaging the target over its own symmetry gives the closest symmetric
    pattern to it, and correlating the two says how much of the photograph is
    reachable at all.  Scores should be read against this, not against 1.0:
    on the client's reference it sits near 0.83, so 0.75 is not "75% of the
    way there", it is about 90% of everything available.
    """
    period = target.polar.shape[1] // target.n_fold
    rotations = [np.roll(target.polar, k * period, axis=1) for k in range(target.n_fold)]
    symmetric = _normalise(np.mean(rotations, axis=0)).ravel()
    original = _normalise(target.polar).ravel()
    denom = np.linalg.norm(symmetric) * np.linalg.norm(original)
    if denom <= 0:
        return 1.0
    return float(np.clip(0.5 * (symmetric @ original / denom + 1.0), 0.0, 1.0))


def relative_match(similarity: float, ceiling: float) -> float:
    """Similarity rescaled so 1.0 means "as close as anything could get".

    0.5 is the floor of :func:`pattern_similarity` (zero correlation), so the
    scale is anchored there rather than at zero.
    """
    span = ceiling - 0.5
    if span <= 1e-9:
        return 0.0
    return float(np.clip((similarity - 0.5) / span, 0.0, 1.0))


# --------------------------------------------------------------------------
# targets that are not photographs
# --------------------------------------------------------------------------


def uniform_target(n_fold: int = 8, name: str = "white") -> Target:
    """"As white as possible" -- the client's first description of the goal.

    Before the hearts-and-arrows photograph arrived, the brief was a stone
    matching a white or mirror image.  That is not a pattern to correlate
    against: a flat field has no structure, and the correlation of anything
    with a constant is undefined.  What it really asks for is *all* of the
    light coming back, which the tracer already reports directly.

    So this target exists to be recognised rather than correlated -- the
    evaluator sees ``uniform`` and scores light return instead of pattern.
    Returning it as a normal Target keeps every caller's plumbing unchanged.
    """
    target = Target(polar=np.zeros((POLAR_R, POLAR_THETA)), name=name, n_fold=n_fold)
    target.uniform = True
    return target


def mirror_target(n_fold: int = 8, name: str = "mirror") -> Target:
    """A perfect reflector: everything returned, nothing leaking, no pattern.

    Optically identical to :func:`uniform_target` as far as scoring goes --
    the difference is only what the client calls it -- so it is the same
    object under a second name rather than a second code path that could
    drift out of step.
    """
    return uniform_target(n_fold=n_fold, name=name)


def is_uniform(target: Target | None) -> bool:
    """Whether a target asks for brightness rather than for a pattern."""
    return bool(target is not None and getattr(target, "uniform", False))
