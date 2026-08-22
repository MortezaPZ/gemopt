"""Fit the virtual instrument to the client's real one.

Every hearts-and-arrows viewer is built differently: the cone has its own
half-angle, its own shade of red, and its own bright aperture at the apex.
Those numbers decide which facet reads bright and which reads dark, so
guessing them means the simulated pattern never quite matches the photograph
-- and then the optimiser is aiming at the wrong picture.

Rather than ask the client to measure their viewer, we recover it from a
photograph they already have.  A reference photo of a *known* stone -- an
ideal-cut diamond -- pins the instrument down: the stone's geometry is not in
question, so any mismatch between simulation and photo must come from the
instrument.  Sweeping the cone angles and keeping whatever reproduces the
photograph best gives a calibrated viewer, and everything measured through it
afterwards is comparable with what the client sees on their bench.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import CutParams, build_stone
from .metrics import Target, pattern_similarity, to_polar, _normalise
from .optics import DIAMOND, Geometry, Material, trace
from .scope import Camera, ProfileEnvironment, ScopeEnvironment, render

__all__ = [
    "CalibrationResult",
    "dominant_colours",
    "calibrate_scope",
    "fit_radiance_profile",
]


@dataclass
class CalibrationResult:
    scope: ScopeEnvironment
    score: float
    inner_deg: float
    mid_deg: float
    outer_deg: float
    order: str
    """``"bright_core"`` or ``"dark_core"`` -- which way round the bands run."""

    def describe(self) -> str:
        return (
            f"{self.order}: inner {self.inner_deg:.0f} deg, "
            f"mid {self.mid_deg:.0f} deg, outer {self.outer_deg:.0f} deg "
            f"(similarity {self.score:.4f})"
        )


def dominant_colours(image: np.ndarray, k: int = 3, iters: int = 25) -> np.ndarray:
    """The ``k`` most common colours, ordered dark to bright.

    Plain Lloyd's algorithm on the pixels.  A photograph of a stone in a red
    viewer really does only contain three families -- shadow, the red cone and
    the bright aperture -- so there is nothing to be gained from anything
    cleverer, and this keeps the dependency list at numpy.
    """
    px = np.asarray(image, dtype=np.float64).reshape(-1, 3)
    if px.max() > 1.5:
        px = px / 255.0
    rng = np.random.default_rng(0)
    centres = px[rng.choice(len(px), size=k, replace=False)]
    for _ in range(iters):
        d = ((px[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        label = d.argmin(axis=1)
        for j in range(k):
            sel = label == j
            if sel.any():
                centres[j] = px[sel].mean(axis=0)
    lum = centres @ np.array([0.2126, 0.7152, 0.0722])
    return centres[np.argsort(lum)]


def _build_scope(colours: np.ndarray, inner: float, mid: float, outer: float,
                 order: str) -> ScopeEnvironment:
    dark, red, bright = colours[0], colours[1], colours[2]
    if order == "bright_core":
        seq = [bright, red, dark]
    else:
        seq = [red, bright, dark]
    return ScopeEnvironment(
        name=f"calibrated_{order}",
        bands=[
            (inner, tuple(seq[0])),
            (mid, tuple(seq[1])),
            (outer, tuple(seq[2])),
            (180.0, tuple(dark)),
        ],
    )


def calibrate_scope(
    target: Target,
    reference_rgb: np.ndarray,
    view: str = "crown",
    params: CutParams | None = None,
    material: Material = DIAMOND,
    resolution: int = 200,
    inner_range: tuple[float, ...] = (10.0, 14.0, 18.0, 22.0, 26.0, 30.0),
    mid_range: tuple[float, ...] = (34.0, 40.0, 46.0, 52.0, 58.0, 64.0),
    outer_range: tuple[float, ...] = (72.0, 80.0, 88.0),
    progress=None,
) -> CalibrationResult:
    """Search cone angles for the best match to ``target``.

    ``params`` defaults to the standard round brilliant, which is what the
    reference photograph shows.  The stone is rebuilt once and reused across
    the whole sweep -- only the instrument changes.
    """
    params = params or CutParams()
    mesh = build_stone(params)
    geom = Geometry(mesh)
    camera = Camera(resolution=resolution, view=view, samples=1)
    colours = dominant_colours(reference_rgb)

    best: CalibrationResult | None = None
    for order in ("bright_core", "dark_core"):
        for inner in inner_range:
            for mid in mid_range:
                if mid <= inner:
                    continue
                for outer in outer_range:
                    if outer <= mid:
                        continue
                    scope = _build_scope(colours, inner, mid, outer, order)
                    result = render(mesh, material, camera, scope, geometry=geom)
                    pol = _normalise(to_polar(result.gray))
                    score = pattern_similarity(pol, target)
                    if best is None or score > best.score:
                        best = CalibrationResult(scope, score, inner, mid, outer, order)
                        if progress is not None:
                            progress(best)
    assert best is not None
    return best


# --------------------------------------------------------------------------
# recovering the instrument's brightness curve from a photograph
# --------------------------------------------------------------------------


def _polar_rays(n_r: int, n_theta: int, view: str, max_radius: float):
    """Camera rays arranged on a polar grid, one per cell.

    Fitting happens in polar coordinates because that is the natural frame for
    a round stone: rotating the reference photo into alignment becomes a plain
    array roll instead of an image resampling.
    """
    rr = (np.arange(n_r) + 0.5) / n_r * max_radius
    tt = np.arange(n_theta) / n_theta * 2.0 * np.pi
    R, T = np.meshgrid(rr, tt, indexing="ij")
    x, y = (R * np.cos(T)).ravel(), (R * np.sin(T)).ravel()
    z = 5.0 if view == "crown" else -5.0
    step = np.array([0.0, 0.0, -1.0]) if view == "crown" else np.array([0.0, 0.0, 1.0])
    origin = np.column_stack([x, y, np.full(x.size, z)])
    direction = np.tile(step, (x.size, 1))
    return origin, direction


def fit_radiance_profile(
    reference_rgb: np.ndarray,
    view: str,
    material: Material = DIAMOND,
    params: CutParams | None = None,
    n_bins: int = 30,
    n_r: int = 56,
    n_theta: int = 144,
    max_radius: float = 1.019,
    smooth: float = 0.05,
):
    """Solve for what the viewer looks like, given a photo of a known stone.

    The reference photograph shows an ideal-cut diamond, so its geometry is
    not in question.  Whatever the simulation gets wrong must therefore be the
    *instrument*, and the instrument enters the image linearly: every pixel is
    a weighted sum of the brightness of whichever directions its light came
    from.  Binning direction into ``n_bins`` angular slices turns that into

        image = A @ radiance

    with ``A`` built straight out of one ray trace.  Recovering the viewer is
    then a non-negative least squares solve per colour channel -- no search,
    no guessing at cone angles, and it takes about a second.

    A small first-difference penalty (``smooth``) keeps neighbouring bins from
    swinging apart where few rays landed to constrain them.
    """
    from scipy.optimize import nnls

    params = params or CutParams()
    mesh = build_stone(params)
    geom = Geometry(mesh)
    axis = np.array([0.0, 0.0, 1.0]) if view == "crown" else np.array([0.0, 0.0, -1.0])

    origin, direction = _polar_rays(n_r, n_theta, view, max_radius)
    ev = trace(geom, origin, direction, material, max_bounces=14)

    edges = np.concatenate([np.linspace(0.0, 90.0, n_bins), [180.0]])
    angle = np.degrees(np.arccos(np.clip(ev.direction @ axis, -1.0, 1.0)))
    bin_of = np.clip(np.searchsorted(edges, angle, side="right") - 1, 0, n_bins - 1)

    n_cells = n_r * n_theta
    A = np.zeros((n_cells, n_bins))
    np.add.at(A, (ev.ray, bin_of), ev.weight)

    target = to_polar(
        reference_rgb,
        centre=(reference_rgb.shape[1] / 2.0 - 0.5, reference_rgb.shape[0] / 2.0 - 0.5),
        radius=min(reference_rgb.shape[:2]) / 2.0 * (max_radius / 1.04),
        n_r=n_r,
        n_theta=n_theta,
    ) if reference_rgb.ndim == 2 else np.stack([
        to_polar(
            reference_rgb[:, :, c],
            centre=(reference_rgb.shape[1] / 2.0 - 0.5, reference_rgb.shape[0] / 2.0 - 0.5),
            radius=min(reference_rgb.shape[:2]) / 2.0 * (max_radius / 1.04),
            n_r=n_r,
            n_theta=n_theta,
        ) for c in range(3)
    ], axis=-1)

    # penalty rows that ask neighbouring bins to stay close
    reg = np.zeros((n_bins - 1, n_bins))
    reg[np.arange(n_bins - 1), np.arange(n_bins - 1)] = -smooth * n_cells / n_bins
    reg[np.arange(n_bins - 1), np.arange(1, n_bins)] = smooth * n_cells / n_bins
    A_reg = np.vstack([A, reg])

    # The photo's rotation is arbitrary; only rolls within one symmetry period
    # can differ, so try each and keep the one that fits best.
    period = n_theta // int(params.n_base)
    best = None
    for shift in range(period):
        rolled = np.roll(target, shift, axis=1)
        values = np.zeros((n_bins, 3))
        residual = 0.0
        for c in range(3):
            b = np.concatenate([rolled[:, :, c].ravel(), np.zeros(n_bins - 1)])
            sol, rnorm = nnls(A_reg, b)
            values[:, c] = sol
            residual += rnorm**2
        if best is None or residual < best[0]:
            best = (residual, shift, values)

    residual, shift, values = best
    scale = float(values.max())
    if scale > 0:
        values = np.clip(values, 0.0, None)
    return ProfileEnvironment(edges=edges, values=values, name=f"fitted_{view}"), {
        "residual": float(residual),
        "shift": int(shift),
        "n_bins": n_bins,
        "peak": scale,
    }


def _angle_map(mesh, material, view, resolution, max_bounces=14):
    """Mean exit angle per pixel -- the geometry's fingerprint.

    Independent of any lighting model: it says only "light leaving this pixel
    came from this far off the viewing axis".  Whatever the instrument is, its
    picture has to be some function of this map, which is what makes it the
    right thing to align a photograph against.
    """
    from .scope import Camera

    geom = Geometry(mesh)
    cam = Camera(resolution=resolution, view=view, samples=1)
    origin, direction, _ = cam.rays()
    axis = cam.axis
    ev = trace(geom, origin, direction, material, max_bounces=max_bounces)

    towards = ev.direction @ axis
    out = ev.entered & (towards > 0)
    angle = np.degrees(np.arccos(np.clip(towards, -1.0, 1.0)))

    n = resolution * resolution
    num = np.zeros(n)
    den = np.zeros(n)
    np.add.at(num, ev.ray[out], ev.weight[out] * angle[out])
    np.add.at(den, ev.ray[out], ev.weight[out])
    lit = den > 1e-6
    amap = np.full(n, np.nan)
    amap[lit] = num[lit] / den[lit]
    return amap.reshape(resolution, resolution), den.reshape(resolution, resolution)


def _correlation_ratio(values: np.ndarray, groups: np.ndarray, n_groups: int) -> float:
    """How much of ``values`` is explained by knowing the group.

    Unlike a correlation coefficient this makes no assumption that the
    relationship is monotonic -- the mapping from exit angle to brightness in
    a gem viewer certainly is not -- so it is the honest way to test whether a
    given rotation lines the photograph up with the geometry.
    """
    total_var = values.var()
    if total_var <= 0:
        return 0.0
    sums = np.bincount(groups, weights=values, minlength=n_groups)
    counts = np.bincount(groups, minlength=n_groups).astype(float)
    ok = counts > 0
    means = np.zeros(n_groups)
    means[ok] = sums[ok] / counts[ok]
    between = (counts[ok] * (means[ok] - values.mean()) ** 2).sum() / len(values)
    return float(between / total_var)


def fit_instrument(
    reference_rgb: np.ndarray,
    view: str,
    material: Material = DIAMOND,
    params: CutParams | None = None,
    resolution: int = 320,
    n_bins: int = 44,
    max_angle: float = 90.0,
    n_fold: int = 8,
):
    """Recover the viewer by asking the photograph what each angle looks like.

    Every pixel of the reference photo shows light that arrived from some
    angle off the viewing axis, and :func:`_angle_map` tells us which angle
    that was.  Group the photo's pixels by that angle and average them, and
    the result *is* the instrument's brightness curve -- the conditional mean
    is exactly the mapping that minimises squared error, so there is nothing
    left to search for once the two images are aligned.

    Alignment is the one unknown, since the photo was taken at an arbitrary
    rotation.  Only rotations inside one symmetry period can differ, and the
    right one is whichever makes the photo most predictable from the angle map
    -- measured with a correlation ratio so a non-monotonic mapping (which is
    what a red cone with a dark centre produces) is not penalised.
    """
    from PIL import Image
    from .scope import ProfileEnvironment

    params = params or CutParams()
    mesh = build_stone(params)
    amap, weight = _angle_map(mesh, material, view, resolution)

    ref = Image.fromarray((np.clip(reference_rgb, 0, 1) * 255).astype("uint8"))
    ref = np.asarray(ref.resize((resolution, resolution), Image.LANCZOS),
                     dtype=np.float64) / 255.0

    edges = np.linspace(0.0, max_angle, n_bins + 1)
    valid = np.isfinite(amap) & (weight > 0.02)
    bin_of = np.clip(np.digitize(amap, edges) - 1, 0, n_bins - 1)

    lum = ref @ np.array([0.2126, 0.7152, 0.0722])
    best_angle, best_eta = 0.0, -1.0
    for deg in np.arange(0.0, 360.0 / n_fold, 0.5):
        rot = np.asarray(
            Image.fromarray((lum * 255).astype("uint8")).rotate(deg, resample=Image.BILINEAR),
            dtype=np.float64,
        ) / 255.0
        eta = _correlation_ratio(rot[valid], bin_of[valid], n_bins)
        if eta > best_eta:
            best_eta, best_angle = eta, deg

    rotated = np.stack([
        np.asarray(
            Image.fromarray((ref[:, :, c] * 255).astype("uint8")).rotate(
                best_angle, resample=Image.BILINEAR),
            dtype=np.float64,
        ) / 255.0
        for c in range(3)
    ], axis=-1)

    values = np.zeros((n_bins, 3))
    counts = np.bincount(bin_of[valid], minlength=n_bins).astype(float)
    for c in range(3):
        sums = np.bincount(bin_of[valid], weights=rotated[:, :, c][valid], minlength=n_bins)
        ok = counts > 0
        values[ok, c] = sums[ok] / counts[ok]
    # bins no ray reached: hold the last measured value rather than go black
    for c in range(3):
        last = 0.0
        for k in range(n_bins):
            if counts[k] > 0:
                last = values[k, c]
            else:
                values[k, c] = last

    full_edges = np.concatenate([edges, [180.0]])
    values = np.vstack([values, values[-1:]])
    env = ProfileEnvironment(edges=full_edges, values=values, name=f"fitted_{view}")
    return env, {"rotation_deg": float(best_angle), "correlation_ratio": float(best_eta),
                 "n_bins": n_bins, "covered_bins": int((counts > 0).sum())}
