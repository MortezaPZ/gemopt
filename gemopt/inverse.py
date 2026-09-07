"""Work backwards from a photograph to the stone that produced it.
The forward problem -- given a cut and a viewer, draw the pattern -- is what
the rest of the package does.  This is the inverse: given a photograph, what
cut is it?

The trick that makes it tractable is that the two unknowns are not equally
hard.  The viewer enters the image *linearly* (a pixel is a weighted sum of
the brightness of the directions its light arrived from), so for any candidate
cut the best possible viewer can be solved for outright, with no search.  Only
the geometry has to be searched, and each candidate is scored by the match it
achieves once its viewer has been solved for.

That has two benefits over guessing the viewer up front.  It removes the
instrument as a source of error -- no candidate is ever penalised for being
paired with the wrong cone angles -- and it turns "which cut is this?" into a
nine-parameter search rather than a twenty-parameter one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fold import align_view_to_n
from .geometry import CutParams, build_stone
from .metrics import _normalise, symmetry_error, to_polar
from .optics import DIAMOND, Geometry, Material, trace
from .scope import ProfileEnvironment

__all__ = [
    "InverseProblem",
    "load_single",
    "load_reference",
    "FixedInstrumentProblem",
    "ViewData",
    "solve_profile",
    "design_matrix",
    "apply_psf",
]

POLAR_EXTENT = 0.94
"""How far out the comparison reaches, in girdle radii.

Two things set this.  It has to be identical for the photograph and the
simulation or the pattern's rings are compared against the wrong rings.  And
it has to stay inside the part of the photograph that actually shows stone:
the client's reference is cropped tight enough that the top and bottom of each
gem run off the edge of the frame, so anything beyond about 0.95 of the girdle
radius is image border rather than gem.  Sampling out there would compare our
girdle against their black background -- which correlates well enough to look
like an improvement while meaning nothing."""


def _polar_rays(n_r: int, n_theta: int, view: str, max_radius: float = POLAR_EXTENT):
    """One camera ray per polar cell."""
    rr = (np.arange(n_r) + 0.5) / n_r * max_radius
    tt = np.arange(n_theta) / n_theta * 2.0 * np.pi
    R, T = np.meshgrid(rr, tt, indexing="ij")
    x, y = (R * np.cos(T)).ravel(), (R * np.sin(T)).ravel()
    z = 5.0 if view == "crown" else -5.0
    step = np.array([0.0, 0.0, -1.0]) if view == "crown" else np.array([0.0, 0.0, 1.0])
    return (np.column_stack([x, y, np.full(x.size, z)]),
            np.tile(step, (x.size, 1)))


@dataclass
class ViewData:
    """A reference photograph of one side of the stone, ready to match."""

    view: str
    target: np.ndarray
    """(n_r, n_theta) polar luminance, z-normalised."""

    rgb: np.ndarray
    """The framed square image, kept for colouring the fitted instrument."""

    shifts: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """Every rotation of the target that symmetry allows, as columns."""

    radius_scale: float = 1.0
    """How far the fitted girdle sits from the one brightness thresholding found."""

    n_fold: int = 8
    """Fold the polar target was aligned to.  Must match the stone."""

    source_fold: int = 8
    """Fold detected in the photograph before any remapping."""

    def prepare(self, n_fold: int) -> None:
        n_fold = max(3, int(n_fold))
        self.n_fold = n_fold
        period = max(1, self.target.shape[1] // n_fold)
        cols = [np.roll(self.target, s, axis=1).ravel() for s in range(period)]
        T = np.stack(cols, axis=1)
        self.shifts = T - T.mean(axis=0, keepdims=True)


def design_matrix(
    mesh_geom: Geometry,
    view: str,
    material: Material,
    n_bins: int,
    n_r: int,
    n_theta: int,
    max_angle: float = 90.0,
    max_bounces: int = 14,
) -> np.ndarray:
    """``A[cell, bin]``: light reaching each polar cell from each angular bin.

    This is the whole forward model, reduced to a matrix.  Multiply it by any
    brightness-versus-angle curve and you get the picture that curve produces,
    which is what lets the instrument be solved for rather than searched.
    """
    axis = np.array([0.0, 0.0, 1.0]) if view == "crown" else np.array([0.0, 0.0, -1.0])
    origin, direction = _polar_rays(n_r, n_theta, view)
    ev = trace(mesh_geom, origin, direction, material, max_bounces=max_bounces)

    edges = np.linspace(0.0, max_angle, n_bins + 1)
    angle = np.degrees(np.arccos(np.clip(ev.direction @ axis, -1.0, 1.0)))
    bin_of = np.clip(np.digitize(angle, edges) - 1, 0, n_bins - 1)

    A = np.zeros((n_r * n_theta, n_bins))
    np.add.at(A, (ev.ray, bin_of), ev.weight)
    return A


def apply_psf(A: np.ndarray, n_r: int, n_theta: int, sigma: float) -> np.ndarray:
    """Blur the forward model the way the camera blurred the photograph.

    Every real instrument has a point spread function: lens, defocus, sensor,
    and in this case JPEG on top.  Our render is pixel sharp, and a faceted gem
    is nothing but edges -- so comparing the two throws away correlation at
    every facet boundary, which is most of the picture.  Modelling it is worth
    more than a decade of geometry tuning: on the client's reference it moves
    the match from about 79% of what is achievable to about 90%.

    Blur is linear, so it can be folded into the design matrix and the
    instrument still has a closed-form solution.  ``sigma`` is in pixels of the
    original image, which means the angular blur has to shrink with radius --
    the same arc subtends fewer pixels near the centre.
    """
    if sigma <= 0.0:
        return A
    from scipy.ndimage import gaussian_filter1d

    cube = gaussian_filter1d(A.reshape(n_r, n_theta, -1), sigma, axis=0,
                             mode="nearest")

    # The angular pass is a different width at every radius, so doing it row by
    # row costs a Python loop over the whole design matrix -- five times the
    # price of the ray trace it is meant to accompany.  Wrapping is periodic,
    # which makes it a multiply in Fourier space: one transform, one scaling,
    # one transform back, for every radius at once.
    radius = (np.arange(n_r) + 0.5) / n_r * POLAR_EXTENT * n_r
    angular = sigma / np.maximum(radius, 1.0) * n_theta / (2.0 * np.pi)
    freq = np.fft.rfftfreq(n_theta) * n_theta
    kernel = np.exp(-0.5 * (2.0 * np.pi * freq[None, :] * angular[:, None] / n_theta) ** 2)
    spectrum = np.fft.rfft(cube, axis=1)
    spectrum *= kernel[:, :, None]
    return np.fft.irfft(spectrum, n=n_theta, axis=1).reshape(n_r * n_theta, -1)


def _smoothness_operator(n_bins: int) -> np.ndarray:
    """Second-difference matrix, used to penalise a jagged profile.

    Light inside a cone falls off gradually; no real viewer flickers between
    bright and dark from one degree to the next.  Without this the solve is
    free to spend its extra degrees of freedom fitting photographic noise,
    which flatters the score on the reference and then transfers badly to a
    different stone -- exactly the case we need it to survive.
    """
    D = np.zeros((n_bins - 2, n_bins))
    rows = np.arange(n_bins - 2)
    D[rows, rows] = 1.0
    D[rows, rows + 1] = -2.0
    D[rows, rows + 2] = 1.0
    return D


def solve_profile(blocks: list[tuple[np.ndarray, np.ndarray]], ridge: float = 2e-3,
                  smooth: float = 0.0):
    """Best single instrument across several views, and the match it gives.

    ``blocks`` pairs each view's design matrix with its rotated targets.  One
    profile has to serve every view, because it is one physical device -- and
    that shared constraint is what stops the fit from quietly inventing a
    different instrument for each picture just to flatter the numbers.

    Rotations are handled by solving every candidate at once: they differ only
    in the right-hand side, so the expensive normal-equation matrix is built a
    single time and reused.
    """
    n_bins = blocks[0][0].shape[1]
    centred = [(A - A.mean(axis=0, keepdims=True), T) for A, T in blocks]

    gram = np.zeros((n_bins, n_bins))
    for Ac, _ in centred:
        gram += Ac.T @ Ac
    scale = np.trace(gram) / n_bins
    gram += ridge * scale * np.eye(n_bins)
    if smooth > 0.0:
        D = _smoothness_operator(n_bins)
        gram += smooth * scale * (D.T @ D)

    # Each view picks its own rotation, so search them independently and add
    # the right-hand sides of the winning combination.  One pass is enough
    # because the profile that maximises the sum is driven by the same normal
    # equations; a second pass with the joint solution refines the choice.
    rhs = np.zeros(n_bins)
    chosen: list[int] = []
    for Ac, T in centred:
        coefs = np.linalg.solve(gram, Ac.T @ T)
        pred = Ac @ coefs
        denom = np.linalg.norm(pred, axis=0) * np.linalg.norm(T, axis=0)
        corr = np.divide((pred * T).sum(axis=0), denom,
                         out=np.zeros(T.shape[1]), where=denom > 0)
        s = int(np.argmax(corr))
        chosen.append(s)
        rhs += Ac.T @ T[:, s]

    profile = np.linalg.solve(gram, rhs)
    if profile.min() < 0.0:
        # Negative radiance is not physical; redo the winning combination
        # under a non-negativity constraint.
        from scipy.optimize import nnls

        penalties = [np.sqrt(ridge * scale) * np.eye(n_bins)]
        if smooth > 0.0:
            penalties.append(np.sqrt(smooth * scale) * _smoothness_operator(n_bins))
        stacked_A = np.vstack([Ac for Ac, _ in centred] + penalties)
        stacked_t = np.concatenate(
            [T[:, s] for (_, T), s in zip(centred, chosen)]
            + [np.zeros(len(pen)) for pen in penalties])
        profile, _ = nnls(stacked_A, stacked_t)

    per_view = []
    for (Ac, T), s in zip(centred, chosen):
        pred = Ac @ profile
        d = np.linalg.norm(pred) * np.linalg.norm(T[:, s])
        corr = float(pred @ T[:, s] / d) if d > 0 else 0.0
        per_view.append(0.5 * (corr + 1.0))

    return profile, per_view, chosen


@dataclass
class InverseProblem:
    """Score a candidate cut by how well it can be made to match the photos."""

    views: list[ViewData]
    material: Material = DIAMOND
    n_bins: int = 36
    n_r: int = 64
    n_theta: int = 256
    max_angle: float = 90.0
    ridge: float = 2e-3
    smooth: float = 0.05
    n_fold: int | None = None
    """Stone fold.  ``None`` reads it off the views so a 16-main job cannot
    silently prepare 8-fold rotation shifts."""
    min_return: float = 0.55
    """Light return below which a stone stops counting as finished.

    Pattern similarity is measured on contrast-normalised images, so a cut
    that leaks nearly everything can still have its remaining light in roughly
    the right places and score well.  Optimising on the pattern alone duly
    finds exactly that -- a shape that draws the reference while being, in the
    hand, a dead piece of glass.  Light return has to be part of the score."""

    return_weight: float = 0.45
    """How much of the score is the stone actually being alive."""

    psf_sigma: float = 2.5
    """Camera blur, in pixels.  Fitted once against the reference; see
    :func:`apply_psf` for why it matters so much."""
    max_bounces: int = 14

    def __post_init__(self) -> None:
        if self.n_fold is None:
            self.n_fold = int(self.views[0].n_fold) if self.views else 8
        for v in self.views:
            v.prepare(int(self.n_fold))

    def score(self, params: CutParams) -> tuple[float, list[float], np.ndarray]:
        """Returns ``(mean similarity, per view, profile)``.

        Invalid geometry scores zero rather than raising, so the optimiser can
        walk right up to the edge of the feasible region without special cases.
        """
        if params.validate():
            return 0.0, [0.0] * len(self.views), np.zeros(self.n_bins)
        mesh = build_stone(params)
        if not mesh.is_watertight():
            return 0.0, [0.0] * len(self.views), np.zeros(self.n_bins)

        geom = Geometry(mesh)
        blocks, returns, predicted = [], [], []
        for v in self.views:
            axis = (np.array([0.0, 0.0, 1.0]) if v.view == "crown"
                    else np.array([0.0, 0.0, -1.0]))
            origin, direction = _polar_rays(self.n_r, self.n_theta, v.view)
            ev = trace(geom, origin, direction, self.material,
                       max_bounces=self.max_bounces)

            edges = np.linspace(0.0, self.max_angle, self.n_bins + 1)
            towards = ev.direction @ axis
            bin_of = np.clip(np.digitize(
                np.degrees(np.arccos(np.clip(towards, -1.0, 1.0))), edges) - 1,
                0, self.n_bins - 1)
            A = np.zeros((self.n_r * self.n_theta, self.n_bins))
            np.add.at(A, (ev.ray, bin_of), ev.weight)
            A = apply_psf(A, self.n_r, self.n_theta, self.psf_sigma)
            blocks.append((A, v.shifts))
            predicted.append(A)

            back = ev.entered & (towards > 0)
            returns.append(float(ev.weight[back].sum() / max(ev.n_rays, 1)))

        profile, per_view, _ = solve_profile(blocks, self.ridge, self.smooth)
        pattern = float(np.mean(per_view))

        fold = int(params.n_base)
        sym = [
            symmetry_error((A @ profile).reshape(self.n_r, self.n_theta), fold)
            for A in predicted
        ]
        pattern = pattern * (1.0 - 0.35 * float(np.mean(sym)))

        # Fade the score out as the stone goes dark rather than cutting it
        # off, so the optimiser keeps a gradient to climb back up.
        alive = float(np.clip(np.mean(returns) / self.min_return, 0.0, 1.0))
        combined = ((1.0 - self.return_weight) * pattern
                    + self.return_weight * alive * pattern)
        return combined, per_view, profile

    def environment(self, profile: np.ndarray, view_index: int = 0) -> ProfileEnvironment:
        """Turn a solved profile into a renderable instrument.

        Brightness comes from the solve; colour is taken from the photograph,
        by asking what hue it shows wherever it is that bright.
        """
        edges = np.linspace(0.0, self.max_angle, self.n_bins + 1)
        rgb = self.views[view_index].rgb.reshape(-1, 3)
        lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
        order = np.argsort(lum)
        ranked = rgb[order]

        level = profile / max(profile.max(), 1e-9)
        span = max(1, len(ranked) // 150)
        idx = np.clip((level * (len(ranked) - 1)).astype(int), 0, len(ranked) - 1)
        values = np.stack([ranked[max(0, i - span):i + span + 1].mean(axis=0)
                           for i in idx])
        return ProfileEnvironment(
            edges=np.concatenate([edges, [180.0]]),
            values=np.vstack([values, values[-1:]]),
            name="solved",
        )


def _stone_mask(img, tol: float = 0.12) -> np.ndarray:
    """Which pixels are stone, whatever colour the background happens to be.

    Brightness alone does not answer this.  A photograph of a stone on black
    velvet is bright where the stone is, but a rendered or vector reference --
    which is what a client hands over as often as a photograph -- is usually a
    coloured disc on *white*, and thresholding on brightness then calls the
    entire canvas stone.  That is not a small error: the girdle radius comes
    straight out of this mask, and on a 3000px white-background reference it
    came out 8% too large, so every ring was compared against the wrong ring
    and the search was fitting the margin as if it were part of the gem.

    The corners are background in either convention, so the background colour
    is read off them and anything far enough from it is the stone.
    """
    corners = np.stack([img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]])
    background = np.median(corners, axis=0)
    return np.linalg.norm(img - background, axis=2) > tol


def _frame(img, sl, centre, radius, n_r, n_theta) -> ViewData:
    from PIL import Image

    cy, cx = centre
    pad = int(round(radius * POLAR_EXTENT / 0.98))
    crop = Image.fromarray((img[:, sl] * 255).astype("uint8")).crop(
        (int(cx - pad), int(cy - pad), int(cx + pad), int(cy + pad))
    ).resize((420, 420), Image.LANCZOS)
    rgb = np.asarray(crop, dtype=np.float64) / 255.0
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    polar = to_polar(lum, radius=min(lum.shape) / 2.0 * 0.98, n_r=n_r, n_theta=n_theta)
    data = ViewData(view="", target=_normalise(polar), rgb=rgb)
    return data


def _align_framed(data: ViewData, n_fold: int) -> ViewData:
    """Put the framed photograph on the stone's fold before it is scored."""
    polar, rgb, src = align_view_to_n(data.target, data.rgb, n_fold)
    data.target = _normalise(polar)
    data.rgb = rgb
    data.source_fold = int(src)
    data.n_fold = int(n_fold)
    return data


def load_single(photo, view: str = "crown", n_r: int = 64, n_theta: int = 256,
                n_fold: int = 8, radius_scale: float | dict[str, float] | None = None,
                scale_range: tuple[float, float] = (0.96, 1.06),
                scale_steps: int = 11) -> list[ViewData]:
    """One photograph of one side of the stone.

    The side-by-side pair was a property of the first reference we were given,
    not of the problem: a client with a single photo of a single face has
    nothing to split.  This frames that one picture the same way -- find the
    stone, fit the girdle radius, resample to polar -- and hands back a list of
    one, so every scorer downstream works unchanged.

    ``view`` says which face it is, because the tracer has to know which way to
    point the camera.  There is no way to tell from the picture alone.
    """
    from PIL import Image

    img = np.asarray(Image.open(photo).convert("RGB"), dtype=np.float64) / 255.0
    mask = _stone_mask(img)
    ys, xs = np.nonzero(mask.any(axis=1))[0], np.nonzero(mask.any(axis=0))[0]
    if len(ys) == 0 or len(xs) == 0:
        raise ValueError("no stone found: the image looks uniformly dark")
    centre = ((ys[0] + ys[-1]) / 2.0, (xs[0] + xs[-1]) / 2.0)
    radius = max(ys[-1] - ys[0], xs[-1] - xs[0]) / 2.0
    geometry = {view: (slice(None), centre, radius)}

    if radius_scale is None:
        scales = _fit_radius_scales(img, geometry, n_r, n_theta,
                                    scale_range, scale_steps, n_fold)
    elif isinstance(radius_scale, dict):
        scales = radius_scale
    else:
        scales = {view: float(radius_scale)}

    data = _frame(img, slice(None), centre, radius * scales[view], n_r, n_theta)
    data.view = view
    data.radius_scale = scales[view]
    data = _align_framed(data, n_fold)
    data.prepare(n_fold)
    return [data]


def load_reference(photo, single: bool = False, view: str = "crown", **kwargs):
    """Load a reference either way, so callers do not branch.

    ``single`` picks between one framed photograph and the side-by-side pair.
    """
    if single:
        return load_single(photo, view=view, **kwargs)
    return load_views(photo, **kwargs)


def load_views(photo, n_r: int = 64, n_theta: int = 256,
               n_fold: int = 8,
               radius_scale: float | dict[str, float] | None = None,
               scale_range: tuple[float, float] = (0.96, 1.06),
               scale_steps: int = 11) -> list[ViewData]:
    """Split the client's side-by-side photograph into two matchable views.

    Finding where the stone *ends* matters more than it looks.  The pattern is
    compared ring by ring against the simulation, so if the girdle is assumed
    to sit at the wrong radius every ring lines up against the wrong one and
    the score collapses -- on this photograph a 40% radius error was costing
    about 0.12 of similarity, more than any amount of geometry tuning could
    win back.

    The radius is therefore fitted rather than thresholded -- the rim of the
    stone is dark in the viewer, so brightness alone finds the wrong edge --
    but only within a narrow band around the measured outline.  Left free the
    fit runs away to 1.6, which simply pads the photograph with black until
    our whole stone is being compared against the middle of theirs; it scores
    better and means nothing.  Radial disagreement is the geometry's job to
    fix, not the cropping's.  Pass ``radius_scale`` to skip the search.
    """
    from PIL import Image

    img = np.asarray(Image.open(photo).convert("RGB"), dtype=np.float64) / 255.0
    mask = _stone_mask(img)
    mid = mask.shape[1] // 2
    split = int(np.argmin(mask.mean(axis=0)[mid - 90:mid + 90])) + mid - 90

    slices = {"pavilion": slice(0, split), "crown": slice(split, mask.shape[1])}
    geometry: dict[str, tuple] = {}
    for view, sl in slices.items():
        sub = mask[:, sl]
        ys, xs = np.nonzero(sub.any(axis=1))[0], np.nonzero(sub.any(axis=0))[0]
        centre = ((ys[0] + ys[-1]) / 2.0, (xs[0] + xs[-1]) / 2.0)
        radius = max(ys[-1] - ys[0], xs[-1] - xs[0]) / 2.0
        geometry[view] = (sl, centre, radius)

    if radius_scale is None:
        scales = _fit_radius_scales(img, geometry, n_r, n_theta,
                                    scale_range, scale_steps, n_fold)
    elif isinstance(radius_scale, dict):
        scales = radius_scale
    else:
        scales = {view: float(radius_scale) for view in slices}

    out = []
    for view, (sl, centre, radius) in geometry.items():
        data = _frame(img, sl, centre, radius * scales[view], n_r, n_theta)
        data.view = view
        data = _align_framed(data, n_fold)
        data.prepare(n_fold)
        data.radius_scale = scales[view]
        out.append(data)
    return out


def _fit_radius_scales(img, geometry, n_r, n_theta, scale_range, steps, n_fold=8):
    """Pick the framing that lets a standard brilliant match the photo best.

    The instrument is solved for at every candidate scale, so a bad framing
    cannot hide behind a flattering instrument -- each scale is judged at its
    own best case.
    """
    from .geometry import CutParams

    geom = Geometry(build_stone(CutParams()))
    scales = {}
    for view, (sl, centre, radius) in geometry.items():
        A = design_matrix(geom, view, DIAMOND, 36, n_r, n_theta)
        best = (None, -1.0)
        for scale in np.linspace(scale_range[0], scale_range[1], steps):
            data = _frame(img, sl, centre, radius * scale, n_r, n_theta)
            data.prepare(n_fold)
            _, per_view, _ = solve_profile([(A, data.shifts)], ridge=2e-3, smooth=0.1)
            if per_view[0] > best[1]:
                best = (float(scale), per_view[0])
        scales[view] = best[0]
    return scales


@dataclass
class FixedInstrumentProblem:
    """Score a cut through an instrument that is already known.

    Once the viewer has been recovered from the reference photograph it must
    stay put.  It is one physical device: letting it be re-solved for every
    new stone would let a bad cut hide behind a flattering instrument, and the
    resulting "match" would not survive contact with the client's bench.

    This is the honest scorer for the real question -- given that viewer and a
    different material, which cut comes closest to the same picture?
    """

    views: list[ViewData]
    profile: np.ndarray
    material: Material
    n_bins: int = 36
    n_r: int = 64
    n_theta: int = 256
    max_angle: float = 90.0
    max_bounces: int = 14
    psf_sigma: float = 2.5
    """Camera blur, in pixels; must match whatever stage one was solved with."""

    n_fold: int | None = None
    """Stone fold; ``None`` reads it off the views."""

    min_return: float = 0.55
    """Light return below which a cut stops counting as a finished stone.

    Pattern similarity is measured on contrast-normalised images, so a stone
    that leaks almost everything can still have its faint remaining light in
    roughly the right places and score well.  Optimising on the pattern alone
    duly produces exactly that: a cut that reproduces the *shape* of the
    reference while being, physically, a piece of dead glass you can see
    through.  Light return has to be part of what "matches" means.
    """

    return_weight: float = 0.45
    """How much of the score is the stone actually being alive."""

    def __post_init__(self) -> None:
        if self.n_fold is None:
            self.n_fold = int(self.views[0].n_fold) if self.views else 8
        for v in self.views:
            v.prepare(int(self.n_fold))

    def _pattern(self, A: np.ndarray, view: ViewData) -> float:
        pred = A @ self.profile
        pred = pred - pred.mean()
        norm = np.linalg.norm(pred)
        if norm <= 0:
            return 0.5
        corr = (view.shifts.T @ pred) / (norm * np.linalg.norm(view.shifts, axis=0))
        return float(0.5 * (np.nanmax(corr) + 1.0))

    def score(self, params: CutParams) -> tuple[float, list[float], list[float]]:
        """Returns ``(combined score, pattern per view, light return per view)``."""
        blank = [0.0] * len(self.views)
        if params.validate():
            return 0.0, blank, blank
        mesh = build_stone(params)
        if not mesh.is_watertight():
            return 0.0, blank, blank

        geom = Geometry(mesh)
        patterns, returns = [], []
        for v in self.views:
            axis = (np.array([0.0, 0.0, 1.0]) if v.view == "crown"
                    else np.array([0.0, 0.0, -1.0]))
            origin, direction = _polar_rays(self.n_r, self.n_theta, v.view)
            ev = trace(geom, origin, direction, self.material,
                       max_bounces=self.max_bounces)

            edges = np.linspace(0.0, self.max_angle, self.n_bins + 1)
            towards = ev.direction @ axis
            angle = np.degrees(np.arccos(np.clip(towards, -1.0, 1.0)))
            bin_of = np.clip(np.digitize(angle, edges) - 1, 0, self.n_bins - 1)
            A = np.zeros((self.n_r * self.n_theta, self.n_bins))
            np.add.at(A, (ev.ray, bin_of), ev.weight)
            A = apply_psf(A, self.n_r, self.n_theta, self.psf_sigma)

            patterns.append(self._pattern(A, v))
            back = ev.entered & (towards > 0)
            returns.append(float(ev.weight[back].sum() / max(ev.n_rays, 1)))

        pattern = float(np.mean(patterns))
        # Fade the score out as the stone goes dark, rather than cutting it
        # off, so the optimiser still sees a gradient to climb back up.
        alive = float(np.clip(np.mean(returns) / self.min_return, 0.0, 1.0))
        combined = (1.0 - self.return_weight) * pattern + self.return_weight * alive * pattern
        return combined, patterns, returns
