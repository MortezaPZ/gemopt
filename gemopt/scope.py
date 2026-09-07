"""Virtual viewing instruments.

A hearts-and-arrows pattern is not something you see under ordinary light.
It only appears inside a specific gadget: the stone sits at the narrow end of
a cone whose inside is coloured, the observer looks straight down the axis,
and the observer's own head blocks a disc of sky right around that axis.  The
pattern *is* the map of which parts of the sky each facet is looking at.

So instead of rendering the stone we ask, for every pixel, "where does this
line of sight end up after bouncing around inside the stone?", and colour the
pixel by whichever zone of the instrument it lands in.  That is one backwards
ray per pixel and no light transport simulation at all -- microseconds
instead of the minutes V-Ray needs, and it is the same information.

Two instruments are modelled:

``ScopeEnvironment``
    concentric bands of colour seen from the viewing axis.  Presets cover the
    red-cone hearts-and-arrows viewer and the IdealScope leakage tester.

``IESEnvironment``
    a real luminaire described by a photometric file, so the simulation can
    be lit by the same lamp as the 3ds Max scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import Mesh
from .ies import IESProfile
from .optics import Geometry, Material, trace

__all__ = [
    "Camera",
    "Environment",
    "ScopeEnvironment",
    "ProfileEnvironment",
    "IESEnvironment",
    "HEARTS_ARROWS_SCOPE",
    "IDEALSCOPE",
    "render",
    "RenderResult",
    "save_png",
]


# --------------------------------------------------------------------------
# camera
# --------------------------------------------------------------------------


@dataclass
class Camera:
    """An orthographic view straight down the stone's axis.

    Gem viewers are effectively telecentric -- the stone is small and the
    instrument looks at it from far away -- so an orthographic camera is not
    an approximation here, it is the right model.
    """

    resolution: int = 192
    extent: float = 1.02
    """Half-width of the image in girdle radii; slightly over 1 leaves a rim."""

    view: str = "crown"
    """``"crown"`` looks down at the table (arrows), ``"pavilion"`` looks up
    from underneath (hearts)."""

    samples: int = 1
    """Jittered samples per pixel.  1 is fine while searching, 4+ for finals."""

    seed: int = 0

    @property
    def axis(self) -> np.ndarray:
        """Direction the light travels to reach the observer."""
        return np.array([0.0, 0.0, 1.0]) if self.view == "crown" else np.array([0.0, 0.0, -1.0])

    def rays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(origins, directions, pixel_index)``."""
        n = self.resolution
        lin = (np.arange(n) + 0.5) / n * 2.0 - 1.0
        gx, gy = np.meshgrid(lin * self.extent, lin * self.extent, indexing="xy")
        px = np.arange(n * n)

        xs = [gx.ravel()]
        ys = [gy.ravel()]
        idx = [px]
        if self.samples > 1:
            rng = np.random.default_rng(self.seed)
            step = 2.0 * self.extent / n
            for _ in range(self.samples - 1):
                jitter = rng.uniform(-0.5, 0.5, size=(2, n * n)) * step
                xs.append(gx.ravel() + jitter[0])
                ys.append(gy.ravel() + jitter[1])
                idx.append(px)

        x = np.concatenate(xs)
        y = np.concatenate(ys)
        pixel = np.concatenate(idx)

        start_z = 5.0 if self.view == "crown" else -5.0
        origin = np.column_stack([x, y, np.full(len(x), start_z)])
        step_dir = np.array([0.0, 0.0, -1.0]) if self.view == "crown" else np.array([0.0, 0.0, 1.0])
        direction = np.tile(step_dir, (len(x), 1))
        return origin, direction, pixel


# --------------------------------------------------------------------------
# environments
# --------------------------------------------------------------------------


class Environment:
    """Maps an outgoing direction to the colour seen from it."""

    def radiance(self, directions: np.ndarray, axis: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class ScopeEnvironment(Environment):
    """Concentric coloured bands around the viewing axis.

    ``bands`` is a list of ``(max_angle_deg, (r, g, b))`` sorted by angle; a
    direction is coloured by the first band whose angle it falls inside.  The
    band that reaches 180 degrees catches everything leaving downwards, which
    is light leaking straight through the stone.
    """

    bands: list[tuple[float, tuple[float, float, float]]]
    name: str = "scope"

    def radiance(self, directions: np.ndarray, axis: np.ndarray) -> np.ndarray:
        cos_a = np.clip(directions @ axis, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_a))
        out = np.zeros((len(directions), 3))
        assigned = np.zeros(len(directions), dtype=bool)
        for limit, colour in self.bands:
            take = (~assigned) & (angle <= limit)
            out[take] = colour
            assigned |= take
        return out


HEARTS_ARROWS_SCOPE = ScopeEnvironment(
    name="hearts_arrows",
    bands=[
        # straight overhead the observer sees the bright open end of the cone:
        # facets aimed here are the ones that draw the pale arrows and hearts
        (26.0, (1.00, 0.97, 0.90)),
        # the body of the cone is red, and it fills most of the field
        (72.0, (0.82, 0.10, 0.10)),
        # a dark rim right at the edge sharpens the outline
        (90.0, (0.05, 0.05, 0.05)),
        # anything heading downwards has leaked through the pavilion
        (180.0, (0.10, 0.10, 0.12)),
    ],
)
"""Matches the red-cone viewer the client's reference photograph came from."""

IDEALSCOPE = ScopeEnvironment(
    name="idealscope",
    bands=[
        # The observer's head blocks the middle -- about 25 degrees for a
        # stone held at reading distance.  Light coming back inside that cone
        # has nowhere bright to come from, which is the contrast that draws
        # the arrows.
        (25.0, (0.04, 0.04, 0.04)),
        # the useful light, returned from a wide angle
        (75.0, (0.85, 0.09, 0.09)),
        (90.0, (0.55, 0.10, 0.10)),
        # leakage reads white, which is exactly how the real tool shows it
        (180.0, (1.00, 1.00, 1.00)),
    ],
)
"""The standard leakage tester: red good, black contrast, white leaking."""


@dataclass
class ProfileEnvironment(Environment):
    """An instrument described by a measured brightness curve.

    Three flat bands are a caricature of a real viewer: the cone is lit
    unevenly, its walls fall off towards the apex, and the aperture has a soft
    edge.  This version carries one RGB value per angular bin instead, which
    is enough freedom to reproduce an actual photograph -- and the values are
    recovered from that photograph rather than guessed.  See
    :func:`gemopt.calibrate.fit_radiance_profile`.
    """

    edges: np.ndarray
    """(K+1,) bin edges in degrees from the viewing axis, ascending to 180."""

    values: np.ndarray
    """(K, 3) radiance in each bin."""

    name: str = "fitted"

    def radiance(self, directions: np.ndarray, axis: np.ndarray) -> np.ndarray:
        cos_a = np.clip(directions @ axis, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_a))
        idx = np.clip(np.searchsorted(self.edges, angle, side="right") - 1,
                      0, len(self.values) - 1)
        return self.values[idx]


@dataclass
class IESEnvironment(Environment):
    """Lit by a real luminaire read from a photometric file.

    The lamp is treated as far away, so its angular intensity profile becomes
    the brightness of the sky in that direction -- which is what an IES file
    describes and what makes the result comparable to the 3ds Max render.
    """

    profile: IESProfile
    lamp_direction: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0])
    )
    """Direction from the stone towards the lamp."""

    tint: tuple[float, float, float] = (1.0, 0.98, 0.94)
    """4000 K white, matching the 840 in the client's luminaire name."""

    ambient: float = 0.04

    def radiance(self, directions: np.ndarray, axis: np.ndarray) -> np.ndarray:
        lamp = self.lamp_direction / np.linalg.norm(self.lamp_direction)
        cos_a = np.clip(directions @ lamp, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_a))
        level = self.profile.normalised(angle) + self.ambient
        return level[:, None] * np.asarray(self.tint)[None, :]


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


@dataclass
class RenderResult:
    image: np.ndarray          # (H, W, 3) float, 0..1 before tone mapping
    coverage: np.ndarray       # (H, W)    fraction of the pixel on the stone
    light_return: float        # share of incident energy sent back to viewer
    leakage: float             # share leaving the far side
    surface: float             # share reflected straight off the surface
    unresolved: float          # share still bouncing when tracing stopped

    @property
    def gray(self) -> np.ndarray:
        return self.image @ np.array([0.2126, 0.7152, 0.0722])


def render(
    mesh: Mesh,
    material: Material,
    camera: Camera | None = None,
    environment: Environment | None = None,
    max_bounces: int = 14,
    dispersion: bool = False,
    geometry: Geometry | None = None,
) -> RenderResult:
    """Backwards-trace one image and collect the energy statistics with it.

    Setting ``dispersion`` traces red, green and blue separately so the
    refractive index differs per channel; that is what shows up as fire, and
    it costs three times as much.
    """
    camera = camera or Camera()
    environment = environment or HEARTS_ARROWS_SCOPE
    geom = geometry or Geometry(mesh)

    origin, direction, pixel = camera.rays()
    axis = camera.axis
    n_pixels = camera.resolution**2
    per_pixel = len(origin) / n_pixels

    channels = (
        [(610.0, 0), (550.0, 1), (465.0, 2)]
        if dispersion and material.dispersion > 0.0
        else [(589.3, None)]
    )

    image = np.zeros((n_pixels, 3))
    totals = {"return": 0.0, "leak": 0.0, "surface": 0.0, "lost": 0.0}

    # Silhouette: which samples land on the stone at all.  Doing it with one
    # extra intersection test is far cheaper than inferring it from the trace
    # and gives a clean mask for the scoring stage.
    _, first_hit = geom.intersect(origin, direction)
    coverage = np.zeros(n_pixels)
    np.add.at(coverage, pixel, (first_hit >= 0).astype(np.float64))
    coverage /= per_pixel

    for wavelength, channel in channels:
        ev = trace(geom, origin, direction, material,
                   wavelength_nm=wavelength, max_bounces=max_bounces)

        colour = environment.radiance(ev.direction, axis)
        contribution = colour * ev.weight[:, None]
        rays_pixel = pixel[ev.ray]

        if channel is None:
            for c in range(3):
                np.add.at(image[:, c], rays_pixel, contribution[:, c])
        else:
            np.add.at(image[:, channel], rays_pixel, contribution[:, channel])

        # energy bookkeeping, independent of the colours above
        towards = ev.direction @ axis
        totals["return"] += float(ev.weight[ev.entered & (towards > 0)].sum())
        totals["leak"] += float(ev.weight[ev.entered & (towards <= 0)].sum())
        totals["surface"] += float(ev.weight[(~ev.entered) & (ev.facet >= 0)].sum())
        totals["lost"] += float(ev.lost.sum())

    image *= 1.0 / per_pixel

    # everything that ever met the stone, used to turn the sums into fractions
    incident = sum(totals.values())
    if incident <= 0.0:
        incident = 1.0

    return RenderResult(
        image=np.clip(image, 0.0, 1.0).reshape(camera.resolution, camera.resolution, 3),
        coverage=np.clip(coverage, 0.0, 1.0).reshape(camera.resolution, camera.resolution),
        light_return=totals["return"] / incident,
        leakage=totals["leak"] / incident,
        surface=totals["surface"] / incident,
        unresolved=totals["lost"] / incident,
    )


def save_png(image: np.ndarray, path, gamma: float = 2.2) -> None:
    """Write a float image, applying the usual display gamma."""
    import io
    from pathlib import Path
    from PIL import Image

    arr = np.nan_to_num(np.asarray(image, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if gamma and gamma != 1.0:
        arr = np.power(arr, 1.0 / gamma)
    buf = io.BytesIO()
    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(buf, format="PNG")
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(buf.getvalue())
