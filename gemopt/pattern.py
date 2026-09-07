"""The N-fold radial layout — one source of truth for PNG and MaxScript.

Changing ``N`` here (or ``CutParams.n_base``) must rebuild the whole pattern:
one sector of ``360/N`` degrees, copied around the centre, with line weight
and halo spacing scaled so N=16 is denser, not busier.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

__all__ = [
    "REFERENCE_N",
    "PatternLayout",
    "layout_from_n",
    "sector_degrees",
    "unit_wedge",
    "radial_array",
]


REFERENCE_N = 8
"""The N the visual weights were tuned against (classic brilliant)."""


def sector_degrees(n: int) -> float:
    n = int(n)
    if n < 3:
        raise ValueError("N must be at least 3")
    return 360.0 / n


def _valley_ratio_for_n(n: int, ref_ratio: float = 0.34, ref_n: int = REFERENCE_N) -> float:
    """Keep each star point's length/width the same as the N=8 drawing.

    A fixed waist radius makes N=16 a bundle of needles: the sector is half
    as wide but the point is just as long.  Matching the aspect ratio pushes
    the waist out as N grows, so the 16-fold picture is a denser compass
    rose, not a thinner copy of the 8-fold one.
    """
    k = (1.0 - ref_ratio) / (2.0 * ref_ratio * math.sin(math.pi / ref_n))
    return 1.0 / (1.0 + 2.0 * k * math.sin(math.pi / n))


@dataclass(frozen=True)
class PatternLayout:
    """Every derived number the PNG and the MaxScript share."""

    n: int
    sector_deg: float
    rotation_offset_deg: float
    """Azimuth of facet 0.  Zero puts a vertex on +X, matching ``geometry.ring``."""

    radius: float
    """Outer tip / N-gon vertex, in model units (girdle radius = 1)."""

    valley_radius: float
    """Inner 'waist' of each kite, as a fraction of ``radius``."""

    stroke_width: float
    """Line weight in model units.  Shrinks as 8/N so lines do not pile up."""

    halo_count: int
    """Fixed number of faint concentric rings — does not grow with N."""

    halo_inner: float
    halo_outer: float
    halo_stroke: float

    n_gon_stroke: float

    def to_dict(self) -> dict:
        return asdict(self)


def layout_from_n(
    n: int,
    *,
    radius: float = 1.0,
    rotation_offset_deg: float | None = None,
    valley_ratio: float | None = None,
    base_stroke: float = 0.020,
    halo_count: int = 8,
) -> PatternLayout:
    """Scale every visual weight off ``N`` against the N=8 reference.

    Waist radius grows with N so each point keeps the same length/width as
    the 8-fold drawing.  Line weight still shrinks as 8/N.
    """
    n = int(n)
    scale = REFERENCE_N / n
    r = float(radius)
    if valley_ratio is None:
        valley_ratio = _valley_ratio_for_n(n)
    # Vertex-up: the crown drawing the client compares against has a tip at
    # the top of the frame, not on +X.
    if rotation_offset_deg is None:
        rotation_offset_deg = 90.0
    # Halo must stay inside the drawing: scale is ~0.42 of the canvas, so
    # anything past ~1.15 radii clips and reads as a dark ring.
    return PatternLayout(
        n=n,
        sector_deg=sector_degrees(n),
        rotation_offset_deg=float(rotation_offset_deg),
        radius=r,
        valley_radius=r * float(valley_ratio),
        stroke_width=max(0.004, base_stroke * scale),
        halo_count=int(halo_count),
        halo_inner=0.16 * r,
        halo_outer=1.12 * r,
        halo_stroke=max(0.0020, 0.0045 * scale),
        n_gon_stroke=max(0.006, 0.014 * scale),
    )


@dataclass(frozen=True)
class Wedge:
    """One sector: two kite halves that tile the plane with no gap or overlap."""

    index: int
    tip: tuple[float, float]
    valley_left: tuple[float, float]
    valley_right: tuple[float, float]
    origin: tuple[float, float]
    left_poly: tuple[tuple[float, float], ...]
    right_poly: tuple[tuple[float, float], ...]


def _xy(radius: float, deg: float) -> tuple[float, float]:
    a = math.radians(deg)
    return (radius * math.cos(a), radius * math.sin(a))


def unit_wedge(layout: PatternLayout, index: int = 0) -> Wedge:
    """The unit cell for facet ``index``.

    The kite tip sits on vertex ``index`` of the outer N-gon.  The two
    valleys sit on the half-azimuths, so neighbouring wedges share those
    points and neither overlap nor leave a gap.
    """
    step = layout.sector_deg
    tip_deg = layout.rotation_offset_deg + index * step
    left_deg = tip_deg - 0.5 * step
    right_deg = tip_deg + 0.5 * step
    origin = (0.0, 0.0)
    tip = _xy(layout.radius, tip_deg)
    valley_left = _xy(layout.valley_radius, left_deg)
    valley_right = _xy(layout.valley_radius, right_deg)
    return Wedge(
        index=index,
        tip=tip,
        valley_left=valley_left,
        valley_right=valley_right,
        origin=origin,
        left_poly=(origin, valley_left, tip),
        right_poly=(origin, tip, valley_right),
    )


def radial_array(layout: PatternLayout) -> list[Wedge]:
    """Copy the unit wedge around the centre, ``i * 360/N`` each time."""
    return [unit_wedge(layout, i) for i in range(layout.n)]


def n_gon_vertices(layout: PatternLayout) -> list[tuple[float, float]]:
    return [_xy(layout.radius, layout.rotation_offset_deg + i * layout.sector_deg)
            for i in range(layout.n)]


def halo_radii(layout: PatternLayout) -> list[float]:
    if layout.halo_count <= 1:
        return [layout.halo_inner]
    span = layout.halo_outer - layout.halo_inner
    return [
        layout.halo_inner + span * k / (layout.halo_count - 1)
        for k in range(layout.halo_count)
    ]
