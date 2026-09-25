"""The N-fold radial layout — one source of truth for PNG and MaxScript.

Every N uses the same crown layers as N=8, only copied more times:

* a dark table N-gon,
* N beige spokes that start on the table vertices (not the origin),
* N outer kites,
* terracotta fillers between neighbouring tips.

Changing N must not turn those spokes into needles from the centre.
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
    "table_vertices",
]


REFERENCE_N = 8
"""The N the visual weights were tuned against (classic brilliant)."""

TABLE_RATIO = 0.32
"""Table radius / outer radius.  Large enough that an N=16 table still reads
as a polygon, not a point the spokes collapse into."""

VALLEY_RATIO = 0.55
"""Spoke waist / outer radius.  Fixed so each sector is the same drawing."""


def sector_degrees(n: int) -> float:
    n = int(n)
    if n < 3:
        raise ValueError("N must be at least 3")
    return 360.0 / n


@dataclass(frozen=True)
class PatternLayout:
    """Every derived number the PNG and the MaxScript share."""

    n: int
    sector_deg: float
    rotation_offset_deg: float
    """Azimuth of facet 0.  Zero puts a vertex on +X, matching ``geometry.ring``."""

    radius: float
    """Outer tip / N-gon vertex, in model units (girdle radius = 1)."""

    table_radius: float
    """Dark central N-gon.  Same fraction of ``radius`` at every N."""

    valley_radius: float
    """Waist of each spoke, on the half-azimuths."""

    stroke_width: float
    """Line weight in model units.  Shrinks as 8/N so lines do not pile up."""

    halo_count: int
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
    table_ratio: float | None = None,
    base_stroke: float = 0.018,
    halo_count: int = 6,
) -> PatternLayout:
    """Same radial fractions as N=8; only the sector angle and stroke change."""
    n = int(n)
    scale = REFERENCE_N / n
    r = float(radius)
    if table_ratio is None:
        table_ratio = TABLE_RATIO
    if valley_ratio is None:
        valley_ratio = VALLEY_RATIO
    if rotation_offset_deg is None:
        rotation_offset_deg = 90.0
    return PatternLayout(
        n=n,
        sector_deg=sector_degrees(n),
        rotation_offset_deg=float(rotation_offset_deg),
        radius=r,
        table_radius=r * float(table_ratio),
        valley_radius=r * float(valley_ratio),
        stroke_width=max(0.008, base_stroke * math.sqrt(scale)),
        halo_count=int(halo_count),
        halo_inner=0.14 * r,
        halo_outer=1.10 * r,
        halo_stroke=max(0.0025, 0.0040 * math.sqrt(scale)),
        n_gon_stroke=max(0.008, 0.012 * math.sqrt(scale)),
    )


@dataclass(frozen=True)
class Wedge:
    """One sector of the crown map: table vertex, two spoke halves, outer tip."""

    index: int
    tip: tuple[float, float]
    table: tuple[float, float]
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

    Spokes start on the table N-gon, not at the origin.  That is what keeps
    N=16 looking like N=8 with more copies: a table, then N kites.
    """
    step = layout.sector_deg
    tip_deg = layout.rotation_offset_deg + index * step
    left_deg = tip_deg - 0.5 * step
    right_deg = tip_deg + 0.5 * step
    origin = (0.0, 0.0)
    table = _xy(layout.table_radius, tip_deg)
    tip = _xy(layout.radius, tip_deg)
    valley_left = _xy(layout.valley_radius, left_deg)
    valley_right = _xy(layout.valley_radius, right_deg)
    return Wedge(
        index=index,
        tip=tip,
        table=table,
        valley_left=valley_left,
        valley_right=valley_right,
        origin=origin,
        left_poly=(table, valley_left, tip),
        right_poly=(table, tip, valley_right),
    )


def radial_array(layout: PatternLayout) -> list[Wedge]:
    """Copy the unit wedge around the centre, ``i * 360/N`` each time."""
    return [unit_wedge(layout, i) for i in range(layout.n)]


def n_gon_vertices(layout: PatternLayout) -> list[tuple[float, float]]:
    return [_xy(layout.radius, layout.rotation_offset_deg + i * layout.sector_deg)
            for i in range(layout.n)]


def table_vertices(layout: PatternLayout) -> list[tuple[float, float]]:
    return [_xy(layout.table_radius,
                layout.rotation_offset_deg + i * layout.sector_deg)
            for i in range(layout.n)]


def halo_radii(layout: PatternLayout) -> list[float]:
    if layout.halo_count <= 1:
        return [layout.halo_inner]
    span = layout.halo_outer - layout.halo_inner
    return [
        layout.halo_inner + span * k / (layout.halo_count - 1)
        for k in range(layout.halo_count)
    ]
