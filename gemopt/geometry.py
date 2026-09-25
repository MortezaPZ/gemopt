"""Parametric faceted-gem geometry.

The model follows the structure the client described:

* everything above the girdle is the *crown*, everything below is the *pavilion*;
* the stone is built from horizontal *rows* (rings) of vertices;
* the number of facets may differ from row to row, but within a row the facets
  must line up with the rows above and below -- i.e. every row count is a
  multiple of one base number ``N`` (8 for a classic round brilliant);
* facet-to-facet symmetry is always preserved: every facet in a row shares the
  same geometry, only rotated around the vertical axis.

Because of that last rule a whole stone is described by a handful of numbers
(a few per row) instead of one number per vertex, which is what makes the
search space small enough to optimise.

The classic 57-facet round brilliant comes out of this scheme exactly:

    crown     = 1 table + N stars + N bezels + mN upper-girdle halves
    pavilion  = mN lower-girdle halves + N pavilion mains
    total     = 3N + 2mN + 1  ->  57 for N = 8, m = 2

All linear dimensions are expressed as a fraction of the girdle *radius*, so
the stone always has radius 1.0 and can be scaled to any physical size at the
end.  Angles are in degrees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Iterable

import numpy as np

from .pattern import sector_degrees

__all__ = [
    "CutParams",
    "Mesh",
    "build_stone",
    "brilliant_for_n",
    "eight_fold_twin",
    "STANDARD_BRILLIANT",
]


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------


@dataclass
class CutParams:
    """Every number the optimiser is allowed to touch.

    Fractions are relative to the girdle radius (which is fixed at 1.0), so
    ``table`` = 0.53 means the table is 53% of the stone's diameter -- the same
    convention gemmologists use.
    """

    # --- structure (discrete, changes the topology) ------------------------
    n_base: int = 8
    """Base symmetry.  Every row has ``n_base`` or ``2 * n_base`` facets."""

    girdle_multiple: int = 2
    """Girdle facets per main facet, on each of the crown and pavilion sides.

    The mains, stars and bezels all come in ``n_base``; the girdle rows come in
    ``girdle_multiple * n_base``.  Two is the classic round brilliant -- eight
    mains, sixteen upper and sixteen lower girdle halves.  Three or four are
    what the trade calls a modified brilliant, and give a busier pattern.

    Keeping it an integer multiple is what preserves the alignment the client
    asked for: every girdle facet still starts and ends on a main facet's
    boundary, so the rows stay in register all the way round.
    """

    extra_crown_rows: int = 0
    """Additional facet tiers between the bezels and the girdle.

    The client's video is him clicking down a column of facets, saying "these
    are all rows, I want to be able to have more of them".  A classic brilliant
    has three on the crown and two on the pavilion and that is all it has.

    Each extra tier gets its *own* angle, interpolated between the main facet
    and ``crown_outer_angle``.  Simply subdividing a row without breaking the
    angle would leave the pieces coplanar -- the same mistake that once turned
    the pavilion into a smooth cone and erased the pattern entirely."""

    extra_pavilion_rows: int = 0
    """Additional facet tiers between the pavilion mains and the girdle."""

    crown_outer_angle: float = 46.0
    """Angle of the outermost crown tier, where the extra rows run out to.

    Only used when ``extra_crown_rows`` is set; the tiers fan between the bezel
    angle and this one."""

    pavilion_outer_angle: float = 44.0
    """Same, for the pavilion side."""

    star_row: bool = True
    """Whether the crown has a star-facet row between table and bezels."""

    upper_girdle_row: bool = True
    """Whether the crown has upper-girdle halves (2N facets)."""

    lower_girdle_row: bool = True
    """Whether the pavilion has lower-girdle halves (2N facets)."""

    culet: float = 0.0
    """Culet radius as a fraction of the girdle radius.  0.0 = pointed."""

    # --- crown geometry (continuous) ---------------------------------------
    table: float = 0.53
    """Table radius / girdle radius."""

    crown_angle: float = 34.5
    """Angle of the bezel (main crown) facets against the girdle plane."""

    star_length: float = 0.55
    """How far the star facets reach from the table edge towards the girdle,
    as a fraction of the remaining distance.  GIA calls this 'star length'."""

    upper_girdle_length: float = 0.80
    """How far the upper-girdle halves reach along the girdle, as a fraction
    of the half-arc between two bezel corners."""

    # --- pavilion geometry --------------------------------------------------
    pavilion_angle: float = 40.75
    """Angle of the pavilion main facets against the girdle plane."""

    lower_girdle_length: float = 0.77
    """Length of the lower-girdle halves, same convention as above."""

    # --- girdle -------------------------------------------------------------
    girdle_thick_main: float = 0.020
    """Girdle thickness at the main (bezel) azimuths -- the thin 'valleys'."""

    girdle_thick_half: float = 0.035
    """Girdle thickness half-way between mains -- the thick 'hills'.
    Ignored when ``flat_girdle`` is set."""

    flat_girdle: bool = True
    """Whether the girdle band is a straight cylinder rather than scalloped.

    A hand-cut brilliant has a wavy girdle -- thin under each bezel, thick
    between -- because that is what falls out of grinding the mains.  The
    client asks for a straight one: it is what a modern machine-cut stone
    looks like, and it is the edge his own models are built against.  With
    this set, ``girdle_thick_half`` is forced to ``girdle_thick_main`` and the
    band has one thickness all the way round."""

    # --- optional extra twist ----------------------------------------------
    crown_twist: float = 0.0
    """Azimuthal offset of the crown against the pavilion, in degrees.
    0 means the crown mains sit exactly above the pavilion mains, which is
    what produces the hearts-and-arrows pattern.  The optimiser is normally
    kept at 0 but it is exposed so misalignment can be studied."""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CutParams":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    # -- derived, reported back to the user in gemmological terms -----------

    def derived(self) -> dict:
        """Numbers a gem cutter recognises, computed from the parameters."""
        crown_h = (1.0 - self.table) * math.tan(math.radians(self.crown_angle))
        pav_d = (1.0 - self.culet) * math.tan(math.radians(self.pavilion_angle))
        girdle = 0.5 * (self.girdle_thick_main + self.girdle_thick_half)
        total = crown_h + girdle + pav_d
        return {
            "table_pct": 100.0 * self.table,
            "crown_height_pct": 50.0 * crown_h,        # % of *diameter*
            "pavilion_depth_pct": 50.0 * pav_d,
            "girdle_thickness_pct": 50.0 * girdle,
            "total_depth_pct": 50.0 * total,
            "crown_angle": self.crown_angle,
            "pavilion_angle": self.pavilion_angle,
            "facet_count": facet_count(self),
            "facet_count_with_girdle": facet_count(self, include_girdle=True),
        }

    # -- validity ------------------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of reasons the stone is not physically sane.

        The optimiser throws away any candidate with a non-empty list, so we
        never waste an evaluation on a shape that could not be cut.
        """
        problems: list[str] = []
        if not (3 <= self.n_base <= 24):
            problems.append("n_base out of range")
        if not (1 <= self.girdle_multiple <= 6):
            problems.append("girdle multiple outside 1..6")
        if not (0 <= self.extra_crown_rows <= 4):
            problems.append("extra crown rows outside 0..4")
        if not (0 <= self.extra_pavilion_rows <= 4):
            problems.append("extra pavilion rows outside 0..4")
        if self.extra_crown_rows and not (
                self.crown_angle < self.crown_outer_angle < 75.0):
            problems.append("crown tiers must fan outward to a steeper angle")
        if self.extra_pavilion_rows and not (
                self.pavilion_angle < self.pavilion_outer_angle < 70.0):
            problems.append("pavilion tiers must fan outward to a steeper angle")
        if not (0.05 < self.table < 0.95):
            problems.append("table outside 5%..95%")
        if not (5.0 < self.crown_angle < 70.0):
            problems.append("crown angle outside 5..70 deg")
        if not (20.0 < self.pavilion_angle < 65.0):
            problems.append("pavilion angle outside 20..65 deg")
        if not (0.0 <= self.star_length <= 1.0):
            problems.append("star length outside 0..1")
        if not (0.0 <= self.upper_girdle_length <= 1.0):
            problems.append("upper girdle length outside 0..1")
        if not (0.0 <= self.lower_girdle_length <= 1.0):
            problems.append("lower girdle length outside 0..1")
        if self.girdle_thick_main <= 0.0 or self.girdle_thick_half <= 0.0:
            problems.append("girdle thickness must stay positive")
        if self.girdle_thick_main > self.girdle_thick_half:
            # the scallop always bulges *between* the mains
            problems.append("girdle thinnest point must be at the mains")
        if not (0.0 <= self.culet < self.table):
            problems.append("culet larger than table")
        d = self.derived()
        if not (30.0 < d["total_depth_pct"] < 95.0):
            problems.append("total depth outside 30%..95%")
        return problems


STANDARD_BRILLIANT = CutParams()
"""Tolkowsky-ish reference: the proportions that give hearts and arrows in
diamond.  Used as a sanity check and as the starting point of a search."""


def brilliant_for_n(n: int, n_d: float | None = None, **overrides) -> CutParams:
    """The same brilliant recipe, with ``n_base`` set to ``n``.

    Elevation angles stay at the Tolkowsky-ish defaults.  A steep shift for
    low-index glass was tried and dumped 75% of the light through the
    pavilion; the search is allowed to walk pavilion from 36 to 54 degrees
    and that is where glass belongs, not in the starting stone.
    ``n_d`` is accepted so callers can pass the material without a second
    code path, and is reserved for a future gentle bias.
    """
    values = STANDARD_BRILLIANT.to_dict()
    values["n_base"] = int(n)
    values.update(overrides)
    return CutParams.from_dict(values)


def eight_fold_twin(params: CutParams) -> CutParams:
    """The same cut with eight mains, so N=16 can be drawn as copies of N=8.

    A 16-main stone traced in a hearts-and-arrows scope is a different
    picture — sixteen needles into the centre, no table.  The still the
    client compares is this twin, copied around the circle.
    """
    values = params.to_dict()
    values["n_base"] = 8
    return CutParams.from_dict(values)


def facet_count(p: CutParams, include_girdle: bool = False) -> int:
    """Number of facets.

    Gem trade convention counts the classic round brilliant as 57 (58 with a
    culet) and leaves the polished girdle band out, so that is the default.
    Pass ``include_girdle=True`` for the count you get by selecting every
    polygon in 3ds Max, which is 2N higher.
    """
    n, m = p.n_base, max(1, int(p.girdle_multiple))
    ec, ep = max(0, int(p.extra_crown_rows)), max(0, int(p.extra_pavilion_rows))

    total = 1 + n                      # table + bezels
    if p.star_row:
        total += n
    if ec:
        total += n                     # the bezel splits in two when tiered
    if p.upper_girdle_row:
        total += m * n
    total += m * n * ec
    total += n                         # pavilion mains
    if ep:
        total += n                     # likewise on the pavilion
    if p.lower_girdle_row:
        total += m * n
    total += m * n * ep
    if p.culet > 0.0:
        total += 1                     # the culet facet itself
    if include_girdle:
        total += m * n
    return total


# --------------------------------------------------------------------------
# mesh
# --------------------------------------------------------------------------


@dataclass
class Mesh:
    """A triangulated stone plus the facet each triangle came from."""

    vertices: np.ndarray                      # (V, 3) float64
    triangles: np.ndarray                     # (T, 3) int32
    facet_id: np.ndarray                      # (T,)   int32
    facet_group: list[str] = field(default_factory=list)
    """Human readable name per facet id, e.g. ``"bezel"``."""

    polygons: list[list[int]] = field(default_factory=list)
    """The facets before triangulation, one vertex loop each.

    Ray tracing wants triangles, but a gem cutter -- and 3ds Max -- wants
    facets.  Carrying both means the stone arrives in Max as 57 polygons the
    client can select and edit, not 110 triangles."""

    # -- basic properties ---------------------------------------------------

    @property
    def n_facets(self) -> int:
        return int(self.facet_id.max()) + 1 if len(self.facet_id) else 0

    def triangle_vertices(self) -> np.ndarray:
        """(T, 3, 3) array of triangle corner coordinates."""
        return self.vertices[self.triangles]

    def normals(self) -> np.ndarray:
        tv = self.triangle_vertices()
        n = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        ln[ln == 0.0] = 1.0
        return n / ln

    def areas(self) -> np.ndarray:
        tv = self.triangle_vertices()
        return 0.5 * np.linalg.norm(
            np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0]), axis=1
        )

    def volume(self) -> float:
        """Signed volume via the divergence theorem; positive for outward normals."""
        tv = self.triangle_vertices()
        return float(np.sum(np.einsum("ij,ij->i", tv[:, 0], np.cross(tv[:, 1], tv[:, 2]))) / 6.0)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def is_watertight(self) -> bool:
        """Every edge must be shared by exactly two triangles."""
        e = np.concatenate(
            [self.triangles[:, [0, 1]], self.triangles[:, [1, 2]], self.triangles[:, [2, 0]]]
        )
        e = np.sort(e, axis=1)
        _, counts = np.unique(e, axis=0, return_counts=True)
        return bool(np.all(counts == 2))

    def scale(self, factor: float) -> "Mesh":
        return Mesh(self.vertices * factor, self.triangles, self.facet_id,
                    list(self.facet_group), [list(p) for p in self.polygons])

    # -- export -------------------------------------------------------------

    def to_obj(self, path, name: str = "stone", triangulate: bool = False) -> None:
        """Wavefront OBJ.  Facets are written as single polygons unless
        ``triangulate`` is set, so the stone opens in 3ds Max with its real
        facet layout intact."""
        from .precision import DECIMALS

        lines = [f"# gemopt export -- {self.n_facets} facets", f"o {name}"]
        for v in self.vertices:
            lines.append(
                f"v {v[0]:.{DECIMALS}f} {v[1]:.{DECIMALS}f} {v[2]:.{DECIMALS}f}")

        if self.polygons and not triangulate:
            for fid, poly in enumerate(self.polygons):
                group = self.facet_group[fid] if fid < len(self.facet_group) else "facet"
                lines.append(f"g {group}_{fid:03d}")
                lines.append("f " + " ".join(str(i + 1) for i in poly))
        else:
            current = -1
            for tri, fid in zip(self.triangles, self.facet_id):
                if fid != current:
                    current = int(fid)
                    group = self.facet_group[current] if current < len(self.facet_group) else "facet"
                    lines.append(f"g {group}_{current:03d}")
                lines.append(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}")

        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


class _Builder:
    """Accumulates vertices and polygons, then triangulates in one go."""

    def __init__(self) -> None:
        self._verts: list[tuple[float, float, float]] = []
        self._polys: list[tuple[list[int], str]] = []

    def vertex(self, x: float, y: float, z: float) -> int:
        self._verts.append((x, y, z))
        return len(self._verts) - 1

    def ring(self, n: int, radius: float, z: float, offset_deg: float = 0.0) -> list[int]:
        """A horizontal ring of ``n`` vertices, returned as vertex indices."""
        off = math.radians(offset_deg)
        return [
            self.vertex(radius * math.cos(off + 2.0 * math.pi * k / n),
                        radius * math.sin(off + 2.0 * math.pi * k / n),
                        z)
            for k in range(n)
        ]

    def facet(self, indices: Iterable[int], group: str) -> None:
        idx = list(indices)
        if len(idx) >= 3:
            self._polys.append((idx, group))

    def finish(self) -> Mesh:
        verts = np.asarray(self._verts, dtype=np.float64)
        tris: list[tuple[int, int, int]] = []
        fids: list[int] = []
        groups: list[str] = []
        for fid, (poly, group) in enumerate(self._polys):
            groups.append(group)
            # fan triangulation -- every facet of a faceted gem is convex and
            # planar by construction, so a fan is exact.
            for i in range(1, len(poly) - 1):
                tris.append((poly[0], poly[i], poly[i + 1]))
                fids.append(fid)
        return Mesh(
            vertices=verts,
            triangles=np.asarray(tris, dtype=np.int32),
            facet_id=np.asarray(fids, dtype=np.int32),
            facet_group=groups,
            polygons=[list(poly) for poly, _ in self._polys],
        )


def _bridge(b: "_Builder", inner: list[int], outer: list[int],
            n: int, m: int, total: int, group: str) -> None:
    """Facet the gap between an ``n``-point ring and an ``m*n``-point one.

    When both rings have the same count the band is a row of quads; when the
    inner one is coarser each of its vertices fans out to the ``m`` segments
    of the sector beneath it.  Either way the facets start and end on a main
    facet's boundary, which is what keeps the rows in register.
    """
    if len(inner) == len(outer):
        for i in range(len(inner)):
            j = (i + 1) % len(inner)
            b.facet([inner[i], outer[i], outer[j], inner[j]], group)
        return
    for k in range(n):
        for j in range(m):
            b.facet([inner[k], outer[(k * m + j) % total],
                     outer[(k * m + j + 1) % total]], group)


def _tier_rings(b: "_Builder", n: int, m: int, total: int, extra: int,
                outer: list[int], inner_radius: float,
                inner_z: float, angle_from: float, angle_to: float,
                z_edge: float, upward: bool, offset_deg: float) -> list[list[int]]:
    """The stack of rings the crown or pavilion steps down through.

    With no extra rows this is just ``[inner, outer]`` and nothing changes.
    Each extra row adds an intermediate ring of ``m*n`` points, placed along a
    plane whose angle is stepped between the main facet's and the outermost
    one's -- the break in angle is the entire point, since coplanar tiers
    would be one flat facet wearing extra edges.
    """
    extra = max(0, int(extra))
    if extra == 0:
        return [outer]

    rings = []
    for step in range(1, extra + 1):
        t = step / (extra + 1)
        radius = inner_radius + t * (1.0 - inner_radius)
        angle = math.radians(angle_from + t * (angle_to - angle_from))
        drop = (radius - inner_radius) * math.tan(angle)
        z = inner_z - drop if upward else inner_z + drop
        z = max(z, z_edge) if upward else min(z, z_edge)
        rings.append(_girdle_ring(b, n, m, z, z, offset_deg=offset_deg,
                                  radius=radius))
    rings.append(outer)
    return rings


def _girdle_ring(b: "_Builder", n: int, m: int, z_main: float, z_half: float,
                 offset_deg: float, radius: float = 1.0) -> list[int]:
    """One scalloped ring of ``m * n`` girdle vertices.

    Height runs from ``z_main`` where a main facet lands on the girdle to
    ``z_half`` at the middle of each sector, following a half sine so the
    scallop is smooth however many segments the sector is cut into.  At m = 2
    the two samples land exactly on the ends of that curve, which is the
    classic thin/thick alternation.
    """
    total = m * n
    out = []
    for i in range(total):
        t = (i % m) / m
        z = z_main + (z_half - z_main) * math.sin(math.pi * t)
        angle = math.radians(offset_deg) + 2.0 * math.pi * i / total
        out.append(b.vertex(radius * math.cos(angle), radius * math.sin(angle), z))
    return out


def build_stone(p: CutParams) -> Mesh:
    """Turn a parameter set into a watertight, outward-facing mesh.

    Layout (N = ``n_base``):

    ``main`` azimuths sit at 0, 360/N, ... and carry the bezel and pavilion
    main facets.  ``half`` azimuths sit exactly between them and carry the
    star tips and the girdle 'hills'.  Keeping the two crown/pavilion mains on
    the same azimuth is what lines the facets up and creates the arrows.
    """
    n = int(p.n_base)
    step = sector_degrees(n)
    b = _Builder()

    # ---- heights ----------------------------------------------------------
    crown_h = (1.0 - p.table) * math.tan(math.radians(p.crown_angle))
    pav_d = (1.0 - p.culet) * math.tan(math.radians(p.pavilion_angle))

    # girdle top/bottom at the two azimuth families; the band is centred on z=0
    zt_main = +0.5 * p.girdle_thick_main
    zb_main = -0.5 * p.girdle_thick_main
    zt_half = +0.5 * p.girdle_thick_half
    zb_half = -0.5 * p.girdle_thick_half

    # A tier ring is flat, so the band joining it to the girdle is only planar
    # if the girdle edge it lands on is flat too: a quad spanning two rings is
    # planar exactly when its two radial edges are parallel, and a scalloped
    # girdle tilts them against each other.  So the scallop is dropped on
    # whichever side carries extra rows -- which is also how a modified
    # brilliant is actually cut, with a straight girdle under the extra tiers.
    if p.extra_crown_rows:
        zt_half = zt_main
    if p.extra_pavilion_rows:
        zb_half = zb_main
    # A straight girdle is the same statement on both edges at once.
    if p.flat_girdle:
        zt_half = zt_main
        zb_half = zb_main

    z_table = zt_main + crown_h
    z_culet = zb_main - pav_d

    tw = p.crown_twist  # crown azimuth offset

    # ---- vertices ---------------------------------------------------------
    # table corners sit above the main azimuths (each bezel apex is a corner)
    table_ring = b.ring(n, p.table, z_table, offset_deg=tw)

    # Girdle: m*N vertices, with every m-th one sitting on a main azimuth.
    # The band is scalloped -- thinnest where a main facet meets it, thickest
    # between two mains -- so the thickness follows a half sine across each
    # sector.  At m = 2 that reproduces the classic alternating thin/thick
    # girdle exactly, and at higher m it just adds intermediate steps.
    m = max(1, int(p.girdle_multiple))
    g_top = _girdle_ring(b, n, m, zt_main, zt_half, offset_deg=tw)
    g_bot = _girdle_ring(b, n, m, zb_main, zb_half, offset_deg=0.0)

    # The shared apex vertices sit at the half azimuths.  Their height is NOT
    # a free interpolation: each one is a corner of a bezel (or pavilion main)
    # facet, and those facets are flat planes cut at a fixed angle.  Placing
    # the apex anywhere else warps the main facet into a slightly folded quad,
    # and -- worse -- leaves the girdle facets almost coplanar with it, so the
    # pavilion becomes a smooth cone.  A smooth cone reflects light uniformly
    # and draws no pattern at all; the hearts and arrows exist precisely
    # because there is a crisp angular break between a main facet and the
    # girdle facets beside it.  So we solve for the height that puts the apex
    # exactly on the main facet's plane.
    cos_half = math.cos(math.pi / n)
    tan_crown = math.tan(math.radians(p.crown_angle))
    tan_pav = math.tan(math.radians(p.pavilion_angle))

    if p.upper_girdle_row or p.star_row:
        # `star_length` is the trade's measurement: how far the star reaches
        # from the table edge towards the girdle, in plan view.
        r_u = p.table + p.star_length * (1.0 - p.table)
        # on the bezel plane through the table corner at the main azimuth
        z_u = z_table + (p.table - r_u * cos_half) * tan_crown
        upper_ring = b.ring(n, r_u, z_u, offset_deg=tw + step / 2.0)
    else:
        upper_ring = None
        # No apex ring, so the tiers start from the table edge instead.
        r_u, z_u = p.table, z_table

    if p.lower_girdle_row:
        # how far the lower-girdle facets reach down towards the culet
        r_l = 1.0 - p.lower_girdle_length * (1.0 - p.culet)
        # On the pavilion main plane -- which, with a flat culet, passes
        # through the culet ring's edge at radius p.culet, not through the
        # centre.  Anchoring on r=0 instead (as if the culet were always a
        # point) tilts this ring off that plane by exactly p.culet * tan_pav,
        # which is small in absolute terms but enough to fold every pavilion
        # main facet that touches it.
        z_l = z_culet + (r_l * cos_half - p.culet) * tan_pav
        lower_ring = b.ring(n, r_l, z_l, offset_deg=step / 2.0)
    else:
        lower_ring = None
        # No lower-girdle apex, so the tiers start from the culet.
        r_l, z_l = max(p.culet, 1e-3), z_culet

    if p.culet > 0.0:
        # A flat culet is a horizontal plane cutting off the point, and each
        # pavilion main is a single flat plane.  Its corners are therefore
        # where two *adjacent* main planes meet that horizontal cut -- which
        # is at the half azimuths, and at radius culet / cos(half step), not
        # at radius culet.  Placing them at `culet` puts them off every main
        # plane by culet * tan(pav) * (1 - cos(half step)), which folds all
        # eight pavilion mains: the exact "these faces should not exist"
        # artefact, since a folded facet has to be split to be drawn at all.
        #
        # `culet` keeps its trade meaning: the flat's half-width measured
        # along a main azimuth, so the number the client reads is unchanged.
        culet_ring = b.ring(n, p.culet / cos_half, z_culet, offset_deg=step / 2.0)
        culet_pt = None
    else:
        culet_ring = None
        culet_pt = b.vertex(0.0, 0.0, z_culet)

    # ---- crown facets -----------------------------------------------------
    # table: seen from above the winding must be counter-clockwise for the
    # normal to point up, which is the order the ring was generated in.
    b.facet(table_ring, "table")

    total = m * n                       # girdle points all the way round

    # A tiered crown steps down through intermediate rings, so the bezel ends
    # on the first of them rather than reaching the girdle directly.  With no
    # extra rows that first ring *is* the girdle and nothing changes.
    crown_stack = _tier_rings(b, n, m, total, p.extra_crown_rows,
                              inner_radius=r_u, inner_z=z_u,
                              angle_from=p.crown_angle,
                              angle_to=p.crown_outer_angle,
                              z_edge=zt_main, upward=True, offset_deg=tw,
                              outer=g_top)
    pavilion_stack = _tier_rings(b, n, m, total, p.extra_pavilion_rows,
                                 inner_radius=r_l, inner_z=z_l,
                                 angle_from=p.pavilion_angle,
                                 angle_to=p.pavilion_outer_angle,
                                 z_edge=zb_main, upward=False, offset_deg=0.0,
                                 outer=g_bot)

    main_at = lambda k: crown_stack[0][(k % n) * m]
    main_bt = lambda k: pavilion_stack[0][(k % n) * m]

    if p.star_row and upper_ring is not None:
        for k in range(n):
            # star facet: table edge k->k+1 with its apex pushed outward
            b.facet([table_ring[k], upper_ring[k], table_ring[(k + 1) % n]], "star")

    for k in range(n):
        # Bezel.  Normally a kite reaching all the way down to the girdle, with
        # its lower corner solved to sit exactly on the bezel plane.  Once
        # extra tiers are added that corner would land on a flat tier ring
        # instead, at a height the plane does not pass through, and the kite
        # would come out folded.  With tiers the bezel is therefore just
        # another row: a triangle between the table and the apex ring, with
        # the tiers carrying the crown down from there.
        if upper_ring is not None and p.extra_crown_rows:
            # Two triangles rather than one kite: a triangle cannot be folded,
            # so both halves stay flat wherever the tier ring happens to sit.
            b.facet([table_ring[k], upper_ring[(k - 1) % n], upper_ring[k]],
                    "bezel")
            b.facet([upper_ring[(k - 1) % n], main_at(k), upper_ring[k]],
                    "bezel_lower")
        elif upper_ring is not None:
            b.facet(
                [table_ring[k], upper_ring[(k - 1) % n], main_at(k), upper_ring[k]],
                "bezel",
            )
        else:
            b.facet(
                [table_ring[k], g_top[(k * m - 1) % total], main_at(k),
                 g_top[(k * m + 1) % total]],
                "bezel",
            )

    if p.upper_girdle_row and upper_ring is not None:
        # The upper girdle normally spans straight from the apex ring to the
        # girdle.  Extra rows put intermediate rings in that gap, each one
        # tilted a little further out, so the crown steps down in tiers
        # instead of in one plane.
        for k in range(n):
            for j in range(m):
                b.facet([upper_ring[k], crown_stack[0][(k * m + j) % total],
                         crown_stack[0][(k * m + j + 1) % total]], "upper_girdle")
        for lo, hi in zip(crown_stack[:-1], crown_stack[1:]):
            _bridge(b, lo, hi, n, m, total, "crown_tier")

    # ---- girdle band ------------------------------------------------------
    for i in range(total):
        b.facet([g_top[i], g_bot[i], g_bot[(i + 1) % total], g_top[(i + 1) % total]],
                "girdle")

    # ---- pavilion facets --------------------------------------------------
    if p.lower_girdle_row and lower_ring is not None:
        # The lower girdle facets fan from the apex ring onto the first tier;
        # any further tiers are plain bands between rings.  Winding is sorted
        # out for the whole mesh at the end by _orient_outward.
        for k in range(n):
            for j in range(m):
                b.facet([lower_ring[k], pavilion_stack[0][(k * m + j) % total],
                         pavilion_stack[0][(k * m + j + 1) % total]],
                        "lower_girdle")
        for lo, hi in zip(pavilion_stack[:-1], pavilion_stack[1:]):
            _bridge(b, lo, hi, n, m, total, "pavilion_tier")

    for k in range(n):
        apex_prev = (lower_ring[(k - 1) % n] if lower_ring is not None
                     else g_bot[(k * m - 1) % total])
        apex_next = (lower_ring[k] if lower_ring is not None
                     else g_bot[(k * m + 1) % total])
        bottom = culet_ring[k] if culet_ring is not None else culet_pt
        if culet_ring is not None and not p.extra_pavilion_rows:
            # One flat cut, so one facet: the kite truncated by the culet is a
            # single five-sided plane, exactly as a cutter would polish it.
            # No split is needed now that the culet corners really do lie on
            # that plane.
            b.facet(
                [main_bt(k), apex_prev, culet_ring[(k - 1) % n],
                 culet_ring[k], apex_next],
                "pavilion_main",
            )
        elif culet_ring is not None:
            # With a genuine extra tier, main_bt(k) lands on a tier ring
            # instead of on the plane through the culet, so that one corner
            # has to become its own facet -- as it already does without a
            # culet, just below.
            b.facet(
                [apex_prev, culet_ring[(k - 1) % n], culet_ring[k], apex_next],
                "pavilion_main",
            )
            b.facet([apex_prev, apex_next, main_bt(k)], "pavilion_main_upper")
        elif p.extra_pavilion_rows:
            # Split for the same reason as the bezel: the kite's upper corner
            # now lands on a tier ring rather than on the plane through the
            # culet, so it is cut into two triangles that cannot fold.
            b.facet([apex_prev, bottom, apex_next], "pavilion_main")
            b.facet([apex_prev, apex_next, main_bt(k)], "pavilion_main_upper")
        else:
            b.facet([main_bt(k), apex_prev, bottom, apex_next], "pavilion_main")

    if culet_ring is not None:
        b.facet(list(reversed(culet_ring)), "culet")

    mesh = b.finish()
    return _orient_outward(mesh)


def _orient_outward(mesh: Mesh) -> Mesh:
    """Flip any triangle whose normal points at the centroid.

    Facets are built by hand above, and getting every winding right by
    inspection is error-prone; the stone is star-shaped about its centroid so
    a dot-product test fixes the whole mesh reliably.
    """
    centroid = mesh.vertices.mean(axis=0)
    tv = mesh.triangle_vertices()
    tri_centres = tv.mean(axis=1)
    normals = mesh.normals()
    outward = np.einsum("ij,ij->i", normals, tri_centres - centroid) >= 0.0
    tris = mesh.triangles.copy()
    tris[~outward] = tris[~outward][:, [0, 2, 1]]

    # A facet's polygon has to wind the same way as its triangles, otherwise
    # the normals disagree between the tracer and anything importing the OBJ.
    flipped = set()
    for tri_index, fid in enumerate(mesh.facet_id):
        if not outward[tri_index]:
            flipped.add(int(fid))
    polygons = [
        list(reversed(poly)) if fid in flipped else list(poly)
        for fid, poly in enumerate(mesh.polygons)
    ]
    return Mesh(mesh.vertices, tris, mesh.facet_id, mesh.facet_group, polygons)
