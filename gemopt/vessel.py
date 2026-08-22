"""Faceted hollow objects: tumblers, pitchers, anything turned and cut.

The client asked for these alongside the stones -- "پارچ و لیوان" -- and the
optics do not change at all: it is the same glass, the same critical angle,
the same tracer.  What changes is the shape, and in two ways that matter.

A vessel is *hollow*, so light crosses two walls instead of one and the inside
surface is as much a part of the design as the outside.  And its silhouette is
a free curve rather than the crown-girdle-pavilion of a gem, so it is
described by a profile of rings rather than by cut angles.

Everything else carries straight over.  The rings are cut into ``n_base``
facets exactly as the gem rows are, they stay in register all the way up, and
facet-to-facet symmetry is preserved -- so a vessel can be scored and
optimised by the same machinery without knowing it is not a stone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np

from .geometry import Mesh, _Builder

__all__ = [
    "VesselParams",
    "build_vessel",
    "TUMBLER",
    "PITCHER",
    "WINE_GLASS",
]


@dataclass
class VesselParams:
    """A turned, faceted, hollow object.

    The profile is given as fractions so the whole thing scales with
    ``height``: radii are fractions of the widest radius, heights are
    fractions of the total height.  That keeps a pitcher a pitcher when it is
    resized, and keeps every parameter in a range an optimiser can search.
    """

    n_base: int = 12
    """Facets around.  Twelve is a common tumbler; sixteen and twenty-four are
    the usual cut-crystal counts."""

    profile: list[tuple[float, float]] = field(default_factory=lambda: [
        (0.62, 0.00),      # foot
        (0.70, 0.10),
        (0.86, 0.35),
        (1.00, 0.72),
        (0.97, 1.00),      # rim
    ])
    """``(radius, height)`` control rings from base to rim, both normalised."""

    wall: float = 0.075
    """Wall thickness as a fraction of the widest radius."""

    floor: float = 0.13
    """Thickness of the solid base, as a fraction of total height.  This is
    where a tumbler does its sparkling, so it is worth having as a knob."""

    height: float = 2.4
    """Total height in girdle-radius units, so the same camera framing works."""

    facet_twist: float = 0.0
    """Degrees of rotation added per ring, for a spiral cut.  Zero keeps every
    row in register, which is what the client asked for."""

    rim_round: float = 0.0
    """Fraction of the top ring pulled inwards, to soften a sharp rim."""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VesselParams":
        known = {f for f in cls.__dataclass_fields__}
        out = {k: v for k, v in d.items() if k in known}
        if "profile" in out:
            out["profile"] = [tuple(p) for p in out["profile"]]
        return cls(**out)

    def derived(self) -> dict:
        radii = [r for r, _ in self.profile]
        return {
            "facets_around": self.n_base,
            "rings": len(self.profile),
            "height": self.height,
            "max_radius": max(radii),
            "rim_radius": self.profile[-1][0],
            "wall_thickness": self.wall,
            "floor_thickness": self.floor * self.height,
            "facet_count": facet_count(self),
        }

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not (3 <= self.n_base <= 48):
            problems.append("facet count around must be 3..48")
        if len(self.profile) < 2:
            problems.append("profile needs at least two rings")
        heights = [h for _, h in self.profile]
        if any(b <= a for a, b in zip(heights, heights[1:])):
            problems.append("profile heights must increase from base to rim")
        if any(r <= 0.0 for r, _ in self.profile):
            problems.append("profile radii must be positive")
        if not (0.01 <= self.wall <= 0.45):
            problems.append("wall thickness outside 1%..45%")
        if not (0.02 <= self.floor <= 0.6):
            problems.append("floor thickness outside 2%..60%")
        if self.wall >= min(r for r, _ in self.profile):
            problems.append("wall thicker than the narrowest part of the body")
        return problems


def facet_count(p: VesselParams) -> int:
    """How many facets the vessel is cut into.

    Per column of ``n_base``: one band for each gap in the profile going up
    the outside, the same number coming back down the inside, then the rim,
    the cavity floor and the underside -- three single rows.
    """
    bands = len(p.profile) - 1
    return p.n_base * (2 * bands + 3)


TUMBLER = VesselParams()

PITCHER = VesselParams(
    n_base=16,
    profile=[(0.55, 0.00), (0.68, 0.12), (1.00, 0.45), (0.92, 0.78), (0.74, 1.00)],
    wall=0.06,
    floor=0.10,
    height=2.8,
)

WINE_GLASS = VesselParams(
    n_base=24,
    profile=[(0.80, 0.00), (0.18, 0.16), (0.16, 0.48), (0.70, 0.70), (0.86, 1.00)],
    wall=0.045,
    floor=0.06,
    height=3.4,
)


def build_vessel(p: VesselParams) -> Mesh:
    """Turn a profile into a closed, faceted, hollow solid.

    Built as one surface that runs up the outside, across the rim, back down
    the inside and closes on the floor -- so it is watertight by construction
    and the tracer sees a genuine two-walled object rather than a shell.
    """
    n = int(p.n_base)
    b = _Builder()

    outer = [(r, h * p.height) for r, h in p.profile]
    floor_z = p.floor * p.height

    # The inside mirrors the outside, pulled in by the wall thickness and
    # stopped at the floor.  Rings that would fall below the floor collapse
    # onto it, which is what gives a tumbler its thick solid base.
    inner = []
    for r, z in outer:
        inner.append((max(r - p.wall, 1e-3), max(z, floor_z)))
    if p.rim_round > 0.0:
        r_out, z_out = outer[-1]
        inner[-1] = (max(r_out - p.wall * (1.0 + p.rim_round), 1e-3), inner[-1][1])

    def ring(radius: float, z: float, index: int) -> list[int]:
        twist = math.radians(p.facet_twist * index)
        return [
            b.vertex(radius * math.cos(twist + 2.0 * math.pi * k / n),
                     radius * math.sin(twist + 2.0 * math.pi * k / n), z)
            for k in range(n)
        ]

    outer_rings = [ring(r, z, i) for i, (r, z) in enumerate(outer)]
    inner_rings = [ring(r, z, i) for i, (r, z) in enumerate(inner)]

    # outside wall, base to rim
    for i in range(len(outer_rings) - 1):
        lo, hi = outer_rings[i], outer_rings[i + 1]
        for k in range(n):
            b.facet([lo[k], lo[(k + 1) % n], hi[(k + 1) % n], hi[k]], f"outside_{i}")

    # rim: the ring joining outside to inside
    for k in range(n):
        b.facet([outer_rings[-1][k], outer_rings[-1][(k + 1) % n],
                 inner_rings[-1][(k + 1) % n], inner_rings[-1][k]], "rim")

    # inside wall, rim back down to the floor
    for i in range(len(inner_rings) - 1, 0, -1):
        hi, lo = inner_rings[i], inner_rings[i - 1]
        for k in range(n):
            b.facet([hi[k], hi[(k + 1) % n], lo[(k + 1) % n], lo[k]], f"inside_{i}")

    # the floor of the cavity and the underside, closing the solid
    cavity = b.vertex(0.0, 0.0, floor_z)
    for k in range(n):
        b.facet([inner_rings[0][k], inner_rings[0][(k + 1) % n], cavity], "cavity_floor")

    underside = b.vertex(0.0, 0.0, outer[0][1])
    for k in range(n):
        b.facet([outer_rings[0][(k + 1) % n], outer_rings[0][k], underside], "underside")

    mesh = b.finish()
    return _orient_outward(mesh)


def _orient_outward(mesh: Mesh) -> Mesh:
    """Point every normal away from the solid.

    A vessel is not star-shaped about its centroid -- the cavity faces inwards
    -- so the gem's dot-product test does not apply.  Winding is fixed instead
    by the sign of the total volume: if it comes out negative the whole surface
    is inside out, and flipping every triangle fixes it in one go.
    """
    if mesh.volume() < 0:
        tris = mesh.triangles[:, [0, 2, 1]]
        return Mesh(mesh.vertices, tris, mesh.facet_id, mesh.facet_group)
    return mesh
