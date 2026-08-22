"""Ray tracing inside a transparent stone.

This is the piece that replaces V-Ray in the optimisation loop.  Rendering a
frame takes minutes; tracing a few hundred thousand rays through 100-odd
triangles with numpy takes a fraction of a second, and it gives us *numbers*
rather than pixels -- which is what the search actually needs.

Physics covered:

* refraction at every surface crossing (Snell's law),
* total internal reflection, which is the whole reason a brilliant cut works,
* Fresnel reflection/transmission with unpolarised light, so partial
  reflections are weighted correctly instead of being all-or-nothing,
* wavelength-dependent refractive index (dispersion), which is what separates
  white sparkle from coloured fire,
* optional Beer-Lambert absorption for tinted material.

Everything is vectorised: a "ray batch" is a set of numpy arrays and every
bounce advances all live rays at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Mesh

__all__ = [
    "Material",
    "DIAMOND",
    "CROWN_GLASS",
    "LEAD_CRYSTAL",
    "CUBIC_ZIRCONIA",
    "MATERIALS",
    "trace",
    "TraceResult",
]

EPS = 1e-7
"""Surface offset used to push a ray off the face it just left.  It has to
stay above the resolution of the single-precision hit test but far below any
facet dimension, and 1e-7 of a girdle radius is both."""


# --------------------------------------------------------------------------
# materials
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Material:
    """Optical constants of the stone.

    ``n_d`` is the refractive index at the sodium D line (589 nm), the number
    people quote when they say "RI".  ``dispersion`` is the Abbe-style
    difference n_F - n_C (486 nm vs 656 nm) that spreads white light into
    colour.  ``absorption`` is in inverse girdle-radii, so 0 means perfectly
    clear.
    """

    name: str
    n_d: float
    dispersion: float = 0.0
    absorption: float = 0.0

    def index(self, wavelength_nm: float | np.ndarray) -> float | np.ndarray:
        """Cauchy fit anchored on ``n_d`` and ``dispersion``.

        Two data points fix the two Cauchy coefficients exactly, which is
        plenty for the visible range and avoids needing full Sellmeier data
        for materials the client may just hand us an "RI" for.
        """
        if self.dispersion == 0.0:
            return self.n_d
        lam = np.asarray(wavelength_nm, dtype=np.float64) * 1e-3  # micrometres
        lam_f, lam_c, lam_d = 0.486133, 0.656273, 0.589293
        # n = A + B/lam^2 ; solve for B from n_F - n_C, then A from n_d
        b = self.dispersion / (1.0 / lam_f**2 - 1.0 / lam_c**2)
        a = self.n_d - b / lam_d**2
        return a + b / lam**2


DIAMOND = Material("diamond", n_d=2.4175, dispersion=0.0440)
CROWN_GLASS = Material("glass", n_d=1.5200, dispersion=0.0080)
LEAD_CRYSTAL = Material("lead_crystal", n_d=1.5450, dispersion=0.0130)
CUBIC_ZIRCONIA = Material("cubic_zirconia", n_d=2.1600, dispersion=0.0600)

MATERIALS = {m.name: m for m in (DIAMOND, CROWN_GLASS, LEAD_CRYSTAL, CUBIC_ZIRCONIA)}


# --------------------------------------------------------------------------
# intersection
# --------------------------------------------------------------------------


class Geometry:
    """Pre-digested triangle soup with a ray/triangle test.

    A gem has of the order of a hundred triangles.  At that size a tree
    structure buys nothing, so every ray is tested against every triangle --
    but the test is arranged so all the heavy work is four ``(R,3) @ (3,T)``
    matrix products, which numpy hands straight to BLAS.  The textbook
    Moller-Trumbore formulation needs per-ray cross products and materialises
    ``(rays, triangles, 3)`` temporaries instead, which is an order of
    magnitude slower and far heavier on memory.

    The trick is to split the ray/plane hit and the inside-triangle test:
    ``t`` comes from the plane equation, and the barycentric coordinates are
    assembled from dot products that are themselves linear in ``t``.
    """

    def __init__(self, mesh: Mesh) -> None:
        tv = mesh.triangle_vertices()
        self.v0 = np.ascontiguousarray(tv[:, 0])
        self.edge1 = np.ascontiguousarray(tv[:, 1] - tv[:, 0])
        self.edge2 = np.ascontiguousarray(tv[:, 2] - tv[:, 0])
        self.normals = mesh.normals()
        self.facet_id = mesh.facet_id
        self.n_tri = len(self.v0)
        self.mesh = mesh

        # plane equation n . x = plane_d
        self._nT = np.ascontiguousarray(self.normals.T)
        self.plane_d = np.einsum("ij,ij->i", self.normals, self.v0)

        # constants of the barycentric solve
        self._e1T = np.ascontiguousarray(self.edge1.T)
        self._e2T = np.ascontiguousarray(self.edge2.T)
        self._d00 = np.einsum("ij,ij->i", self.edge1, self.edge1)
        self._d01 = np.einsum("ij,ij->i", self.edge1, self.edge2)
        self._d11 = np.einsum("ij,ij->i", self.edge2, self.edge2)
        denom = self._d00 * self._d11 - self._d01**2
        self._inv_denom = np.where(np.abs(denom) < 1e-18, 0.0,
                                   1.0 / np.where(np.abs(denom) < 1e-18, 1.0, denom))
        self._v0e1 = np.einsum("ij,ij->i", self.v0, self.edge1)
        self._v0e2 = np.einsum("ij,ij->i", self.v0, self.edge2)

        # radius of a sphere around the origin that contains the whole stone,
        # used to reject rays that obviously miss before any real work
        self.radius = float(np.linalg.norm(mesh.vertices, axis=1).max())

        # Single precision copies for the hit test.  The stone is one unit
        # across and float32 resolves ~1e-7 of that, far finer than any facet
        # feature, while halving the memory traffic of the (rays x triangles)
        # intermediates -- and that traffic is what the tracer is limited by.
        # Which triangle was last touched is tracked exactly (see `ignore`),
        # so self-intersection never relies on this precision.
        self._nT32 = self._nT.astype(np.float32)
        self._e1T32 = self._e1T.astype(np.float32)
        self._e2T32 = self._e2T.astype(np.float32)
        self._plane_d32 = self.plane_d.astype(np.float32)
        self._d00_32 = self._d00.astype(np.float32)
        self._d01_32 = self._d01.astype(np.float32)
        self._d11_32 = self._d11.astype(np.float32)
        self._inv_denom32 = self._inv_denom.astype(np.float32)
        self._v0e1_32 = self._v0e1.astype(np.float32)
        self._v0e2_32 = self._v0e2.astype(np.float32)

        # A facet is a flat polygon split into a fan of triangles, so a ray
        # leaving one of them sits exactly in the plane of its neighbours and
        # will "hit" them again at t ~ 0.  Suppressing the whole facet rather
        # than the single triangle is what keeps the seams from showing up as
        # dark cracks across the render.
        n_facets = int(mesh.facet_id.max()) + 1 if len(mesh.facet_id) else 0
        self.facet_mask = np.zeros((n_facets + 1, self.n_tri), dtype=bool)
        self.facet_mask[mesh.facet_id, np.arange(self.n_tri)] = True
        self.facet_mask[-1] = False        # index -1 means "suppress nothing"

    def intersect(
        self, origin: np.ndarray, direction: np.ndarray, ignore: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Closest hit per ray.

        Returns ``(t, tri_index)``.  ``t`` is ``inf`` and ``tri_index`` is -1
        where the ray escapes.  ``ignore`` optionally suppresses one *facet*
        per ray -- the one it is leaving -- which also covers the other
        triangles that facet was split into.
        """
        o32 = origin.astype(np.float32, copy=False)
        d32 = direction.astype(np.float32, copy=False)

        denom = d32 @ self._nT32                           # (R, T)
        # Rays parallel to a facet plane never hit it.  Substituting a dummy
        # denominator keeps the barycentric maths free of inf/NaN; the
        # `parallel` mask discards those columns at the end anyway.
        parallel = np.abs(denom) < 1e-9
        np.copyto(denom, np.float32(1.0), where=parallel)

        t = o32 @ self._nT32
        np.subtract(self._plane_d32[None, :], t, out=t)
        np.divide(t, denom, out=t)
        del denom

        # d20 = (o + t*d - v0) . e1, and likewise for e2, all built from
        # matrix products so nothing is evaluated per ray in Python.
        scratch = d32 @ self._e1T32
        scratch *= t
        d20 = o32 @ self._e1T32
        d20 += scratch
        d20 -= self._v0e1_32[None, :]

        np.matmul(d32, self._e2T32, out=scratch)
        scratch *= t
        d21 = o32 @ self._e2T32
        d21 += scratch
        d21 -= self._v0e2_32[None, :]
        del scratch

        u = self._d11_32[None, :] * d20
        u -= self._d01_32[None, :] * d21
        u *= self._inv_denom32[None, :]

        v = self._d00_32[None, :] * d21
        v -= self._d01_32[None, :] * d20
        v *= self._inv_denom32[None, :]
        del d20, d21

        ok = ~parallel
        ok &= u >= -1e-6
        ok &= v >= -1e-6
        u += v
        ok &= u <= 1.0 + 1e-6
        ok &= t > 1e-6
        del u, v, parallel

        if ignore is not None:
            ok &= ~self.facet_mask[ignore]

        np.copyto(t, np.float32(np.inf), where=~ok)
        tri = np.argmin(t, axis=1)
        tmin = t[np.arange(len(t)), tri].astype(np.float64)
        tri = np.where(np.isfinite(tmin), tri, -1)
        return tmin, tri


# --------------------------------------------------------------------------
# tracing
# --------------------------------------------------------------------------


@dataclass
class TraceResult:
    """Every packet of light that left the stone.

    One input ray normally produces several events: a little glare reflected
    off the crown on the way in, then one event each time an internal bounce
    lets part of the energy through a facet.  ``ray`` says which input ray an
    event belongs to, and the weights of all events sharing a ``ray`` sum to
    (at most) that ray's starting energy.
    """

    ray: np.ndarray           # (E,)   index of the originating ray
    origin: np.ndarray        # (E, 3) point on the surface where it left
    direction: np.ndarray     # (E, 3) exit direction, unit length
    weight: np.ndarray        # (E,)   fraction of the original energy
    bounces: np.ndarray       # (E,)   internal reflections before exiting
    entered: np.ndarray       # (E,)   bool, False for surface glare / misses
    facet: np.ndarray         # (E,)   facet the light left through, -1 if none
    n_rays: int = 0
    """Number of input rays, so fractions can be normalised."""

    lost: np.ndarray | None = None
    """(R,) energy per input ray still bouncing when the trace gave up."""

    def energy_per_ray(self) -> np.ndarray:
        out = np.zeros(self.n_rays)
        np.add.at(out, self.ray, self.weight)
        return out


def _refract(
    direction: np.ndarray, normal: np.ndarray, eta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Snell's law in vector form.

    ``normal`` must face *against* the incoming direction.  ``eta`` is
    n_from / n_to.  The boolean tells which rays survived; the rest hit total
    internal reflection.
    """
    cos_i = -np.einsum("ij,ij->i", direction, normal)
    sin2_t = eta**2 * (1.0 - cos_i**2)
    ok = sin2_t <= 1.0
    cos_t = np.sqrt(np.clip(1.0 - sin2_t, 0.0, None))
    out = eta[:, None] * direction + (eta * cos_i - cos_t)[:, None] * normal
    return out, ok


def _fresnel(cos_i: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Unpolarised reflectance at a dielectric boundary.

    ``eta`` = n_from / n_to, ``cos_i`` >= 0.  Returns 1.0 where the geometry
    implies total internal reflection.
    """
    sin2_t = eta**2 * (1.0 - cos_i**2)
    tir = sin2_t >= 1.0
    cos_t = np.sqrt(np.clip(1.0 - sin2_t, 0.0, None))
    rs = (eta * cos_i - cos_t) / np.where(
        np.abs(eta * cos_i + cos_t) < 1e-12, 1e-12, eta * cos_i + cos_t
    )
    rp = (cos_i - eta * cos_t) / np.where(
        np.abs(cos_i + eta * cos_t) < 1e-12, 1e-12, cos_i + eta * cos_t
    )
    r = 0.5 * (rs**2 + rp**2)
    return np.where(tir, 1.0, np.clip(r, 0.0, 1.0))


def trace(
    geom: Geometry,
    origin: np.ndarray,
    direction: np.ndarray,
    material: Material,
    wavelength_nm: float = 589.3,
    max_bounces: int = 24,
    min_weight: float = 1e-3,
    chunk: int = 4096,
) -> TraceResult:
    """Trace a batch of rays, splitting it so the ``(rays, triangles)``
    intermediates stay a sane size.  See :func:`_trace_chunk` for the physics.
    """
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    n_rays = len(origin)
    if n_rays <= chunk:
        return _trace_chunk(geom, origin, direction, material, wavelength_nm,
                            max_bounces, min_weight)

    parts: list[TraceResult] = []
    lost = np.zeros(n_rays)
    for start in range(0, n_rays, chunk):
        stop = min(start + chunk, n_rays)
        part = _trace_chunk(geom, origin[start:stop], direction[start:stop],
                            material, wavelength_nm, max_bounces, min_weight)
        part.ray = part.ray + start
        lost[start:stop] = part.lost
        parts.append(part)

    merged = TraceResult(
        ray=np.concatenate([p.ray for p in parts]),
        origin=np.concatenate([p.origin for p in parts]),
        direction=np.concatenate([p.direction for p in parts]),
        weight=np.concatenate([p.weight for p in parts]),
        bounces=np.concatenate([p.bounces for p in parts]),
        entered=np.concatenate([p.entered for p in parts]),
        facet=np.concatenate([p.facet for p in parts]),
        n_rays=n_rays,
    )
    merged.lost = lost
    return merged


def _trace_chunk(
    geom: Geometry,
    origin: np.ndarray,
    direction: np.ndarray,
    material: Material,
    wavelength_nm: float = 589.3,
    max_bounces: int = 24,
    min_weight: float = 1e-3,
) -> TraceResult:
    """Push a batch of rays through the stone, recording everything that leaves.

    At each surface the Fresnel split sends part of the energy through and
    part back.  Rather than branch into a tree (exponential) or gamble on one
    side (noisy, and it throws away exactly the leakage we are trying to
    measure in low-index material), the tracer *records the transmitted part
    as a finished exit event* and carries on with the reflected part.

    That is exact for the question we ask -- where does the light come out and
    in which direction -- because a ray that refracts out of a convex stone
    does not come back.  The cost stays one chain per input ray, so the whole
    thing runs thousands of times inside an optimisation.
    """
    n_stone = float(material.index(wavelength_nm))
    origin = np.asarray(origin, dtype=np.float64).copy()
    direction = np.asarray(direction, dtype=np.float64).copy()
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    n_rays = len(origin)
    ray_id = np.arange(n_rays)
    weight = np.ones(n_rays)
    bounces = np.zeros(n_rays, dtype=np.int32)
    inside = np.zeros(n_rays, dtype=bool)
    last_facet = np.full(n_rays, -1, dtype=np.int64)
    live = np.ones(n_rays, dtype=bool)

    ev_ray: list[np.ndarray] = []
    ev_pos: list[np.ndarray] = []
    ev_dir: list[np.ndarray] = []
    ev_w: list[np.ndarray] = []
    ev_b: list[np.ndarray] = []
    ev_in: list[np.ndarray] = []
    ev_facet: list[np.ndarray] = []

    def emit(rays, pos, dirs, w, b, entered, facet):
        keep = w > 0.0
        if not np.any(keep):
            return
        ev_ray.append(rays[keep])
        ev_pos.append(pos[keep])
        ev_dir.append(dirs[keep])
        ev_w.append(w[keep])
        ev_b.append(b[keep])
        ev_in.append(np.broadcast_to(np.asarray(entered), keep.shape)[keep].astype(bool))
        ev_facet.append(facet[keep] if facet is not None else np.full(keep.sum(), -1, np.int32))

    for _ in range(max_bounces):
        idx = np.nonzero(live)[0]
        if idx.size == 0:
            break

        t, tri = geom.intersect(origin[idx], direction[idx], ignore=last_facet[idx])

        miss = tri < 0
        if np.any(miss):
            gone = idx[miss]
            # Outside the stone, hitting nothing just means the ray flew past
            # -- that is the background, and the camera needs to know about it.
            #
            # Inside the stone it means the ray was sitting exactly on the
            # seam between two facets and slipped out through the crack.  That
            # happens along every symmetry line of the stone, so silently
            # dropping those rays paints dark spokes across the render and
            # loses their energy.  Treating them as having left at that point
            # keeps the books balanced and the picture clean.
            was_in = inside[gone]
            emit(ray_id[gone], origin[gone], direction[gone], weight[gone],
                 bounces[gone], was_in, None)
            live[gone] = False

        hit = ~miss
        if not np.any(hit):
            continue
        idx = idx[hit]
        t = t[hit]
        tri = tri[hit]

        # advance to the surface
        origin[idx] = origin[idx] + t[:, None] * direction[idx]
        was_inside = inside[idx]

        if material.absorption > 0.0:
            weight[idx] *= np.where(was_inside, np.exp(-material.absorption * t), 1.0)

        normal = geom.normals[tri]
        facing = np.einsum("ij,ij->i", direction[idx], normal) > 0.0
        normal = np.where(facing[:, None], -normal, normal)

        eta = np.where(was_inside, n_stone, 1.0 / n_stone)
        cos_i = np.clip(-np.einsum("ij,ij->i", direction[idx], normal), 0.0, 1.0)
        reflectance = _fresnel(cos_i, eta)

        refr, refr_ok = _refract(direction[idx], normal, eta)
        refl = direction[idx] - 2.0 * np.einsum(
            "ij,ij->i", direction[idx], normal
        )[:, None] * normal
        refl /= np.linalg.norm(refl, axis=1, keepdims=True)
        norm_refr = np.linalg.norm(refr, axis=1, keepdims=True)
        refr = refr / np.where(norm_refr < 1e-12, 1.0, norm_refr)

        # under total internal reflection nothing gets through
        transmit = np.where(refr_ok, 1.0 - reflectance, 0.0)
        reflect = np.where(refr_ok, reflectance, 1.0)

        facet_of = geom.facet_id[tri]
        surf = origin[idx]

        inbound = ~was_inside
        if np.any(inbound):
            # entering: the reflected share is surface glare and is finished
            s = np.nonzero(inbound)[0]
            emit(ray_id[idx[s]], surf[s], refl[s], weight[idx[s]] * reflect[s],
                 bounces[idx[s]], False, facet_of[s])
        if np.any(was_inside):
            # leaving: the transmitted share has escaped and is finished
            s = np.nonzero(was_inside)[0]
            emit(ray_id[idx[s]], surf[s], refr[s], weight[idx[s]] * transmit[s],
                 bounces[idx[s]], True, facet_of[s])

        # carry on with the share that stays in play
        cont_w = np.where(inbound, transmit, reflect)
        cont_dir = np.where(inbound[:, None], refr, refl)

        weight[idx] *= cont_w
        direction[idx] = cont_dir
        origin[idx] = surf + EPS * cont_dir
        inside[idx] = np.where(inbound, refr_ok, True)
        bounces[idx] += was_inside.astype(np.int32)
        last_facet[idx] = facet_of

        # an entering ray that could not refract (grazing) is fully reflected
        # away and has already been emitted
        stuck = inbound & (~refr_ok)
        if np.any(stuck):
            live[idx[stuck]] = False

        faded = weight[idx] < min_weight
        if np.any(faded):
            live[idx[faded]] = False

    if ev_ray:
        result = TraceResult(
            ray=np.concatenate(ev_ray),
            origin=np.concatenate(ev_pos),
            direction=np.concatenate(ev_dir),
            weight=np.concatenate(ev_w),
            bounces=np.concatenate(ev_b),
            entered=np.concatenate(ev_in),
            facet=np.concatenate(ev_facet),
            n_rays=n_rays,
        )
    else:
        empty = np.zeros(0)
        result = TraceResult(
            ray=np.zeros(0, np.int64), origin=np.zeros((0, 3)), direction=np.zeros((0, 3)),
            weight=empty, bounces=np.zeros(0, np.int32), entered=np.zeros(0, bool),
            facet=np.zeros(0, np.int32), n_rays=n_rays,
        )

    lost = np.zeros(n_rays)
    np.add.at(lost, ray_id[live], weight[live])
    result.lost = lost
    return result
