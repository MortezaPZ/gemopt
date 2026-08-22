"""Optics: the tracer has to obey physics, not merely produce pretty pictures.

These tests check it against closed-form answers -- Snell's law, the critical
angle, the Fresnel coefficient at normal incidence -- because a scoring
function built on subtly wrong physics would rank stones confidently and
wrongly.
"""

import math

import numpy as np
import pytest

from gemopt.geometry import STANDARD_BRILLIANT, Mesh, build_stone
from gemopt.optics import (
    CROWN_GLASS,
    DIAMOND,
    Geometry,
    Material,
    trace,
)


@pytest.fixture(scope="module")
def stone():
    mesh = build_stone(STANDARD_BRILLIANT)
    return mesh, Geometry(mesh)


def _slab(thickness: float = 1.0, half: float = 6.0) -> Mesh:
    """A big flat box, so a ray meets one horizontal interface at a time."""
    z0, z1 = -thickness / 2.0, thickness / 2.0
    v = np.array([
        [-half, -half, z0], [half, -half, z0], [half, half, z0], [-half, half, z0],
        [-half, -half, z1], [half, -half, z1], [half, half, z1], [-half, half, z1],
    ], dtype=np.float64)
    quads = [
        [0, 3, 2, 1],   # bottom
        [4, 5, 6, 7],   # top
        [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
    ]
    tris, fids, polys = [], [], []
    for fid, q in enumerate(quads):
        polys.append(list(q))
        for i in range(1, len(q) - 1):
            tris.append((q[0], q[i], q[i + 1]))
            fids.append(fid)
    return Mesh(v, np.asarray(tris, np.int32), np.asarray(fids, np.int32),
                ["face"] * len(quads), polys)


# ---------------------------------------------------------------- physics --


def test_normal_incidence_reflectance_matches_fresnel():
    """At 90 degrees to the surface R = ((n-1)/(n+1))^2 exactly."""
    geom = Geometry(_slab())
    for material in (DIAMOND, CROWN_GLASS):
        n = material.n_d
        expected = ((n - 1.0) / (n + 1.0)) ** 2
        ev = trace(geom,
                   np.array([[0.0, 0.0, 3.0]]),
                   np.array([[0.0, 0.0, -1.0]]),
                   material, max_bounces=2)
        glare = ev.weight[(~ev.entered) & (ev.direction[:, 2] > 0)].sum()
        assert abs(glare - expected) < 1e-6


@pytest.mark.parametrize("angle_deg", [10.0, 20.0, 35.0])
def test_refraction_obeys_snells_law(angle_deg):
    geom = Geometry(_slab(thickness=2.0))
    theta = math.radians(angle_deg)
    direction = np.array([[math.sin(theta), 0.0, -math.cos(theta)]])
    ev = trace(geom, np.array([[0.0, 0.0, 3.0]]), direction, CROWN_GLASS, max_bounces=3)

    # the strongest transmitted packet is the one that went straight through
    inside = ev.entered & (ev.direction[:, 2] < 0)
    assert inside.any()
    best = ev.direction[inside][np.argmax(ev.weight[inside])]
    # it exits the far side parallel to how it came in
    exit_angle = math.degrees(math.acos(min(1.0, abs(best[2]))))
    assert abs(exit_angle - angle_deg) < 0.5


def _wedge(beta_deg: float, half: float = 2.0, height: float = 2.0) -> Mesh:
    """A box with its top face tilted by ``beta_deg``.

    A ray entering the flat bottom straight up passes through unbent and then
    meets the top face at exactly ``beta_deg``, which makes the critical angle
    directly testable.  A plain slab cannot do this: light entering from
    outside can never reach the critical angle at the far face, which is the
    same reason a window pane does not trap light.
    """
    t = math.tan(math.radians(beta_deg))
    v = np.array([
        [-half, -half, 0.0], [half, -half, 0.0], [half, half, 0.0], [-half, half, 0.0],
        [-half, -half, height - half * t], [half, -half, height + half * t],
        [half, half, height + half * t], [-half, half, height - half * t],
    ], dtype=np.float64)
    quads = [
        [0, 3, 2, 1], [4, 5, 6, 7],
        [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
    ]
    tris, fids, polys = [], [], []
    for fid, q in enumerate(quads):
        polys.append(list(q))
        for i in range(1, len(q) - 1):
            tris.append((q[0], q[i], q[i + 1]))
            fids.append(fid)
    return Mesh(v, np.asarray(tris, np.int32), np.asarray(fids, np.int32),
                ["face"] * len(quads), polys)


@pytest.mark.parametrize("offset, expect_escape", [(-4.0, True), (+4.0, False)])
def test_total_internal_reflection_starts_at_the_critical_angle(offset, expect_escape):
    """Just below the critical angle light gets out of the top; just above
    it, none does.  This is the effect the whole brilliant cut is built on,
    and the reason glass needs different angles from diamond."""
    material = CROWN_GLASS
    critical = math.degrees(math.asin(1.0 / material.n_d))
    assert 41.0 < critical < 41.3

    geom = Geometry(_wedge(critical + offset))
    ev = trace(geom,
               np.array([[0.0, 0.0, -3.0]]),
               np.array([[0.0, 0.0, 1.0]]),
               material, max_bounces=4)

    # energy leaving upwards through the tilted face, having been inside
    out = ev.weight[ev.entered & (ev.direction[:, 2] > 0.1)].sum()
    if expect_escape:
        assert out > 0.5
    else:
        assert out < 1e-6


def test_dispersion_splits_the_index():
    """n_F - n_C is the dispersion figure quoted for a gem material."""
    blue = DIAMOND.index(486.133)
    red = DIAMOND.index(656.273)
    assert blue > red
    assert abs((blue - red) - DIAMOND.dispersion) < 1e-9
    assert abs(DIAMOND.index(589.293) - DIAMOND.n_d) < 1e-9


def test_material_without_dispersion_is_flat():
    m = Material("flat", n_d=1.5)
    assert m.index(400.0) == m.index(700.0) == 1.5


# ------------------------------------------------------------ bookkeeping --


def test_energy_is_conserved_for_every_ray(stone):
    """Nothing may vanish: emitted plus still-bouncing must equal what went in.

    This is the test that caught rays slipping through the seams between
    facets, which had been quietly painting dark spokes across every render.
    """
    _, geom = stone
    rng = np.random.default_rng(4)
    r = np.sqrt(rng.uniform(0, 1, 3000))
    th = rng.uniform(0, 2 * np.pi, 3000)
    origin = np.column_stack([r * np.cos(th), r * np.sin(th), np.full(3000, 4.0)])
    direction = np.tile(np.array([0.0, 0.0, -1.0]), (3000, 1))

    for material in (DIAMOND, CROWN_GLASS):
        ev = trace(geom, origin, direction, material, max_bounces=16)
        total = ev.energy_per_ray() + ev.lost
        assert np.all(total > 0.998)
        assert np.all(total < 1.0001)


def test_weights_never_go_negative(stone):
    _, geom = stone
    rng = np.random.default_rng(9)
    origin = rng.uniform(-1, 1, (500, 3)) + np.array([0, 0, 4.0])
    direction = np.tile(np.array([0.0, 0.0, -1.0]), (500, 1))
    ev = trace(geom, origin, direction, DIAMOND)
    assert np.all(ev.weight >= 0.0)


def test_rays_that_miss_are_reported_as_background(stone):
    _, geom = stone
    origin = np.array([[5.0, 5.0, 4.0]])
    direction = np.array([[0.0, 0.0, -1.0]])
    ev = trace(geom, origin, direction, DIAMOND)
    assert len(ev.weight) == 1
    assert not ev.entered[0]
    assert ev.weight[0] == pytest.approx(1.0)


def test_diamond_returns_more_light_than_glass_under_diffuse_light(stone):
    """The headline physical fact the whole project rests on: at the same
    geometry, a high index keeps light in and a low one lets it out."""
    _, geom = stone
    rng = np.random.default_rng(11)
    n = 4000
    # a hemisphere of incoming directions, not just straight down -- a single
    # vertical beam sits right on the glass critical angle and is misleading
    r = np.sqrt(rng.uniform(0, 0.8, n))
    th = rng.uniform(0, 2 * np.pi, n)
    origin = np.column_stack([r * np.cos(th), r * np.sin(th), np.full(n, 4.0)])
    tilt = rng.uniform(0, 0.5, n)
    az = rng.uniform(0, 2 * np.pi, n)
    direction = np.column_stack([
        tilt * np.cos(az), tilt * np.sin(az), -np.ones(n)
    ])
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    leaks = {}
    for material in (DIAMOND, CROWN_GLASS):
        ev = trace(geom, origin, direction, material, max_bounces=16)
        down = ev.entered & (ev.direction[:, 2] < 0)
        leaks[material.name] = ev.weight[down].sum() / n

    assert leaks["glass"] > leaks["diamond"]


# ------------------------------------------------------------ intersection --


def test_intersection_agrees_with_a_reference_implementation(stone):
    """The fast path uses single precision and matrix products; check it
    against a textbook Moller-Trumbore in double precision."""
    mesh, geom = stone
    rng = np.random.default_rng(3)
    n = 4000
    origin = rng.uniform(-1.5, 1.5, (n, 3))
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    tv = mesh.triangle_vertices()
    v0, e1, e2 = tv[:, 0], tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0]
    pv = np.cross(direction[:, None, :], e2[None])
    det = np.einsum("tk,rtk->rt", e1, pv)
    par = np.abs(det) < 1e-12
    inv = np.where(par, 0.0, 1.0 / np.where(par, 1.0, det))
    tvec = origin[:, None, :] - v0[None]
    u = np.einsum("rtk,rtk->rt", tvec, pv) * inv
    qv = np.cross(tvec, e1[None])
    v = np.einsum("rk,rtk->rt", direction, qv) * inv
    t = np.einsum("tk,rtk->rt", e2, qv) * inv
    ok = (~par) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-7)
    t = np.where(ok, t, np.inf)
    ref_tri = np.argmin(t, axis=1)
    ref_t = t[np.arange(n), ref_tri]
    ref_tri = np.where(np.isfinite(ref_t), ref_tri, -1)

    got_t, got_tri = geom.intersect(origin, direction)
    assert np.array_equal(got_tri, ref_tri)
    hit = ref_tri >= 0
    assert np.max(np.abs(got_t[hit] - ref_t[hit])) < 1e-4
