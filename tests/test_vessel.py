"""Hollow faceted objects: tumblers, pitchers, stemware.

Same optics as the stones; the thing that has to be proved here is that the
shape itself is sound -- closed, correctly wound, and cut into rows that stay
in register the way the client asked for.
"""

import collections

import numpy as np
import pytest

from gemopt.vessel import (
    PITCHER,
    TUMBLER,
    WINE_GLASS,
    VesselParams,
    build_vessel,
    facet_count,
)


@pytest.mark.parametrize("name,params", [("tumbler", TUMBLER), ("pitcher", PITCHER),
                                         ("wine", WINE_GLASS)])
def test_presets_are_closed_solids(name, params):
    assert params.validate() == []
    mesh = build_vessel(params)
    assert mesh.is_watertight()
    assert mesh.volume() > 0, "surface is inside out"


@pytest.mark.parametrize("n", [6, 8, 12, 16, 24])
def test_any_facet_count_still_closes(n):
    mesh = build_vessel(VesselParams(n_base=n))
    assert mesh.is_watertight()
    assert mesh.n_facets == facet_count(VesselParams(n_base=n))


def test_every_facet_is_flat():
    """A facet is one polished surface; its triangles must agree."""
    mesh = build_vessel(TUMBLER)
    normals = mesh.normals()
    for fid in range(mesh.n_facets):
        f = normals[mesh.facet_id == fid]
        assert np.allclose(f, f[0], atol=1e-9), f"facet {fid} is bent"


def test_it_is_actually_hollow():
    """There has to be a cavity, or it is a paperweight rather than a glass.

    A solid of the same outline would displace the full cylinder; the vessel
    must come out well under that.
    """
    params = TUMBLER
    mesh = build_vessel(params)
    widest = max(r for r, _ in params.profile)
    solid = np.pi * widest**2 * params.height
    assert mesh.volume() < 0.6 * solid


def test_rows_stay_in_register_without_twist():
    """Every ring must share the same facet azimuths.

    This is the client's alignment rule applied to vessels: rows may differ in
    radius and height, but they line up all the way from base to rim.
    """
    mesh = build_vessel(VesselParams(n_base=12, facet_twist=0.0))
    xy = mesh.vertices[np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1]) > 1e-6]
    azimuths = np.degrees(np.arctan2(xy[:, 1], xy[:, 0])) % 30.0
    assert np.allclose(np.minimum(azimuths, 30.0 - azimuths), 0.0, atol=1e-6)


def test_twist_breaks_the_register_on_purpose():
    """The spiral option has to actually do something."""
    mesh = build_vessel(VesselParams(n_base=12, facet_twist=4.0))
    xy = mesh.vertices[np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1]) > 1e-6]
    azimuths = np.degrees(np.arctan2(xy[:, 1], xy[:, 0])) % 30.0
    assert not np.allclose(np.minimum(azimuths, 30.0 - azimuths), 0.0, atol=1e-6)


def test_bad_shapes_are_rejected_before_anything_is_built():
    assert VesselParams(n_base=2).validate()
    assert VesselParams(wall=0.9).validate()
    assert VesselParams(profile=[(1.0, 0.0)]).validate()
    assert VesselParams(profile=[(1.0, 0.0), (1.0, -0.5)]).validate()


def test_light_can_be_traced_through_one():
    """The tracer must handle two walls without losing energy."""
    from gemopt.optics import CROWN_GLASS, Geometry, trace

    mesh = build_vessel(TUMBLER)
    geom = Geometry(mesh)
    rng = np.random.default_rng(0)
    r = np.sqrt(rng.uniform(0, 1, 2000)) * 0.9
    th = rng.uniform(0, 2 * np.pi, 2000)
    origin = np.column_stack([r * np.cos(th), r * np.sin(th), np.full(2000, 6.0)])
    direction = np.tile([0.0, 0.0, -1.0], (2000, 1))

    ev = trace(geom, origin, direction, CROWN_GLASS, max_bounces=14)
    energy = (ev.weight.sum() + ev.lost.sum()) / 2000
    assert 0.99 < energy < 1.01, f"energy not conserved: {energy}"


def test_round_trip_through_the_dict_form():
    restored = VesselParams.from_dict(PITCHER.to_dict())
    assert restored.profile == PITCHER.profile
    assert restored.n_base == PITCHER.n_base
    assert build_vessel(restored).n_facets == build_vessel(PITCHER).n_facets
