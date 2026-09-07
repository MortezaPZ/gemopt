"""Geometry: the stone has to be a real, closed, correctly counted solid."""

import collections
import math

import numpy as np
import pytest

from gemopt.geometry import (
    STANDARD_BRILLIANT,
    CutParams,
    build_stone,
    facet_count,
)


def test_standard_brilliant_has_57_facets():
    """The trade count for a round brilliant, and the whole point of the
    parameterisation is that it reproduces it exactly rather than approximately."""
    assert facet_count(STANDARD_BRILLIANT) == 57
    assert facet_count(STANDARD_BRILLIANT, include_girdle=True) == 73


@pytest.mark.parametrize("n", [6, 8, 10, 12])
def test_facet_count_follows_7n_plus_1(n):
    p = CutParams(n_base=n)
    assert facet_count(p) == 7 * n + 1
    assert build_stone(p).n_facets == 7 * n + 1 + 2 * n


def test_mesh_is_closed_and_solid():
    mesh = build_stone(STANDARD_BRILLIANT)
    assert mesh.is_watertight()
    # positive volume means every normal ended up pointing outwards
    assert mesh.volume() > 0.0


def test_all_normals_point_away_from_the_centre():
    mesh = build_stone(STANDARD_BRILLIANT)
    centres = mesh.triangle_vertices().mean(axis=1)
    outward = np.einsum("ij,ij->i", mesh.normals(), centres - mesh.vertices.mean(axis=0))
    assert np.all(outward > 0.0)


def test_polygon_winding_matches_triangles():
    """3ds Max reads the polygons, the tracer reads the triangles; if they
    disagree about which way is out, the render and the export differ."""
    mesh = build_stone(STANDARD_BRILLIANT)
    normals = mesh.normals()
    for fid, poly in enumerate(mesh.polygons):
        v = mesh.vertices[poly]
        pn = np.cross(v[1] - v[0], v[2] - v[0])
        pn /= np.linalg.norm(pn)
        tn = normals[mesh.facet_id == fid][0]
        assert pn @ tn > 0.99


def test_girdle_radius_is_one():
    mesh = build_stone(STANDARD_BRILLIANT)
    assert math.isclose(np.linalg.norm(mesh.vertices[:, :2], axis=1).max(), 1.0, rel_tol=1e-9)


def test_facet_angles_match_the_requested_parameters():
    """Crown and pavilion angles are what a cutter dials in; the mesh has to
    actually come out at those angles."""
    p = CutParams(crown_angle=34.5, pavilion_angle=40.75)
    mesh = build_stone(p)
    groups = np.array(mesh.facet_group)[mesh.facet_id]
    normals = mesh.normals()

    bezel = normals[groups == "bezel"]
    tilt = np.degrees(np.arccos(np.abs(bezel[:, 2]))).mean()
    assert abs(tilt - p.crown_angle) < 1.0

    pav = normals[groups == "pavilion_main"]
    tilt = np.degrees(np.arccos(np.abs(pav[:, 2]))).mean()
    assert abs(tilt - p.pavilion_angle) < 1.5


def test_derived_depth_matches_the_mesh():
    p = STANDARD_BRILLIANT
    mesh = build_stone(p)
    lo, hi = mesh.bounds()
    measured = (hi[2] - lo[2]) / 2.0 * 100.0     # % of the diameter
    assert abs(measured - p.derived()["total_depth_pct"]) < 1.5


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"pavilion_angle": 80.0}, "pavilion angle"),
        ({"crown_angle": 80.0}, "crown angle"),
        ({"table": 0.99}, "table"),
        ({"girdle_thick_main": 0.05, "girdle_thick_half": 0.02}, "thinnest point"),
        ({"girdle_thick_main": -0.01}, "positive"),
        ({"culet": 0.9}, "culet"),
    ],
)
def test_invalid_stones_are_rejected(kwargs, reason):
    """Rejecting before tracing is what keeps the search from wasting a
    second per impossible candidate."""
    problems = CutParams(**kwargs).validate()
    assert problems, f"expected {kwargs} to be rejected"
    assert any(reason in p for p in problems)


def test_standard_brilliant_is_valid():
    assert STANDARD_BRILLIANT.validate() == []


def test_optional_rows_change_the_count():
    p = CutParams(star_row=False)
    assert facet_count(p) == 57 - 8
    p = CutParams(lower_girdle_row=False)
    assert facet_count(p) == 57 - 16
    assert build_stone(p).is_watertight()


def test_culet_adds_a_facet_and_stays_closed():
    p = CutParams(culet=0.04)
    assert facet_count(p) == 58
    assert build_stone(p).is_watertight()


def test_a_flat_culet_leaves_every_pavilion_main_in_one_plane():
    """The trade counts 58 facets, not 66: a truncated kite is one polished
    plane, so its five corners have to be coplanar.  They are only coplanar if
    the culet ring sits where adjacent main planes actually meet the flat --
    at radius culet/cos(half step), not at radius culet."""
    import numpy as np

    for p in (CutParams(culet=0.04),
              CutParams(culet=0.06, n_base=6),
              CutParams(culet=0.02, n_base=16, girdle_multiple=3)):
        mesh = build_stone(p)
        mains = [poly for fid, poly in enumerate(mesh.polygons)
                 if mesh.facet_group[fid] == "pavilion_main"]
        assert len(mains) == p.n_base, "the mains must not be split up"
        for poly in mains:
            assert len(poly) == 5, "kite truncated by the culet is a pentagon"
            v = mesh.vertices[poly]
            normal = np.cross(v[1] - v[0], v[2] - v[0])
            normal /= np.linalg.norm(normal)
            assert np.abs((v - v[0]) @ normal).max() < 1e-9


def test_params_roundtrip_through_dict():
    p = CutParams(crown_angle=31.2, table=0.61, culet=0.02)
    assert CutParams.from_dict(p.to_dict()) == p


def test_scale_preserves_shape():
    mesh = build_stone(STANDARD_BRILLIANT)
    big = mesh.scale(6.5)
    assert big.is_watertight()
    assert math.isclose(big.volume(), mesh.volume() * 6.5**3, rel_tol=1e-9)


def _facet_tilts(mesh):
    """Mean tilt from horizontal, in degrees, per facet group."""
    import collections
    n = mesh.normals()
    groups = np.array(mesh.facet_group)[mesh.facet_id]
    out = {}
    for name in set(groups):
        sel = groups == name
        out[name] = float(
            np.degrees(np.arccos(np.clip(np.abs(n[sel][:, 2]), 0, 1))).mean()
        )
    return out


def test_main_facets_are_exactly_flat():
    """Every facet must be planar to machine precision.

    A facet is a single polished surface; if the triangles it was split into
    disagree even slightly, the stone reflects light along a fold that does
    not exist on a real gem.
    """
    mesh = build_stone(CutParams())
    normals = mesh.normals()
    for fid in range(mesh.n_facets):
        sel = mesh.facet_id == fid
        n = normals[sel]
        assert np.allclose(n, n[0], atol=1e-9), f"facet {fid} is not planar"


def test_bezel_and_pavilion_match_their_requested_angles():
    """The two angles the user sets must be the angles they get."""
    p = CutParams(crown_angle=34.5, pavilion_angle=40.75)
    tilts = _facet_tilts(build_stone(p))
    assert tilts["bezel"] == pytest.approx(34.5, abs=0.01)
    assert tilts["pavilion_main"] == pytest.approx(40.75, abs=0.01)


def test_girdle_facets_break_away_from_the_mains():
    """The angular break that makes hearts and arrows exist.

    Girdle facets lying flush with the main facets turn the pavilion into a
    smooth cone, which reflects light uniformly and draws no pattern at all.
    This guards the geometry against sliding back into that state.
    """
    tilts = _facet_tilts(build_stone(CutParams()))
    assert tilts["lower_girdle"] - tilts["pavilion_main"] > 0.5
    assert tilts["upper_girdle"] - tilts["bezel"] > 2.0
    assert tilts["bezel"] - tilts["star"] > 2.0


def test_pattern_has_structure_in_the_middle_not_just_at_the_rim():
    """A brilliant's arrows live inside the table, not around the edge."""
    from gemopt.calibrate import _angle_map
    from gemopt.metrics import to_polar
    from gemopt.optics import DIAMOND

    amap, _ = _angle_map(build_stone(CutParams()), DIAMOND, "crown", 160)
    polar = to_polar(np.nan_to_num(amap, nan=float(np.nanmean(amap))),
                     radius=80 * 0.94, n_r=48, n_theta=128)
    variance = polar.var(axis=1)
    inner = variance[6:24].mean()      # the table region
    outer = variance[24:].mean()
    assert inner > 0.2 * outer, "the middle of the stone is featureless"


@pytest.mark.parametrize("n,m", [(6, 2), (8, 1), (8, 2), (8, 3), (8, 4),
                                 (12, 2), (16, 2), (16, 3)])
def test_row_counts_can_be_changed_without_breaking_the_stone(n, m):
    """Any base symmetry and girdle multiple must still close up.

    This is the client's requirement in code: rows may be re-divided, but the
    facets have to stay in register and the solid has to stay a solid.
    """
    mesh = build_stone(CutParams(n_base=n, girdle_multiple=m))
    assert mesh.is_watertight()
    assert mesh.volume() > 0

    groups = collections.Counter(mesh.facet_group)
    assert groups["pavilion_main"] == n
    assert groups["star"] == n
    assert groups["bezel"] == n
    assert groups["upper_girdle"] == m * n
    assert groups["lower_girdle"] == m * n
    assert groups["girdle"] == m * n

    normals = mesh.normals()
    for fid in range(mesh.n_facets):
        f = normals[mesh.facet_id == fid]
        assert np.allclose(f, f[0], atol=1e-9), "a facet came out bent"


def test_facet_count_matches_the_formula():
    """3N + 2mN + 1, and 57 for the classic round brilliant."""
    assert facet_count(CutParams(n_base=8, girdle_multiple=2)) == 57
    for n in (6, 8, 12, 16):
        for m in (1, 2, 3, 4):
            p = CutParams(n_base=n, girdle_multiple=m)
            assert facet_count(p) == 3 * n + 2 * m * n + 1


def test_rows_stay_in_register():
    """Every main facet must land on a girdle vertex, at every multiple.

    If they drift apart the pattern loses the alignment that makes hearts and
    arrows appear at all.
    """
    for m in (1, 2, 3, 4):
        p = CutParams(n_base=8, girdle_multiple=m)
        mesh = build_stone(p)
        groups = np.array(mesh.facet_group)[mesh.facet_id]
        mains = mesh.triangle_vertices()[groups == "pavilion_main"].reshape(-1, 3)
        rim = mains[np.isclose(np.hypot(mains[:, 0], mains[:, 1]), 1.0, atol=1e-9)]
        step = 360.0 / p.n_base
        azimuths = np.degrees(np.arctan2(rim[:, 1], rim[:, 0])) % step
        assert np.allclose(np.minimum(azimuths, step - azimuths), 0.0, atol=1e-6)


@pytest.mark.parametrize("crown_rows", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("pavilion_rows", [0, 1, 2])
def test_extra_rows_keep_the_stone_a_stone(crown_rows, pavilion_rows):
    """More rows is what the client asked for in the video.

    Every combination has to stay a closed solid made of flat facets.  Both
    are easy to lose here: adding a tier moves where the bezel and pavilion
    main terminate, and a quad spanning two rings is only flat when its radial
    edges stay parallel.
    """
    p = CutParams(extra_crown_rows=crown_rows, extra_pavilion_rows=pavilion_rows)
    assert p.validate() == []
    mesh = build_stone(p)
    assert mesh.is_watertight(), "adding a row opened a hole"
    assert mesh.volume() > 0

    normals = mesh.normals()
    for fid in range(mesh.n_facets):
        f = normals[mesh.facet_id == fid]
        assert np.allclose(f, f[0], atol=1e-9), f"facet {fid} came out folded"

    assert facet_count(p, include_girdle=True) == mesh.n_facets


def test_more_rows_means_more_facets():
    """The count has to actually grow, or the control does nothing."""
    counts = [build_stone(CutParams(extra_crown_rows=r)).n_facets for r in range(4)]
    assert counts == sorted(counts) and len(set(counts)) == 4


def test_tiers_break_the_angle_rather_than_just_subdividing():
    """A row that shares its neighbour's plane is not a row.

    This is the failure that once turned the pavilion into a smooth cone and
    erased the pattern; the tiers have to fan outwards.
    """
    mesh = build_stone(CutParams(extra_crown_rows=2, crown_angle=34.5,
                                 crown_outer_angle=50.0))
    groups = np.array(mesh.facet_group)[mesh.facet_id]
    normals = mesh.normals()
    tilts = sorted({round(float(np.degrees(np.arccos(abs(n[2])))), 3)
                    for n in normals[groups == "crown_tier"]})
    assert len(tilts) >= 2, "every tier ended up on the same plane"
    assert max(tilts) - min(tilts) > 2.0


def _girdle_thicknesses(p):
    """Every distinct top-to-bottom thickness around the girdle band."""
    mesh = build_stone(p)
    columns = {}
    for fid, poly in enumerate(mesh.polygons):
        if mesh.facet_group[fid] != "girdle":
            continue
        for i in poly:
            v = mesh.vertices[i]
            key = round(math.degrees(math.atan2(v[1], v[0])) % 360, 3)
            columns.setdefault(key, []).append(v[2])
    return sorted({round(max(z) - min(z), 9)
                   for z in columns.values() if len(z) > 1})


def test_a_flat_girdle_has_one_thickness_all_the_way_round():
    """The client's stones are machine-cut, so the band is a straight
    cylinder -- not the scallop that falls out of grinding the mains."""
    assert _girdle_thicknesses(CutParams(flat_girdle=True)) == [0.020]


def test_the_scalloped_girdle_is_still_available():
    """Turning the option off has to actually bring the waviness back."""
    thicknesses = _girdle_thicknesses(CutParams(flat_girdle=False))
    assert thicknesses == [0.020, 0.035]
