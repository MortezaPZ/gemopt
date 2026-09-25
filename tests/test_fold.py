"""N-fold remapping: an 8-arrow photo must score a 16-main stone as 16 arrows."""

import numpy as np
import pytest

from gemopt.fold import align_view_to_n, detect_fold, remap_polar_fold
from gemopt.geometry import brilliant_for_n, build_stone, eight_fold_twin, facet_count
from gemopt.params import SearchSpace, bind_symmetry


def _arrow_polar(n: int, n_r: int = 64, n_theta: int = 256) -> np.ndarray:
    """Synthetic hearts-and-arrows: N bright shafts, dark between them."""
    rr = (np.arange(n_r) + 0.5) / n_r
    tt = np.arange(n_theta) / n_theta * 2.0 * np.pi
    R, T = np.meshgrid(rr, tt, indexing="ij")
    # shafts: raised-cosine lobes, N of them, stronger in the mid-radius
    shafts = 0.55 + 0.45 * np.cos(n * T)
    radial = np.clip(1.0 - 2.0 * np.abs(R - 0.55), 0.1, 1.0)
    return shafts * radial


def test_detects_eight_fold_arrows():
    assert detect_fold(_arrow_polar(8)) == 8


def test_detects_sixteen_fold_arrows():
    assert detect_fold(_arrow_polar(16)) == 16


def test_remap_turns_eight_arrows_into_sixteen():
    src = _arrow_polar(8)
    dst = remap_polar_fold(src, 8, 16)
    assert detect_fold(dst) == 16
    # one destination sector is the source sector, compressed
    assert dst.shape == src.shape


def test_eight_to_eight_stays_eight_fold():
    src = _arrow_polar(8)
    dst = remap_polar_fold(src, 8, 8)
    assert detect_fold(dst) == 8


def test_align_rewrites_rgb_only_when_the_fold_changes():
    rgb = np.zeros((40, 40, 3))
    rgb[:, :, 0] = 0.8
    polar = _arrow_polar(8, n_r=32, n_theta=128)
    p8, rgb8, src = align_view_to_n(polar, rgb, 8)
    assert src == 8
    assert np.allclose(rgb8, rgb)
    p16, rgb16, src16 = align_view_to_n(polar, rgb, 16)
    assert src16 == 8
    assert detect_fold(p16) == 16
    assert rgb16.shape == rgb.shape


def test_brilliant_for_n_changes_topology_not_angles():
    a = brilliant_for_n(8)
    b = brilliant_for_n(16)
    assert a.n_base == 8 and b.n_base == 16
    assert a.pavilion_angle == b.pavilion_angle
    assert a.crown_angle == b.crown_angle
    mesh = build_stone(b)
    assert mesh.is_watertight()
    assert facet_count(b) == 3 * 16 + 2 * 2 * 16 + 1


def test_brilliant_for_n_accepts_n_d_without_killing_the_cut():
    diamond = brilliant_for_n(16, n_d=2.4175)
    glass = brilliant_for_n(16, n_d=1.52)
    assert diamond.n_base == glass.n_base == 16
    # starting angles stay cuttable; the search, not the start, owns glass
    assert not glass.validate()
    assert glass.pavilion_angle == pytest.approx(40.75)


def test_bind_symmetry_uses_brilliant_for_n():
    space = bind_symmetry(SearchSpace(), 16)
    params = space.to_params(space.start())
    assert params.n_base == 16
    assert params.pavilion_angle == pytest.approx(40.75)


def test_eight_fold_twin_keeps_the_cut_and_drops_to_eight_mains():
    src = brilliant_for_n(16, extra_crown_rows=1, table=0.56)
    twin = eight_fold_twin(src)
    assert twin.n_base == 8
    assert twin.table == src.table
    assert twin.crown_angle == src.crown_angle
    assert twin.extra_crown_rows == src.extra_crown_rows
    assert build_stone(twin).is_watertight()

