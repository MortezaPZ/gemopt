"""Unit wedge + N-fold copy: PNG and MaxScript share one layout."""

import math
from pathlib import Path

import pytest

from gemopt.diagram import (
    TABLE,
    assert_n_fold_symmetric,
    draw_pattern,
    homologous_optics,
    rotation_symmetry_error,
    save_pattern_png,
)
from gemopt.params import SearchSpace, bind_symmetry
from gemopt.pattern import layout_from_n, radial_array, sector_degrees, unit_wedge
from gemopt.pattern_maxscript import write_pattern_ms


@pytest.mark.parametrize("n", [6, 8, 12, 16])
def test_sector_is_exactly_360_over_n(n):
    assert sector_degrees(n) == pytest.approx(360.0 / n)
    layout = layout_from_n(n)
    assert layout.sector_deg == pytest.approx(360.0 / n)
    assert layout.n == n


def test_table_and_valley_fractions_match_across_n():
    a = layout_from_n(8)
    b = layout_from_n(16)
    assert a.table_radius / a.radius == pytest.approx(b.table_radius / b.radius)
    assert a.valley_radius / a.radius == pytest.approx(b.valley_radius / b.radius)
    assert a.valley_radius < a.radius
    assert a.table_radius < a.valley_radius


def test_spokes_start_on_the_table_not_the_origin():
    for n in (8, 16):
        layout = layout_from_n(n)
        w = unit_wedge(layout, 0)
        assert w.left_poly[0] == w.table
        assert w.right_poly[0] == w.table
        assert math.hypot(*w.table) == pytest.approx(layout.table_radius)
        assert w.table != w.origin


def test_stroke_shrinks_with_sqrt_n():
    a = layout_from_n(8)
    b = layout_from_n(16)
    assert b.stroke_width == pytest.approx(a.stroke_width * math.sqrt(0.5))
    assert b.halo_count == a.halo_count
    assert b.n_gon_stroke == pytest.approx(a.n_gon_stroke * math.sqrt(0.5))


def test_wedges_tile_without_gap_or_overlap():
    layout = layout_from_n(16)
    wedges = radial_array(layout)
    assert len(wedges) == 16
    for i, w in enumerate(wedges):
        nxt = wedges[(i + 1) % 16]
        assert w.valley_right == pytest.approx(nxt.valley_left)
    first = wedges[0]
    last = wedges[-1]
    assert first.valley_left == pytest.approx(last.valley_right)


def test_unit_wedge_spread_is_the_sector():
    layout = layout_from_n(8)
    w = unit_wedge(layout, 0)
    a = math.degrees(math.atan2(w.valley_left[1], w.valley_left[0]))
    b = math.degrees(math.atan2(w.valley_right[1], w.valley_right[0]))
    span = (b - a) % 360.0
    assert span == pytest.approx(layout.sector_deg, abs=1e-6)


@pytest.mark.parametrize("n", [8, 16])
def test_rotation_symmetry_passes(n):
    image = draw_pattern(n, size=900)
    err = assert_n_fold_symmetric(image, n, limit=0.04)
    assert err < 0.04


def test_png_and_maxscript_use_the_same_n_and_offset(tmp_path):
    layout = layout_from_n(16)
    png = save_pattern_png(tmp_path / "gemstone_16_facets.png", layout, size=400)
    ms = write_pattern_ms(tmp_path / "gemstone_16_facets.ms", layout)
    text = Path(ms).read_text(encoding="utf-8")
    assert png.exists() and png.stat().st_size > 1000
    assert "n = 16" in text
    assert "sides:n" in text
    assert f"{layout.sector_deg:.5f}" in text
    assert f"{layout.rotation_offset_deg:.5f}" in text
    assert "wedge_00" in text and "wedge_15" in text
    assert "tableRadius" in text
    assert "table_outline" in text
    assert "ngon_outline" in text
    assert "gemopt_table" in text


def test_bind_symmetry_resets_n_and_does_not_reuse_n8_defaults():
    space = bind_symmetry(SearchSpace(), 16)
    params = space.to_params(space.start())
    assert params.n_base == 16
    assert "n_base" not in [s.name for s in space.free()]
    layout = layout_from_n(params.n_base)
    assert layout.sector_deg == pytest.approx(22.5)


@pytest.mark.parametrize("n", [8, 16])
def test_center_is_the_dark_table(n):
    import numpy as np

    img = np.asarray(draw_pattern(n, size=400))
    patch = img[195:206, 195:206].reshape(-1, 3).mean(axis=0)
    assert np.allclose(patch, TABLE, atol=18)


def test_homologous_optics_is_sixteen_fold():
    import numpy as np

    src = np.asarray(draw_pattern(8, size=256), dtype=np.float64) / 255.0
    dst = homologous_optics(src, 16)
    err = rotation_symmetry_error(dst * 255.0, 16)
    assert err < 0.08
    # the copied still still has a dark table, not needles at the origin
    patch = (dst[124:133, 124:133] * 255.0).reshape(-1, 3).mean(axis=0)
    assert patch.mean() < 80.0

