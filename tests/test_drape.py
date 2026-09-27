"""The two ways a pattern sits on the stone this program already builds.

Tight keeps every horizontal position, so an orthographic view from above
reproduces the picture inside the silhouette.  Loose measures arc length, so
on the flat table the same motif is seen farther from the centre than it sits
in the picture.
"""

from pathlib import Path

import numpy as np
import pytest

from gemopt.drape import frame_half_width, planar_uv, surface_uv, top_view
from gemopt.geometry import CutParams, build_stone
from gemopt.maxbridge import read_gem, write_gem
from gemopt.scope import Camera

SCRIPT = Path("maxscript/import_stone.ms")


def _stone():
    return build_stone(CutParams())


def _picture(mesh, size=192):
    """Pixel centres in the same frame ``top_view`` casts its rays through."""
    half = frame_half_width(mesh)
    col = (np.arange(size) + 0.5) / size
    row = (np.arange(size) + 0.5) / size
    xs = (col * 2.0 - 1.0) * half
    ys = (1.0 - row * 2.0) * half
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    image = np.stack((grid_x, grid_y, np.ones_like(grid_x)), axis=2)
    return image, grid_x, grid_y


def test_tight_top_view_matches_the_pattern():
    mesh = _stone()
    image, _, _ = _picture(mesh)
    view = top_view(mesh, planar_uv(mesh), image)

    assert view.hit.any()
    assert np.array_equal(view.image[view.hit], image[view.hit])


def test_loose_top_view_stretches_on_the_table():
    """The table is flat, but the crown beyond it makes the arc longer than
    the plan radius, so a motif is seen farther out than where it was drawn.

    The band stays inside the table octagon.  Its apothem is the table radius
    times cos(22.5°), about 0.49, and 0.40 is short of that.
    """
    mesh = _stone()
    image, grid_x, grid_y = _picture(mesh)
    view = top_view(mesh, surface_uv(mesh), image)

    screen_r = np.hypot(grid_x, grid_y)[view.hit]
    motif_r = np.hypot(view.image[view.hit][:, 0], view.image[view.hit][:, 1])
    on_table = (screen_r > 0.15) & (screen_r < 0.40)

    assert on_table.sum() > 20
    ratio = screen_r[on_table] / motif_r[on_table]
    assert np.min(ratio) > 1.05
    # A profile that walked the crown twice would push this well past the
    # arc-to-radius ratio of a normal brilliant, which sits near 1.10.
    assert np.median(ratio) < 1.25
    assert not np.array_equal(view.image[view.hit], image[view.hit])


def test_the_max_script_maps_both_ways_without_drape():
    source = SCRIPT.read_text(encoding="utf-8-sig")
    assert "gemPlanarUV" in source
    assert "gemSurfaceUV" in source
    assert "Drape" not in source
    show = source.split("fn gemShowPattern", 1)[1]
    assert "gemPatternPath = gemPattern" in show
    assert "destroyDialog gemPatternRollout" in show
    assert "fn gemProfile" in source
    assert "fn gemFitTol" in source
    frame = source.split("fn gemFrameHalf", 1)[1].split("\nfn ", 1)[0]
    assert "and (given > 0" not in frame


def _image_radius(uv, half):
    xy = (uv - 0.5) * (2.0 * half)
    return np.hypot(xy[:, 0], xy[:, 1])


def _loose_reaches_the_girdle(mesh, half=None):
    """Every vertex lies on its profile, the crown edge keeps its plan radius,
    and the pavilion runs past that edge."""
    half = frame_half_width(mesh, half)
    rho = np.linalg.norm(mesh.vertices[:, :2], axis=1)
    uv = surface_uv(mesh, half)
    assert np.isfinite(uv).all()
    image_r = _image_radius(uv, half)
    scale = max(1.0, float(rho.max()))
    z = mesh.vertices[:, 2]
    top = np.flatnonzero(rho >= rho.max() - 1e-4 * scale)
    crown = top[z[top] >= z[top].max() - 1e-4 * scale]
    assert crown.size
    assert np.allclose(image_r[crown], rho[crown], atol=1e-3 * scale)
    # A cut with no break facets has no vertex between the girdle and the
    # culet.  Where those vertices exist, the arc has already left the picture.
    low = (z < z[crown].min() - 1e-3 * scale) & (rho > 0.05 * scale) & (rho < rho.max() - 1e-3 * scale)
    if low.any():
        assert np.min(image_r[low] - rho[low]) > 1e-3 * scale
    on_axis = rho <= 1e-6 * scale
    if on_axis.any():
        assert np.allclose(uv[on_axis], 0.5, atol=1e-8)


@pytest.mark.parametrize("params", [
    CutParams(table=0.85),
    CutParams(n_base=24),
    CutParams(extra_crown_rows=2, extra_pavilion_rows=2),
    CutParams(n_base=16, culet=0.04, flat_girdle=False, extra_crown_rows=1,
              extra_pavilion_rows=1, girdle_multiple=2, crown_twist=7.0),
    CutParams(star_row=False),
    CutParams(star_row=False, upper_girdle_row=False, lower_girdle_row=False),
])
def test_loose_survives_shapes_that_split_the_profile(params, tmp_path):
    mesh = build_stone(params)
    _loose_reaches_the_girdle(mesh)
    path = write_gem(tmp_path / "stone.gem", mesh, params=params, scale=8.0)
    rounded = read_gem(path)
    _loose_reaches_the_girdle(rounded, half=Camera.extent * 8.0)


def test_tight_matches_inside_the_camera_frame():
    """The picture the program saves is wider than the girdle, and not always square."""
    mesh = _stone()
    radius = float(np.linalg.norm(mesh.vertices[:, :2], axis=1).max())
    half = Camera.extent * radius
    width, height = 70, 36
    col = (np.arange(width) + 0.5) / width
    row = (np.arange(height) + 0.5) / height
    xs = (col * 2.0 - 1.0) * half
    ys = (1.0 - row * 2.0) * half
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    image = np.stack((grid_x, grid_y, np.ones_like(grid_x)), axis=2)
    view = top_view(mesh, planar_uv(mesh, half), image, half)

    assert np.array_equal(view.image[view.hit], image[view.hit])
    interior = np.hypot(grid_x, grid_y) < 0.40 * radius
    assert interior.any() and view.hit[interior].all()


def test_a_steep_face_packs_the_tight_picture_along_the_surface():
    mesh = _stone()
    uv = planar_uv(mesh)
    half = frame_half_width(mesh)
    found = False
    for tri in mesh.triangles:
        pairs = ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
        for a, b in pairs:
            edge = mesh.vertices[b] - mesh.vertices[a]
            horizontal = float(np.linalg.norm(edge[:2]))
            length = float(np.linalg.norm(edge))
            if length < 0.05 or horizontal < 1e-3 or length < horizontal * 1.2:
                continue
            along_picture = float(np.linalg.norm(uv[b] - uv[a])) * 2.0 * half
            assert abs(along_picture - horizontal) < 1e-8
            assert along_picture < length * 0.95
            found = True
            break
        if found:
            break
    assert found
