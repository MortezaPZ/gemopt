"""One drawing stays one drawing. A repeating stone still changes its count."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]

from gemopt.app import _crop_to_stone
from gemopt.design import render_design, retarget_views
from gemopt.diagram import rotation_symmetry_error
from gemopt.fold import detect_fold, remap_image_fold
from gemopt.geometry import CutParams
from gemopt.inverse import InverseProblem, load_single, to_polar
from gemopt.optics import MATERIALS


def _save(path, image):
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8)).save(path)


def _circle(draw, size):
    margin = 10
    draw.ellipse((margin, margin, size - margin, size - margin), outline=(50, 50, 50), width=2)


def _butterfly(size=320):
    image = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    _circle(draw, size)
    cx = cy = size / 2
    draw.polygon(
        [(cx, cy - 8), (cx - 128, cy - 72), (cx - 108, cy + 4), (cx - 126, cy + 78), (cx, cy + 18)],
        fill=(150, 196, 230),
    )
    draw.polygon(
        [(cx, cy - 8), (cx + 128, cy - 72), (cx + 108, cy + 4), (cx + 126, cy + 78), (cx, cy + 18)],
        fill=(150, 196, 230),
    )
    draw.polygon([(cx - 7, cy - 52), (cx + 7, cy - 52), (cx + 4, cy + 58), (cx - 4, cy + 58)], fill=(28, 48, 78))
    return np.asarray(image, dtype=np.float64) / 255.0


def _leaf(size=320):
    image = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    _circle(draw, size)
    cx = cy = size / 2
    draw.polygon(
        [(cx - 18, cy + 36), (cx + 24, cy - 96), (cx + 108, cy - 8), (cx + 30, cy + 70)],
        fill=(46, 158, 72),
    )
    draw.line([(cx - 6, cy + 28), (cx + 70, cy - 20)], fill=(18, 72, 32), width=4)
    return np.asarray(image, dtype=np.float64) / 255.0


def _arrows(n=8, size=321):
    yy, xx = np.mgrid[:size, :size]
    radius = np.hypot(xx - (size - 1) / 2, yy - (size - 1) / 2) / (size * 0.46)
    theta = np.arctan2(yy - (size - 1) / 2, xx - (size - 1) / 2)
    shaft = np.exp(-((radius - 0.54) / 0.29) ** 8)
    opening = (1 + np.cos(n * theta)) / 2
    arrow = shaft * opening ** (2 + 12 * np.exp(-((radius - 0.45) / 0.13) ** 2))
    image = np.zeros((size, size, 3)) + np.array((0.85, 0.12, 0.25))
    image = image * (1 - arrow[..., None]) + 0.93 * arrow[..., None]
    image[radius < 0.25] = np.array((0.85, 0.12, 0.25)) * 0.35
    image[radius > 1] = 0
    return image


def _loaded(tmp_path, image, name):
    path = tmp_path / name
    _save(path, image)
    return load_single(path, view="crown", n_fold=None, source_n=None, radius_scale=1.0)[0]


def _stays_itself(view):
    assert view.repeats is False
    assert detect_fold(view.target) is None
    for count in (8, 16):
        shown = retarget_views([view], count)[0]
        assert shown.repeats is False
        np.testing.assert_allclose(shown.rgb, view.rgb_native, atol=1e-6)
        assert rotation_symmetry_error(shown.rgb * 255.0, 8) > 0.03
        tiled = remap_image_fold(view.rgb_native, view.source_fold, count)
        assert np.mean(np.abs(shown.rgb - tiled)) > 0.02


def test_a_butterfly_stays_one_butterfly(tmp_path):
    view = _loaded(tmp_path, _butterfly(), "butterfly.png")
    _stays_itself(view)
    rgb = retarget_views([view], 8)[0].rgb
    mid = rgb.shape[1] // 2
    assert rgb[:, :mid, 2].mean() > 0.02
    assert rgb[:, mid:, 2].mean() > 0.02


def test_another_single_drawing_is_not_tiled(tmp_path):
    view = _loaded(tmp_path, _leaf(), "leaf.png")
    _stays_itself(view)
    rgb = retarget_views([view], 8)[0].rgb
    green = rgb[:, :, 1]
    assert green.max() > 0.4
    assert rotation_symmetry_error(rgb * 255.0, 3) > 0.03


def test_section_two_puts_the_drawing_on_a_round_stone(tmp_path, monkeypatch):
    from queue import Queue

    from gemopt.app import Job, _export_design
    from gemopt.tomax import MaxResult

    path = tmp_path / "butterfly.png"
    _save(path, _butterfly())
    job = Job(photo=str(path), single=True, view="crown", material="glass", ior=None,
              target_kind="pattern", n_base=8, girdle_multiple=2, extra_crown_rows=0,
              extra_pavilion_rows=0, generations=1, popsize=1, workers=1,
              outdir=str(tmp_path / "out"), diameter_mm=6.5, section="design")

    def _no_max(*_args, **_kwargs):
        return MaxResult(ok=False, path=None, message="no max", script=None)

    monkeypatch.setattr("gemopt.tomax.gem_to_max", _no_max)
    _export_design(job, Queue())
    out = np.asarray(__import__("PIL").Image.open(tmp_path / "out" / "out_crown.png").convert("RGB"))
    assert rotation_symmetry_error(out, 8) > 0.03
    mid = out.shape[1] // 2
    assert out[:, :mid, 2].mean() > 5
    assert out[:, mid:, 2].mean() > 5
    assert (tmp_path / "out" / "stone.obj").exists()
    assert (tmp_path / "out" / "stone.gem").exists()


def test_the_employer_butterfly_is_not_an_eight_column_stone():
    from gemopt.design_output import stone_with_design

    path = ROOT / "tests" / "fixtures" / "employer_butterfly.png"
    view = load_single(path, view="crown", n_fold=None, source_n=None, radius_scale=1.0)[0]
    assert view.repeats is False
    assert detect_fold(view.target) is None
    shown = stone_with_design(view.rgb_native)
    assert rotation_symmetry_error(shown * 255.0, 8) > 0.04
    mid = shown.shape[1] // 2
    assert shown[:, :mid, 2].mean() > 0.05
    assert shown[:, mid:, 2].mean() > 0.05
    tiled = remap_image_fold(view.rgb_native, 8, 8)
    assert np.mean(np.abs(shown - tiled)) > 0.05


def test_a_real_multi_arrow_stone_still_changes_its_count():
    path = ROOT / "gemstone_8_facets.png"
    view = load_single(path, view="crown", n_fold=None, source_n=None, radius_scale=1.0)[0]
    assert view.repeats is True
    assert detect_fold(view.target) == 8
    shown = retarget_views([view], 16)[0]
    assert shown.repeats is True
    assert detect_fold(to_polar(shown.rgb)) == 16
    assert np.mean(np.abs(shown.rgb - view.rgb_native)) > 0.02


def test_a_repeating_stone_still_changes_its_count(tmp_path):
    view = _loaded(tmp_path, _arrows(8), "arrows.png")
    assert view.repeats is True
    assert detect_fold(view.target) == 8
    shown = retarget_views([view], 16)[0]
    assert shown.repeats is True
    assert detect_fold(to_polar(shown.rgb)) == 16
    assert np.mean(np.abs(shown.rgb - view.rgb_native)) > 0.02
    assert rotation_symmetry_error(shown.rgb * 255.0, 16) < 0.04


def test_the_cap_sits_on_the_pavilion_inside_the_table():
    """Fixed angles. The drawing is on the lower side and no larger than the table."""
    from gemopt.design_output import (
        pavilion_with_design, round_stone, table_half_width,
    )
    from gemopt.drape import pavilion_uv
    from gemopt.geometry import CutParams

    photo = np.asarray(
        Image.open(ROOT / "tests" / "fixtures" / "employer_butterfly.png").convert("RGB"),
        dtype=np.float64,
    ) / 255.0
    mesh = round_stone()
    table = float(CutParams().table)
    half = table_half_width(mesh, 1.0)
    girdle = float(np.linalg.norm(mesh.vertices[:, :2], axis=1).max())
    assert abs(half - table * girdle) < 1e-12
    assert half < girdle

    uv = pavilion_uv(mesh, half)
    rho = np.linalg.norm(mesh.vertices[:, :2], axis=1)
    lower = mesh.vertices[:, 2] < -1e-6
    upper = mesh.vertices[:, 2] > 1e-6
    assert np.all(uv[upper, 0] < 0.0)
    offset = np.hypot(uv[lower, 0] - 0.5, uv[lower, 1] - 0.5)
    np.testing.assert_allclose(offset, rho[lower] / (2.0 * half), atol=1e-9)
    culet = int(np.argmin(mesh.vertices[:, 2]))
    np.testing.assert_allclose(uv[culet], (0.5, 0.5), atol=1e-9)
    # The table circle is the picture edge. Past it the coordinate leaves the picture.
    assert np.all(offset[rho[lower] <= half + 1e-9] <= 0.5 + 1e-9)
    assert np.any(offset[rho[lower] > half + 0.05] > 0.5)

    shown = pavilion_with_design(photo)
    assert rotation_symmetry_error(shown * 255.0, 8) > 0.04
    height, width = shown.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    centre = (width - 1) / 2.0
    radial = np.hypot(xx - centre, yy - centre)
    cap = radial <= (min(height, width) * 0.48 * table)
    blue = shown[:, :, 2] - shown[:, :, 0]
    assert float(blue[cap].mean()) > 0.02
    assert float(blue[~cap].mean()) < float(blue[cap].mean())
    mid = width // 2
    assert shown[:, :mid, 2].mean() > 0.02
    assert shown[:, mid:, 2].mean() > 0.02


def test_the_picture_stops_on_the_table_face():
    """From above, the drawing is on the table polygon and not on the stars."""
    from gemopt.design_output import round_stone, table_half_width, table_with_design
    from gemopt.drape import cap_face_uv
    from gemopt.geometry import CutParams

    mesh = round_stone()
    half = table_half_width(mesh, 1.0)
    faces = cap_face_uv(mesh, half)
    for uv, group in zip(faces, mesh.facet_group):
        offset = np.hypot(uv[:, 0] - 0.5, uv[:, 1] - 0.5)
        if group == "table":
            np.testing.assert_allclose(offset, 0.5, atol=1e-6)
        if group in ("star", "bezel", "upper_girdle", "girdle"):
            assert np.all(uv[:, 0] < 0.0)

    size = 220
    image = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((8, 8, size - 9, size - 9), fill=(70, 120, 190))
    photo = np.asarray(image, dtype=np.float64) / 255.0
    shown = table_with_design(photo)
    table = float(CutParams().table)
    n = int(CutParams().n_base)
    centre = (size - 1) / 2.0
    reach = 0.48 * (size - 1) * table
    apothem = reach * np.cos(np.pi / n)
    angle = np.pi / n
    ox = int(round(centre + (apothem + 8) * np.cos(angle)))
    oy = int(round(centre - (apothem + 8) * np.sin(angle)))
    ix, iy = int(round(centre)), int(round(centre))
    assert shown[iy, ix, 2] > shown[oy, ox, 2] + 0.05


def test_a_smooth_heart_becomes_origami_and_a_faceted_drawing_stays():
    from gemopt.origami import origami_pattern

    size = 360
    image = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((12, 12, size - 13, size - 13), fill=(255, 255, 255))
    t = np.linspace(0, 2 * np.pi, 480, endpoint=False)
    hx = 16 * np.sin(t) ** 3
    hy = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    hx = (hx - hx.min()) / (hx.max() - hx.min())
    hy = (hy - hy.min()) / (hy.max() - hy.min())
    pts = [((0.22 + 0.56 * x) * size, (0.18 + 0.58 * (1 - y)) * size) for x, y in zip(hx, hy)]
    draw.polygon(pts, fill=(220, 90, 110))
    photo = np.asarray(image, dtype=np.float64) / 255.0
    folded = origami_pattern(photo)
    assert float(np.mean(np.abs(folded - photo))) > 0.01
    facets = folded[:, :, 2] > folded[:, :, 0] + 0.04
    assert int(facets.sum()) > 200
    from gemopt.origami import _simplify, _walk
    corners = _simplify(_walk(facets), 2.0)
    assert 4 <= len(corners) <= 24

    butterfly = np.asarray(
        Image.open(ROOT / "tests" / "fixtures" / "employer_butterfly.png").convert("RGB"),
        dtype=np.float64,
    ) / 255.0
    kept = origami_pattern(butterfly)
    assert float(np.mean(np.abs(kept - butterfly))) < 0.02


def test_an_offcentre_circle_is_moved_onto_the_stone_without_shifting_the_motif():
    from gemopt.design_output import seated_pattern

    image = Image.new("RGB", (400, 300), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.ellipse((40, 30, 240, 230), outline=(20, 20, 20), width=3)
    draw.rectangle((60, 110, 90, 150), fill=(40, 90, 200))
    seated = seated_pattern(np.asarray(image, dtype=np.float64) / 255.0, size=120)
    blue = seated[:, :, 2] > seated[:, :, 0] + 0.25
    ys, xs = np.nonzero(blue)
    assert len(xs) > 10
    assert float(xs.mean()) < (seated.shape[1] - 1) / 2.0


def test_section_two_puts_the_cap_on_the_pavilion(tmp_path, monkeypatch):
    from queue import Queue

    from gemopt.app import Job, _export_design
    from gemopt.design_output import round_stone, table_half_width
    from gemopt.tomax import MaxResult

    captured = {}

    def _no_max(*_args, **kwargs):
        captured.update(kwargs)
        return MaxResult(ok=False, path=None, message="no max", script=None)

    monkeypatch.setattr("gemopt.tomax.gem_to_max", _no_max)
    job = Job(photo=str(ROOT / "tests" / "fixtures" / "employer_butterfly.png"),
              single=True, view="crown", material="glass", ior=None,
              target_kind="pattern", n_base=8, girdle_multiple=2, extra_crown_rows=0,
              extra_pavilion_rows=0, generations=1, popsize=1, workers=1,
              outdir=str(tmp_path / "out"), diameter_mm=6.5, section="design")
    _export_design(job, Queue())
    expect = table_half_width(round_stone(), 3.25)
    assert abs(captured["half_width"] - expect) < 1e-9
    assert captured["pattern_face"] == "pavilion"
    assert captured["half_width"] < 0.53 * 3.25 + 1e-9
    pattern = np.asarray(Image.open(tmp_path / "out" / "pattern.png").convert("RGB"),
                         dtype=np.float64)
    assert pattern.shape[0] == pattern.shape[1]
    lum = pattern.mean(axis=2)
    centre = (pattern.shape[1] - 1) / 2.0
    yy, xx = np.mgrid[:pattern.shape[0], :pattern.shape[1]]
    radial = np.hypot(xx - centre, yy - centre)
    edge = float(lum[np.abs(radial - centre) < 1.2].mean())
    inner = float(lum[np.abs(radial - 0.82 * centre) < 1.2].mean())
    assert edge < 190
    assert inner > edge + 30


def test_preserve_form_render_keeps_the_drawing(tmp_path):
    view = _loaded(tmp_path, _butterfly(), "butterfly.png")
    problem = InverseProblem(views=[view], material=MATERIALS["glass"],
                             n_fold=view.source_fold, max_bounces=3)
    result = render_design(CutParams(n_base=view.source_fold), 16, problem,
                           np.ones(problem.n_bins), resolution=36, samples=1)
    np.testing.assert_allclose(result.pattern, _crop_to_stone(view.rgb_native), atol=1e-6)
    assert result.optics.shape[0] == 36
    assert rotation_symmetry_error(result.pattern * 255.0, 8) > 0.03
