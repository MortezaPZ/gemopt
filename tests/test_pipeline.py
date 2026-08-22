"""The parts around the physics: file formats, scoring, and the search itself."""

import math
from pathlib import Path

import numpy as np
import pytest

from gemopt.evaluate import EvalConfig, evaluate, make_reference
from gemopt.geometry import STANDARD_BRILLIANT, CutParams, build_stone
from gemopt.ies import load_ies
from gemopt.maxbridge import read_gem, read_obj, write_gem, write_vertex_csv
from gemopt.metrics import (
    ScoreWeights,
    align_rotation,
    contrast,
    pattern_similarity,
    symmetry_error,
    target_from_image,
    to_polar,
)
from gemopt.optics import CROWN_GLASS, DIAMOND
from gemopt.optimize import optimise
from gemopt.params import SearchSpace, load_space, write_template
from gemopt.report import write_report, write_trials_csv
from gemopt.scope import (
    HEARTS_ARROWS_SCOPE,
    IDEALSCOPE,
    Camera,
    IESEnvironment,
    render,
    save_png,
)

ASSETS = Path(__file__).resolve().parent.parent / "assets"
IES_FILE = ASSETS / "075 SPARK 5000 840 30D.ies"


# --------------------------------------------------------------------- IES --


@pytest.mark.skipif(not IES_FILE.exists(), reason="client IES file not bundled")
def test_reads_the_clients_luminaire():
    """The real photometric file from the client's 3ds Max scene."""
    p = load_ies(IES_FILE)
    assert p.lumens == pytest.approx(5410.0)
    assert p.candela.max() == pytest.approx(2166.70 * 5.41, rel=1e-6)
    assert 10.0 < p.beam_angle < 40.0
    assert p.cutoff_angle == pytest.approx(70.0)
    # a spot is brightest on axis and dies away off it
    assert p.normalised(0.0) == pytest.approx(1.0)
    assert p.normalised(80.0) == pytest.approx(0.0)
    assert p.normalised(15.0) > p.normalised(30.0)


def test_ies_rejects_a_file_that_is_not_one(tmp_path):
    bad = tmp_path / "not.ies"
    bad.write_text("hello\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_ies(bad)


# ------------------------------------------------------------------- scope --


@pytest.fixture(scope="module")
def diamond_render():
    mesh = build_stone(STANDARD_BRILLIANT)
    return render(mesh, DIAMOND, Camera(resolution=96, view="pavilion"), HEARTS_ARROWS_SCOPE)


def test_render_energy_adds_up(diamond_render):
    total = (diamond_render.light_return + diamond_render.leakage
             + diamond_render.surface + diamond_render.unresolved)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_render_covers_most_of_the_frame(diamond_render):
    # the stone is a disc inside a square: pi/4 of the area, near enough
    assert 0.65 < diamond_render.coverage.mean() < 0.82


def test_glass_leaks_far_more_than_diamond_through_the_pavilion():
    """The physical heart of the brief: diamond proportions do not work in glass."""
    mesh = build_stone(STANDARD_BRILLIANT)
    cam = Camera(resolution=96, view="pavilion")
    d = render(mesh, DIAMOND, cam, IDEALSCOPE)
    g = render(mesh, CROWN_GLASS, cam, IDEALSCOPE)
    assert g.leakage > 3.0 * d.leakage


@pytest.mark.skipif(not IES_FILE.exists(), reason="client IES file not bundled")
def test_ies_environment_renders():
    mesh = build_stone(STANDARD_BRILLIANT)
    env = IESEnvironment(profile=load_ies(IES_FILE))
    result = render(mesh, DIAMOND, Camera(resolution=64), env)
    assert np.isfinite(result.image).all()
    assert result.image.max() > 0.0


def test_save_png_roundtrip(tmp_path):
    from PIL import Image

    image = np.zeros((8, 8, 3))
    image[:, :, 0] = 1.0
    path = tmp_path / "x.png"
    save_png(image, path)
    back = np.asarray(Image.open(path))
    assert back.shape == (8, 8, 3)
    assert back[0, 0, 0] == 255 and back[0, 0, 1] == 0


# ----------------------------------------------------------------- metrics --


def test_polar_transform_turns_rotation_into_a_shift():
    """The reason everything is compared in polar coordinates."""
    n = 128
    y, x = np.mgrid[0:n, 0:n]
    cx = cy = n / 2 - 0.5
    angle = np.arctan2(y - cy, x - cx)
    wedge = (np.cos(8 * angle) > 0).astype(float)

    pol = to_polar(wedge, n_r=32, n_theta=256)
    rolled = np.roll(pol, 17, axis=1)
    shift, _ = align_rotation(rolled, pol)
    assert shift == 17


def test_identical_patterns_score_one():
    rng = np.random.default_rng(0)
    pol = rng.random((32, 256))
    target = target_from_image(np.zeros((4, 4)))
    target.polar = (pol - pol.mean()) / pol.std()
    assert pattern_similarity(pol, target) == pytest.approx(1.0, abs=1e-6)


def test_symmetry_error_is_zero_for_a_symmetric_pattern():
    theta = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    pol = np.tile(np.cos(8 * theta), (32, 1))
    assert symmetry_error(pol, 8) < 1e-9
    # the same pattern is not 6-fold symmetric
    assert symmetry_error(pol, 6) > 0.1


def test_contrast_separates_flat_from_patterned():
    flat = np.full((32, 256), 0.5)
    assert contrast(flat) == pytest.approx(0.0)
    theta = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    assert contrast(np.tile(np.cos(8 * theta), (32, 1))) > 0.5


def test_low_contrast_cannot_win_on_pattern_alone():
    """A washed-out stone must not score well just because its noise happens
    to correlate; the gate is what stops the search cheating."""
    w = ScoreWeights()
    strong = w.combine({"pattern": 0.9, "contrast": 0.6, "light_return": 0.5,
                        "leakage": 0.1, "symmetry": 0.0})
    washed = w.combine({"pattern": 0.9, "contrast": 0.02, "light_return": 0.5,
                        "leakage": 0.1, "symmetry": 0.0})
    assert strong > washed


def test_score_weights_are_bounded():
    w = ScoreWeights()
    perfect = w.combine({"pattern": 1.0, "light_return": 1.0, "leakage": 0.0,
                         "symmetry": 0.0, "contrast": 1.0})
    worst = w.combine({"pattern": 0.0, "light_return": 0.0, "leakage": 1.0,
                       "symmetry": 1.0, "contrast": 0.0})
    assert perfect == pytest.approx(1.0)
    assert worst == pytest.approx(0.0)


# ------------------------------------------------------------------ params --


@pytest.mark.parametrize("suffix", [".csv", ".xlsx"])
def test_parameter_sheet_roundtrip(tmp_path, suffix):
    space = SearchSpace()
    path = write_template(tmp_path / f"parameters{suffix}", space)
    back = load_space(path)
    assert [s.name for s in back.specs] == [s.name for s in space.specs]
    assert [s.name for s in back.free()] == [s.name for s in space.free()]
    for a, b in zip(back.specs, space.specs):
        assert a.low == pytest.approx(b.low)
        assert a.high == pytest.approx(b.high)
        assert a.enabled == b.enabled


def test_unknown_parameter_name_is_reported(tmp_path):
    """A typo silently dropping a parameter from the search would be very
    hard to spot in the results, so it has to be an error."""
    path = tmp_path / "bad.csv"
    path.write_text("parameter,enabled,min,max,start\nnot_a_parameter,1,0,1,0.5\n",
                    encoding="utf-8")
    with pytest.raises(ValueError, match="does not have"):
        load_space(path)


def test_vector_roundtrip():
    space = SearchSpace()
    v = space.start()
    assert np.allclose(space.to_vector(space.to_params(v)), v)


def test_disabled_rows_are_held_at_their_start_value():
    space = SearchSpace()
    for spec in space.specs:
        if spec.name == "table":
            spec.enabled = False
            spec.default = 0.60
    assert "table" not in [s.name for s in space.free()]
    assert space.to_params(space.start()).table == pytest.approx(0.60)


# --------------------------------------------------------------- max bridge --


def test_gem_roundtrip_preserves_the_stone(tmp_path):
    mesh = build_stone(STANDARD_BRILLIANT)
    write_gem(tmp_path / "s.gem", mesh, STANDARD_BRILLIANT, scale=6.5)
    back = read_gem(tmp_path / "s.gem")
    assert len(back.vertices) == len(mesh.vertices)
    assert len(back.polygons) == len(mesh.polygons)
    assert back.is_watertight()
    assert np.allclose(back.vertices, mesh.vertices * 6.5)
    assert (tmp_path / "s.json").exists()


def test_obj_roundtrip_preserves_the_stone(tmp_path):
    mesh = build_stone(STANDARD_BRILLIANT)
    mesh.to_obj(tmp_path / "s.obj")
    back = read_obj(tmp_path / "s.obj")
    assert len(back.polygons) == len(mesh.polygons)
    assert back.is_watertight()
    assert back.volume() == pytest.approx(mesh.volume(), rel=1e-6)


def test_obj_handles_slash_separated_faces(tmp_path):
    path = tmp_path / "t.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1/1 2/2/2 3/3/3\n", encoding="utf-8"
    )
    mesh = read_obj(path)
    assert mesh.polygons == [[0, 1, 2]]


def test_vertex_csv_lists_every_vertex(tmp_path):
    mesh = build_stone(STANDARD_BRILLIANT)
    path = write_vertex_csv(tmp_path / "v.csv", mesh, scale=6.5)
    lines = path.read_text(encoding="utf-8-sig").strip().splitlines()
    assert len(lines) == len(mesh.vertices) + 1
    assert lines[0].startswith("vertex,x,y,z")


# ---------------------------------------------------------------- pipeline --


@pytest.fixture(scope="module")
def small_config():
    cfg = EvalConfig(material=CROWN_GLASS, view="crown", resolution=64)
    cfg.target = make_reference(cfg, resolution=96)
    return cfg


def test_ideal_diamond_matches_its_own_reference(small_config):
    """A sanity check on the whole chain: score the very stone the target was
    made from and the pattern term should be almost exactly 1."""
    cfg = EvalConfig(material=DIAMOND, view="crown", resolution=64,
                     target=small_config.target)
    ev = evaluate(STANDARD_BRILLIANT, cfg)
    assert ev.ok
    assert ev.parts["pattern"] > 0.95


def test_invalid_parameters_are_rejected_without_tracing(small_config):
    ev = evaluate(CutParams(pavilion_angle=85.0), small_config)
    assert not ev.ok
    assert ev.score == 0.0
    assert ev.render is None


def test_evaluation_row_is_flat_and_complete(small_config):
    row = evaluate(STANDARD_BRILLIANT, small_config).row()
    assert "score" in row and "crown_angle" in row
    assert "m_light_return" in row and "d_table_pct" in row
    assert all(not isinstance(v, (list, dict)) for v in row.values())


def test_search_improves_on_its_starting_point(small_config):
    """The end-to-end promise: given glass, the optimiser finds a better cut
    than the diamond proportions it starts from."""
    space = SearchSpace()
    start = evaluate(space.to_params(space.start()), small_config)
    result = optimise(space, small_config, max_evaluations=40, seed=2)
    assert result.evaluations == 40
    assert result.best.score > start.score
    assert result.best.render is not None
    assert len(result.history) == 40


def test_search_respects_its_evaluation_budget(small_config):
    result = optimise(SearchSpace(), small_config, max_evaluations=15, seed=1)
    assert result.evaluations == 15


def test_glass_wants_a_steeper_pavilion_than_diamond(small_config):
    """Physics says so -- glass has a much larger critical angle -- and the
    search should discover it without being told.

    The budget is 250 rather than 90 because at 90 this measures the luck of
    the sampling rather than the physics: seeds 5 and 11 land just under 40.75
    and seed 23 well over it.  From 250 upwards every seed tried clears it, so
    the claim is what is being tested.  Checked at 250 and 600 evaluations
    across seeds 5, 11 and 23.
    """
    result = optimise(SearchSpace(), small_config, max_evaluations=250, seed=5)
    assert result.best.params.pavilion_angle > STANDARD_BRILLIANT.pavilion_angle


def test_reports_are_written(tmp_path, small_config):
    result = optimise(SearchSpace(), small_config, max_evaluations=12, seed=3)
    csv_path = write_trials_csv(tmp_path / "trials.csv", result.history)
    html_path = write_report(tmp_path / "report.html", result, small_config)
    assert csv_path.exists() and html_path.exists()
    body = html_path.read_text(encoding="utf-8")
    assert "data:image/png;base64," in body      # pictures are embedded
    assert "Winning cut" in body
    assert len(csv_path.read_text(encoding="utf-8-sig").splitlines()) == 13
