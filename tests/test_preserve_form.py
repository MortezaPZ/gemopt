"""Regression coverage for changing the count without inventing a new motif."""

from dataclasses import replace
import json
from queue import Queue
from threading import Event

import numpy as np
from PIL import Image
import pytest

from gemopt.app import Job, _prepare_job, _worker
from gemopt.design import retarget_views, transfer_cut
from gemopt.diagram import homologous_optics
from gemopt.fold import align_view_to_n, detect_fold, remap_image_fold, remap_polar_fold
from gemopt.geometry import CutParams, build_stone
from gemopt.inverse import load_single
from gemopt.metrics import symmetry_error, to_polar


def motif(n=8, size=321, colour=(0.85, 0.12, 0.25), phase=0.0, waist=0.25):
    yy, xx = np.mgrid[:size, :size]
    radius = np.hypot(xx - (size - 1) / 2, yy - (size - 1) / 2) / (size * 0.46)
    theta = np.arctan2(yy - (size - 1) / 2, xx - (size - 1) / 2) - phase
    # A recognisable head, waist and centre, not merely a cosine frequency.
    shaft = np.exp(-((radius - 0.54) / 0.29) ** 8)
    opening = (1 + np.cos(n * theta)) / 2
    arrow = shaft * opening ** (2 + 12 * np.exp(-((radius - 0.45) / 0.13) ** 2))
    image = np.zeros((size, size, 3)) + np.array(colour)
    image = image * (1 - arrow[..., None]) + 0.93 * arrow[..., None]
    image[radius < waist] = np.array(colour) * 0.35
    image[radius > 1] = 0
    return image


@pytest.mark.parametrize("n", [6, 8, 12, 16])
def test_same_count_is_exact_identity(n):
    rgb = motif(n, phase=0.17)
    polar = to_polar(rgb)
    np.testing.assert_array_equal(remap_polar_fold(polar, n, n), polar)
    np.testing.assert_array_equal(remap_image_fold(rgb, n, n), rgb)


def test_detection_does_not_cancel_oppositely_lit_rings():
    theta = np.arange(256) * 2 * np.pi / 256
    polar = np.stack([np.cos(8 * theta), -np.cos(8 * theta)] * 32)
    assert detect_fold(polar) == 8


def test_different_inputs_produce_different_sixteen_fold_designs():
    a = homologous_optics(motif(colour=(0.8, 0.1, 0.2), waist=0.2), 16)
    b = homologous_optics(motif(colour=(0.1, 0.3, 0.8), waist=0.34), 16)
    assert np.mean(abs(a - b)) > 0.1
    np.testing.assert_allclose(a[160, 160], np.array([0.8, 0.1, 0.2]) * 0.35)
    np.testing.assert_allclose(b[160, 160], np.array([0.1, 0.3, 0.8]) * 0.35)


@pytest.mark.parametrize("destination", [6, 12, 16, 24])
def test_mapping_preserves_radial_profile_colour_and_symmetry(destination):
    src = motif(size=513, phase=0.12)
    dst = remap_image_fold(src, 8, destination)
    before, after = to_polar(src, radius=235), to_polar(dst, radius=235)
    assert detect_fold(after) == destination
    assert symmetry_error(after, destination) < 0.008
    np.testing.assert_allclose(before.mean(axis=1), after.mean(axis=1), atol=0.009)
    assert np.max(dst) <= np.max(src) + 1e-12
    np.testing.assert_array_equal(dst[0, 0], src[0, 0])


def test_score_target_and_preview_share_the_same_motif_and_phase():
    src = motif(size=513, phase=0.19)
    polar = to_polar(src, radius=230, n_theta=1024)
    transformed, preview, count = align_view_to_n(polar, src, 16)
    shown = to_polar(preview, radius=230, n_theta=1024)
    assert count == 8
    assert np.mean(abs(transformed - shown)) < 0.004


@pytest.mark.parametrize("count", [0, 2, 25, 8.5, True])
def test_invalid_fold_is_not_silently_clamped(count):
    with pytest.raises(ValueError, match="fold count"):
        remap_image_fold(motif(size=33), 8, count)


@pytest.fixture(scope="module")
def source_photo(tmp_path_factory):
    path = tmp_path_factory.mktemp("form") / "arrows.png"
    Image.fromarray((motif() * 255).astype("uint8")).save(path)
    return path


def test_framing_is_independent_of_destination_and_native_pixels_survive(source_photo):
    v8 = load_single(source_photo, n_fold=8, scale_steps=3)[0]
    v16 = load_single(source_photo, n_fold=16, scale_steps=3)[0]
    assert v8.radius_scale == v16.radius_scale
    np.testing.assert_array_equal(v8.rgb_native, v16.rgb_native)
    again = retarget_views([v16], 8)[0]
    np.testing.assert_array_equal(again.rgb, v8.rgb)


def _job(photo, out, count=8, **kwargs):
    return Job(photo=str(photo), single=True, view="crown", material="diamond", ior=None,
               target_kind="photo", n_base=count, girdle_multiple=2,
               extra_crown_rows=0, extra_pavilion_rows=0, generations=0, popsize=1,
               workers=1, outdir=str(out), polish=False, **kwargs)


def test_destination_does_not_trigger_an_unrelated_new_search(source_photo, tmp_path):
    job = _job(source_photo, tmp_path, source_n=8)
    p8, s8 = _prepare_job(job)
    p16, s16 = _prepare_job(replace(job, n_base=16))
    assert s8.base == s16.base
    np.testing.assert_array_equal(s8.start(), s16.start())
    np.testing.assert_array_equal(s8.bounds(), s16.bounds())
    np.testing.assert_array_equal(p8.views[0].target, p16.views[0].target)
    assert p8.material == p16.material
    assert p16.n_fold == s16.base.n_base == 8


def test_transfer_changes_only_count_and_exports_a_closed_symmetric_mesh():
    source = CutParams(table=0.61, crown_angle=32.0, pavilion_angle=42.0)
    destination = transfer_cut(source, 16)
    assert {**source.to_dict(), "n_base": 16} == destination.to_dict()
    mesh = build_stone(destination)
    assert mesh.is_watertight()
    angle = 2 * np.pi / 16
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    rotated = mesh.vertices @ rotation.T
    for point in rotated:
        assert np.min(np.linalg.norm(mesh.vertices - point, axis=1)) < 1e-10


def test_real_worker_preserves_form_and_keeps_true_optics_separate(source_photo, tmp_path, monkeypatch):
    from gemopt.tomax import MaxResult, write_build_script

    # Test our actual optimiser, renderer and exporters; 3ds Max is external.
    def export_script(gem, **kwargs):
        return MaxResult(False, None, "Max not launched in regression test",
                         script=write_build_script(gem, **kwargs))

    monkeypatch.setattr("gemopt.tomax.gem_to_max", export_script)
    results = []
    for count in (8, 16):
        out = tmp_path / str(count)
        queue = Queue()
        _worker(_job(source_photo, out, count, source_n=8), queue, Event())
        messages = list(queue.queue)
        assert messages[-1][0] == "done", messages[-1]
        result = json.loads((out / "result.json").read_text(encoding="utf-8"))
        results.append(result)
        assert result["params"]["n_base"] == count
        assert result["ior"] == pytest.approx(2.4175)
        assert result["optics_kind"] == "destination_mesh_raytrace"
        assert (out / "stone.obj").is_file()
        assert (out / "stone_build.ms").is_file()
    assert results[0]["source_params"] == results[1]["source_params"]
    assert {**results[0]["params"], "n_base": 16} == results[1]["params"]
    def read(count, name):
        return np.asarray(Image.open(tmp_path / str(count) / name), dtype=float) / 255
    np.testing.assert_array_equal(read(8, "out_crown.png"), read(16, "source_crown.png"))
    expected = remap_image_fold(read(8, "out_crown.png"), 8, 16)
    np.testing.assert_allclose(read(16, "out_crown.png"), expected, atol=2 / 255)
    assert np.mean(abs(read(16, "out_crown.png") - read(16, "out_crown_optics.png"))) > 0.005
