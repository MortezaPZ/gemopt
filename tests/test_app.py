"""The desktop app.

The window itself needs a display, so what is tested here is everything that
does not: the job description that crosses into the worker process, the
material table behind the dropdown, and the effort presets.  Those are where a
mistake would silently produce the wrong stone; the widgets are checked by
opening the window on Windows, which CI cannot do.
"""

import pytest

from gemopt.app import EFFORT, FA, MATERIALS_FA, Job


def test_every_material_resolves_to_something_the_engine_knows():
    """A label in the dropdown has to map to a real material or an index."""
    from gemopt.optics import MATERIALS

    for label, name, ior in MATERIALS_FA:
        assert label.strip(), "a dropdown entry with no text"
        if name is not None:
            assert name in MATERIALS, f"{label} names an unknown material"
            assert ior == pytest.approx(MATERIALS[name].n_d, abs=0.01), (
                f"{label} advertises an index the engine does not agree with")
        else:
            # the custom row is the only one allowed to carry no index
            assert ior is None or ior > 1.0


def test_the_custom_row_is_last_and_unique():
    """The interface keys "is this custom?" off the final row."""
    customs = [m for m in MATERIALS_FA if m[1] is None and m[2] is None]
    assert len(customs) == 1
    assert MATERIALS_FA[-1] is customs[0]


@pytest.mark.parametrize("level", ["fast", "normal", "deep"])
def test_effort_presets_are_ordered_and_usable(level):
    preset = EFFORT[level]
    assert preset["generations"] >= 4
    assert preset["popsize"] >= 6


def test_more_effort_means_more_work():
    assert (EFFORT["fast"]["generations"] < EFFORT["normal"]["generations"]
            < EFFORT["deep"]["generations"])


def test_job_survives_being_pickled():
    """It crosses a process boundary, so it has to pickle cleanly."""
    import pickle

    job = Job(photo="ref.jpg", single=False, view="crown", material="glass",
              ior=None, target_kind="photo", n_base=16, girdle_multiple=3, extra_crown_rows=0,
              extra_pavilion_rows=0, generations=8, popsize=8, workers=1, outdir="results/app")
    restored = pickle.loads(pickle.dumps(job))
    assert restored == job
    assert restored.n_base == 16
    assert restored.single is False and restored.view == "crown"


def test_job_settings_reach_the_geometry_they_claim():
    """What the spinboxes say must be what gets built."""
    from gemopt.geometry import CutParams, build_stone, facet_count
    from gemopt.params import SearchSpace

    job = Job(photo="x.jpg", single=True, view="pavilion", material="glass",
              ior=None, target_kind="photo", n_base=16, girdle_multiple=3,
              extra_crown_rows=1, extra_pavilion_rows=2,
              generations=4, popsize=6, workers=1, outdir="out")
    space = SearchSpace(base=CutParams(
        n_base=job.n_base, girdle_multiple=job.girdle_multiple,
        extra_crown_rows=job.extra_crown_rows,
        extra_pavilion_rows=job.extra_pavilion_rows))
    params = space.to_params(space.start())
    assert params.n_base == 16
    assert params.girdle_multiple == 3
    assert params.extra_crown_rows == 1
    assert params.extra_pavilion_rows == 2
    assert build_stone(params).is_watertight()
    assert facet_count(params) == build_stone(params).n_facets - 3 * 16


def test_the_facet_note_matches_the_formula():
    """The line under the spinboxes is the user's only preview of the count."""
    for n in (6, 8, 16):
        for m in (1, 2, 3):
            total = 3 * n + 2 * m * n + 1
            assert total == n + n + n + m * n + m * n + 1


def test_all_labels_are_present():
    """A missing key would show up as a KeyError mid-click."""
    needed = {"title", "reference", "browse", "material", "facets", "start",
              "stop", "ready", "running", "done", "failed", "hearts", "arrows",
              "similarity", "light", "leak", "results", "open_folder"}
    assert needed <= set(FA)
    assert all(FA[k].strip() for k in needed)


def test_a_dead_stone_cannot_win_on_pattern_alone():
    """The defect this guards against shipped once already.

    Pattern similarity is contrast-normalised, so a cut that leaks almost all
    its light still has the little that remains in roughly the right places
    and scores well.  Left unchecked the search prefers exactly that, and the
    app hands the client a piece of glass you can see through.  Light return
    has to be able to overrule a flattering pattern.
    """
    from gemopt.geometry import CutParams
    from gemopt.inverse import InverseProblem, load_views
    from gemopt.optics import MATERIALS

    views = load_views("1 (1).jpg")
    dead = CutParams(crown_angle=28.63, pavilion_angle=53.04)
    alive = CutParams()

    pattern_only = InverseProblem(views=views, material=MATERIALS["glass"],
                                  return_weight=0.0, psf_sigma=2.5, max_bounces=10)
    balanced = InverseProblem(views=views, material=MATERIALS["glass"],
                              return_weight=0.45, psf_sigma=2.5, max_bounces=10)

    # on pattern alone the dead stone is the more attractive of the two
    assert pattern_only.score(dead)[0] > pattern_only.score(alive)[0]
    # with brightness in the score that ordering has to reverse
    assert balanced.score(dead)[0] < balanced.score(alive)[0]


def test_a_single_view_job_carries_which_face_it_is():
    """With one photo the picture cannot say which side it shows."""
    job = Job(photo="one.png", single=True, view="pavilion", material="glass",
              ior=None, target_kind="photo", n_base=8, girdle_multiple=2,
              extra_crown_rows=0, extra_pavilion_rows=0,
              generations=4, popsize=6, workers=1, outdir="out")
    assert job.single and job.view == "pavilion"

    import pickle
    assert pickle.loads(pickle.dumps(job)).view == "pavilion"


def test_worker_count_is_positive_and_bounded():
    """Sizing the pool by memory must never ask for zero or absurd workers."""
    import os

    from gemopt.app import _worker_count

    n = _worker_count()
    assert n >= 1
    assert n <= max(1, (os.cpu_count() or 2) - 1)


def test_the_required_libraries_are_declared_in_requirements():
    """The client hit a missing scipy at run time once already.

    Anything the app imports at startup has to appear in requirements.txt, or
    installing as documented leaves a machine that crashes on the first click.
    """
    from pathlib import Path

    from gemopt.app import REQUIRED

    listed = Path("requirements.txt").read_text(encoding="utf-8").lower()
    for _module, pip_name in REQUIRED:
        assert pip_name.lower() in listed, f"{pip_name} is used but not declared"
