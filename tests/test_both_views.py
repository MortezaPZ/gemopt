"""Scoring a stone through both faces at once.

A cut can draw clean arrows through the crown while leaking badly when turned
over, so optimising a single view quietly sacrifices the other.  These tests
pin down the combined mode.
"""

import numpy as np
import pytest

from gemopt.evaluate import EvalConfig, evaluate, make_reference
from gemopt.geometry import STANDARD_BRILLIANT
from gemopt.optics import CROWN_GLASS, DIAMOND
from gemopt.optimize import optimise
from gemopt.params import SearchSpace


@pytest.fixture(scope="module")
def both_config():
    cfg = EvalConfig(material=CROWN_GLASS, view="both", resolution=64)
    cfg.target = make_reference(cfg, resolution=96)
    return cfg


def test_views_property():
    assert EvalConfig(view="crown").views == ("crown",)
    assert EvalConfig(view="pavilion").views == ("pavilion",)
    assert EvalConfig(view="both").views == ("crown", "pavilion")


def test_make_reference_builds_one_target_per_view(both_config):
    assert isinstance(both_config.target, dict)
    assert set(both_config.target) == {"crown", "pavilion"}
    for view, target in both_config.target.items():
        assert view in target.name
        assert target.preview is not None


def test_single_view_target_is_not_a_dict():
    cfg = EvalConfig(material=DIAMOND, view="crown", resolution=48)
    target = make_reference(cfg, resolution=64)
    assert not isinstance(target, dict)
    assert cfg.target_for("crown") is None      # not assigned yet
    cfg.target = target
    assert cfg.target_for("crown") is target
    # a single target stands in for whichever view is asked for
    assert cfg.target_for("pavilion") is target


def test_both_views_reports_each_side_separately(both_config):
    ev = evaluate(STANDARD_BRILLIANT, both_config, keep_render=True)
    assert ev.ok
    assert set(ev.renders) == {"crown", "pavilion"}
    for key in ("light_return", "leakage", "pattern"):
        assert f"crown_{key}" in ev.parts
        assert f"pavilion_{key}" in ev.parts
        # the headline figure is the mean of the two sides
        assert ev.parts[key] == pytest.approx(
            0.5 * (ev.parts[f"crown_{key}"] + ev.parts[f"pavilion_{key}"])
        )


def test_glass_leaks_through_the_pavilion_but_not_the_crown(both_config):
    """The asymmetry that makes the combined mode worth having.

    Turned crown-up the stone still returns most of its light, because the
    girdle facets sit above the critical angle; turned over there is nothing
    to catch the light and it pours out.  The factor here is deliberately
    loose -- it is the asymmetry that matters, not its exact size, which moves
    with the girdle facet angles.
    """
    ev = evaluate(STANDARD_BRILLIANT, both_config)
    assert ev.parts["pavilion_leakage"] > 3 * ev.parts["crown_leakage"]


def test_combined_score_is_the_mean_of_the_two_view_scores(both_config):
    ev = evaluate(STANDARD_BRILLIANT, both_config)
    crown = EvalConfig(material=CROWN_GLASS, view="crown", resolution=64,
                       target=both_config.target["crown"])
    pav = EvalConfig(material=CROWN_GLASS, view="pavilion", resolution=64,
                     target=both_config.target["pavilion"])
    a = evaluate(STANDARD_BRILLIANT, crown).score
    b = evaluate(STANDARD_BRILLIANT, pav).score
    assert ev.score == pytest.approx(0.5 * (a + b), abs=1e-9)


def test_one_good_side_cannot_hide_a_hopeless_other(both_config):
    """Averaging the scores, not the raw ingredients, is what enforces this."""
    ev = evaluate(STANDARD_BRILLIANT, both_config)
    crown_only = EvalConfig(material=CROWN_GLASS, view="crown", resolution=64,
                            target=both_config.target["crown"])
    assert evaluate(STANDARD_BRILLIANT, crown_only).score > ev.score


def test_search_works_in_combined_mode(both_config):
    space = SearchSpace()
    start = evaluate(space.to_params(space.start()), both_config)
    result = optimise(space, both_config, max_evaluations=30, seed=7)
    assert result.best.score > start.score
    assert set(result.best.renders) == {"crown", "pavilion"}
    assert "m_crown_leakage" in result.history[0]


def test_evaluation_row_stays_flat_with_both_views(both_config):
    row = evaluate(STANDARD_BRILLIANT, both_config).row()
    assert "m_crown_leakage" in row and "m_pavilion_leakage" in row
    assert all(not isinstance(v, (list, dict, np.ndarray)) for v in row.values())
