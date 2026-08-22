"""One photograph, one face of the stone.

The side-by-side pair was a property of the first reference we happened to be
given, not of the problem.  A client with a single photo has nothing to split,
and everything downstream has to work the same either way.
"""

import numpy as np
import pytest
from PIL import Image

from gemopt.geometry import CutParams
from gemopt.inverse import InverseProblem, load_reference, load_single, load_views


@pytest.fixture(scope="module")
def single_photo(tmp_path_factory):
    """The right-hand half of the client's reference, on its own."""
    img = np.asarray(Image.open("1 (1).jpg").convert("RGB"))
    path = tmp_path_factory.mktemp("single") / "one.png"
    Image.fromarray(img[:, img.shape[1] // 2:]).save(path)
    return str(path)


def test_one_photo_gives_exactly_one_view(single_photo):
    views = load_single(single_photo, view="crown")
    assert len(views) == 1
    assert views[0].view == "crown"


def test_the_view_can_be_either_face(single_photo):
    for face in ("crown", "pavilion"):
        assert load_single(single_photo, view=face)[0].view == face


def test_it_is_ready_to_score_without_further_setup(single_photo):
    """`shifts` has to be populated or the solver divides by an empty array."""
    view = load_single(single_photo)[0]
    assert view.shifts.size > 0
    assert view.target.shape == (64, 256)


def test_a_single_view_scores(single_photo):
    views = load_single(single_photo, view="crown")
    score, per_view, profile = InverseProblem(
        views=views, psf_sigma=2.5, max_bounces=10).score(CutParams())
    assert len(per_view) == 1
    assert 0.0 < score <= 1.0
    assert profile.shape[0] == 36


def test_the_loader_switch_picks_the_right_one(single_photo):
    assert len(load_reference(single_photo, single=True)) == 1
    assert len(load_reference("1 (1).jpg")) == 2


def test_a_blank_image_is_rejected_with_a_useful_message(tmp_path):
    """Better a clear error than a stone fitted to nothing."""
    blank = tmp_path / "dark.png"
    Image.fromarray(np.zeros((80, 80, 3), dtype="uint8")).save(blank)
    with pytest.raises(ValueError, match="no stone found"):
        load_single(str(blank))


def test_the_pair_loader_still_works():
    views = load_views("1 (1).jpg")
    assert {v.view for v in views} == {"crown", "pavilion"}
