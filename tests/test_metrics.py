"""The scoring layer: what "looks like the reference" is worth."""

import numpy as np
import pytest

from gemopt.metrics import _normalise

def test_a_perfectly_symmetric_target_has_ceiling_one():
    """Nothing is lost when the target is already symmetric."""
    from gemopt.metrics import Target, achievable_ceiling

    theta = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    polar = np.tile(np.cos(8 * theta), (64, 1))
    target = Target(polar=_normalise(polar), n_fold=8)
    assert achievable_ceiling(target) == pytest.approx(1.0, abs=1e-6)


def test_asymmetric_noise_lowers_the_ceiling():
    """Whatever is not symmetric cannot be reproduced by a symmetric render."""
    from gemopt.metrics import Target, achievable_ceiling

    rng = np.random.default_rng(0)
    theta = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    clean = np.tile(np.cos(8 * theta), (64, 1))
    noisy = clean + rng.normal(scale=1.5, size=clean.shape)
    ceiling = achievable_ceiling(Target(polar=_normalise(noisy), n_fold=8))
    assert 0.5 < ceiling < 0.95


def test_relative_match_rescales_against_the_ceiling():
    """A score at the ceiling reads as a perfect match; chance reads as zero."""
    from gemopt.metrics import relative_match

    assert relative_match(0.83, 0.83) == pytest.approx(1.0)
    assert relative_match(0.50, 0.83) == pytest.approx(0.0)
    assert relative_match(0.75, 0.83) == pytest.approx(0.7576, abs=1e-3)
