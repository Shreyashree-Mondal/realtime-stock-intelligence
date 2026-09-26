"""The A/B decision logic should detect a real difference and ignore noise."""
import numpy as np

from ml.ab_test import compare


def test_detects_clearly_better_model():
    rng = np.random.default_rng(1)
    a = rng.normal(1.0, 0.2, 500).clip(0)
    b = rng.normal(0.8, 0.2, 500).clip(0)
    res = compare(a, b, n_boot=1000)
    assert res["p_value"] < 0.05
    assert res["diff_ci_high"] < 0
    assert res["decision"].startswith("promote B")


def test_no_difference_keeps_champion():
    rng = np.random.default_rng(2)
    a = rng.normal(1.0, 0.2, 500).clip(0)
    b = rng.normal(1.0, 0.2, 500).clip(0)
    res = compare(a, b, n_boot=1000)
    assert not res["decision"].startswith("promote")


def test_small_samples_wait_for_more_data():
    res = compare([1.0, 1.1, 0.9], [0.5, 0.6, 0.4], n_boot=200)
    assert res["decision"].startswith("keep collecting")
