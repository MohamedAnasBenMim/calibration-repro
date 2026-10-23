import numpy as np
import pytest

from calrepro.metrics import bin_stats, ece_mce, evaluate_probs, nll, softmax


def test_perfect_confident_predictions_have_zero_ece():
    conf = np.ones(10)
    correct = np.ones(10)
    assert ece_mce(conf, correct) == (0.0, 0.0)


def test_single_bin_ece_hand_computed():
    # All predictions at confidence 0.8, half of them correct: |0.5 - 0.8| = 0.3.
    conf = np.full(10, 0.8)
    correct = np.array([1, 0] * 5)
    ece, mce = ece_mce(conf, correct, n_bins=15)
    assert ece == pytest.approx(0.3)
    assert mce == pytest.approx(0.3)


def test_two_bin_ece_hand_computed():
    # M = 2 bins: (0, 0.5] and (0.5, 1].
    # Bin 1: confidences 0.3, 0.4 (mean 0.35), 1 of 2 correct (acc 0.5) -> gap 0.15.
    # Bin 2: confidences 0.9, 0.9, 0.6, 0.6 (mean 0.75), 4/4 correct -> gap 0.25.
    # ECE = 2/6 * 0.15 + 4/6 * 0.25 = 0.05 + 0.1666... = 0.21666...
    conf = np.array([0.3, 0.4, 0.9, 0.9, 0.6, 0.6])
    correct = np.array([1, 0, 1, 1, 1, 1])
    ece, mce = ece_mce(conf, correct, n_bins=2)
    assert ece == pytest.approx(2 / 6 * 0.15 + 4 / 6 * 0.25)
    assert mce == pytest.approx(0.25)


def test_bins_are_right_closed():
    # 0.5 belongs to the bin (0, 0.5], not (0.5, 1], with 2 bins.
    s = bin_stats(np.array([0.5, 0.51]), np.array([1, 1]), n_bins=2)
    assert s.counts.tolist() == [1, 1]
    # Exact 0 goes to the first bin.
    s = bin_stats(np.array([0.0]), np.array([0]), n_bins=2)
    assert s.counts.tolist() == [1, 0]


def test_ece_is_weighted_mce_upper_bounds_it():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.1, 1, 1000)
    correct = rng.uniform(size=1000) < conf**2
    ece, mce = ece_mce(conf, correct)
    assert 0 <= ece <= mce <= 1


def test_nll_and_evaluate_probs():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    labels = np.array([0, 1])
    assert nll(probs, labels) == pytest.approx(-(np.log(0.7) + np.log(0.1)) / 2)
    m = evaluate_probs(probs, labels)
    assert m["error"] == pytest.approx(50.0)
    assert m["avg_confidence"] == pytest.approx(75.0)


def test_softmax_rows_sum_to_one_and_stable():
    z = np.array([[1000.0, 0.0, -1000.0], [1.0, 2.0, 3.0]])
    p = softmax(z)
    assert np.allclose(p.sum(1), 1.0)
    assert np.isfinite(p).all()
