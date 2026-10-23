import numpy as np
import pytest

from calrepro.calibration import (
    HistogramBinning,
    IsotonicCalibration,
    MatrixScaling,
    TemperatureScaling,
    Uncalibrated,
    VectorScaling,
)
from calrepro.metrics import expected_calibration_error, softmax


def _calibrated_synthetic(n=6000, k=10, seed=0, scale=1.0):
    """Logits with labels drawn from softmax(logits): calibrated by construction.

    Multiplying the logits by ``scale`` > 1 afterwards makes the model
    overconfident while keeping the true conditional distribution fixed.
    """
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 2.0, size=(n, k))
    p = softmax(logits)
    labels = np.array([rng.choice(k, p=row) for row in p])
    return logits * scale, labels


def test_temperature_scaling_preserves_argmax():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(500, 10)) * 5
    labels = rng.integers(0, 10, 500)
    ts = TemperatureScaling().fit(logits, labels)
    for t in (ts.temperature, 0.1, 3.0, 100.0):
        ts.temperature = t
        assert (ts.predict_proba(logits).argmax(1) == logits.argmax(1)).all()


def test_temperature_above_one_for_overconfident_model():
    logits, labels = _calibrated_synthetic(scale=3.0)
    ts = TemperatureScaling().fit(logits, labels)
    assert ts.temperature > 1.0
    # The true generating temperature is 3; L-BFGS on 6000 samples should be close.
    assert ts.temperature == pytest.approx(3.0, rel=0.15)


def test_temperature_close_to_one_for_calibrated_model():
    logits, labels = _calibrated_synthetic(scale=1.0)
    ts = TemperatureScaling().fit(logits, labels)
    assert ts.temperature == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize(
    "cls", [HistogramBinning, IsotonicCalibration, TemperatureScaling, VectorScaling, MatrixScaling]
)
def test_calibrators_reduce_ece_and_output_distributions(cls):
    val_logits, val_labels = _calibrated_synthetic(seed=0, scale=3.0)
    test_logits, test_labels = _calibrated_synthetic(seed=1, scale=3.0)
    cal = cls().fit(val_logits, val_labels)
    probs = cal.predict_proba(test_logits)
    assert probs.shape == test_logits.shape
    assert np.allclose(probs.sum(1), 1.0)
    assert (probs >= 0).all()
    before = expected_calibration_error(Uncalibrated().predict_proba(test_logits), test_labels)
    after = expected_calibration_error(probs, test_labels)
    assert after < before


def test_vector_and_matrix_scaling_start_from_identity():
    # On logits that are already calibrated, the fitted maps should stay near identity.
    logits, labels = _calibrated_synthetic(n=8000, scale=1.0)
    vs = VectorScaling().fit(logits, labels)
    assert np.allclose(vs.w, 1.0, atol=0.15)
    ms = MatrixScaling().fit(logits, labels)
    assert np.allclose(np.diag(ms.w), 1.0, atol=0.2)
