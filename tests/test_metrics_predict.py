import numpy as np
import pytest

from cad_hybrid.evaluation.metrics import bootstrap_ci, compute_metrics
from cad_hybrid.fusion.calibrate import Calibrator
from cad_hybrid.predict import predict


def test_perfect_model_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y, p)
    assert m["auroc"] == 1.0
    assert m["sensitivity"] == 1.0 and m["specificity"] == 1.0


def test_bootstrap_interval_contains_point():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 200), 0, 1)
    point, lo, hi = bootstrap_ci(y, p, n_boot=200)
    assert lo <= point <= hi


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_calibrator_outputs_probabilities(method):
    rng = np.random.default_rng(1)
    scores = rng.random(100)
    y = (scores + rng.normal(0, 0.2, 100) > 0.5).astype(int)
    out = Calibrator(method).fit(scores, y).transform(scores)
    assert out.min() >= 0 and out.max() <= 1


def test_predict_requires_an_input():
    with pytest.raises(ValueError):
        predict()
