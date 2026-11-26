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


# ---- ECG image check -------------------------------------------------------------------

def _fake_details(image_check):
    return lambda path: {"cad_probability": 0.8, "arrhythmia_probability": 0.1,
                         "image_check": image_check}


def test_predict_ecg_refuses_images_that_are_not_ecgs(monkeypatch):
    import sys
    import types

    import pytest

    import cad_hybrid.predict as predict_module
    fake = types.ModuleType("cad_hybrid.ecg.predict")
    fake.predict_ecg = _fake_details({"passed": False, "distance": 0.4, "threshold": 0.1})
    monkeypatch.setitem(sys.modules, "cad_hybrid.ecg.predict", fake)
    with pytest.raises(ValueError, match="doesn't look like"):
        predict_module.predict_ecg("cat.jpg")
    fake.predict_ecg = _fake_details({"passed": True, "distance": 0.05, "threshold": 0.1})
    assert predict_module.predict_ecg("ecg.jpg")["cad_probability"] == 0.8
    fake.predict_ecg = _fake_details(None)  # check not built yet: still scores
    assert predict_module.predict_ecg("ecg.jpg")["image_check"] is None


def test_image_check_flags_features_unlike_the_training_ecgs(tmp_path):
    import numpy as np

    from cad_hybrid.ecg.image_check import build_reference, check, load_reference, save_reference

    rng = np.random.default_rng(0)
    train = np.ones(512) + rng.normal(0, 0.05, (200, 512))
    validation = np.ones(512) + rng.normal(0, 0.05, (50, 512))
    ref = load_reference(save_reference(build_reference(train, validation), tmp_path / "r.npz"))
    assert check(ref, np.ones((1, 512)))[0] is True
    assert check(ref, rng.normal(0, 1, (1, 512)))[0] is False
