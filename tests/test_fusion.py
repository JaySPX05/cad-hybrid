import math

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import cad_hybrid.predict as predict_module
from cad_hybrid.fusion.calibrate import Calibrator
from cad_hybrid.fusion.fuse import fuse, risk_category
from cad_hybrid.predict import predict, predict_tabular
from cad_hybrid.tabular.artifact import build_artifact, save_artifact


def logit(p):
    return math.log(p / (1 - p))


def test_no_branches_returns_target_prior():
    assert fuse({"tabular": None, "ecg": None}, {}, target_prior=0.3) == pytest.approx(0.3)


def test_single_branch_trained_at_target_rate_returns_its_probability():
    p = fuse({"tabular": 0.7, "ecg": None}, {"tabular": 0.3}, target_prior=0.3)
    assert p == pytest.approx(0.7)


def test_uninformative_branch_returns_target_prior():
    # a probability equal to the branch's own training rate carries no evidence
    p = fuse({"ecg": 0.59}, {"ecg": 0.59}, target_prior=0.2)
    assert p == pytest.approx(0.2)


def test_each_branch_uses_its_own_training_rate():
    p = fuse({"tabular": 0.8, "ecg": 0.7}, {"tabular": 0.565, "ecg": 0.592}, target_prior=0.3)
    expected = logit(0.3) + (logit(0.8) - logit(0.565)) + (logit(0.7) - logit(0.592))
    assert logit(p) == pytest.approx(expected)


def test_agreeing_branches_strengthen_evidence():
    priors = {"tabular": 0.5, "ecg": 0.5}
    both = fuse({"tabular": 0.7, "ecg": 0.7}, priors, target_prior=0.5)
    assert both > 0.7


def test_zero_weight_ignores_branch():
    p = fuse({"tabular": 0.7, "ecg": 0.95}, {"tabular": 0.3, "ecg": 0.6}, target_prior=0.3,
             weights={"ecg": 0.0})
    assert p == pytest.approx(0.7)


def test_missing_branch_prior_rejected():
    with pytest.raises(ValueError, match="branch_priors.ecg"):
        fuse({"tabular": 0.7, "ecg": 0.6}, {"tabular": 0.5}, target_prior=0.5)


def test_risk_category_bounds():
    cats = {"low": [0.0, 0.1], "moderate": [0.1, 0.2], "high": [0.2, 1.0]}
    assert risk_category(0.05, cats) == "low"
    assert risk_category(0.1, cats) == "moderate"
    assert risk_category(1.0, cats) == "high"


# ---------- predict(): priors come from the config and the saved lab model ----------

CATS = {"low": [0.0, 0.1], "moderate": [0.1, 0.2], "high": [0.2, 1.0]}


@pytest.fixture
def lab_model(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"age": rng.uniform(20, 80, 200), "ldl": rng.uniform(60, 200, 200)})
    y = (0.02 * (X["ldl"] - 110) + rng.normal(0, 0.5, 200) > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    calibrator = Calibrator("platt").fit(model.predict_proba(X)[:, 1], y.to_numpy())
    artifact = build_artifact(model, calibrator, ["age", "ldl"], {},
                              {"positive_rate": round(float(y.mean()), 4)})
    return save_artifact(artifact, tmp_path / "lab.joblib")


def config(**overrides):
    cfg = {"target_prior": None, "branch_priors": {"tabular": None, "ecg": None},
           "branch_weights": {"tabular": 1.0, "ecg": 1.0}, "risk_categories": CATS}
    cfg.update(overrides)
    return cfg


def test_lab_only_result_equals_lab_model_by_default(lab_model):
    labs = {"age": 60, "ldl": 170}
    result = predict(labs, tabular_model_path=lab_model, fusion_config=config())
    assert result.probability == pytest.approx(predict_tabular(labs, lab_model))


def test_lower_target_prior_lowers_lab_only_result(lab_model):
    labs = {"age": 60, "ldl": 170}
    default = predict(labs, tabular_model_path=lab_model, fusion_config=config())
    screening = predict(labs, tabular_model_path=lab_model,
                        fusion_config=config(target_prior=0.1))
    assert screening.probability < default.probability


def test_ecg_without_training_rate_gives_clear_error(lab_model, monkeypatch):
    monkeypatch.setattr(predict_module, "predict_ecg", lambda path: 0.8)
    with pytest.raises(ValueError, match="branch_priors.ecg"):
        predict({"age": 60, "ldl": 170}, "ecg.png", tabular_model_path=lab_model,
                fusion_config=config())


def test_both_branches_fused(lab_model, monkeypatch):
    monkeypatch.setattr(predict_module, "predict_ecg", lambda path: 0.9)
    labs = {"age": 60, "ldl": 170}
    cfg = config(branch_priors={"tabular": None, "ecg": 0.592})
    both = predict(labs, "ecg.png", tabular_model_path=lab_model, fusion_config=cfg)
    lab_only = predict(labs, tabular_model_path=lab_model, fusion_config=cfg)
    assert both.probability > lab_only.probability  # a confident CAD ECG adds evidence
    assert both.branch_probabilities["ecg"] == 0.9


def test_ecg_only_without_lab_model_needs_target_prior(tmp_path, monkeypatch):
    monkeypatch.setattr(predict_module, "predict_ecg", lambda path: 0.9)
    missing = tmp_path / "none.joblib"
    cfg = config(branch_priors={"tabular": None, "ecg": 0.592})
    with pytest.raises(ValueError, match="target_prior"):
        predict(ecg_image_path="ecg.png", tabular_model_path=missing, fusion_config=cfg)
    cfg["target_prior"] = 0.592
    assert predict(ecg_image_path="ecg.png", tabular_model_path=missing,
                   fusion_config=cfg).probability == pytest.approx(0.9)
