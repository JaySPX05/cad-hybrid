"""Tests for SHAP explanations of the lab model (fake data only)."""

import numpy as np
import pandas as pd
import pytest

from cad_hybrid.explain.shap_tabular import (
    explain_patient, global_importance, make_background, plot_importance, plot_patient,
    shap_values, write_report)
from cad_hybrid.fusion.calibrate import Calibrator
from cad_hybrid.predict import explain_tabular
from cad_hybrid.tabular.artifact import build_artifact, predict_calibrated, save_artifact
from cad_hybrid.tabular.baseline import logistic_regression

FEATURES = ["age", "ldl", "noise", "sex_male"]
RANGES = {"age": [18, 110], "ldl": [10, 500]}


def fake_patients(n: int = 300, seed: int = 0):
    """LDL drives the outcome strongly, age a little, 'noise' not at all."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "age": rng.uniform(20, 80, n),
        "ldl": rng.uniform(60, 200, n),
        "noise": rng.normal(0, 1, n),
        "sex_male": rng.integers(0, 2, n).astype(float),
    })
    risk = 0.005 * (X["age"] - 50) + 0.04 * (X["ldl"] - 130)
    y = pd.Series((risk + rng.normal(0, 0.5, n) > 0).astype(int))
    return X, y


@pytest.fixture(scope="module")
def artifact():
    X, y = fake_patients()
    model = logistic_regression(["age", "ldl", "noise"], ["sex_male"]).fit(X, y)
    calibrator = Calibrator("platt").fit(model.predict_proba(X)[:, 1], y.to_numpy())
    return build_artifact(model, calibrator, FEATURES, RANGES, {"model": "fake"},
                          explain_background=make_background(X, ["sex_male"], size=8),
                          explain_nsamples=100)


def test_background_is_small_weighted_and_binary_safe():
    X, _ = fake_patients()
    X.loc[:10, "ldl"] = np.nan
    bg = make_background(X, ["sex_male"], size=5)
    assert bg["centres"].shape == (5, len(FEATURES))
    assert not bg["centres"].isna().any().any()
    assert set(bg["centres"]["sex_male"].unique()) <= {0.0, 1.0}
    assert bg["weights"].sum() == pytest.approx(1.0)


def test_contributions_add_up_to_the_prediction(artifact):
    X, _ = fake_patients(5, seed=1)
    values, baseline = shap_values(artifact, X)
    assert values.shape == (5, len(FEATURES))
    assert baseline + values.sum(axis=1) == pytest.approx(predict_calibrated(artifact, X))


def test_explanations_are_repeatable(artifact):
    X, _ = fake_patients(3, seed=2)
    assert np.allclose(shap_values(artifact, X)[0], shap_values(artifact, X)[0])


def test_driving_feature_ranks_first_with_right_direction(artifact):
    X, _ = fake_patients(40, seed=3)
    values, _ = shap_values(artifact, X)
    importance = global_importance(values, X)
    assert importance.iloc[0]["feature"] == "ldl" and importance.iloc[0]["direction"] > 0.5
    assert importance.set_index("feature").loc["noise", "mean_abs_shap"] < 0.02


def test_patient_explanation_marks_blank_and_out_of_range_fields(artifact):
    X = pd.DataFrame([{"age": 60, "ldl": 9999, "noise": np.nan, "sex_male": 1}])
    e = explain_patient(artifact, X)
    table = e["contributions"].set_index("feature")
    assert not table.loc["ldl", "given"] and not table.loc["noise", "given"]
    assert table.loc["age", "given"]
    assert e["probability"] == pytest.approx(float(predict_calibrated(artifact, X)[0]))
    # largest effect first
    assert list(e["contributions"]["contribution"].abs()) == sorted(
        e["contributions"]["contribution"].abs(), reverse=True)


def test_model_without_background_gives_clear_error(artifact):
    old = dict(artifact, explain_background=None)
    with pytest.raises(ValueError, match="explain_tabular.py"):
        shap_values(old, pd.DataFrame([{"age": 60, "ldl": 150, "noise": 0, "sex_male": 1}]))


def test_explain_tabular_from_saved_model(artifact, tmp_path):
    path = save_artifact(artifact, tmp_path / "lab.joblib")
    e = explain_tabular({"age": 70, "ldl": 190, "sex_male": 1}, model_path=path)
    top = e["contributions"].iloc[0]
    assert top["feature"] == "ldl" and top["contribution"] > 0


def test_figures_and_report_written(artifact, tmp_path):
    X, _ = fake_patients(10, seed=4)
    values, _ = shap_values(artifact, X)
    importance = global_importance(values, X)
    fig_path = plot_importance(importance, tmp_path / "fig.png")
    report = write_report(importance, tmp_path / "shap.md", "fake", "fake_ds", 10,
                          {"background_size": 8, "nsamples": 100}, "fig.png", "2026-01-01")
    assert fig_path.stat().st_size > 0
    text = report.read_text(encoding="utf-8")
    assert "| 1 | ldl |" in text and "not causes of disease" in text
    fig = plot_patient(explain_patient(artifact, X.iloc[:1]))
    assert fig.axes and len(fig.axes[0].patches) == len(FEATURES)
