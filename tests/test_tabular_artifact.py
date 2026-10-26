"""Tests for saving, loading and using the final lab model (fake data only)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cad_hybrid.fusion.calibrate import Calibrator
from cad_hybrid.predict import predict_tabular, tabular_threshold
from cad_hybrid.tabular.artifact import build_artifact, load_artifact, save_artifact, to_frame
from cad_hybrid.tabular.baseline import logistic_regression

FEATURES = ["age", "ldl", "sex_male"]
RANGES = {"age": [18, 110], "ldl": [10, 500]}


def fake_patients(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "age": rng.uniform(20, 80, n),
        "ldl": rng.uniform(60, 200, n),
        "sex_male": rng.integers(0, 2, n).astype(float),
    })
    risk = 0.03 * (X["age"] - 50) + 0.02 * (X["ldl"] - 130)
    y = pd.Series((risk + rng.normal(0, 0.5, n) > 0).astype(int))
    return X, y


@pytest.fixture
def model_path(tmp_path) -> Path:
    """Train a small model on fake data and save it as an artifact."""
    X, y = fake_patients()
    model = logistic_regression(["age", "ldl"], ["sex_male"]).fit(X, y)
    calibrator = Calibrator("platt").fit(model.predict_proba(X)[:, 1], y.to_numpy())
    artifact = build_artifact(model, calibrator, FEATURES, RANGES, {"model": "fake"})
    return save_artifact(artifact, tmp_path / "lab_model.joblib")


def test_save_writes_model_and_readable_metadata(model_path):
    assert model_path.exists()
    assert json.loads(model_path.with_suffix(".json").read_text())["model"] == "fake"


def test_predict_returns_probability(model_path):
    p = predict_tabular({"age": 60, "ldl": 180, "sex_male": 1}, model_path=model_path)
    assert isinstance(p, float) and 0.0 <= p <= 1.0


def test_higher_risk_values_give_higher_probability(model_path):
    low = predict_tabular({"age": 25, "ldl": 70, "sex_male": 0}, model_path=model_path)
    high = predict_tabular({"age": 75, "ldl": 190, "sex_male": 1}, model_path=model_path)
    assert high > low


def test_missing_fields_are_allowed(model_path):
    p = predict_tabular({"age": 60}, model_path=model_path)
    assert 0.0 <= p <= 1.0


def test_out_of_range_value_is_treated_as_missing(model_path):
    impossible = predict_tabular({"age": 60, "ldl": 9999}, model_path=model_path)
    missing = predict_tabular({"age": 60}, model_path=model_path)
    assert impossible == pytest.approx(missing)


def test_unknown_field_rejected(model_path):
    with pytest.raises(ValueError, match="Unknown lab fields"):
        predict_tabular({"age": 60, "cholestrol": 200}, model_path=model_path)


def test_non_numeric_value_rejected():
    with pytest.raises(ValueError, match="must be numbers"):
        to_frame({"age": "sixty"}, FEATURES)


def test_numeric_strings_accepted():
    frame = to_frame({"age": "60", "ldl": "150.5"}, FEATURES)
    assert frame.loc[0, "age"] == 60.0 and frame.loc[0, "ldl"] == 150.5
    assert np.isnan(frame.loc[0, "sex_male"])


@pytest.fixture
def strict_model_path(tmp_path) -> Path:
    """Same fake model, but allowing at most 1 blank field."""
    X, y = fake_patients()
    model = logistic_regression(["age", "ldl"], ["sex_male"]).fit(X, y)
    calibrator = Calibrator("platt").fit(model.predict_proba(X)[:, 1], y.to_numpy())
    artifact = build_artifact(model, calibrator, FEATURES, RANGES, {"model": "fake"},
                              max_missing=1)
    return save_artifact(artifact, tmp_path / "strict.joblib")


def test_too_many_blank_fields_rejected(strict_model_path):
    with pytest.raises(ValueError, match="Too many missing"):
        predict_tabular({"age": 60}, model_path=strict_model_path)  # 2 of 3 blank


def test_out_of_range_counts_as_blank(strict_model_path):
    with pytest.raises(ValueError, match="ldl"):
        predict_tabular({"age": 60, "ldl": 9999}, model_path=strict_model_path)


def test_within_blank_limit_accepted(strict_model_path):
    p = predict_tabular({"age": 60, "ldl": 150}, model_path=strict_model_path)  # 1 blank
    assert 0.0 <= p <= 1.0


def test_missing_model_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_tabular.py"):
        predict_tabular({"age": 60}, model_path=tmp_path / "nope.joblib")


# ---------- train_final_model in the training script ----------

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_tabular.py"
spec = importlib.util.spec_from_file_location("train_tabular_for_artifact", SCRIPT)
train_tabular = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_tabular)


def test_train_final_model_saves_usable_model(tmp_path):
    X, y = fake_patients()
    models = {"lr": LogisticRegression()}
    oof = {"lr": train_tabular.out_of_fold_predictions(models["lr"], X, y, 5, 1, 0)}
    results = {"lr": train_tabular.summarise(y.to_numpy(), oof["lr"][0])}
    config = {
        "data": {"target": "heart_disease", "plausible_ranges": RANGES},
        "final": {"calibration": "platt", "threshold_method": "youden"},
    }
    path = train_tabular.train_final_model(
        config, models, oof, results, X, y, "lr", "fake_dataset",
        output_path=tmp_path / "final.joblib")

    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["model"] == "lr" and meta["n_patients"] == len(y)
    assert set(meta["cv_metrics"]) == set(train_tabular.METRICS)
    p = predict_tabular({"age": 60, "ldl": 150, "sex_male": 1}, model_path=path)
    assert 0.0 <= p <= 1.0
    # threshold chosen on the held-out predictions and stored with the model
    assert 0.0 < meta["threshold"] < 1.0 and meta["threshold_method"] == "youden"
    assert tabular_threshold(path) == meta["threshold"]
    # explanation background stored with the model (synthetic centres, not patients)
    background = load_artifact.__wrapped__(str(path))["explain_background"]
    assert len(background["centres"]) <= 10 and list(background["centres"].columns) == list(X.columns)


def test_train_final_model_uses_configured_threshold_rule(tmp_path):
    X, y = fake_patients()
    models = {"lr": LogisticRegression()}
    oof = {"lr": train_tabular.out_of_fold_predictions(models["lr"], X, y, 5, 1, 0)}
    results = {"lr": train_tabular.summarise(y.to_numpy(), oof["lr"][0])}
    config = {
        "data": {"target": "heart_disease", "plausible_ranges": RANGES},
        "final": {"calibration": "platt", "threshold_method": "fixed", "threshold_target": 0.3},
    }
    path = train_tabular.train_final_model(
        config, models, oof, results, X, y, "lr", "fake_dataset",
        output_path=tmp_path / "final.joblib")
    assert tabular_threshold(path) == 0.3


def test_model_without_threshold_returns_none(model_path):
    assert tabular_threshold(model_path) is None


def test_save_oof_writes_labels_and_mean_predictions(tmp_path):
    X, y = fake_patients()
    oof = {"lr": np.vstack([np.full(len(y), 0.2), np.full(len(y), 0.4)])}
    path = train_tabular.save_oof(oof, y, tmp_path / "oof.csv")
    frame = pd.read_csv(path)
    assert list(frame.columns) == ["y", "lr"] and frame["lr"].iloc[0] == pytest.approx(0.3)


# ---------- scripts/choose_threshold.py ----------

CHOOSE = Path(__file__).resolve().parents[1] / "scripts" / "choose_threshold.py"
spec2 = importlib.util.spec_from_file_location("choose_threshold", CHOOSE)
choose_threshold_script = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(choose_threshold_script)


def test_rechoose_threshold_without_retraining(tmp_path):
    X, y = fake_patients()
    models = {"lr": LogisticRegression()}
    oof = {"lr": train_tabular.out_of_fold_predictions(models["lr"], X, y, 5, 1, 0)}
    results = {"lr": train_tabular.summarise(y.to_numpy(), oof["lr"][0])}
    config = {
        "data": {"target": "heart_disease", "plausible_ranges": RANGES},
        "final": {"calibration": "platt", "threshold_method": "youden"},
    }
    path = train_tabular.train_final_model(
        config, models, oof, results, X, y, "lr", "fake_dataset",
        output_path=tmp_path / "final.joblib")
    oof_path = train_tabular.save_oof(oof, y, tmp_path / "oof.csv")
    youden = tabular_threshold(path)

    meta = choose_threshold_script.rechoose(
        config, path, oof_path, tmp_path / "report.md", method="min_sensitivity", target=0.99)
    assert meta["threshold_method"] == "min_sensitivity"
    assert tabular_threshold(path) == meta["threshold"] <= youden
    assert "(chosen)" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_rechoose_rejects_mismatched_predictions(model_path, tmp_path):
    pd.DataFrame({"y": [0, 1], "other_model": [0.1, 0.9]}).to_csv(tmp_path / "oof.csv", index=False)
    with pytest.raises(ValueError, match="doesn't match"):
        choose_threshold_script.rechoose({"final": {}}, model_path, tmp_path / "oof.csv",
                                         tmp_path / "r.md")


def test_train_final_model_rejects_unknown_model(tmp_path):
    with pytest.raises(ValueError, match="isn't enabled"):
        train_tabular.train_final_model({"final": {}}, {}, {}, {}, None, None, "nope", "x",
                                        output_path=tmp_path / "f.joblib")
