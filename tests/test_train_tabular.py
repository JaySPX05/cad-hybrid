"""Tests for the lab-branch training script, using fake data only."""

import importlib.util
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# scripts/ isn't a package, so load the script file directly
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_tabular.py"
spec = importlib.util.spec_from_file_location("train_tabular", SCRIPT)
train_tabular = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_tabular)


def fake_data(n: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series((X["a"] + rng.normal(0, 0.5, n) > 0).astype(int))
    return X, y


def test_every_patient_gets_a_held_out_prediction():
    X, y = fake_data()
    preds = train_tabular.out_of_fold_predictions(
        LogisticRegression(), X, y, n_splits=5, n_repeats=2, seed=0)
    assert preds.shape == (2, len(y))
    # predictions start as 0, so any 0 left means a patient was never predicted
    assert ((preds > 0) & (preds < 1)).all()


def test_summarise_returns_all_metrics_with_intervals():
    X, y = fake_data()
    preds = train_tabular.out_of_fold_predictions(LogisticRegression(), X, y, 5, 1, 0)[0]
    results = train_tabular.summarise(y.to_numpy(), preds)
    assert list(results) == train_tabular.METRICS
    for point, low, high in results.values():
        assert low <= point <= high


def test_save_results_writes_csv_and_markdown(tmp_path):
    results = {"model_a": {m: (0.9, 0.8, 0.95) for m in train_tabular.METRICS}}
    csv_path, md_path = train_tabular.save_results(results, "fake_dataset", 100, out_dir=tmp_path)
    assert len(pd.read_csv(csv_path)) == len(train_tabular.METRICS)
    text = md_path.read_text(encoding="utf-8")
    assert "| model_a |" in text and "overstate" in text

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "tabular.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_build_models_without_tabicl():
    config = load_config()
    config["models"]["tabicl"]["enabled"] = False
    assert set(train_tabular.build_models(config)) == {"logistic_regression", "xgboost"}


def test_xgboost_imputes_so_missingness_is_not_a_signal():
    config = load_config()
    config["models"]["tabicl"]["enabled"] = False
    xgb = train_tabular.build_models(config)["xgboost"]
    assert type(xgb.steps[0][1]).__name__ == "SimpleImputer"


def test_build_models_with_tabicl_adds_stack():
    config = load_config()
    config["models"]["tabicl"]["enabled"] = True
    config["models"]["stack"]["enabled"] = True
    assert set(train_tabular.build_models(config)) == {
        "logistic_regression", "xgboost", "tabicl", "stack_xgb_tabicl"}