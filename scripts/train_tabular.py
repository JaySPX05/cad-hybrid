"""Train and evaluate the lab (tabular) branch baselines.

    python scripts/train_tabular.py --config configs/tabular.yaml
"""

import argparse

import yaml

from cad_hybrid.data.tabular_loader import load_dataset

from cad_hybrid.tabular.baseline import logistic_regression, xgboost_model

from datetime import date
from pathlib import Path

import pandas as pd
from cad_hybrid.tabular.stack import make_stack

import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from cad_hybrid.evaluation.metrics import bootstrap_ci

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from cad_hybrid.tabular.deep_tabular import tabicl_model

METRICS = ["auroc", "auprc", "brier", "sensitivity", "specificity"]

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def save_results(results: dict, dataset_id: str, n_patients: int, out_dir: Path = REPORTS_DIR) -> list[Path]:
    """Save metrics as a CSV (one row per model and metric) and a Markdown table."""
    rows = [
        {"model": model, "metric": metric, "value": round(point, 3),
         "ci_low": round(low, 3), "ci_high": round(high, 3)}
        for model, metrics in results.items()
        for metric, (point, low, high) in metrics.items()
    ]
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "baseline_results_tabular.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    lines = [
        "# Lab branch: baseline results",
        "",
        f"Dataset: `{dataset_id}` ({n_patients} patients). Run date: {date.today()}.",
        "Held-out predictions from repeated stratified 5-fold CV (3 repeats), "
        "95% bootstrap confidence intervals, threshold 0.5.",
        "",
        "| Model | " + " | ".join(METRICS) + " |",
        "|---" * (len(METRICS) + 1) + "|",
    ]
    for model, metrics in results.items():
        cells = [f"{p:.3f} ({lo:.3f} to {hi:.3f})" for p, lo, hi in metrics.values()]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines += ["", "> These scores overstate real-world performance; "
                  "see `reports/data_audit_tabular.md`."]
    md_path = out_dir / "baseline_results_tabular.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [csv_path, md_path]


def summarise(y: np.ndarray, probs: np.ndarray, threshold: float = 0.5, seed: int = 42) -> dict:
    """Point estimate and 95% bootstrap confidence interval for each metric."""
    return {
        metric: bootstrap_ci(y, probs, metric=metric, n_boot=1000, threshold=threshold, seed=seed)
        for metric in METRICS
    }

def out_of_fold_predictions(model, X, y, n_splits: int, n_repeats: int, seed: int) -> np.ndarray:
    """Predict every patient with a model that never saw them.

    Returns an array of shape (n_repeats, n_patients).
    """
    preds = np.zeros((n_repeats, len(y)))
    for r in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for train_idx, test_idx in cv.split(X, y):
            fold_model = clone(model)  # fresh, untrained copy for every fold
            fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds[r, test_idx] = fold_model.predict_proba(X.iloc[test_idx])[:, 1]
    return preds

def build_models(config: dict, numeric: list[str] | None = None,
                 categorical: list[str] | None = None) -> dict:
    """Create the lab-branch models (baselines, TabICL and the stack, as enabled in the config)."""
    features = config["data"]["features"]
    numeric = features["numeric"] if numeric is None else numeric
    categorical = features["categorical"] if categorical is None else categorical
    params = config["models"]
    models = {
        "logistic_regression": logistic_regression(numeric, categorical, **params["logistic_regression"]),
        "xgboost": xgboost_model(random_state=config["seed"], **params["xgboost"]),
    }
    if params["tabicl"]["enabled"]:
        models["tabicl"] = make_pipeline(
            SimpleImputer(strategy="median"),
            tabicl_model(finetune=params["tabicl"]["finetune"]),
        )
    stack_cfg = params["stack"]
    if stack_cfg.get("enabled", False) and "tabicl" in models:
        models["stack_xgb_tabicl"] = make_stack(
            {"xgboost": models["xgboost"], "tabicl": models["tabicl"]},
            cv=stack_cfg.get("inner_cv", 3),
        )
    return models

def pipeline_check(config: dict) -> None:
    """Train on the main dataset with shared features only; predict the pipeline-test dataset."""
    shared = config["data"]["shared_features"]
    features = config["data"]["features"]
    numeric = [c for c in shared if c in features["numeric"]]
    categorical = [c for c in shared if c in features["categorical"]]

    train_id = config["data"]["train_sources"][0]
    test_id = config["data"]["pipeline_test_sources"][0]
    X_train, y_train = load_dataset(train_id, config, features=shared)
    X_test, y_test = load_dataset(test_id, config, features=shared)

    print(f"\nPipeline check: train {train_id} ({len(y_train)}) -> test {test_id} ({len(y_test)}), "
          f"features: {shared}")
    for name, model in build_models(config, numeric, categorical).items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        print(f"  {name}: AUROC on {test_id} = {roc_auc_score(y_test, probs):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tabular.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_id = config["data"]["train_sources"][0]
    X, y = load_dataset(dataset_id, config)

    print(f"Loaded {dataset_id}: {X.shape[0]} patients, {X.shape[1]} features")
    print(f"Positives: {int(y.sum())}  Negatives: {int((y == 0).sum())}")
    print("Missing values per feature (%):")
    print((X.isna().mean() * 100).round(1).sort_values(ascending=False).to_string())

    models = build_models(config)
    split = config["split"]
    oof = {}
    for name, model in models.items():
        oof[name] = out_of_fold_predictions(
            model, X, y, split["cv_folds"], split["cv_repeats"], config["seed"]
        )
        aucs = [roc_auc_score(y, oof[name][r]) for r in range(len(oof[name]))]
        print(f"{name}: AUROC per repeat = {[round(a, 3) for a in aucs]}")

    print("\nResults (held-out predictions averaged over repeats, 95% CI):")
    results = {}
    for name, preds in oof.items():
        mean_pred = preds.mean(axis=0)
        results[name] = summarise(y.to_numpy(), mean_pred, seed=config["seed"])
        print(f"\n{name}")
        for metric, (point, low, high) in results[name].items():
            print(f"  {metric:12s} {point:.3f}  ({low:.3f} to {high:.3f})")

    saved = save_results(results, dataset_id, len(y))
    print("\nSaved:", *saved, sep="\n  ")
    pipeline_check(config)

if __name__ == "__main__":
    main()