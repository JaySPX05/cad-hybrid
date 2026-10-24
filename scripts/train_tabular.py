"""Train and evaluate the lab (tabular) branch, then save the final model.

    python scripts/train_tabular.py --config configs/tabular.yaml
    python scripts/train_tabular.py --final-model logistic_regression   # save a different model

Steps: load + clean data -> repeated cross-validation for every enabled model ->
metrics with 95% CIs saved to reports/ -> final model trained on all patients,
calibrated on its held-out predictions, decision threshold chosen on them, saved to
models/ -> pipeline check on the India dataset.

Held-out predictions are also saved to data/processed/oof_tabular.csv (gitignored), so
the threshold can be re-chosen later with scripts/choose_threshold.py without retraining.
"""

import argparse
import platform
import sys
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

from cad_hybrid.data.tabular_loader import load_dataset
from cad_hybrid.evaluation.metrics import bootstrap_ci
from cad_hybrid.evaluation.thresholds import threshold_summary, write_threshold_report
from cad_hybrid.fusion.calibrate import Calibrator
from cad_hybrid.tabular.artifact import PROJECT_ROOT, build_artifact, load_artifact, save_artifact
from cad_hybrid.tabular.baseline import logistic_regression, xgboost_model
from cad_hybrid.tabular.deep_tabular import tabicl_model
from cad_hybrid.tabular.stack import make_stack

METRICS = ["auroc", "auprc", "brier", "sensitivity", "specificity"]

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
THRESHOLD_REPORT = REPORTS_DIR / "threshold_tabular.md"
OOF_PATH = PROJECT_ROOT / "data" / "processed" / "oof_tabular.csv"  # gitignored


def save_results(results: dict, dataset_id: str, n_patients: int, out_dir: Path = REPORTS_DIR,
                 n_folds: int = 5, n_repeats: int = 3) -> list[Path]:
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
        "# Lab branch: model results",
        "",
        f"Dataset: `{dataset_id}` ({n_patients} patients). Run date: {date.today()}.",
        f"Held-out predictions from repeated stratified {n_folds}-fold CV ({n_repeats} repeats), "
        "95% bootstrap confidence intervals. Sensitivity and specificity at threshold 0.5 "
        "for comparison; the final model's threshold is in `reports/threshold_tabular.md`.",
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


def save_oof(oof: dict, y: pd.Series, path: Path = OOF_PATH) -> Path:
    """Save each model's held-out predictions (averaged over repeats) with the labels.

    Patient-level data: goes to data/processed/, which is gitignored. Never commit it.
    """
    frame = pd.DataFrame({"y": y.to_numpy()})
    for name, preds in oof.items():
        frame[name] = preds.mean(axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def build_models(config: dict, numeric: list[str] | None = None,
                 categorical: list[str] | None = None) -> dict:
    """Create the lab-branch models (baselines, TabICL and the stack, as enabled in the config)."""
    features = config["data"]["features"]
    numeric = features["numeric"] if numeric is None else numeric
    categorical = features["categorical"] if categorical is None else categorical
    params = config["models"]
    models = {
        "logistic_regression": logistic_regression(numeric, categorical, **params["logistic_regression"]),
        # Impute before XGBoost (as the other models do) so "value is missing" can't be used
        # as a signal: missing creatinine only occurs in disease-positive patients (audit).
        "xgboost": make_pipeline(
            SimpleImputer(strategy="median"),
            xgboost_model(random_state=config["seed"], **params["xgboost"]),
        ),
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


def library_versions() -> dict:
    """Versions of the libraries a saved model depends on (models may not load on others)."""
    versions = {"python": platform.python_version()}
    for package in ["scikit-learn", "xgboost", "numpy", "pandas", "joblib", "tabicl", "torch"]:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def train_final_model(config: dict, models: dict, oof: dict, results: dict, X: pd.DataFrame,
                      y: pd.Series, name: str, dataset_id: str,
                      output_path: Path | None = None) -> Path:
    """Train the chosen model on all patients, calibrate it and save it for predict_tabular().

    Calibration is fitted on the model's cross-validation (held-out) predictions, so it
    learns how the model behaves on patients it hasn't seen, not on its own training data.
    The decision threshold is chosen on the same held-out predictions, after calibration.
    """
    if name not in models:
        raise ValueError(f"Final model '{name}' isn't enabled in the config. "
                         f"Available: {sorted(models)}")
    final_cfg = config["final"]

    held_out = oof[name].mean(axis=0)
    calibrator = Calibrator(final_cfg["calibration"]).fit(held_out, y.to_numpy())
    thresholds = threshold_summary(
        y.to_numpy(), np.clip(calibrator.transform(held_out), 0.0, 1.0),
        method=final_cfg.get("threshold_method", "youden"),
        target=final_cfg.get("threshold_target"), seed=config.get("seed", 42))
    model = clone(models[name]).fit(X, y)

    metadata = {
        "model": name,
        "trained_on": str(date.today()),
        "dataset_id": dataset_id,
        "n_patients": int(len(y)),
        "positive_rate": round(float(y.mean()), 4),
        "target": config["data"]["target"],
        "target_meaning": "heart disease present (not CAD-specific)",
        "calibration": final_cfg["calibration"],
        "threshold": thresholds["threshold"],
        "threshold_method": thresholds["method"],
        "threshold_target": thresholds["target"],
        "threshold_options": thresholds["options"],
        "cv_metrics": {m: {"value": round(p, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3)}
                       for m, (p, lo, hi) in results[name].items()},
        "library_versions": library_versions(),
        "note": "Research prototype, not for clinical use. CV scores overstate real-world "
                "performance; see reports/data_audit_tabular.md.",
    }
    metadata["max_missing_features"] = final_cfg.get("max_missing_features")
    artifact = build_artifact(model, calibrator, list(X.columns),
                              config["data"].get("plausible_ranges", {}), metadata,
                              max_missing=final_cfg.get("max_missing_features"),
                              threshold=thresholds["threshold"])
    path = output_path or PROJECT_ROOT / final_cfg["output_path"]
    return save_artifact(artifact, path)


def print_threshold(meta: dict) -> None:
    """Print the chosen threshold and the comparison of rules."""
    print(f"\nDecision threshold: {meta['threshold']:.3f} ({meta['threshold_method']}"
          + (f", target {meta['threshold_target']}" if meta["threshold_target"] is not None else "")
          + "). Cross-validated estimates per rule:")
    for o in meta["threshold_options"]:
        m = o["cv_metrics"]
        mark = "  <- chosen" if o["chosen"] else ""
        print(f"  {o['rule']:20s} t={o['threshold']:.3f}  sens {m['sensitivity']:.3f}  "
              f"spec {m['specificity']:.3f}  flagged {m['flagged']:.0%}{mark}")
    print(f"Saved: {THRESHOLD_REPORT}")


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
    parser.add_argument("--final-model", default=None,
                        help="model to save for predict_tabular() (default: final.model in the config)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    final_name = args.final_model or config["final"]["model"]

    dataset_id = config["data"]["train_sources"][0]
    X, y = load_dataset(dataset_id, config)

    print(f"Loaded {dataset_id}: {X.shape[0]} patients, {X.shape[1]} features")
    print(f"Positives: {int(y.sum())}  Negatives: {int((y == 0).sum())}")
    print("Missing values per feature (%):")
    print((X.isna().mean() * 100).round(1).sort_values(ascending=False).to_string())

    models = build_models(config)
    if final_name not in models:  # fail early, before the slow cross-validation
        sys.exit(f"Final model '{final_name}' isn't enabled in the config. Available: {sorted(models)}")

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

    saved = save_results(results, dataset_id, len(y),
                         n_folds=split["cv_folds"], n_repeats=split["cv_repeats"])
    saved.append(save_oof(oof, y))
    print("\nSaved:", *saved, sep="\n  ")

    model_path = train_final_model(config, models, oof, results, X, y, final_name, dataset_id)
    meta = load_artifact(str(model_path))["metadata"]
    write_threshold_report({"threshold": meta["threshold"], "options": meta["threshold_options"],
                            "prevalence": meta["positive_rate"]},
                           final_name, dataset_id, len(y), THRESHOLD_REPORT, meta["trained_on"])
    print(f"\nFinal model '{final_name}' trained on all {len(y)} patients, "
          f"calibrated ({config['final']['calibration']}), saved to:\n  {model_path}")
    print_threshold(meta)

    pipeline_check(config)


if __name__ == "__main__":
    main()
