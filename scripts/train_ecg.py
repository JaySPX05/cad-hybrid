
"""Train the ECG SVM classifier using locally extracted ResNet18 features.

Usage:
    python scripts/train_ecg.py --config configs/ecg.yaml
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]

CLASS_NAMES = [
    "cad_positive",
    "negative",
    "arrhythmia",
]


def load_features(path):
    path = Path(path)

    if not path.is_absolute():
        path = ROOT / path

    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}"
        )

    with np.load(path) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(int)
        groups = data["groups"].astype(str)

    if X.ndim != 2 or X.shape[1] != 512:
        raise ValueError(
            f"Expected 512-dimensional features, got {X.shape}"
        )

    if not (
        len(X) == len(y) == len(groups)
    ):
        raise ValueError(
            "Feature, label and group counts do not match."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "Features contain NaN or infinite values."
        )

    return X, y, groups


def evaluate_model(model, X, y, split_name):
    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        y,
        predictions,
        labels=[0, 1, 2],
    )

    report = classification_report(
        y,
        predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    print(f"\n{split_name} evaluation")
    print("------------------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("Confusion matrix:")
    print(matrix)

    print("\nClassification report:")
    print(
        classification_report(
            y,
            predictions,
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--config",
        default="configs/ecg.yaml",
    )

    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = ROOT / config_path

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Automatically extract features before SVM training.
    print("\nChecking and extracting ECG features...")

    extraction_script = (
        ROOT / "scripts" / "extract_ecg_features.py"
    )

    subprocess.run(
        [sys.executable, str(extraction_script)],
        cwd=ROOT,
        check=True,
    )

    # Load locally extracted features.
    print("\nLoading local ECG features...")

    X_train, y_train, train_groups = load_features(
        "data/features/train_features.npz"
    )

    X_val, y_val, _ = load_features(
        "data/features/validation_features.npz"
    )

    print("\nTraining data:", X_train.shape)
    print("Validation data:", X_val.shape)
    print("Training groups:", len(np.unique(train_groups)))

    unique_classes, class_counts = np.unique(
        y_train,
        return_counts=True,
    )

    print("\nTraining class distribution:")

    for class_id, count in zip(unique_classes, class_counts):
        print(f"{CLASS_NAMES[class_id]}: {count}")

    if len(unique_classes) != len(CLASS_NAMES):
        raise ValueError(
            "Training data must contain all three ECG classes."
        )

    # Verify that each duplicate group has only one label.
    group_counts = (
        pd.DataFrame({
            "label": y_train,
            "group": train_groups,
        })
        .groupby("group")["label"]
        .nunique()
    )

    if (group_counts > 1).any():
        raise ValueError(
            "A duplicate group contains multiple labels."
        )

    # Group-aware cross-validation.
    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # Original SVM hyperparameter search.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "svm",
            SVC(
                kernel="rbf",
                probability=False,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    parameter_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", 0.1, 0.01, 0.001],
    }

    print("\nStarting group-aware SVM grid search...")

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
        refit=True,
        verbose=2,
        error_score="raise",
    )

    search.fit(
        X_train,
        y_train,
        groups=train_groups,
    )

    best_model = search.best_estimator_

    print("\nBest SVM parameters:")
    print(search.best_params_)

    print(
        "Best cross-validation macro F1:",
        round(search.best_score_, 4),
    )

    # Evaluate on the independent validation split.
    validation_results = evaluate_model(
        best_model,
        X_val,
        y_val,
        "Validation",
    )

    # Save model locally.
    model_config = config.get("model", {})

    model_path = Path(
        model_config.get(
            "classifier_path",
            "models/ecg_svm_classifier.pkl",
        )
    )

    if not model_path.is_absolute():
        model_path = ROOT / model_path

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(best_model, model_path)

    # Save training results.
    results = {
        "model": "ResNet18 features + RBF SVM",
        "training_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "training_groups": int(len(np.unique(train_groups))),
        "best_parameters": search.best_params_,
        "best_cv_macro_f1": float(search.best_score_),
        "validation": validation_results,
    }

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    results_path = (
        reports_dir / "ecg_training_results.json"
    )

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nSVM training completed successfully.")
    print("Saved model:", model_path)
    print("Saved results:", results_path)


if __name__ == "__main__":
    main()
