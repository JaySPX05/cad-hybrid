
"""Evaluate the ECG model on the held-out test set.

Usage:
    python -m cad_hybrid.ecg.evaluate
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
)

from cad_hybrid.ecg.predict import predict_ecg


# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[3]

TEST_CSV = ROOT / "data" / "processed" / "test.csv"

RESULTS_DIR = ROOT / "results"

RANDOM_STATE = 42
N_BOOTSTRAP = 1000
THRESHOLD = 0.5


# =====================================================
# METRIC CALCULATION
# =====================================================

def calculate_metrics(y, scores):
    """Calculate binary classification metrics."""

    predictions = (scores >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "auroc": roc_auc_score(y, scores),
        "sensitivity": (
            tp / (tp + fn) if tp + fn else np.nan
        ),
        "specificity": (
            tn / (tn + fp) if tn + fp else np.nan
        ),
        "brier_score": brier_score_loss(y, scores),
    }


# =====================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =====================================================

def bootstrap_ci(y, scores):
    """Calculate 95% bootstrap confidence intervals."""

    rng = np.random.default_rng(RANDOM_STATE)

    metric_names = [
        "auroc",
        "sensitivity",
        "specificity",
        "brier_score",
    ]

    results = {
        name: [] for name in metric_names
    }

    for _ in range(N_BOOTSTRAP):

        indices = rng.integers(
            0,
            len(y),
            len(y),
        )

        y_sample = y[indices]
        score_sample = scores[indices]

        if len(np.unique(y_sample)) < 2:
            continue

        metrics = calculate_metrics(
            y_sample,
            score_sample,
        )

        for key, value in metrics.items():

            if np.isfinite(value):
                results[key].append(value)

    return {
        key: [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]
        for key, values in results.items()
        if values
    }


# =====================================================
# MAIN EVALUATION
# =====================================================

def main():

    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Test CSV not found: {TEST_CSV}"
        )

    df = pd.read_csv(TEST_CSV)

    required_columns = {
        "path",
        "label",
        "dup_group",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # CAD versus normal only.
    df = df[
        df["label"].isin(
            ["cad_positive", "negative"]
        )
    ].copy()

    if df.empty:
        raise ValueError(
            "No CAD-positive or normal samples found in test CSV."
        )

    # One image per exact-duplicate group.
    df = (
        df.sort_values("path")
        .drop_duplicates(
            subset="dup_group",
            keep="first",
        )
        .reset_index(drop=True)
    )

    y = (
        df["label"] == "cad_positive"
    ).astype(int).to_numpy()

    if len(np.unique(y)) < 2:
        raise ValueError(
            "Evaluation requires both CAD-positive and normal samples."
        )

    scores = []

    print("=" * 50)
    print("ECG MODEL EVALUATION")
    print("=" * 50)

    print("Test images:", len(df))
    print("CAD-positive:", int(y.sum()))
    print("Normal:", int((y == 0).sum()))

    # -----------------------------------------------
    # PREDICTIONS
    # -----------------------------------------------

    for i, row in df.iterrows():

        result = predict_ecg(row["path"])

        scores.append(
            float(result["cad_probability"])
        )

        if (i + 1) % 10 == 0:
            print(
                f"Processed {i + 1}/{len(df)}"
            )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    if not np.all(np.isfinite(scores)):
        raise ValueError(
            "Model produced invalid or non-finite prediction scores."
        )

    if np.any((scores < 0) | (scores > 1)):
        raise ValueError(
            "Prediction scores must be between 0 and 1."
        )

    # -----------------------------------------------
    # METRICS
    # -----------------------------------------------

    metrics = calculate_metrics(
        y,
        scores,
    )

    confidence_intervals = bootstrap_ci(
        y,
        scores,
    )

    print("\nTEST RESULTS")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nBOOTSTRAP 95% CONFIDENCE INTERVALS")

    for name, interval in confidence_intervals.items():

        print(
            f"{name}: "
            f"{interval[0]:.4f} - {interval[1]:.4f}"
        )

    # -----------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "test_images": len(df),
        "cad_positive": int(y.sum()),
        "normal": int((y == 0).sum()),
        "threshold": THRESHOLD,
        "metrics": {
            key: float(value)
            for key, value in metrics.items()
        },
        "bootstrap_95_ci": confidence_intervals,
        "bootstrap_iterations": N_BOOTSTRAP,
    }

    output_path = (
        RESULTS_DIR / "ecg_test_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=4,
        )

    print("\nResults saved to:", output_path)

    print("\nECG EVALUATION COMPLETED")


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()
