
"""Evaluate the ECG CAD score on the held-out test set.

Usage:
    python -m cad_hybrid.ecg.evaluate

CAD vs normal ECGs only, one image per duplicate group. The decision threshold is
chosen on the validation set (Youden), never on the test set, then applied to test;
results at a fixed 0.5 are reported alongside for comparison. Saved to
reports/ecg_test_results.json.
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
from cad_hybrid.ecg.scoring import cad_vs_normal
from cad_hybrid.evaluation.thresholds import choose_threshold


# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[3]

TEST_CSV = ROOT / "data" / "processed" / "test.csv"
VALIDATION_CSV = ROOT / "data" / "processed" / "validation.csv"

RESULTS_DIR = ROOT / "reports"

RANDOM_STATE = 42
N_BOOTSTRAP = 1000
FIXED_THRESHOLD = 0.5


# =====================================================
# METRIC CALCULATION
# =====================================================

def calculate_metrics(y, scores, threshold):
    """Calculate binary classification metrics at a threshold."""

    predictions = (scores >= threshold).astype(int)

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

def bootstrap_ci(y, scores, threshold):
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
            threshold,
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

def load_split(csv_path):
    """CAD and normal ECGs from a split, one image per duplicate group."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"dup_group": str})

    missing = {"path", "label", "dup_group"} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing)}")

    df, y = cad_vs_normal(df)

    if len(np.unique(y)) < 2:
        raise ValueError(f"{csv_path.name} needs both CAD and normal ECGs.")

    return df, y


def score_split(df, name):
    """Calibrated P(CAD) for every image in the split."""

    scores = []
    for i, path in enumerate(df["path"], start=1):
        scores.append(float(predict_ecg(path)["cad_probability"]))
        if i % 10 == 0 or i == len(df):
            print(f"{name}: scored {i}/{len(df)}")

    scores = np.asarray(scores, dtype=float)
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Model produced scores outside 0 to 1.")
    return scores


def main():

    print("=" * 50)
    print("ECG MODEL EVALUATION")
    print("=" * 50)

    validation_df, y_validation = load_split(VALIDATION_CSV)
    test_df, y_test = load_split(TEST_CSV)

    print(f"Validation ECGs: {len(y_validation)} "
          f"(CAD {int(y_validation.sum())}, normal {int((y_validation == 0).sum())})")
    print(f"Test ECGs: {len(y_test)} "
          f"(CAD {int(y_test.sum())}, normal {int((y_test == 0).sum())})")

    # Threshold chosen on validation only; the test set is not looked at.
    validation_scores = score_split(validation_df, "validation")
    threshold = choose_threshold(y_validation, validation_scores, "youden")
    print(f"\nThreshold chosen on validation (Youden): {threshold:.4f}")

    test_scores = score_split(test_df, "test")

    metrics = calculate_metrics(y_test, test_scores, threshold)
    confidence_intervals = bootstrap_ci(y_test, test_scores, threshold)
    fixed_metrics = calculate_metrics(y_test, test_scores, FIXED_THRESHOLD)

    print("\nTEST RESULTS (validation-chosen threshold)")
    for name, value in metrics.items():
        interval = confidence_intervals.get(name)
        ci = f"  ({interval[0]:.4f} - {interval[1]:.4f})" if interval else ""
        print(f"{name}: {value:.4f}{ci}")

    print(f"\nAt fixed {FIXED_THRESHOLD}: sensitivity "
          f"{fixed_metrics['sensitivity']:.4f}, specificity {fixed_metrics['specificity']:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "unit": "one image per duplicate group, CAD vs normal only",
        "validation_ecgs": int(len(y_validation)),
        "test_ecgs": int(len(y_test)),
        "test_cad_positive": int(y_test.sum()),
        "test_normal": int((y_test == 0).sum()),
        "threshold": float(threshold),
        "threshold_method": "youden, chosen on validation",
        "metrics": {key: float(value) for key, value in metrics.items()},
        "bootstrap_95_ci": confidence_intervals,
        "bootstrap_iterations": N_BOOTSTRAP,
        "at_fixed_0_5": {key: float(value) for key, value in fixed_metrics.items()},
    }

    output_path = RESULTS_DIR / "ecg_test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print("\nResults saved to:", output_path)
    print("\nECG EVALUATION COMPLETED")


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()
