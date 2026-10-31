
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
)

from src.cad_hybrid.ecg.predict import predict_ecg

ROOT = Path(__file__).resolve().parents[3]
TEST_CSV = ROOT / "data" / "processed" / "test.csv"
RESULTS_DIR = ROOT / "results"
RANDOM_STATE = 42
N_BOOTSTRAP = 1000
THRESHOLD = 0.5


def calculate_metrics(y, scores):
    predictions = (scores >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    return {
        "auroc": roc_auc_score(y, scores),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "brier_score": brier_score_loss(y, scores),
    }


def bootstrap_ci(y, scores):
    rng = np.random.default_rng(RANDOM_STATE)
    results = {k: [] for k in
               ["auroc", "sensitivity", "specificity", "brier_score"]}

    for _ in range(N_BOOTSTRAP):
        indices = rng.integers(0, len(y), len(y))
        y_sample = y[indices]
        score_sample = scores[indices]

        if len(np.unique(y_sample)) < 2:
            continue

        metrics = calculate_metrics(y_sample, score_sample)

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


def main():
    df = pd.read_csv(TEST_CSV)

    # CAD versus normal only.
    df = df[df["label"].isin(["cad_positive", "negative"])].copy()

    # One image per exact-duplicate group.
    df = df.sort_values("path").drop_duplicates(
        subset="dup_group", keep="first"
    ).reset_index(drop=True)

    y = (df["label"] == "cad_positive").astype(int).to_numpy()
    scores = []

    print("Test images:", len(df))
    print("CAD-positive:", int(y.sum()))
    print("Normal:", int((y == 0).sum()))

    for i, row in df.iterrows():
        result = predict_ecg(row["path"])
        scores.append(result["cad_probability"])

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(df)}")

    scores = np.asarray(scores, dtype=float)

    metrics = calculate_metrics(y, scores)
    confidence_intervals = bootstrap_ci(y, scores)

    print("\nTEST RESULTS")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nBOOTSTRAP 95% CONFIDENCE INTERVALS")
    for name, interval in confidence_intervals.items():
        print(f"{name}: {interval[0]:.4f} - {interval[1]:.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)

    output = {
        "test_images": len(df),
        "cad_positive": int(y.sum()),
        "normal": int((y == 0).sum()),
        "threshold": THRESHOLD,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "bootstrap_95_ci": confidence_intervals,
        "bootstrap_iterations": N_BOOTSTRAP,
    }

    with open(RESULTS_DIR / "ecg_test_results.json", "w") as f:
        json.dump(output, f, indent=4)

    print("\nResults saved to:", RESULTS_DIR / "ecg_test_results.json")


if __name__ == "__main__":
    main()
