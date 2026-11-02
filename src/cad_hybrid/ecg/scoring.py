"""Pure helpers for the ECG CAD score: no torch, no files, so they can be unit-tested.

The ECG model predicts three classes (CAD, normal, arrhythmia). For fusion it needs one
number, P(CAD vs normal), so arrhythmia is left out of the score:

    score = P(CAD) / (P(CAD) + P(normal))

The score is then Platt-calibrated (logistic regression on its log-odds) on validation
ECGs that are CAD or normal, one image per duplicate group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

CAD, NORMAL, ARRHYTHMIA = 0, 1, 2
LABELS = {"cad_positive": CAD, "negative": NORMAL, "arrhythmia": ARRHYTHMIA}
EPS = 1e-7


def cad_vs_normal_score(probabilities, classes) -> np.ndarray:
    """P(CAD) / (P(CAD) + P(normal)) for each row of 3-class probabilities."""
    probabilities = np.atleast_2d(np.asarray(probabilities, dtype=float))
    classes = list(classes)
    p_cad = probabilities[:, classes.index(CAD)]
    p_normal = probabilities[:, classes.index(NORMAL)]
    total = p_cad + p_normal
    if np.any(total <= 0):
        raise ValueError("CAD and normal probabilities sum to zero.")
    return p_cad / total


def score_logit(score) -> np.ndarray:
    """Log-odds of a score in (0, 1), clipped away from 0 and 1, as one column."""
    score = np.clip(np.asarray(score, dtype=float), EPS, 1 - EPS)
    return np.log(score / (1 - score)).reshape(-1, 1)


def fit_binary_calibrator(score, y) -> LogisticRegression:
    """Platt calibration of the CAD-vs-normal score (y: 1 = CAD, 0 = normal)."""
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError("Binary calibration needs both CAD and normal ECGs.")
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000).fit(score_logit(score), y)


def apply_binary_calibrator(calibrator: LogisticRegression, score) -> np.ndarray:
    """Calibrated P(CAD) for each score."""
    positive = list(calibrator.classes_).index(1)
    return calibrator.predict_proba(score_logit(score))[:, positive]


def one_per_group(df: pd.DataFrame, group_col: str = "dup_group") -> pd.DataFrame:
    """Keep the first image (by path) of each duplicate group, so copies count once."""
    return (df.sort_values("path").drop_duplicates(subset=group_col, keep="first")
            .reset_index(drop=True))


def cad_vs_normal(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """CAD and normal ECGs only, one per duplicate group, with labels (1 = CAD)."""
    kept = one_per_group(df[df["label"].isin(["cad_positive", "negative"])])
    return kept, (kept["label"] == "cad_positive").astype(int).to_numpy()


def positive_rate_per_ecg(df: pd.DataFrame) -> dict:
    """CAD rate among CAD + normal ECGs, counting each duplicate group once."""
    kept, y = cad_vs_normal(df)
    return {
        "cad_ecgs": int(y.sum()),
        "normal_ecgs": int((y == 0).sum()),
        "positive_rate": float(y.mean()) if len(y) else float("nan"),
    }
