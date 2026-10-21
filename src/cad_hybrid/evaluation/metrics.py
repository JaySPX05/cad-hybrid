"""Metrics for binary risk models, with bootstrap confidence intervals.

Small datasets give noisy numbers, so always report the interval, not just the point.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def sensitivity_specificity(y_true, y_prob, threshold: float = 0.5) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return float(sens), float(spec)


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    """AUROC, AUPRC, Brier score, sensitivity and specificity."""
    sens, spec = sensitivity_specificity(y_true, y_prob, threshold)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "sensitivity": sens,
        "specificity": spec,
    }


def bootstrap_ci(
    y_true, y_prob, metric: str = "auroc", n_boot: int = 1000, alpha: float = 0.05,
    threshold: float = 0.5, seed: int = 42,
) -> tuple[float, float, float]:
    """Point estimate and (1 - alpha) percentile bootstrap interval for one metric."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    point = compute_metrics(y_true, y_prob, threshold)[metric]
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:  # AUROC needs both classes
            continue
        samples.append(compute_metrics(y_true[idx], y_prob[idx], threshold)[metric])
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)
