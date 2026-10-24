"""Choosing the decision threshold for a risk model, and measuring it honestly.

A threshold turns a probability into a yes/no flag ("heart disease likely"). Where to
put it is a trade-off between catching sick patients (sensitivity) and not raising
false alarms (specificity), so this module offers a few standard rules:

- ``youden``: maximise sensitivity + specificity - 1 (balanced).
- ``min_sensitivity``: the highest threshold that still catches at least ``target`` of
  sick patients (screening-style: few missed cases, more false alarms).
- ``fixed``: use ``target`` as the threshold.

Choosing a threshold on the same predictions you report results on is optimistic, so
``cross_validated_metrics`` picks the threshold on 4/5 of the patients and measures it
on the other 1/5, rotating through all of them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

METHODS = ("youden", "min_sensitivity", "fixed")


def choose_threshold(y, p, method: str = "youden", target: float | None = None) -> float:
    """Return the threshold chosen by ``method`` on labels ``y`` and probabilities ``p``."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if method == "fixed":
        if target is None:
            raise ValueError("method 'fixed' needs a target threshold")
        return float(target)

    fpr, tpr, thresholds = roc_curve(y, p)
    # roc_curve's first threshold is +inf (flag nobody); drop it.
    fpr, tpr, thresholds = fpr[1:], tpr[1:], thresholds[1:]

    if method == "youden":
        return float(thresholds[np.argmax(tpr - fpr)])
    if method == "min_sensitivity":
        if target is None or not 0 < target <= 1:
            raise ValueError("method 'min_sensitivity' needs a target between 0 and 1")
        # thresholds are in decreasing order, sensitivity in increasing order:
        # the first one that reaches the target is the highest such threshold.
        return float(thresholds[np.argmax(tpr >= target)])
    raise ValueError(f"Unknown threshold method '{method}'. Use one of {METHODS}")


def metrics_at(y, p, threshold: float) -> dict[str, float]:
    """Sensitivity, specificity, PPV, NPV and share of patients flagged at a threshold."""
    y = np.asarray(y)
    flagged = np.asarray(p, dtype=float) >= threshold
    tp = np.sum(flagged & (y == 1))
    fp = np.sum(flagged & (y == 0))
    tn = np.sum(~flagged & (y == 0))
    fn = np.sum(~flagged & (y == 1))

    def ratio(a, b):
        return float(a / b) if b else float("nan")

    return {
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "ppv": ratio(tp, tp + fp),         # of those flagged, share who are sick
        "npv": ratio(tn, tn + fn),         # of those cleared, share who are healthy
        "flagged": ratio(tp + fp, len(y)),
    }


def cross_validated_metrics(y, p, method: str, target: float | None = None,
                            n_splits: int = 5, seed: int = 42) -> dict[str, float]:
    """Honest performance of a threshold *rule*.

    For each fold, the threshold is chosen on the other folds and applied to this one;
    the flags from all folds are then pooled and scored together.
    """
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    applied = np.zeros(len(y))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for choose_idx, apply_idx in cv.split(p.reshape(-1, 1), y):
        t = choose_threshold(y[choose_idx], p[choose_idx], method, target)
        applied[apply_idx] = p[apply_idx] - t  # >= 0 means flagged
    return metrics_at(y, applied, 0.0)


# Standard rules compared in the threshold report (method, target).
STANDARD_RULES = [("fixed", 0.5), ("youden", None),
                  ("min_sensitivity", 0.90), ("min_sensitivity", 0.95)]


def rule_label(method: str, target: float | None) -> str:
    if method == "fixed":
        return f"fixed {target:g}"
    if method == "min_sensitivity":
        return f"sensitivity >= {target:.0%}"
    return method


def threshold_summary(y, p, method: str = "youden", target: float | None = None,
                      seed: int = 42) -> dict:
    """The chosen threshold plus a comparison of the standard rules.

    Each rule's threshold is chosen on all of ``p``; its metrics are the cross-validated
    (honest) estimates from ``cross_validated_metrics``.
    """
    rules = list(STANDARD_RULES)
    if (method, target) not in rules:
        rules.append((method, target))
    options = []
    for m, t in rules:
        cv = cross_validated_metrics(y, p, m, t, seed=seed)
        options.append({
            "rule": rule_label(m, t),
            "method": m,
            "target": t,
            "threshold": round(choose_threshold(y, p, m, t), 4),
            "cv_metrics": {k: round(v, 3) for k, v in cv.items()},
            "chosen": (m, t) == (method, target),
        })
    return {
        "method": method,
        "target": target,
        "threshold": round(choose_threshold(y, p, method, target), 4),
        "prevalence": round(float(np.mean(y)), 3),
        "options": options,
    }


def write_threshold_report(summary: dict, model_name: str, dataset_id: str, n_patients: int,
                           path, run_date: str) -> None:
    """Markdown table comparing the threshold rules, with the chosen one marked."""
    lines = [
        "# Lab branch: decision threshold",
        "",
        f"Model: `{model_name}`. Dataset: `{dataset_id}` ({n_patients} patients). "
        f"Run date: {run_date}.",
        "",
        "A patient is flagged \"heart disease likely\" when the calibrated lab probability is "
        "at or above the threshold. Each rule's threshold is chosen on all held-out (cross-"
        "validation) predictions; the metrics are honest estimates, with the threshold chosen "
        "on 4/5 of patients and measured on the other 1/5.",
        "",
        "| Rule | Threshold | Sensitivity | Specificity | PPV | NPV | Flagged |",
        "|---|---|---|---|---|---|---|",
    ]
    for o in summary["options"]:
        m = o["cv_metrics"]
        name = f"**{o['rule']} (chosen)**" if o["chosen"] else o["rule"]
        lines.append(
            f"| {name} | {o['threshold']:.3f} | {m['sensitivity']:.3f} | {m['specificity']:.3f} "
            f"| {m['ppv']:.3f} | {m['npv']:.3f} | {m['flagged']:.0%} |")
    lines += [
        "",
        "- PPV: of patients flagged, the share who have heart disease. NPV: of patients "
        "cleared, the share who don't. Both depend on how common the disease is: "
        f"{summary['prevalence']:.0%} in this hospital dataset, far higher than in screening, "
        "so real-world PPV will be lower and NPV higher.",
        "- The threshold applies to the lab probability only, not to the fused (lab + ECG) "
        "probability, which uses the risk categories in `configs/fusion.yaml`.",
        "- Which trade-off is right (balanced vs few missed cases) is a clinical decision; "
        "change `final.threshold_method` in `configs/tabular.yaml` and re-run "
        "`python scripts/choose_threshold.py` (no retraining needed).",
        "",
        "> These scores overstate real-world performance; see `reports/data_audit_tabular.md`.",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
