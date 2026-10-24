# Lab branch: decision threshold

Model: `stack_xgb_tabicl`. Dataset: `bangladesh_cardiac` (1035 patients). Run date: 2026-10-24.

A patient is flagged "heart disease likely" when the calibrated lab probability is at or above the threshold. Each rule's threshold is chosen on all held-out (cross-validation) predictions; the metrics are honest estimates, with the threshold chosen on 4/5 of patients and measured on the other 1/5.

| Rule | Threshold | Sensitivity | Specificity | PPV | NPV | Flagged |
|---|---|---|---|---|---|---|
| fixed 0.5 | 0.500 | 0.969 | 0.967 | 0.974 | 0.960 | 56% |
| **youden (chosen)** | 0.540 | 0.968 | 0.976 | 0.981 | 0.959 | 56% |
| sensitivity >= 90% | 0.971 | 0.909 | 0.984 | 0.987 | 0.893 | 52% |
| sensitivity >= 95% | 0.841 | 0.945 | 0.980 | 0.984 | 0.932 | 54% |

- PPV: of patients flagged, the share who have heart disease. NPV: of patients cleared, the share who don't. Both depend on how common the disease is: 57% in this hospital dataset, far higher than in screening, so real-world PPV will be lower and NPV higher.
- The threshold applies to the lab probability only, not to the fused (lab + ECG) probability, which uses the risk categories in `configs/fusion.yaml`.
- Which trade-off is right (balanced vs few missed cases) is a clinical decision; change `final.threshold_method` in `configs/tabular.yaml` and re-run `python scripts/choose_threshold.py` (no retraining needed).

> These scores overstate real-world performance; see `reports/data_audit_tabular.md`.
