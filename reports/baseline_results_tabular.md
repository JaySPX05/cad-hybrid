# Lab branch: baseline results

Dataset: `bangladesh_cardiac` (1035 patients). Run date: 2026-10-24.
Held-out predictions from repeated stratified 5-fold CV (3 repeats), 95% bootstrap confidence intervals, threshold 0.5.

| Model | auroc | auprc | brier | sensitivity | specificity |
|---|---|---|---|---|---|
| logistic_regression | 0.988 (0.982 to 0.993) | 0.992 (0.988 to 0.995) | 0.036 (0.028 to 0.045) | 0.947 (0.928 to 0.965) | 0.962 (0.943 to 0.979) |
| xgboost | 0.990 (0.983 to 0.994) | 0.993 (0.990 to 0.996) | 0.034 (0.027 to 0.042) | 0.947 (0.929 to 0.964) | 0.960 (0.942 to 0.976) |

> These scores overstate real-world performance; see `reports/data_audit_tabular.md`.
