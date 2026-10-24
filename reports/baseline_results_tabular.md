# Lab branch: baseline results

Dataset: `bangladesh_cardiac` (1035 patients). Run date: 2026-10-24.
Held-out predictions from repeated stratified 5-fold CV (3 repeats), 95% bootstrap confidence intervals, threshold 0.5.

| Model | auroc | auprc | brier | sensitivity | specificity |
|---|---|---|---|---|---|
| logistic_regression | 0.988 (0.982 to 0.993) | 0.992 (0.988 to 0.995) | 0.036 (0.028 to 0.045) | 0.947 (0.928 to 0.965) | 0.962 (0.943 to 0.979) |
| xgboost | 0.990 (0.983 to 0.994) | 0.993 (0.990 to 0.996) | 0.034 (0.027 to 0.042) | 0.947 (0.929 to 0.964) | 0.960 (0.942 to 0.976) |
| tabicl | 0.993 (0.989 to 0.997) | 0.996 (0.993 to 0.998) | 0.025 (0.018 to 0.033) | 0.964 (0.948 to 0.978) | 0.973 (0.959 to 0.987) |
| stack_xgb_tabicl | 0.992 (0.986 to 0.996) | 0.995 (0.992 to 0.997) | 0.026 (0.019 to 0.034) | 0.964 (0.949 to 0.978) | 0.973 (0.958 to 0.987) |

> These scores overstate real-world performance; see `reports/data_audit_tabular.md`.
