# Lab branch: model results

Dataset: `bangladesh_cardiac` (1035 patients). Run date: 2026-11-01.
Held-out predictions from repeated stratified 5-fold CV (3 repeats), 95% bootstrap confidence intervals. Sensitivity and specificity at threshold 0.5 for comparison; the final model's threshold is in `reports/threshold_tabular.md`.

| Model | auroc | auprc | brier | sensitivity | specificity |
|---|---|---|---|---|---|
| logistic_regression | 0.988 (0.982 to 0.993) | 0.992 (0.988 to 0.995) | 0.036 (0.028 to 0.045) | 0.947 (0.928 to 0.965) | 0.962 (0.943 to 0.979) |
| xgboost | 0.988 (0.982 to 0.993) | 0.993 (0.989 to 0.995) | 0.036 (0.028 to 0.044) | 0.950 (0.932 to 0.967) | 0.951 (0.932 to 0.969) |
| tabicl | 0.993 (0.989 to 0.997) | 0.996 (0.993 to 0.998) | 0.025 (0.018 to 0.033) | 0.964 (0.948 to 0.978) | 0.973 (0.959 to 0.987) |
| stack_xgb_tabicl | 0.990 (0.984 to 0.995) | 0.995 (0.991 to 0.997) | 0.027 (0.020 to 0.035) | 0.964 (0.948 to 0.978) | 0.971 (0.956 to 0.984) |

> These scores overstate real-world performance; see `reports/data_audit_tabular.md`.
