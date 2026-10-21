# Model card: cad-hybrid

_Draft. Fill in as the project progresses._

## Overview
- **What it predicts:** TODO (CAD vs general CVD; decided after the data audit)
- **Inputs:** lab values, a 12-lead paper ECG image, or both
- **Output:** probability and risk category (low / moderate / high), with per-branch results and explanations
- **Status:** research prototype, **not for clinical use**

## Architecture
- Lab branch: gradient boosting + deep tabular model (TabICL), stacked
- ECG branch: ImageNet-pretrained CNN fine-tuned on ECG images, embeddings -> XGBoost/SVM
- Fusion: calibrated late fusion; weights hand-set (no paired lab + ECG data yet)

## Training data
South Asian data only (India, Pakistan, Bangladesh). See `data/data_register.csv` and `ATTRIBUTIONS.md`.

| Dataset | Country | Used for | N |
|---|---|---|---|
| TODO | | | |

## Evaluation
| Test set | AUROC (95% CI) | AUPRC | Sensitivity | Specificity | Brier |
|---|---|---|---|---|---|
| Internal held-out | TODO | | | | |
| External: Pakistan -> India (ECG) | TODO | | | | |
| External: India -> Bangladesh (labs) | TODO | | | | |

Subgroups (sex, age bands): TODO

## Limitations
- Small training sets; wide confidence intervals.
- ECG labels are MI / history of MI, so the ECG branch detects heart attacks, not early CAD.
- No dataset has lab values and ECGs for the same patients; fusion weights are not learned.
- No longitudinal outcomes; no risk-over-time prediction.
- Not validated prospectively or in clinical settings.

## Ethical and legal
- De-identified public or approved data only; no re-identification.
- CC BY 4.0 attributions in `ATTRIBUTIONS.md`; data-use agreements logged in the data register.
