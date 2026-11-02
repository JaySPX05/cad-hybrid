# Model card: cad-hybrid

_Draft. Fill in as the project progresses._

## Overview
- **What it predicts:** CAD. ECG branch: MI or history of MI vs normal ECG (arrhythmia excluded). Lab branch: presence of heart disease in general, used as a stand-in for CAD because the lab data has no CAD label (see `reports/data_audit_tabular.md`); how close the two are is unknown for this data.
- **Inputs:** lab values, a 12-lead paper ECG image, or both
- **Output:** probability and risk category (low / moderate / high), with per-branch results and explanations (lab: SHAP contributions per value; ECG: Grad-CAM, TODO)
- **Status:** research prototype, **not for clinical use**

## Architecture
- Lab branch: XGBoost (after median imputation) + TabICL, stacked with a logistic-regression meta-model; Platt calibration on held-out predictions; decision threshold 0.540 (Youden), see `reports/threshold_tabular.md`
- ECG branch: ImageNet-pretrained CNN fine-tuned on ECG images, embeddings -> XGBoost/SVM
- Fusion: each branch's calibrated probability is converted to a likelihood ratio against its own training positive rate (lab 0.565), then combined on the log-odds scale from a target prior (default: the lab training rate). Weights 1.0 each (naive-Bayes assumption), hand-set because no paired lab + ECG data exists; the fused probability is not validated.

## Training data
South Asian data only (India, Pakistan, Bangladesh). See `data/data_register.csv` and `ATTRIBUTIONS.md`.

| Dataset | Country | Used for | N |
|---|---|---|---|
| Northern Bangladesh cardiac patients (Zia Heart Foundation) | Bangladesh | Lab branch training and cross-validation | 1,035 adults (585 with heart disease) |
| Mendeley Cardiovascular_Disease_Dataset | India | Lab branch pipeline check only (failed the audit) | 1,000 |
| ECG images of cardiac patients (Multan) | Pakistan | ECG branch training, validation and test | 928 images, 491 unique ECGs (CAD 116, normal 142, arrhythmia 233) |
| ECG images for MI complication (Bhopal) | India | ECG branch external test | TODO |

## Evaluation
| Test set | AUROC (95% CI) | AUPRC | Sensitivity | Specificity | Brier |
|---|---|---|---|---|---|
| Lab branch, internal (repeated 5-fold CV, Bangladesh) | 0.991 (0.984 to 0.995) | 0.995 | 0.968 | 0.976 | 0.027 |
| ECG branch, internal | TODO | | | | |
| External: Pakistan -> India (ECG) | TODO | | | | |
| External: Bangladesh -> India (labs, 5 shared features) | 0.557 | | | | |

Lab sensitivity and specificity are cross-validated estimates at the 0.540 threshold. The internal lab scores are far higher than real risk models achieve (hospital selection, very sick vs relatively healthy patients) and will overstate real-world performance. The India test uses a dataset that failed the audit, so it checks the pipeline rather than transfer. Credible evaluation needs CARRS / South Asia Biobank.

Subgroups (sex, age bands): TODO

## Limitations
- Small training sets; wide confidence intervals.
- Lab training data comes from one hospital in Northern Bangladesh; patients with more than 4 of 19 lab values missing are refused.
- ECG labels are MI / history of MI, so the ECG branch detects heart attacks, not early CAD.
- No dataset has lab values and ECGs for the same patients; fusion weights are not learned.
- No longitudinal outcomes; no risk-over-time prediction.
- Not validated prospectively or in clinical settings.

## Ethical and legal
- De-identified public or approved data only; no re-identification.
- CC BY 4.0 attributions in `ATTRIBUTIONS.md`; data-use agreements logged in the data register.
