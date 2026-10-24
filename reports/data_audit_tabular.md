# Lab (tabular) data audit — 24 Oct 2026

## Decision
- **Northern Bangladesh dataset = main training set** for the lab branch, after cleaning.
- **Mendeley India dataset = pipeline testing only.** Its red flags make it unreliable for training.
- Both targets mean "heart disease present", not CAD specifically.

## Datasets
| | Mendeley India CVD | Northern Bangladesh |
|---|---|---|
| Rows × columns | 1,000 × 14 | 1,048 × 27 (1,035 adults after cleaning) |
| Source | Unnamed Indian hospital | Zia Heart Foundation Hospital, Northern Bangladesh (CCU + General units) |
| Target | `target` (580 yes / 420 no) | `Heart Disease` (598 yes / 450 no) |
| Missing values | None recorded; 53 cholesterol = 0 (missing coded as 0) | 2–10% in most columns |
| Schema | Same 13 features as UCI Cleveland | Lab-report values: lipids, glucose, kidney, electrolytes, troponin |

## India: red flags
- `slope` alone gives AUROC 0.956 (slope 0 → 0% disease, slope 3 → 100%). Slope 0 (180 rows) isn't in the dataset description (documented 1–3).
- Age has no relationship with the outcome (mean 49.1 vs 49.4) and is near-uniform 20–80. Sex and exercise angina show no relationship either.
- All 53 rows with cholesterol = 0 are positive (missingness leaks the label). Max 602 vs documented 564.
- Gradient boosting CV AUROC 0.998 (0.994 without slope): far above realistic risk models (~0.70–0.80).

## Bangladesh: cleaning rules (implemented in `tabular_loader.py`)
| Issue | Action |
|---|---|
| `UNIT`: every CCU patient is positive (AUROC 0.81 alone) | Excluded |
| Troponin-I: two incompatible assays, censored values (">25000", "<2.50"), acute marker | Excluded |
| `MaxHR`: correlates −0.93 with age; 59% of rows equal 208 − 0.7 × age | Excluded (estimated, not measured) |
| 13 patients under 18 (ages 4–97) | Removed |
| Typos: "\`2.6", "4,3", "106,4", "106. 8" | Parsed to numbers |
| BMI disagrees with height/weight in ~9% of rows | Recomputed where height and weight exist |
| Values outside plausible ranges | Set to missing (ranges in `configs/tabular.yaml`) |
| Missing creatinine occurs only in disease-positive patients (0% vs 6.7% missing); XGBoost learned "blank creatinine → disease" (found 24 Oct 2026) | All models impute (median) before fitting, so missingness can't be used as a signal |
| Training patients almost always have full panels (97% have ≤ 4 of 19 values missing); mostly-blank inputs scored ~0.84–0.999 regardless of values | `predict_tabular()` refuses patients with more than 4 blank fields (`final.max_missing_features`) |

Even after cleaning, CV AUROC ~0.99; lipids alone 0.95; LDL alone 0.89 within the General unit (median 94 vs 150 mg/dL). Separation is much stronger than in real populations, likely selection (very sick vs relatively healthy hospital patients). **Scores on this data will overstate real-world performance.**

## Shared features (India ↔ Bangladesh)
Comparable: age, sex, blood pressure, total cholesterol, chest pain (India types 0–2 = yes, 3 asymptomatic = no).
Not comparable: fasting vs random blood sugar; exercise-test vs age-formula max heart rate; ECG/slope/vessels (India only); lipid subfractions, kidney, electrolytes (Bangladesh only).

## Follow-up
- Model results on the cleaned Bangladesh data: `reports/baseline_results_tabular.md`; decision threshold: `reports/threshold_tabular.md`.
- Credible evaluation needs CARRS / South Asia Biobank.
