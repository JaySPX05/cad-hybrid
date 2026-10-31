# cad-hybrid

A hybrid machine-learning + deep-learning model for coronary artery disease (CAD) / cardiovascular disease (CVD) risk, built only on South Asian data (India, Pakistan, Bangladesh).

> Research prototype. Not for clinical use.

## Approach

The model has two branches that are trained separately and combined at the end:

- **Lab branch (tabular):** XGBoost and a pretrained deep tabular model (TabICL), stacked with a logistic-regression meta-model, then calibrated (Platt) with a decision threshold chosen on held-out predictions. Explained with SHAP (per patient in the app, overall in `reports/shap_tabular.md`).
- **ECG branch (images):** an ImageNet-pretrained CNN (PyTorch / timm) fine-tuned on paper ECG images, with its learned features fed into an XGBoost / SVM classifier. Explained with Grad-CAM.
- **Fusion:** each branch is calibrated on its own data; its probability is turned into a likelihood ratio against its own training positive rate, and the ratios are combined from a target prior (`configs/fusion.yaml`). The model works with lab values only, an ECG only, or both.
- **Target:** CAD. The ECG branch predicts CAD (MI or history of MI vs normal); the lab branch predicts heart disease in general and is used as a stand-in, because the lab data has no CAD label.

The two branches use different patients (no public South Asian dataset has both labs and ECGs for the same person), so they can't be trained as one network.

## Data rule

Only Indian, Pakistani or Bangladeshi datasets are used for training and evaluation. Raw data, processed data and trained model files are never committed to this repository. See [`data/README.md`](data/README.md) for where each dataset comes from and [`ATTRIBUTIONS.md`](ATTRIBUTIONS.md) for CC BY 4.0 credits.

## Repository layout

```
configs/            model and data settings (features, label maps, fusion weights)
data/               NOT committed except README + data_register.csv
  raw/              original downloads, one folder per dataset
  interim/          cleaned, de-duplicated
  processed/        derived files, e.g. held-out (CV) predictions
src/cad_hybrid/
  data/             loaders, near-duplicate check, patient-level splits   (shared)
  tabular/          lab branch: baselines, deep tabular, stacking         (Person A)
  ecg/              ECG branch: preprocessing, CNN, embeddings -> GBM     (Person B)
  fusion/           calibration and late fusion                           (shared)
  evaluation/       metrics with bootstrap CIs, cross-country tests       (shared)
  explain/          SHAP (labs), Grad-CAM (ECG)
  predict.py        single prediction interface used everywhere
scripts/            train_tabular.py, choose_threshold.py, explain_tabular.py, train_ecg.py, evaluate.py
notebooks/          data audits and exploration only
app/                internal Streamlit test app
models/             trained weights, NOT committed
reports/            data audit, results, threshold report, model card
tests/              unit tests
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                 # makes `import cad_hybrid` work everywhere
pytest                           # run the tests
```

Then download the datasets into `data/raw/` as described in `data/README.md`.

## Common commands

```bash
python scripts/train_tabular.py --config configs/tabular.yaml
python scripts/choose_threshold.py          # re-choose the lab threshold, no retraining
python scripts/explain_tabular.py           # SHAP report for the saved lab model
python scripts/train_ecg.py --config configs/ecg.yaml
python scripts/evaluate.py
streamlit run app/streamlit_app.py
```

## Workflow

- `main` stays working; don't commit to it directly.
- Create a branch per task, e.g. `tabular/xgboost-baseline` or `ecg/two-stage-finetune`.
- Open a pull request; the other person reviews before merging.
- Keep exploration in `notebooks/`; move code you keep into `src/cad_hybrid/` with a test.
- Never commit data, model weights or secrets (the `.gitignore` blocks the common cases).

## Status

| Part | State |
|---|---|
| Project structure, fusion, metrics, splits, near-duplicate check | Done (with tests) |
| Lab data audit | Done: `reports/data_audit_tabular.md` |
| Lab branch | Trained: baselines, TabICL, stack, calibration, decision threshold, SHAP explanations (`reports/`) |
| ECG branch | In progress (teammate) |
| Fusion | Per-branch training rates and target prior in place; weights (1.0) and risk categories are unvalidated placeholders until paired lab + ECG data exists |
| Streamlit test app | Works with lab values, incl. per-patient SHAP; ECG pending |
