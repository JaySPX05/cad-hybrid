# cad-hybrid

A hybrid machine-learning + deep-learning model for coronary artery disease (CAD) / cardiovascular disease (CVD) risk, built only on South Asian data (India, Pakistan, Bangladesh).

## Approach

The model has two branches that are trained separately and combined at the end:

- **Lab branch (tabular):** gradient boosting (XGBoost / LightGBM) and a deep tabular model, stacked together. Explained with SHAP.
- **ECG branch (images):** an ImageNet-pretrained CNN fine-tuned on paper ECG images, with its learned features fed into a gradient-boosting / SVM classifier. Explained with Grad-CAM.
- **Fusion:** each branch is calibrated, then the two are combined. The model works with lab values only, an ECG only, or both.

The two branches use different patients (no public South Asian dataset has both labs and ECGs for the same person), so they can't be trained as one network.

## Data rule

Only Indian, Pakistani or Bangladeshi datasets are used for training and evaluation. Raw data, processed data and trained model files are never committed to this repository.

## Status

Project setup. See open pull requests for the initial structure.

> Research prototype. Not for clinical use.
