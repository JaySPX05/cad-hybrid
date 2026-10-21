"""The ECG-branch hybrid: CNN features -> gradient boosting / SVM.

After fine-tuning, the CNN's penultimate-layer features (embeddings) are extracted and
a traditional classifier is trained on them. Compare against the plain CNN and keep
whichever does better on the validation set.
"""

from __future__ import annotations

import numpy as np


def extract_embeddings(model, loader) -> tuple[np.ndarray, np.ndarray]:
    """Return (embeddings, labels) for every image in ``loader``.

    Uses timm's ``forward_features`` + pooling, i.e. everything except the head.
    """
    import torch

    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            z = model.forward_head(model.forward_features(x), pre_logits=True)
            feats.append(z.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def make_head(kind: str = "xgboost", **params):
    """Traditional classifier trained on CNN embeddings."""
    if kind == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(eval_metric="logloss", **params)
    if kind == "svm":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        return make_pipeline(StandardScaler(), SVC(probability=True, **params))
    raise ValueError(f"Unknown head: {kind}")
