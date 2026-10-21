"""Cross-country external tests.

Train on one country's data and test on another's to check the model learned heart
patterns rather than one hospital's quirks:
- ECG branch: Pakistan (Multan) -> India (Bhopal)
- Lab branch: India (Mendeley) -> Bangladesh
A large drop from internal to external performance is a warning sign.
"""

from __future__ import annotations

from .metrics import compute_metrics


def external_test(model, X_train, y_train, X_ext, y_ext, threshold: float = 0.5) -> dict:
    """Fit on the training country, report metrics on both the training data
    (optimistic reference) and the external country."""
    model.fit(X_train, y_train)
    internal = compute_metrics(y_train, model.predict_proba(X_train)[:, 1], threshold)
    external = compute_metrics(y_ext, model.predict_proba(X_ext)[:, 1], threshold)
    return {"internal_train": internal, "external": external,
            "auroc_drop": internal["auroc"] - external["auroc"]}
