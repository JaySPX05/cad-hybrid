"""SHAP explanations for the lab branch: which values pushed risk up or down."""

from __future__ import annotations


def explain(model, X_background, X_explain):
    """Return SHAP values for ``X_explain`` using ``X_background`` as reference data.

    Uses TreeExplainer for gradient-boosting models (fast, exact) and falls back to
    the model-agnostic explainer otherwise.
    """
    import shap

    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.Explainer(model.predict_proba, X_background)
    return explainer(X_explain)
