"""The lab-branch hybrid: stack gradient boosting and the deep tabular model.

Out-of-fold probabilities from each base model are combined by a logistic-regression
meta-model, so the stack learns how much to trust each one.
"""

from __future__ import annotations

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression


def make_stack(base_models: dict, cv: int = 5) -> StackingClassifier:
    """Build a stacking classifier from named base models.

    ``base_models`` maps names to fitted-or-unfitted sklearn-compatible classifiers,
    e.g. ``{"xgb": xgboost_model(...), "tabicl": tabicl_model()}``.
    """
    return StackingClassifier(
        estimators=list(base_models.items()),
        final_estimator=LogisticRegression(max_iter=1000),
        stack_method="predict_proba",
        cv=cv,
    )
