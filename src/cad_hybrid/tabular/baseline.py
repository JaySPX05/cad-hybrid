"""Baseline lab models: logistic regression and gradient boosting.

Every hybrid model has to beat these on the same splits.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Impute + scale numeric columns; impute + one-hot encode categorical columns."""
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )


def logistic_regression(numeric: list[str], categorical: list[str], **params) -> Pipeline:
    """Logistic regression baseline with preprocessing."""
    return Pipeline([
        ("prep", make_preprocessor(numeric, categorical)),
        ("model", LogisticRegression(**params)),
    ])


def xgboost_model(**params):
    """XGBoost classifier (no scaling needed).

    XGBoost can handle missing values itself, but the lab branch imputes first
    (see build_models in scripts/train_tabular.py) so missingness can't act as a signal.
    """
    from xgboost import XGBClassifier

    return XGBClassifier(eval_metric="logloss", **params)


def lightgbm_model(**params):
    """LightGBM classifier."""
    from lightgbm import LGBMClassifier

    return LGBMClassifier(verbose=-1, **params)
