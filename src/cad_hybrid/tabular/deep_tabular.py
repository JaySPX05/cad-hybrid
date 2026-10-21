"""Deep tabular model for the lab branch.

Primary option: TabICL (pretrained tabular foundation model, BSD-3-Clause).
Fallback: a small MLP if TabICL underperforms or causes dependency issues.
"""

from __future__ import annotations


def tabicl_model(finetune: bool = False, **params):
    """Return a TabICL classifier (in-context learning; optional fine-tuning)."""
    if finetune:
        from tabicl import FinetunedTabICLClassifier

        return FinetunedTabICLClassifier(**params)
    from tabicl import TabICLClassifier

    return TabICLClassifier(**params)


def mlp_model(**params):
    """Small multilayer perceptron fallback (scikit-learn)."""
    from sklearn.neural_network import MLPClassifier

    defaults = {"hidden_layer_sizes": (64, 32), "early_stopping": True, "max_iter": 500}
    return MLPClassifier(**{**defaults, **params})
