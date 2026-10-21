"""Late fusion of calibrated branch probabilities.

Each branch's calibrated probability is turned into a likelihood ratio against the
prior (base rate), the ratios are combined on the log-odds scale with per-branch
weights, and the result is converted back to a probability.

Assumes the branches give roughly independent evidence. The weights are hand-set
(configs/fusion.yaml) until we have patients with both lab values and an ECG to
learn them from. A missing branch (None) is simply skipped, so the model works with
labs only, ECG only, or both.
"""

from __future__ import annotations

import math

EPS = 1e-6


def _logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def fuse(
    branch_probs: dict[str, float | None],
    prior: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Combine calibrated branch probabilities into one probability.

    Args:
        branch_probs: e.g. ``{"tabular": 0.4, "ecg": None}``; None = branch not used.
        prior: base rate of the positive class the branches were calibrated against.
        weights: per-branch weights (default 1.0). 1.0 = full trust, 0 = ignore.

    Returns:
        Fused probability. Equals ``prior`` if no branch is available.
    """
    weights = weights or {}
    log_odds = _logit(prior)
    for name, p in branch_probs.items():
        if p is None:
            continue
        w = weights.get(name, 1.0)
        log_odds += w * (_logit(p) - _logit(prior))  # weighted log likelihood ratio
    return _sigmoid(log_odds)


def risk_category(prob: float, categories: dict[str, list[float]]) -> str:
    """Map a probability to a category using ``[low, high)`` bounds from the config."""
    for name, (lo, hi) in categories.items():
        if lo <= prob < hi or (hi >= 1.0 and prob >= lo):
            return name
    raise ValueError(f"No category covers probability {prob}")
