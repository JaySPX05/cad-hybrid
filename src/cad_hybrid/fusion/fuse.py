"""Late fusion of calibrated branch probabilities.

Each branch is calibrated on its own training data, so its probability reflects how
common the disease was there (its training positive rate). To combine branches:

1. turn each branch's probability into a likelihood ratio against ITS OWN training
   rate: LR = odds(p_branch) / odds(branch_prior);
2. start from the target prior (how common the disease is among the people the tool
   is used on), and add each branch's weighted log likelihood ratio;
3. convert back to a probability.

    logit(p_fused) = logit(target_prior) + sum_i w_i * (logit(p_i) - logit(prior_i))

With one branch whose training rate equals the target prior, the fused probability is
just that branch's probability. With weight 1 for every branch this is the naive-Bayes
assumption: the branches give independent evidence given the disease status. The
weights are hand-set (configs/fusion.yaml) until there are patients with both lab
values and an ECG to learn or check them. A missing branch (None) is skipped, so the
model works with labs only, ECG only, or both.
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
    branch_priors: dict[str, float | None],
    target_prior: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Combine calibrated branch probabilities into one probability.

    Args:
        branch_probs: e.g. ``{"tabular": 0.4, "ecg": None}``; None = branch not used.
        branch_priors: each branch's training positive rate (what it was calibrated on).
        target_prior: base rate in the population the result is meant for.
        weights: per-branch weights (default 1.0). 1.0 = full trust, 0 = ignore.

    Returns:
        Fused probability. Equals ``target_prior`` if no branch is available.
    """
    weights = weights or {}
    log_odds = _logit(target_prior)
    for name, p in branch_probs.items():
        if p is None:
            continue
        prior = branch_priors.get(name)
        if prior is None:
            raise ValueError(f"No training positive rate for the '{name}' branch; "
                             f"set fusion branch_priors.{name} in configs/fusion.yaml")
        w = weights.get(name, 1.0)
        log_odds += w * (_logit(p) - _logit(prior))  # weighted log likelihood ratio
    return _sigmoid(log_odds)


def risk_category(prob: float, categories: dict[str, list[float]]) -> str:
    """Map a probability to a category using ``[low, high)`` bounds from the config."""
    for name, (lo, hi) in categories.items():
        if lo <= prob < hi or (hi >= 1.0 and prob >= lo):
            return name
    raise ValueError(f"No category covers probability {prob}")
