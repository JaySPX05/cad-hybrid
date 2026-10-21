"""Train/test splitting that never lets one patient (or near-duplicate image group)
appear on both sides."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def holdout_split(
    y: np.ndarray,
    groups: np.ndarray | None = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train and test indices.

    With ``groups`` (patient ids or duplicate-group ids), whole groups go to one side
    and class balance is kept as close as possible. Without groups, a plain stratified
    split is used.
    """
    y = np.asarray(y)
    idx = np.arange(len(y))
    if groups is None:
        train, test = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
        return np.sort(train), np.sort(test)

    n_splits = max(2, round(1 / test_size))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train, test = next(cv.split(idx, y, groups))
    return train, test
