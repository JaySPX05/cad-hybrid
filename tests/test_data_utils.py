import numpy as np

from cad_hybrid.data.dedupe import group_near_duplicates
from cad_hybrid.data.ecg_loader import apply_label_map
from cad_hybrid.data.splits import holdout_split

import pandas as pd


def test_near_duplicates_share_a_group():
    hashes = [0b0000, 0b0001, 0b1111_0000_1111]  # first two differ by 1 bit
    groups = group_near_duplicates(hashes, max_distance=2)
    assert groups[0] == groups[1]
    assert groups[2] != groups[0]


def test_groups_never_split_across_train_and_test():
    y = np.array([0, 1] * 20)
    groups = np.repeat(np.arange(10), 4)
    train, test = holdout_split(y, groups=groups, test_size=0.2)
    assert set(groups[train]).isdisjoint(groups[test])


def test_unmapped_classes_are_dropped():
    idx = pd.DataFrame({"path": ["a", "b"], "original_class": ["Normal", "Unknown"],
                        "dataset_id": ["x", "x"]})
    out = apply_label_map(idx, {"Normal": "negative"})
    assert list(out["label"]) == ["negative"]
