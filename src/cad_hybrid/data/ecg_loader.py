"""Index the ECG image datasets and map their class names to our labels.

The label mapping lives in ``configs/ecg.yaml`` (e.g. 'History of MI' -> cad_positive,
'Abnormal Heartbeat' -> arrhythmia).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def index_images(dataset_id: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """List every image in ``raw/<dataset_id>/`` with its original class.

    Assumes one subfolder per class (the usual layout for these datasets); adjust
    after the data audit if a dataset uses a CSV of labels instead.

    Returns a DataFrame with columns: ``path``, ``original_class``, ``dataset_id``.
    """
    root = raw_dir / dataset_id
    rows = [
        {"path": str(p), "original_class": p.parent.name, "dataset_id": dataset_id}
        for p in sorted(root.rglob("*"))
        if p.suffix.lower() in IMAGE_SUFFIXES
    ]
    return pd.DataFrame(rows, columns=["path", "original_class", "dataset_id"])


def apply_label_map(index: pd.DataFrame, label_map: dict[str, str]) -> pd.DataFrame:
    """Add a ``label`` column using the dataset's label map.

    Rows whose original class isn't in the map are dropped and reported, so nothing
    is silently mislabelled.
    """
    out = index.copy()
    out["label"] = out["original_class"].map(label_map)
    unmapped = out[out["label"].isna()]["original_class"].unique()
    if len(unmapped):
        print(f"Dropping unmapped classes: {sorted(unmapped)}")
    return out.dropna(subset=["label"]).reset_index(drop=True)
