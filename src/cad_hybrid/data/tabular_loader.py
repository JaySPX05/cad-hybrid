"""Load and harmonise the lab (tabular) datasets.

Each source dataset has its own column names and units. This module maps them onto
one shared schema defined in ``configs/tabular.yaml`` so models can be trained on one
dataset and tested on another (e.g. India -> Bangladesh).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"


def load_raw(dataset_id: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read a raw tabular dataset as-is.

    TODO (week 1 data audit): implement once the real file names and formats are known.
    """
    raise NotImplementedError(f"Loader for '{dataset_id}' not written yet; see data/README.md")


def harmonise(df: pd.DataFrame, dataset_id: str, config: dict) -> pd.DataFrame:
    """Rename columns, convert units and encode categories into the shared schema.

    TODO: per-dataset column mapping (e.g. 'chol' -> 'total_cholesterol'), unit
    conversion (mmol/L <-> mg/dL), and flagging values outside ``plausible_ranges``.
    """
    raise NotImplementedError


def load_dataset(dataset_id: str, config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Return features ``X`` and target ``y`` for one dataset in the shared schema."""
    df = harmonise(load_raw(dataset_id), dataset_id, config)
    target = config["data"]["target"]
    return df.drop(columns=[target]), df[target]
