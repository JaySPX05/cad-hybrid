"""Save, load and use the final lab-branch model.

The "artifact" is one file bundling everything needed to score a new patient:
the trained model, its calibrator, the feature list and the plausible-range rules,
plus metadata about how it was trained. A JSON copy of the metadata is written next
to it so it can be read without loading the model.

Artifacts are written to ``models/`` (gitignored) and are tied to the library versions
they were trained with; the versions are recorded in the metadata.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..data.tabular_loader import apply_plausible_ranges

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATH = PROJECT_ROOT / "models" / "tabular" / "lab_model.joblib"


def build_artifact(model, calibrator, features: list[str], plausible_ranges: dict,
                   metadata: dict, max_missing: int | None = None) -> dict:
    """Bundle a trained model and calibrator with what's needed to use them.

    ``max_missing``: the most blank fields a patient may have (None = no limit).
    """
    return {
        "model": model,
        "calibrator": calibrator,
        "features": list(features),
        "plausible_ranges": dict(plausible_ranges),
        "max_missing": max_missing,
        "metadata": dict(metadata),
    }


def save_artifact(artifact: dict, path: Path | str = DEFAULT_PATH) -> Path:
    """Write the artifact (.joblib) and a readable copy of its metadata (.json)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    path.with_suffix(".json").write_text(
        json.dumps(artifact["metadata"], indent=2, default=str), encoding="utf-8")
    load_artifact.cache_clear()  # make sure the next load sees the new file
    return path


@lru_cache(maxsize=4)
def load_artifact(path: str) -> dict:
    """Load an artifact once and keep it in memory (path as str so it can be cached)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"No trained lab model at {p}. Train one with: python scripts/train_tabular.py")
    return joblib.load(p)


def to_frame(lab_values: dict, features: list[str]) -> pd.DataFrame:
    """Turn one patient's lab values into a one-row DataFrame in the model's feature order.

    Missing fields become NaN (the models handle missing values). Unknown field names
    and values that aren't numbers raise ValueError, so typos don't pass silently.
    """
    unknown = set(lab_values) - set(features)
    if unknown:
        raise ValueError(f"Unknown lab fields: {sorted(unknown)}. Expected any of: {features}")
    row = {f: lab_values.get(f) for f in features}
    frame = pd.DataFrame([row], columns=features)
    try:
        return frame.apply(lambda col: pd.to_numeric(col, errors="raise")).astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Lab values must be numbers: {exc}") from exc


def check_completeness(artifact: dict, X: pd.DataFrame) -> None:
    """Refuse rows with more blank fields than the model was trained to handle.

    Out-of-range values count as blank, because they are treated as missing.
    """
    limit = artifact.get("max_missing")
    if limit is None:
        return
    X = apply_plausible_ranges(X[artifact["features"]], artifact["plausible_ranges"])
    for _, row in X.iterrows():
        blank = [f for f in artifact["features"] if pd.isna(row[f])]
        if len(blank) > limit:
            raise ValueError(
                f"Too many missing lab values ({len(blank)} of {len(artifact['features'])}; "
                f"at most {limit} allowed). Missing or out of range: {blank}")


def predict_calibrated(artifact: dict, X: pd.DataFrame) -> np.ndarray:
    """Calibrated probability of the positive class for each row of ``X``.

    Values outside the plausible ranges used in training are treated as missing,
    exactly as during training.
    """
    X = apply_plausible_ranges(X[artifact["features"]], artifact["plausible_ranges"])
    raw = artifact["model"].predict_proba(X)[:, 1]
    return np.clip(artifact["calibrator"].transform(raw), 0.0, 1.0)
