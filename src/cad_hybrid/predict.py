"""One prediction interface used by the evaluation scripts, the Streamlit app and any
future API, so everything scores the same way.

Usage:
    result = predict(lab_values={"age": 54, ...}, ecg_image_path="ecg.jpg")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .fusion.fuse import fuse, risk_category

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@dataclass
class Prediction:
    probability: float
    category: str
    branch_probabilities: dict[str, float | None] = field(default_factory=dict)


def load_config(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def predict_tabular(lab_values: dict) -> float:
    """Calibrated CAD probability from lab values.

    TODO: load the trained lab model + calibrator from ``models/`` and score.
    """
    raise NotImplementedError("Lab model not trained yet")


def predict_ecg(image_path: str) -> float:
    """Calibrated CAD probability from an ECG image.

    TODO: load the trained ECG model + calibrator from ``models/`` and score.
    """
    raise NotImplementedError("ECG model not trained yet")


def predict(lab_values: dict | None = None, ecg_image_path: str | None = None) -> Prediction:
    """Score whatever inputs are available and fuse them."""
    if lab_values is None and ecg_image_path is None:
        raise ValueError("Provide lab values, an ECG image, or both")
    cfg = load_config("fusion")
    branches = {
        "tabular": predict_tabular(lab_values) if lab_values is not None else None,
        "ecg": predict_ecg(ecg_image_path) if ecg_image_path is not None else None,
    }
    prob = fuse(branches, prior=cfg["prior_probability"], weights=cfg["branch_weights"])
    return Prediction(prob, risk_category(prob, cfg["risk_categories"]), branches)
