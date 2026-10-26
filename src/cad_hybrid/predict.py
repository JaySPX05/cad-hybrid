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
from .tabular.artifact import DEFAULT_PATH as TABULAR_MODEL_PATH
from .tabular.artifact import check_completeness, load_artifact, predict_calibrated, to_frame

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@dataclass
class Prediction:
    probability: float
    category: str
    branch_probabilities: dict[str, float | None] = field(default_factory=dict)


def load_config(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def predict_tabular(lab_values: dict, model_path: Path | str = TABULAR_MODEL_PATH) -> float:
    """Calibrated probability of heart disease from one patient's lab values.

    ``lab_values`` uses the shared-schema names from ``configs/tabular.yaml``
    (e.g. ``{"age": 54, "sex_male": 1, "ldl": 150}``). Fields may be left out, up to
    ``final.max_missing_features`` (out-of-range values count as missing); more raises
    ValueError, as do unknown field names and values that aren't numbers.

    The model is trained by ``scripts/train_tabular.py`` and saved to
    ``models/tabular/lab_model.joblib``; FileNotFoundError if it hasn't been trained.
    """
    artifact = load_artifact(str(model_path))
    X = to_frame(lab_values, artifact["features"])
    check_completeness(artifact, X)  # ValueError if too many fields are blank
    return float(predict_calibrated(artifact, X)[0])


def tabular_threshold(model_path: Path | str = TABULAR_MODEL_PATH) -> float | None:
    """Decision threshold stored with the lab model: probabilities at or above it are
    flagged "heart disease likely". None if the model was saved without one.

    It applies to ``predict_tabular()``'s probability only, not to the fused probability.
    """
    return load_artifact(str(model_path)).get("threshold")


def explain_tabular(lab_values: dict, model_path: Path | str = TABULAR_MODEL_PATH,
                    nsamples: int | None = None) -> dict:
    """Which lab values pushed this patient's probability up or down (SHAP).

    Returns ``{"baseline", "probability", "contributions"}``; contributions is a
    DataFrame (feature, value, contribution, given), largest effect first, and
    ``baseline + sum(contributions) == probability``. Slow with TabICL on CPU.
    """
    from .explain.shap_tabular import explain_patient

    artifact = load_artifact(str(model_path))
    X = to_frame(lab_values, artifact["features"])
    check_completeness(artifact, X)
    return explain_patient(artifact, X, nsamples)


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
