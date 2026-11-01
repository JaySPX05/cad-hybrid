
"""Independent prediction interface for the ECG and tabular branches.

Usage:
    result = predict(
        lab_values={"age": 54, ...},
        ecg_image_path="ecg.jpg"
    )

The tabular branch predicts general heart disease.
The ECG branch predicts CAD.
The two probabilities are returned separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .tabular.artifact import DEFAULT_PATH as TABULAR_MODEL_PATH
from .tabular.artifact import (
    check_completeness,
    load_artifact,
    predict_calibrated,
    to_frame,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@dataclass
class Prediction:
    """Independent predictions from available model branches."""

    tabular_probability: float | None = None
    cad_probability: float | None = None
    branch_probabilities: dict[str, float | None] = field(
        default_factory=dict
    )


def load_config(name: str) -> dict:
    """Load a YAML configuration file."""

    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def predict_tabular(
    lab_values: dict,
    model_path: Path | str = TABULAR_MODEL_PATH
) -> float:
    """Return calibrated general heart disease probability."""

    artifact = load_artifact(str(model_path))
    X = to_frame(lab_values, artifact["features"])
    check_completeness(artifact, X)

    return float(predict_calibrated(artifact, X)[0])


def tabular_threshold(
    model_path: Path | str = TABULAR_MODEL_PATH
) -> float | None:
    """Return the decision threshold stored with the tabular model."""

    return load_artifact(str(model_path)).get("threshold")


def explain_tabular(
    lab_values: dict,
    model_path: Path | str = TABULAR_MODEL_PATH,
    nsamples: int | None = None
) -> dict:
    """Explain the tabular prediction using SHAP."""

    from .explain.shap_tabular import explain_patient

    artifact = load_artifact(str(model_path))
    X = to_frame(lab_values, artifact["features"])
    check_completeness(artifact, X)

    return explain_patient(artifact, X, nsamples)


def predict_ecg(image_path: str) -> float:
    """Return calibrated CAD probability from an ECG image."""

    from .ecg.predict import predict_ecg as predict_ecg_details

    result = predict_ecg_details(image_path)

    return float(result["cad_probability"])


def predict(
    lab_values: dict | None = None,
    ecg_image_path: str | None = None,
    tabular_model_path: Path | str = TABULAR_MODEL_PATH,
) -> Prediction:
    """Return independent tabular and ECG predictions.

    No combined probability is calculated because the tabular model
    predicts general heart disease, whereas the ECG model predicts CAD.
    """

    if lab_values is None and ecg_image_path is None:
        raise ValueError(
            "Provide lab values, an ECG image, or both"
        )

    tabular_probability = (
        predict_tabular(lab_values, tabular_model_path)
        if lab_values is not None
        else None
    )

    cad_probability = (
        predict_ecg(ecg_image_path)
        if ecg_image_path is not None
        else None
    )

    branches = {
        "tabular": tabular_probability,
        "ecg": cad_probability,
    }

    return Prediction(
        tabular_probability=tabular_probability,
        cad_probability=cad_probability,
        branch_probabilities=branches,
    )
