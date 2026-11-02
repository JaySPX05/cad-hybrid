
"""Prediction interface for the ECG and tabular branches, with CAD fusion."""

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
from .fusion.fuse import fuse, risk_category

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@dataclass
class Prediction:
    """Independent branch predictions and combined CAD prediction."""

    tabular_probability: float | None = None
    cad_probability: float | None = None
    arrhythmia_probability: float | None = None
    fused_probability: float | None = None
    category: str | None = None
    branch_probabilities: dict[str, float | None] = field(
        default_factory=dict
    )


def load_config(name: str) -> dict:
    """Load a YAML configuration file."""

    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def predict_tabular(
    lab_values: dict,
    model_path: Path | str = TABULAR_MODEL_PATH,
) -> float:
    """Return calibrated general heart disease probability."""

    artifact = load_artifact(str(model_path))
    X = to_frame(lab_values, artifact["features"])
    check_completeness(artifact, X)

    return float(predict_calibrated(artifact, X)[0])


def tabular_threshold(
    model_path: Path | str = TABULAR_MODEL_PATH,
) -> float | None:
    """Return the decision threshold stored with the tabular model."""

    return load_artifact(str(model_path)).get("threshold")


def explain_tabular(
    lab_values: dict,
    model_path: Path | str = TABULAR_MODEL_PATH,
    nsamples: int | None = None,
) -> dict:
    """Explain the tabular prediction using SHAP."""

    from .explain.shap_tabular import explain_patient

    artifact = load_artifact(str(model_path))
    X = to_frame(lab_values, artifact["features"])
    check_completeness(artifact, X)

    return explain_patient(artifact, X, nsamples)


def predict_ecg(image_path: str) -> dict[str, float]:
    """Return calibrated CAD and arrhythmia probabilities from an ECG."""

    from .ecg.predict import predict_ecg as predict_ecg_details

    result = predict_ecg_details(image_path)

    return {
        "cad_probability": float(result["cad_probability"]),
        "arrhythmia_probability": float(
            result["arrhythmia_probability"]
        ),
    }


def fusion_priors(
    cfg: dict,
    tabular_model_path: Path | str = TABULAR_MODEL_PATH,
) -> tuple[dict, float]:
    """Get branch training rates and target prior from configuration/model."""

    priors = dict(cfg.get("branch_priors") or {})

    try:
        lab_rate = load_artifact(
            str(tabular_model_path)
        )["metadata"].get("positive_rate")
    except FileNotFoundError:
        lab_rate = None

    if priors.get("tabular") is None:
        priors["tabular"] = lab_rate

    target = cfg.get("target_prior")

    if target is None:
        target = lab_rate

    if target is None:
        raise ValueError(
            "target_prior is null and no tabular model is available. "
            "Set target_prior in configs/fusion.yaml."
        )

    return priors, float(target)


def predict(
    lab_values: dict | None = None,
    ecg_image_path: str | None = None,
    tabular_model_path: Path | str = TABULAR_MODEL_PATH,
    fusion_config: dict | None = None,
) -> Prediction:
    """Return independent branch probabilities and fused CAD probability."""

    if lab_values is None and ecg_image_path is None:
        raise ValueError(
            "Provide lab values, an ECG image, or both"
        )

    cfg = fusion_config or load_config("fusion")

    tabular_probability = (
        predict_tabular(lab_values, tabular_model_path)
        if lab_values is not None
        else None
    )

    ecg_result = (
        predict_ecg(ecg_image_path)
        if ecg_image_path is not None
        else None
    )

    if isinstance(ecg_result, dict):
        cad_probability = float(ecg_result["cad_probability"])
        arrhythmia_probability = float(
            ecg_result["arrhythmia_probability"]
        )
    elif ecg_result is not None:
        # Backward compatibility for callers/tests returning a float.
        cad_probability = float(ecg_result)
        arrhythmia_probability = None
    else:
        cad_probability = None
        arrhythmia_probability = None

    # Arrhythmia is deliberately excluded from CAD fusion.
    branches = {
        "tabular": tabular_probability,
        "ecg": cad_probability,
    }

    priors, target = fusion_priors(
        cfg,
        tabular_model_path,
    )

    fused_probability = fuse(
        branches,
        priors,
        target,
        weights=cfg.get("branch_weights"),
    )

    category = risk_category(
        fused_probability,
        cfg["risk_categories"],
    )

    all_branches = {
        **branches,
        "arrhythmia": arrhythmia_probability,
    }

    return Prediction(
        tabular_probability=tabular_probability,
        cad_probability=cad_probability,
        arrhythmia_probability=arrhythmia_probability,
        fused_probability=fused_probability,
        category=category,
        branch_probabilities=all_branches,
    )
