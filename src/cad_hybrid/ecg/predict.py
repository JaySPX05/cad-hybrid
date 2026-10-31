
from pathlib import Path

import joblib
import numpy as np
import torch

from src.predict_ecg import load_models, preprocess_image
from src.model_config import DEVICE

ROOT = Path(__file__).resolve().parents[3]

CALIBRATOR_PATH = ROOT / "models" / "cad_platt_calibrator.pkl"

_feature_model = None
_classifier = None
_calibrator = None


def _load():
    global _feature_model, _classifier, _calibrator

    if _feature_model is None:
        _feature_model, _classifier = load_models()
        _feature_model.eval()

    if _calibrator is None:
        _calibrator = joblib.load(CALIBRATOR_PATH)

    return _feature_model, _classifier, _calibrator


def predict_ecg(image_path):
    feature_model, classifier, calibrator = _load()

    image = preprocess_image(image_path)

    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(image)

    if image.ndim == 3:
        image = image.unsqueeze(0)

    image = image.to(DEVICE)

    with torch.no_grad():
        features = feature_model(image)

    features = features.detach().cpu().numpy().reshape(1, -1)

    probabilities = classifier.predict_proba(features)[0]
    classes = list(classifier.classes_)

    p_cad = float(probabilities[classes.index(0)])
    p_normal = float(probabilities[classes.index(1)])
    p_arrhythmia = float(probabilities[classes.index(2)])

    binary_score = p_cad / (p_cad + p_normal)
    binary_score = np.clip(binary_score, 1e-7, 1 - 1e-7)

    score_logit = np.log(binary_score / (1 - binary_score))
    calibrated_cad = float(
        calibrator.predict_proba([[score_logit]])[0, 1]
    )

    return {
        "cad_probability": calibrated_cad,
        "arrhythmia_probability": p_arrhythmia
    }
