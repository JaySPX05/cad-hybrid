
from pathlib import Path

import joblib
import numpy as np
import torch

from .feature_extractor import create_feature_extractor
from .preprocess import preprocess_image


ROOT = Path(__file__).resolve().parents[3]
DEVICE = torch.device("cpu")

SVM_PATH = ROOT / "models" / "ecg_svm_classifier.pkl"
BUNDLE_PATH = ROOT / "models" / "ecg_calibration_bundle.pkl"

_feature_model = None
_svm_model = None
_calibration_bundle = None


def _load():
    """Load the ECG models once and reuse them."""

    global _feature_model, _svm_model, _calibration_bundle

    if _feature_model is None:
        _feature_model = create_feature_extractor()
        _feature_model.eval()

    if _svm_model is None:
        if not SVM_PATH.exists():
            raise FileNotFoundError(
                f"ECG SVM model not found: {SVM_PATH}"
            )
        _svm_model = joblib.load(SVM_PATH)

    if _calibration_bundle is None:
        if not BUNDLE_PATH.exists():
            raise FileNotFoundError(
                f"ECG calibration bundle not found: {BUNDLE_PATH}"
            )
        _calibration_bundle = joblib.load(BUNDLE_PATH)

    return _feature_model, _svm_model, _calibration_bundle


def predict_ecg(image_path):
    """
    Return calibrated CAD probability and separate
    multiclass-calibrated arrhythmia probability.
    """

    feature_model, svm_model, bundle = _load()

    multiclass_calibrator = bundle["multiclass_calibrator"]
    binary_calibrator = bundle["binary_calibrator"]

    image = preprocess_image(image_path).to(DEVICE)

    with torch.no_grad():
        features = feature_model(image)

    features = features.detach().cpu().numpy().reshape(1, -1)

    # Get decision scores from the original trained SVM.
    decision_scores = svm_model.decision_function(features)

    # Apply the new multiclass calibration layer.
    probabilities = multiclass_calibrator.predict_proba(
        decision_scores
    )[0]

    classes = list(multiclass_calibrator.classes_)

    p_cad = float(probabilities[classes.index(0)])
    p_normal = float(probabilities[classes.index(1)])
    p_arrhythmia = float(probabilities[classes.index(2)])

    denominator = p_cad + p_normal

    if denominator <= 0:
        raise ValueError(
            "CAD and normal probabilities sum to zero."
        )

    # CAD vs normal score; arrhythmia is excluded.
    binary_score = p_cad / denominator
    binary_score = np.clip(binary_score, 1e-7, 1 - 1e-7)

    score_logit = np.log(
        binary_score / (1 - binary_score)
    )

    # Apply binary Platt calibration.
    positive_index = list(binary_calibrator.classes_).index(1)

    calibrated_cad = float(
        binary_calibrator.predict_proba(
            [[score_logit]]
        )[0, positive_index]
    )

    return {
        "cad_probability": calibrated_cad,
        "arrhythmia_probability": p_arrhythmia
    }
