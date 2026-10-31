
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from src.predict_ecg import load_models, preprocess_image
from src.model_config import DEVICE

ROOT = Path(__file__).resolve().parents[3]
VALIDATION_CSV = ROOT / "data" / "processed" / "validation.csv"
MODEL_PATH = ROOT / "models" / "ecg_calibrated_classifier.pkl"
CALIBRATOR_PATH = ROOT / "models" / "cad_platt_calibrator.pkl"

def main():
    df = pd.read_csv(VALIDATION_CSV)

    # Exclude arrhythmia; retain CAD and normal only.
    df = df[df["label"].isin(["cad_positive", "negative"])].copy()
    y = (df["label"] == "cad_positive").astype(int).to_numpy()

    feature_model, classifier = load_models()
    feature_model.eval()

    features = []

    print(f"Validation images for calibration: {len(df)}")

    with torch.no_grad():
        for i, path in enumerate(df["path"]):
            image = preprocess_image(path)

            if not isinstance(image, torch.Tensor):
                image = torch.as_tensor(image)

            if image.ndim == 3:
                image = image.unsqueeze(0)

            image = image.to(DEVICE)
            output = feature_model(image)
            features.append(output.detach().cpu().numpy().reshape(-1))

            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1}/{len(df)}")

    X = np.vstack(features)

    # Existing three-class classifier; no retraining.
    probabilities = classifier.predict_proba(X)
    classes = list(classifier.classes_)

    p_cad = probabilities[:, classes.index(0)]
    p_normal = probabilities[:, classes.index(1)]

    # Binary CAD-vs-normal score.
    score = p_cad / (p_cad + p_normal)

    # Platt calibration on the validation set.
    eps = 1e-7
    score = np.clip(score, eps, 1 - eps)
    score_logit = np.log(score / (1 - score)).reshape(-1, 1)

    calibrator = LogisticRegression(C=1e6, solver="lbfgs")
    calibrator.fit(score_logit, y)

    ROOT.joinpath("models").mkdir(exist_ok=True)
    joblib.dump(calibrator, CALIBRATOR_PATH)

    print("\nCalibration completed successfully.")
    print("CAD validation images:", int(y.sum()))
    print("Normal validation images:", int((y == 0).sum()))
    print("Calibrator saved to:", CALIBRATOR_PATH)

if __name__ == "__main__":
    main()
