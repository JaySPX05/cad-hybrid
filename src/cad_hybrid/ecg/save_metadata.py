"""Write reports/ecg_metadata.json: how the ECG CAD score was built and its CAD rates.

    python -m cad_hybrid.ecg.save_metadata

Rates count each duplicate group once (one ECG), CAD vs normal only, because the
dataset contains many copies of the same ECG (the MI folder is 30 ECGs repeated about
8 times). The fusion prior (configs/fusion.yaml, branch_priors.ecg) is the validation
rate, since the binary CAD score is Platt-calibrated on the validation ECGs.
"""

import json
from pathlib import Path

import pandas as pd

from cad_hybrid.ecg.scoring import positive_rate_per_ecg

ROOT = Path(__file__).resolve().parents[3]
SPLIT_DIR = ROOT / "data" / "processed"
OUTPUT = ROOT / "reports" / "ecg_metadata.json"


def build_metadata(train: pd.DataFrame, validation: pd.DataFrame) -> dict:
    """Metadata dict from the train and validation split tables."""
    train_rate = positive_rate_per_ecg(train)
    calibration_rate = positive_rate_per_ecg(validation)
    return {
        "fusion_prior": round(calibration_rate["positive_rate"], 4),
        "fusion_prior_source": "CAD rate of the calibration (validation) set, one ECG per duplicate group",
        "calibration_set": calibration_rate,
        "training_set": train_rate,
        "training_images": {
            "cad_positive": int((train["label"] == "cad_positive").sum()),
            "negative": int((train["label"] == "negative").sum()),
            "arrhythmia": int((train["label"] == "arrhythmia").sum()),
        },
        "label_mapping": {"0": "cad_positive", "1": "negative", "2": "arrhythmia"},
        "feature_extractor": "ResNet18 pretrained on ImageNet (frozen)",
        "base_classifier": "StandardScaler + RBF SVC, trained by scripts/train_ecg.py "
                           "(group-aware 5-fold CV on the training split)",
        "calibration_method": "Group-aware OOF multiclass logistic calibration, then binary "
                              "Platt calibration on validation CAD + normal ECGs",
        "calibration_bundle": "models/ecg_calibration_bundle.pkl",
        "binary_score": "P(CAD) / (P(CAD) + P(normal))",
        "arrhythmia_handling": "Excluded from the CAD score and from fusion; returned separately",
    }


def main():
    train = pd.read_csv(SPLIT_DIR / "train.csv", dtype={"dup_group": str})
    validation = pd.read_csv(SPLIT_DIR / "validation.csv", dtype={"dup_group": str})
    metadata = build_metadata(train, validation)

    OUTPUT.parent.mkdir(exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(json.dumps(metadata, indent=4))
    print("\nMetadata saved:", OUTPUT)
    print(f"Set branch_priors.ecg in configs/fusion.yaml to {metadata['fusion_prior']}")


if __name__ == "__main__":
    main()
