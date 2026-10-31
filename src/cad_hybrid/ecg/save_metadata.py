
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TRAIN_CSV = ROOT / "data" / "processed" / "train.csv"
OUTPUT = ROOT / "models" / "ecg_metadata.json"


def main():
    df = pd.read_csv(TRAIN_CSV)

    cad_count = int((df["label"] == "cad_positive").sum())
    normal_count = int((df["label"] == "negative").sum())

    positive_rate = cad_count / (cad_count + normal_count)

    metadata = {
        "training_positive_rate": positive_rate,
        "positive_rate_unit": "image_level",
        "training_cad_images": cad_count,
        "training_normal_images": normal_count,
        "label_mapping": {
            "0": "cad_positive",
            "1": "negative",
            "2": "arrhythmia"
        },
        "calibration_method": "Platt sigmoid calibration",
        "base_classifier": "Existing CalibratedClassifierCV",
        "binary_score": "P(CAD) / (P(CAD) + P(normal))",
        "arrhythmia_handling": "Excluded from binary calibration; returned separately",
        "model_retrained": False
    }

    OUTPUT.parent.mkdir(exist_ok=True)

    with open(OUTPUT, "w") as f:
        json.dump(metadata, f, indent=4)

    print(json.dumps(metadata, indent=4))
    print("\nMetadata saved:", OUTPUT)


if __name__ == "__main__":
    main()
