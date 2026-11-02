
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cad_hybrid.ecg.dataset import ECGDataset
from cad_hybrid.ecg.feature_extractor import create_feature_extractor


ROOT = Path(__file__).resolve().parents[1]

SPLITS = {
    "train": "data/processed/train.csv",
    "validation": "data/processed/validation.csv",
    "test": "data/processed/test.csv",
}

BATCH_SIZE = 8
FEATURE_SIZE = 512


def extract_features(model, loader):

    features = []
    labels = []
    groups = []

    model.eval()

    with torch.no_grad():

        for images, targets, duplicate_groups in loader:

            embeddings = model(images)

            embeddings = embeddings.flatten(start_dim=1)

            features.append(embeddings.cpu().numpy())
            labels.append(targets.numpy())
            groups.extend(duplicate_groups)

    return (
        np.concatenate(features),
        np.concatenate(labels),
        np.asarray(groups, dtype=str),
    )


def is_valid_feature_file(output_path, expected_count):

    if not output_path.exists():
        return False

    try:
        with np.load(output_path) as saved:

            X = saved["X"]
            y = saved["y"]
            groups = saved["groups"]

            return (
                X.shape == (expected_count, FEATURE_SIZE)
                and y.shape == (expected_count,)
                and groups.shape == (expected_count,)
                and np.isfinite(X).all()
            )

    except (OSError, ValueError, KeyError):
        return False


def main():

    output_dir = ROOT / "data" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = None

    for split_name, csv_path in SPLITS.items():

        print(f"\nProcessing {split_name} features...")

        dataset = ECGDataset(csv_path)

        output_path = (
            output_dir / f"{split_name}_features.npz"
        )

        # Reuse existing valid features.
        if is_valid_feature_file(
            output_path,
            len(dataset)
        ):

            print("Valid feature file already exists.")
            print("Skipping extraction:", output_path)
            continue

        # Load ResNet18 only when extraction is required.
        if model is None:
            print("Loading pretrained ResNet18...")
            model = create_feature_extractor()

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        X, y, groups = extract_features(
            model,
            loader
        )

        np.savez_compressed(
            output_path,
            X=X,
            y=y,
            groups=groups,
        )

        print("Feature shape:", X.shape)
        print("Label shape:", y.shape)
        print("Unique groups:", len(np.unique(groups)))
        print("Saved:", output_path)

    print("\nAll ECG feature extraction completed.")


if __name__ == "__main__":
    main()
