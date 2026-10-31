
from pathlib import Path
import hashlib
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = Path(
    r"C:\Users\Mamatha\Arohan_internship\ecg-model\data\raw\ecg_multan_pk"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_STATE = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

LABEL_MAPPING = {
    "ECG Images of Myocardial Infarction Patients (240x12=2880)": "cad_positive",
    "ECG Images of Patient that have History of MI (172x12=2064)": "cad_positive",
    "ECG Images of Patient that have abnormal heartbeat (233x12=2796)": "arrhythmia",
    "Normal Person ECG Images (284x12=3408)": "negative",
}


def sha256_file(path):
    """Calculate SHA-256 hash of an image."""
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def select_best_fold(data, n_splits, target_fraction):
    """Select a stratified group fold closest to target size."""
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    y = data["label"].to_numpy()
    groups = data["dup_group"].to_numpy()

    overall_distribution = data["label"].value_counts(normalize=True)

    best_indices = None
    best_score = float("inf")

    for _, candidate_indices in splitter.split(data, y, groups):
        candidate = data.iloc[candidate_indices]

        size_error = abs(
            len(candidate) / len(data) - target_fraction
        )

        candidate_distribution = candidate["label"].value_counts(
            normalize=True
        )

        class_error = sum(
            abs(
                candidate_distribution.get(label, 0)
                - overall_distribution.get(label, 0)
            )
            for label in LABEL_MAPPING.values()
        )

        score = size_error + class_error

        if score < best_score:
            best_score = score
            best_indices = candidate_indices

    return best_indices


def main():

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_ROOT}"
        )

    records = []
    hash_to_group = {}
    hash_to_files = {}
    next_group = 1

    print("Scanning Multan ECG images...")

    image_files = sorted(
        path for path in DATASET_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image_path in image_files:

        original_class = image_path.parent.name

        if original_class not in LABEL_MAPPING:
            raise ValueError(
                f"Unknown class folder: {original_class}"
            )

        image_hash = sha256_file(image_path)

        if image_hash not in hash_to_group:
            hash_to_group[image_hash] = next_group
            hash_to_files[image_hash] = []
            next_group += 1

        group_id = hash_to_group[image_hash]
        hash_to_files[image_hash].append(image_path)

        records.append({
            "path": str(image_path.resolve()),
            "dataset_id": "multan_pk",
            "original_class": original_class,
            "label": LABEL_MAPPING[original_class],
            "dup_group": group_id,
            "sha256": image_hash,
        })

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No ECG images found.")

    # A duplicate group must not contain conflicting labels.
    if (df.groupby("dup_group")["label"].nunique() > 1).any():
        raise ValueError("A duplicate group contains conflicting labels.")

    print(f"Total images: {len(df)}")
    print(f"Unique image groups: {df['dup_group'].nunique()}")

    # Approximately 70% train, 15% validation, 15% test.
    test_indices = select_best_fold(
        df, n_splits=7, target_fraction=0.15
    )

    test_df = df.iloc[test_indices].copy()
    remaining_df = df.drop(index=test_df.index).copy()

    validation_indices = select_best_fold(
        remaining_df,
        n_splits=5,
        target_fraction=0.15 / 0.85,
    )

    validation_df = remaining_df.iloc[validation_indices].copy()
    train_df = remaining_df.drop(index=validation_df.index).copy()

    # Verify duplicate-group leakage.
    train_groups = set(train_df["dup_group"])
    validation_groups = set(validation_df["dup_group"])
    test_groups = set(test_df["dup_group"])

    overlaps = {
        "train_validation": train_groups & validation_groups,
        "train_test": train_groups & test_groups,
        "validation_test": validation_groups & test_groups,
    }

    leakage_count = sum(len(groups) for groups in overlaps.values())

    print("\nDuplicate-group leakage report")
    for split_pair, groups in overlaps.items():
        print(f"{split_pair}: {len(groups)} overlapping groups")

    print(f"Total overlapping groups: {leakage_count}")

    if leakage_count != 0:
        raise RuntimeError("Duplicate-group leakage detected.")

    print("Duplicate-group leakage check: PASSED")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    validation_df.to_csv(
        OUTPUT_DIR / "validation.csv", index=False
    )
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("\nSplit summary")
    for name, split in [
        ("Train", train_df),
        ("Validation", validation_df),
        ("Test", test_df),
    ]:
        print(f"\n{name}: {len(split)} images")
        print(split["label"].value_counts().to_string())

    print(f"\nCSV files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
