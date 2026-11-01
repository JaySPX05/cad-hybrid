
"""
Create verified ECG dataset splits using original leak-free assignments.

The original ecg-model split CSVs are the source of truth.
Original duplicate-group IDs and split assignments are preserved.
Image paths are updated for the cad-hybrid project.
"""

from pathlib import Path
import hashlib

import pandas as pd


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "ecg_multan_pk"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Original project paths
ORIGINAL_PROJECT = PROJECT_ROOT.parent / "ecg-model"
ORIGINAL_SPLIT_DIR = ORIGINAL_PROJECT / "data" / "processed"

SPLIT_FILES = {
    "train": ORIGINAL_SPLIT_DIR / "train.csv",
    "validation": ORIGINAL_SPLIT_DIR / "validation.csv",
    "test": ORIGINAL_SPLIT_DIR / "test.csv",
}

EXPECTED_COUNTS = {
    "train": 638,
    "validation": 158,
    "test": 132,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

OUTPUT_COLUMNS = [
    "path",
    "dataset_id",
    "original_class",
    "label",
    "dup_group",
    "sha256",
]


# --------------------------------------------------
# SHA-256 helper
# --------------------------------------------------

def sha256_file(path: Path) -> str:
    """Calculate SHA-256 hash of an image."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


# --------------------------------------------------
# Path conversion
# --------------------------------------------------

def remap_image_path(original_path: str) -> str:
    """Convert original project image paths to hybrid paths."""

    normalized = original_path.replace("\\", "/")

    prefix = "data/interim/"

    if not normalized.startswith(prefix):
        raise ValueError(
            f"Unexpected original image path: {original_path}"
        )

    relative_image = normalized[len(prefix):]

    return (
        Path("data/raw/ecg_multan_pk") / relative_image
    ).as_posix()


# --------------------------------------------------
# Leakage verification
# --------------------------------------------------

def check_no_group_leakage(splits: dict) -> None:
    """Check whether duplicate groups occur in multiple splits."""

    group_sets = {
        name: set(df["dup_group"].astype(str))
        for name, df in splits.items()
    }

    comparisons = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    total_overlaps = 0

    print("\nDUPLICATE-GROUP LEAKAGE REPORT")
    print("=" * 50)

    for first, second in comparisons:

        overlap = group_sets[first] & group_sets[second]

        print(f"{first} vs {second}: {len(overlap)}")

        total_overlaps += len(overlap)

    print(f"Total overlapping groups: {total_overlaps}")

    if total_overlaps != 0:
        raise RuntimeError(
            "Duplicate-group leakage detected. "
            "Final CSV files were not saved."
        )

    print("Duplicate-group leakage check: PASSED")


# --------------------------------------------------
# Split summary
# --------------------------------------------------

def print_summary(name: str, df: pd.DataFrame) -> None:
    """Display split statistics."""

    print(f"\n{name.upper()} SUMMARY")
    print("-" * 50)

    print("Total images:", len(df))

    print(
        "Unique duplicate groups:",
        df["dup_group"].nunique()
    )

    print("\nClass distribution:")

    print(
        df["label"]
        .value_counts()
        .to_string()
    )


# --------------------------------------------------
# Main function
# --------------------------------------------------

def main():

    print("Preparing verified ECG dataset splits...")
    print("=" * 50)

    # Check dataset directory.

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_ROOT}"
        )

    # Step 1: Load original split CSVs.

    splits = {}

    for split_name, csv_path in SPLIT_FILES.items():

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Original split file not found: {csv_path}"
            )

        df = pd.read_csv(
            csv_path,
            dtype={"dup_group": str},
        )

        required_columns = {
            "path",
            "dataset_id",
            "original_class",
            "label",
            "dup_group",
        }

        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"{split_name} is missing columns: {missing_columns}"
            )

        if len(df) != EXPECTED_COUNTS[split_name]:
            raise ValueError(
                f"Unexpected {split_name} count: {len(df)}"
            )

        splits[split_name] = df.copy()

    print("Original split files loaded successfully.")

    # Step 2: Update paths and verify image files.

    all_records = []

    for split_name, df in splits.items():

        df["path"] = df["path"].apply(remap_image_path)

        df["sha256"] = None

        for index, row in df.iterrows():

            image_path = PROJECT_ROOT / row["path"]

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image missing: {image_path}"
                )

            df.at[index, "sha256"] = sha256_file(image_path)

        splits[split_name] = df

        all_records.append(df)

        print(
            f"{split_name}: {len(df)} images verified."
        )

    combined = pd.concat(
        all_records,
        ignore_index=True,
    )

    # Step 3: Verify total image count.

    if len(combined) != 928:
        raise ValueError(
            f"Expected 928 images, found {len(combined)}."
        )

    # Check duplicate paths.

    duplicate_paths = combined[
        combined["path"].duplicated(keep=False)
    ]

    if not duplicate_paths.empty:
        raise ValueError(
            "An image path appears more than once across splits."
        )

    print("\nTotal images:", len(combined))
    print("Unique image paths:", combined["path"].nunique())

    # Step 4: Check duplicate-group consistency.

    group_label_counts = (
        combined.groupby("dup_group")["label"].nunique()
    )

    if (group_label_counts > 1).any():
        raise ValueError(
            "A duplicate group contains conflicting labels."
        )

    hash_group_counts = (
        combined.groupby("sha256")["dup_group"].nunique()
    )

    if (hash_group_counts > 1).any():
        raise ValueError(
            "Identical images have different duplicate-group IDs."
        )

    print("Duplicate-group consistency: PASSED")

    # Step 5: Check dataset inventory.

    actual_images = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in DATASET_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    }

    assigned_images = set(combined["path"])

    missing_images = actual_images - assigned_images
    extra_images = assigned_images - actual_images

    if missing_images or extra_images:
        raise RuntimeError(
            f"Dataset mismatch. Missing: {len(missing_images)}, "
            f"Extra: {len(extra_images)}"
        )

    print("Dataset inventory check: PASSED")

    # Step 6: Verify duplicate-group leakage.

    check_no_group_leakage(splits)

    # Step 7: Print split summaries.

    for split_name, df in splits.items():
        print_summary(split_name, df)

    # Step 8: Save final CSV files.

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split_name, df in splits.items():

        output_path = OUTPUT_DIR / f"{split_name}.csv"

        df[OUTPUT_COLUMNS].to_csv(
            output_path,
            index=False,
        )

        print(f"\nSaved: {output_path}")

    print("\nAll verified split CSV files saved successfully.")


if __name__ == "__main__":
    main()
