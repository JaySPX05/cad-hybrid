
"""Verify existing ECG splits and generate a reproducible split manifest.

This script does not modify the original train/validation/test CSV files.
It saves the manifest and verification report under reports/.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

SPLITS = {
    "train": 638,
    "validation": 158,
    "test": 132,
}


def main():
    print("=" * 60)
    print("ECG SPLIT VERIFICATION")
    print("=" * 60)

    frames = []

    # 1. Load and validate the original split CSVs.
    for split, expected_count in SPLITS.items():
        file_path = DATA_DIR / f"{split}.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

        df = pd.read_csv(file_path, dtype={"dup_group": str})

        required = {"path", "label", "dup_group"}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(f"{split}.csv missing columns: {missing}")

        if df[["path", "label", "dup_group"]].isna().any().any():
            raise ValueError(f"Missing required values in {split}.csv")

        if len(df) != expected_count:
            raise ValueError(
                f"{split}: expected {expected_count}, found {len(df)}"
            )

        df = df.copy()
        df["split"] = split
        frames.append(df)

        print(f"{split}: {len(df)} images - PASSED")

    combined = pd.concat(frames, ignore_index=True)

    # 2. Verify total images and unique paths.
    if len(combined) != 928:
        raise ValueError(f"Expected 928 images, found {len(combined)}")

    duplicate_paths = combined[
        combined["path"].duplicated(keep=False)
    ]

    if not duplicate_paths.empty:
        raise RuntimeError("Duplicate image paths found across splits.")

    print("\nTotal images:", len(combined))
    print("Unique image paths:", combined["path"].nunique())
    print("Image path uniqueness: PASSED")

    # 3. Check duplicate-group overlap between splits.
    group_sets = {
        split: set(
            combined.loc[combined["split"] == split, "dup_group"]
        )
        for split in SPLITS
    }

    comparisons = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    overlap_report = {}
    total_overlap = 0

    print("\nDUPLICATE-GROUP LEAKAGE CHECK")

    for first, second in comparisons:
        overlap = group_sets[first] & group_sets[second]
        overlap_report[f"{first}_vs_{second}"] = len(overlap)
        total_overlap += len(overlap)

        print(f"{first} vs {second}: {len(overlap)}")

    print("Total overlapping groups:", total_overlap)

    # 4. Check that each duplicate group has one label.
    label_counts = combined.groupby("dup_group")["label"].nunique()

    if (label_counts > 1).any():
        raise RuntimeError("Conflicting labels found within duplicate groups.")

    print("Duplicate-group label consistency: PASSED")

    # 5. Print split and class summaries.
    print("\nSPLIT CLASS DISTRIBUTION")

    for split in SPLITS:
        subset = combined[combined["split"] == split]

        print(f"\n{split.upper()}")
        print("Images:", len(subset))
        print("Unique groups:", subset["dup_group"].nunique())
        print(subset["label"].value_counts().to_string())

    # 6. Print unique groups per class and split.
    print("\nUNIQUE GROUPS PER CLASS")

    group_summary = (
        combined.groupby(["split", "label"])["dup_group"]
        .nunique()
        .unstack(fill_value=0)
    )

    print(group_summary.to_string())

    # 7. Generate manifest.
    manifest = combined[["path", "label", "dup_group", "split"]].copy()
    manifest.insert(
        1,
        "filename",
        manifest["path"].apply(lambda p: Path(p).name),
    )

    # 8. Save manifest and verification report.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = REPORT_DIR / "split_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    report_path = REPORT_DIR / "split_verification.txt"

    lines = [
        "ECG SPLIT VERIFICATION REPORT",
        "=" * 45,
        f"Total images: {len(combined)}",
        f"Unique paths: {combined['path'].nunique()}",
        "",
        "Split sizes:",
    ]

    for split, expected in SPLITS.items():
        actual = int((combined["split"] == split).sum())
        lines.append(f"{split}: {actual} (expected {expected})")

    lines.extend([
        "",
        "Duplicate-group overlaps:",
    ])

    for pair, count in overlap_report.items():
        lines.append(f"{pair}: {count}")

    lines.extend([
        f"Total overlapping groups: {total_overlap}",
        "",
        "Unique groups per class:",
        group_summary.to_string(),
        "",
        f"Duplicate-group leakage check: "
        f"{'PASSED' if total_overlap == 0 else 'FAILED'}",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\nFILES GENERATED")
    print("Manifest:", manifest_path)
    print("Verification report:", report_path)

    if total_overlap != 0:
        raise RuntimeError(
            "Task 3 verification failed: duplicate-group overlap exists."
        )

    print("\n" + "=" * 60)
    print("TASK 3 SPLIT VERIFICATION: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
