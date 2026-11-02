
"""Identify direct cross-split pHash matches and prepare clean ECG splits.

Safety:
- Original train/validation/test CSV files are never overwritten.
- Training images are never excluded.
- Only same-class images are compared.
- Only direct pHash matches within MAX_DISTANCE are counted.
- Run without --write for a dry run.
- Run with --write only after reviewing the dry-run results.
"""

from pathlib import Path
import argparse
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cad_hybrid.data.dedupe import image_phash

INPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = INPUT_DIR / "phash_cleaned"

SPLITS = ["train", "validation", "test"]
MAX_DISTANCE = 4


def load_splits():
    """Load the original split CSV files without modifying them."""
    splits = {}

    for name in SPLITS:
        path = INPUT_DIR / f"{name}.csv"

        if not path.exists():
            raise FileNotFoundError(f"Missing split CSV: {path}")

        df = pd.read_csv(path, dtype={"dup_group": str})

        required = {"path", "label", "dup_group"}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{name}.csv is missing columns: {missing}"
            )

        df = df.copy()
        df["_split"] = name
        splits[name] = df

    return splits


def calculate_phashes(splits):
    """Calculate one pHash for every image."""
    combined = pd.concat(splits.values(), ignore_index=True)

    if combined["path"].duplicated().any():
        raise ValueError(
            "An image path occurs more than once across splits."
        )

    hashes = []

    print("\nCalculating pHash for all images...")

    for i, relative_path in enumerate(combined["path"], start=1):
        image_path = ROOT / relative_path

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        hashes.append(image_phash(str(image_path)))

        if i % 100 == 0 or i == len(combined):
            print(f"Processed {i}/{len(combined)} images")

    combined["_phash"] = hashes

    return combined


def find_direct_cross_split_matches(combined):
    """Find direct same-class pHash matches across different splits.

    Every pair is compared independently. No connected-component
    grouping or transitive matching is used.
    """
    affected = {name: set() for name in SPLITS}
    pair_counts = {}
    total_matches = 0

    print("\nChecking direct pHash matches within each class...")

    for label, group in combined.groupby("label"):
        records = group.to_dict("records")

        for i in range(len(records)):
            first = records[i]

            for j in range(i + 1, len(records)):
                second = records[j]

                # Only compare images belonging to different splits.
                if first["_split"] == second["_split"]:
                    continue

                # Compare the actual pHash values directly.
                distance = (
                    int(first["_phash"]) ^ int(second["_phash"])
                ).bit_count()

                if distance > MAX_DISTANCE:
                    continue

                total_matches += 1

                split_pair = tuple(
                    sorted([first["_split"], second["_split"]])
                )
                key = "_vs_".join(split_pair)

                pair_counts[key] = pair_counts.get(key, 0) + 1

                # Preserve every training image.
                # Exclude matching validation/test images.
                if first["_split"] in ("validation", "test"):
                    affected[first["_split"]].add(first["path"])

                if second["_split"] in ("validation", "test"):
                    affected[second["_split"]].add(second["path"])

        print(f"Checked class: {label}")

    return total_matches, affected, pair_counts


def prepare_cleaned_splits(splits, affected):
    """Prepare separate cleaned DataFrames without changing originals."""
    cleaned = {}

    for name in SPLITS:
        original = splits[name].drop(columns=["_split"]).copy()

        if name == "train":
            cleaned[name] = original.copy()
        else:
            cleaned[name] = original[
                ~original["path"].isin(affected[name])
            ].copy()

    return cleaned


def verify_remaining_matches(cleaned, original_combined):
    """Check whether direct cross-split matches remain after exclusions."""
    remaining_paths = {
        path
        for df in cleaned.values()
        for path in df["path"]
    }

    remaining = original_combined[
        original_combined["path"].isin(remaining_paths)
    ].copy()

    remaining_count, _, remaining_pair_counts = (
        find_direct_cross_split_matches(remaining)
    )

    return remaining_count, remaining_pair_counts


def print_report(
    splits,
    combined,
    total_matches,
    affected,
    pair_counts,
):
    """Print the cleaning proposal and verify its safety."""
    print("\n" + "=" * 60)
    print("DIRECT PHASH CLEANING DRY-RUN REPORT")
    print("=" * 60)

    print(f"pHash distance threshold: {MAX_DISTANCE}")
    print(f"Total images analysed: {len(combined)}")
    print(f"Direct cross-split matches: {total_matches}")

    print("\nDirect cross-split matches by split pair:")

    if pair_counts:
        for key, count in sorted(pair_counts.items()):
            print(f"{key}: {count}")
    else:
        print("No direct cross-split matches found.")

    print("\nTotal direct matches:", total_matches)

    cleaned = prepare_cleaned_splits(splits, affected)

    for name in SPLITS:
        original = splits[name].drop(columns=["_split"])

        print(f"\n{name.upper()}")
        print(f"Original images: {len(original)}")
        print(f"Images to exclude: {len(affected[name])}")
        print(f"Remaining images: {len(cleaned[name])}")
        print(
            "Remaining unique dup_group values:",
            cleaned[name]["dup_group"].nunique(),
        )

        print("Remaining class counts:")
        print(
            cleaned[name]["label"]
            .value_counts()
            .to_string()
        )

    # Verify that training has not changed.
    if not cleaned["train"]["path"].equals(
        splits["train"]["path"]
    ):
        raise RuntimeError(
            "Training split was unexpectedly changed."
        )

    # Verify no direct same-class cross-split matches remain.
    remaining_count, remaining_pair_counts = (
        verify_remaining_matches(cleaned, combined)
    )

    print("\n" + "-" * 60)
    print("FINAL VERIFICATION")
    print("-" * 60)

    print(
        "Remaining direct cross-split matches:",
        remaining_count,
    )

    if remaining_pair_counts:
        for key, count in sorted(remaining_pair_counts.items()):
            print(f"{key}: {count}")

    if remaining_count != 0:
        raise RuntimeError(
            "Direct pHash matches remain after proposed exclusions."
        )

    print("Training preservation: PASSED")
    print("Direct pHash separation: PASSED")

    report = {
        "method": "direct_pairwise_phash",
        "comparison_scope": "same_class_only",
        "threshold": MAX_DISTANCE,
        "total_images": len(combined),
        "direct_cross_split_matches": total_matches,
        "pair_counts": pair_counts,
        "excluded_validation": len(affected["validation"]),
        "excluded_test": len(affected["test"]),
        "remaining_cross_split_matches": remaining_count,
        "original_split_sizes": {
            name: len(splits[name]) for name in SPLITS
        },
        "cleaned_split_sizes": {
            name: len(cleaned[name]) for name in SPLITS
        },
        "excluded_images": {
            name: sorted(affected[name]) for name in SPLITS
        },
    }

    return cleaned, report


def main():
    parser = argparse.ArgumentParser(
        description="Check direct cross-split ECG pHash matches."
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help="Save cleaned CSVs into a separate folder.",
    )

    args = parser.parse_args()

    splits = load_splits()

    combined = calculate_phashes(splits)

    total_matches, affected, pair_counts = (
        find_direct_cross_split_matches(combined)
    )

    cleaned, report = print_report(
        splits,
        combined,
        total_matches,
        affected,
        pair_counts,
    )

    if not args.write:
        print("\nDRY RUN ONLY: No files were written.")
        print("Review the results before using --write.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, df in cleaned.items():
        output_path = OUTPUT_DIR / f"{name}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

    report_path = OUTPUT_DIR / "phash_cleaning_report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"\nSaved report: {report_path}")
    print("\nCleaned splits and report saved successfully.")


if __name__ == "__main__":
    main()
