"""Build the check that an uploaded image looks like the ECGs the model was trained on.

    python scripts/build_ecg_image_check.py
    python scripts/build_ecg_image_check.py --not-ecg-dir path/to/other/pictures

Reads the ResNet18 features saved by scripts/extract_ecg_features.py
(data/features/train_features.npz and validation_features.npz), sets the distance limit
from the validation ECGs and saves models/ecg_image_check.npz (gitignored). Prints how
many validation and test ECGs would be flagged (should be about 1% or fewer).

With --not-ecg-dir, it also extracts features for the pictures in that folder (photos,
X-rays, documents: anything that isn't an ECG) and prints how many are caught; this
needs torch. Run it again after retraining the ECG model or re-extracting features.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cad_hybrid.ecg.image_check import (REFERENCE_PATH, build_reference,  # noqa: E402
                                        nearest_distance, save_reference)

FEATURES = ROOT / "data" / "features"


def load_features(name: str) -> np.ndarray | None:
    path = FEATURES / f"{name}_features.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return data["X"].astype(np.float32)


def not_ecg_features(folder: Path) -> np.ndarray:
    import torch

    from cad_hybrid.ecg.feature_extractor import create_feature_extractor
    from cad_hybrid.ecg.preprocess import preprocess_image

    model = create_feature_extractor().eval()
    rows = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        with torch.no_grad():
            rows.append(model(preprocess_image(str(path))).cpu().numpy().reshape(-1))
    if not rows:
        sys.exit(f"No .jpg/.jpeg/.png pictures in {folder}")
    return np.vstack(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--margin", type=float, default=1.25)
    parser.add_argument("--not-ecg-dir", type=Path, default=None)
    args = parser.parse_args()

    train, validation = load_features("train"), load_features("validation")
    if train is None or validation is None:
        sys.exit(f"Feature files missing in {FEATURES}. Run python scripts/extract_ecg_features.py")

    ref = build_reference(train, validation, args.quantile, args.margin)
    path = save_reference(ref)
    t = ref["threshold"]
    print(f"Distance limit: {t:.4f} (validation median {ref['validation_median']:.4f}, "
          f"max {ref['validation_max']:.4f})")
    for name, X in (("validation", validation), ("test", load_features("test"))):
        if X is not None:
            d = nearest_distance(ref["reference"], X)
            print(f"  {name}: {np.mean(d > t):.1%} of {len(d)} ECG images flagged")
    if args.not_ecg_dir:
        d = nearest_distance(ref["reference"], not_ecg_features(args.not_ecg_dir))
        print(f"  not ECGs: {np.mean(d > t):.1%} of {len(d)} pictures caught")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
