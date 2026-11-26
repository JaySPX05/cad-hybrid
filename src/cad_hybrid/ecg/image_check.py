"""Is an uploaded image like the ECGs the model was trained on?

The ECG model gives a CAD probability for any picture, including ones that aren't ECGs
(a photo, a chest X-ray, a lab report). This check compares the image's ResNet18
features with the training ECGs' features: the distance to the most similar training
ECG (1 - cosine similarity). Real ECGs from the validation split set the threshold, so
about 1% of genuine ECGs like the training ones would be flagged (``quantile``), with a
safety margin on top.

The reference is built by ``scripts/build_ecg_image_check.py`` from the saved training
and validation features and stored in ``models/ecg_image_check.npz`` (gitignored).
No torch needed here: it works on feature vectors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REFERENCE_PATH = ROOT / "models" / "ecg_image_check.npz"


def normalise(X) -> np.ndarray:
    """Rows scaled to length 1 (for cosine similarity)."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float32))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms == 0, 1.0, norms)


def nearest_distance(reference: np.ndarray, X) -> np.ndarray:
    """1 - cosine similarity to the most similar reference row, for each row of X."""
    return 1.0 - (normalise(X) @ reference.T).max(axis=1)


def build_reference(train_X, validation_X, quantile: float = 0.99,
                    margin: float = 1.25) -> dict:
    """Reference features and the distance above which an image is flagged."""
    if not 0.5 < quantile < 1.0 or margin < 1.0:
        raise ValueError("quantile must be in (0.5, 1) and margin at least 1")
    reference = normalise(train_X)
    validation_d = nearest_distance(reference, validation_X)
    threshold = float(np.quantile(validation_d, quantile) * margin)
    return {
        "reference": reference,
        "threshold": threshold,
        "quantile": quantile,
        "margin": margin,
        "validation_median": float(np.median(validation_d)),
        "validation_max": float(validation_d.max()),
    }


def save_reference(ref: dict, path: Path | str = REFERENCE_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in ref.items()})
    return path


def load_reference(path: Path | str = REFERENCE_PATH) -> dict | None:
    """The saved reference, or None if it hasn't been built."""
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path) as data:
        ref = {k: data[k] for k in data.files}
    ref["threshold"] = float(ref["threshold"])
    return ref


def check(ref: dict, features) -> tuple[bool, float]:
    """``(looks_like_training_ecgs, distance)`` for one image's features."""
    distance = float(nearest_distance(ref["reference"], features)[0])
    return distance <= ref["threshold"], distance
