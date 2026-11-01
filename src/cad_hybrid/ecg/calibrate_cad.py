from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[3]

ORIGINAL_ROOT = ROOT.parent / "ecg-model"

TRAIN_FEATURES = ORIGINAL_ROOT / "data" / "features" / "train_features.npz"
VALIDATION_FEATURES = ORIGINAL_ROOT / "data" / "features" / "validation_features.npz"
VALIDATION_CSV = ROOT / "data" / "processed" / "validation.csv"

SVM_PATH = ORIGINAL_ROOT / "models" / "ecg_svm_classifier.pkl"
OUTPUT_PATH = ROOT / "models" / "ecg_calibration_bundle.pkl"

RANDOM_STATE = 42
N_SPLITS = 5


def main():
    print("Loading training features...")

    train = np.load(TRAIN_FEATURES)
    X_train = train["X"]
    y_train = train["y"].astype(int)
    groups_train = train["groups"].astype(str)

    print("Training feature shape:", X_train.shape)
    print("Training groups:", len(np.unique(groups_train)))

    print("\nLoading original trained SVM...")
    svm_model = joblib.load(SVM_PATH)

    print("SVM classes:", svm_model.classes_)

    # Generate out-of-fold decision scores.
    print("\nGenerating group-aware out-of-fold scores...")

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    oof_scores = np.zeros((len(y_train), 3))
    assigned = np.zeros(len(y_train), dtype=bool)

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(X_train, y_train, groups_train), start=1
    ):
        fold_model = clone(svm_model)

        fold_model.fit(X_train[train_idx], y_train[train_idx])

        scores = fold_model.decision_function(X_train[valid_idx])

        oof_scores[valid_idx] = scores
        assigned[valid_idx] = True

        print(
            f"Fold {fold}/{N_SPLITS}: "
            f"training={len(train_idx)}, validation={len(valid_idx)}"
        )

    if not assigned.all():
        raise RuntimeError("Some training samples did not receive OOF scores.")

    print("\nFitting multiclass calibration on OOF scores...")

    # Give each duplicate group equal total weight.
    _, inverse, counts = np.unique(
        groups_train, return_inverse=True, return_counts=True
    )
    sample_weights = 1.0 / counts[inverse]

    multiclass_calibrator = LogisticRegression(
        C=1.0,
        max_iter=2000
    )

    multiclass_calibrator.fit(
        oof_scores,
        y_train,
        sample_weight=sample_weights
    )

    # Load validation features.
    print("\nLoading validation features...")

    validation = np.load(VALIDATION_FEATURES)

    X_validation = validation["X"]
    groups_validation = validation["groups"].astype(str)

    validation_df = pd.read_csv(VALIDATION_CSV)

    validation_df["dup_group"] = (
        validation_df["dup_group"].astype(str)
    )

    # Check group labels.
    label_map = {
        "cad_positive": 0,
        "negative": 1,
        "arrhythmia": 2
    }

    validation_df["numeric_label"] = (
        validation_df["label"].map(label_map)
    )

    if validation_df["numeric_label"].isna().any():
        raise ValueError("Unknown validation labels found.")

    group_labels = (
        validation_df.groupby("dup_group")["numeric_label"]
        .first()
        .to_dict()
    )

    # Select one feature vector per duplicate group.
    selected_indices = []
    selected_groups = []
    seen_groups = set()

    for index, group in enumerate(groups_validation):
        if group not in seen_groups:
            seen_groups.add(group)
            selected_indices.append(index)
            selected_groups.append(group)

    X_selected = X_validation[selected_indices]

    y_selected = np.array(
        [group_labels[group] for group in selected_groups],
        dtype=int
    )

    # Exclude arrhythmia from binary CAD calibration.
    binary_mask = y_selected != 2

    X_binary = X_selected[binary_mask]
    y_binary = (y_selected[binary_mask] == 0).astype(int)

    print("Validation groups:", len(selected_groups))
    print("CAD groups:", int(y_binary.sum()))
    print("Normal groups:", int((y_binary == 0).sum()))

    # Use the unchanged final SVM to produce validation scores.
    print("\nCalculating validation probabilities...")

    final_decision_scores = svm_model.decision_function(X_binary)

    validation_probabilities = multiclass_calibrator.predict_proba(
        final_decision_scores
    )

    classes = list(multiclass_calibrator.classes_)

    p_cad = validation_probabilities[:, classes.index(0)]
    p_normal = validation_probabilities[:, classes.index(1)]

    denominator = p_cad + p_normal

    if np.any(denominator <= 0):
        raise ValueError("Invalid CAD/normal probability denominator.")

    cad_score = p_cad / denominator

    eps = 1e-7
    cad_score = np.clip(cad_score, eps, 1 - eps)

    cad_logit = np.log(cad_score / (1 - cad_score)).reshape(-1, 1)

    print("\nFitting binary Platt calibrator...")

    binary_calibrator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000
    )

    binary_calibrator.fit(cad_logit, y_binary)

    # Save both calibration layers.
    ROOT.joinpath("models").mkdir(exist_ok=True)

    bundle = {
        "multiclass_calibrator": multiclass_calibrator,
        "binary_calibrator": binary_calibrator,
        "label_mapping": {
            0: "cad_positive",
            1: "negative",
            2: "arrhythmia"
        },
        "binary_score": "P(CAD) / (P(CAD) + P(normal))",
        "calibration_method": "Group-aware OOF multiclass + binary Platt",
        "validation_unit": "one image per duplicate group",
        "training_splits": N_SPLITS
    }

    joblib.dump(bundle, OUTPUT_PATH)

    print("\nCalibration completed successfully.")
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()