
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from cad_hybrid.ecg.scoring import cad_vs_normal_score, fit_binary_calibrator


ROOT = Path(__file__).resolve().parents[3]

# All files are loaded from the cad-hybrid project.
TRAIN_FEATURES = ROOT / "data" / "features" / "train_features.npz"
VALIDATION_FEATURES = ROOT / "data" / "features" / "validation_features.npz"
VALIDATION_CSV = ROOT / "data" / "processed" / "validation.csv"

SVM_PATH = ROOT / "models" / "ecg_svm_classifier.pkl"
OUTPUT_PATH = ROOT / "models" / "ecg_calibration_bundle.pkl"

RANDOM_STATE = 42
N_SPLITS = 5


def check_required_files():
    required_files = [
        TRAIN_FEATURES,
        VALIDATION_FEATURES,
        VALIDATION_CSV,
        SVM_PATH,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )


def main():
    check_required_files()

    print("Project root:", ROOT)
    print("\nLoading local training features...")

    train = np.load(TRAIN_FEATURES)

    X_train = train["X"]
    y_train = train["y"].astype(int)
    groups_train = train["groups"].astype(str)

    if not (len(X_train) == len(y_train) == len(groups_train)):
        raise ValueError("Training feature, label and group counts do not match.")

    print("Training feature shape:", X_train.shape)
    print("Training samples:", len(y_train))
    print("Training groups:", len(np.unique(groups_train)))

    print("\nLoading local trained SVM...")
    svm_model = joblib.load(SVM_PATH)

    print("SVM classes:", svm_model.classes_)

    if not hasattr(svm_model, "decision_function"):
        raise ValueError("The SVM model must support decision_function.")

    # Generate group-aware out-of-fold decision scores.
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

        fold_model.fit(
            X_train[train_idx],
            y_train[train_idx]
        )

        scores = fold_model.decision_function(
            X_train[valid_idx]
        )

        oof_scores[valid_idx] = scores
        assigned[valid_idx] = True

        print(
            f"Fold {fold}/{N_SPLITS}: "
            f"training={len(train_idx)}, "
            f"validation={len(valid_idx)}"
        )

    if not assigned.all():
        raise RuntimeError(
            "Some training samples did not receive OOF scores."
        )

    print("\nFitting multiclass calibration on OOF scores...")

    # Give each duplicate group equal total weight.
    _, inverse, counts = np.unique(
        groups_train,
        return_inverse=True,
        return_counts=True
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

    # Load local validation features.
    print("\nLoading local validation features...")

    validation = np.load(VALIDATION_FEATURES)

    X_validation = validation["X"]
    groups_validation = validation["groups"].astype(str)

    if len(X_validation) != len(groups_validation):
        raise ValueError(
            "Validation feature and group counts do not match."
        )

    validation_df = pd.read_csv(VALIDATION_CSV)

    required_columns = {"label", "dup_group"}

    if not required_columns.issubset(validation_df.columns):
        raise ValueError(
            "Validation CSV must contain label and dup_group columns."
        )

    validation_df["dup_group"] = (
        validation_df["dup_group"].astype(str)
    )

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

    missing_groups = set(selected_groups) - set(group_labels)

    if missing_groups:
        raise ValueError(
            f"Validation groups missing from CSV: {len(missing_groups)}"
        )

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
    print("Arrhythmia groups excluded:", int((y_selected == 2).sum()))

    if len(np.unique(y_binary)) != 2:
        raise ValueError(
            "Binary calibration requires both CAD and normal groups."
        )

    # Use the unchanged final SVM to produce validation scores.
    print("\nCalculating validation decision scores...")

    final_decision_scores = svm_model.decision_function(X_binary)

    validation_probabilities = multiclass_calibrator.predict_proba(
        final_decision_scores
    )

    # CAD score = P(CAD) / (P(CAD) + P(normal)), then binary Platt calibration.
    cad_score = cad_vs_normal_score(
        validation_probabilities, multiclass_calibrator.classes_
    )

    print("\nFitting binary Platt calibrator...")

    binary_calibrator = fit_binary_calibrator(cad_score, y_binary)

    # The calibrated probabilities reflect this CAD rate (one ECG per group);
    # fusion needs it as the ECG branch prior (configs/fusion.yaml).
    calibration_positive_rate = float(y_binary.mean())
    print(f"Calibration CAD rate (fusion prior): {calibration_positive_rate:.4f}")

    # Save both calibration layers locally.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

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
        "calibration_positive_rate": calibration_positive_rate,
        "training_splits": N_SPLITS
    }

    joblib.dump(bundle, OUTPUT_PATH)

    print("\nCalibration completed successfully.")
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
