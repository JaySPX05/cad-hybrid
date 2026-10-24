"""Load and harmonise the lab (tabular) datasets.

Each source dataset has its own column names, codings and quirks. This module maps
them onto one shared schema (snake_case names, numeric values, binary flags as 0/1)
so models can be trained on one dataset and tested on another.

Decisions behind the cleaning rules are documented in reports/data_audit_tabular.md.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
TARGET = "heart_disease"

# ---------------------------------------------------------------------------
# Column maps: original name -> shared-schema name
# ---------------------------------------------------------------------------

BANGLADESH_COLUMNS = {
    "SL": "record_id",
    "Age": "age",
    "Sex": "sex",
    "Height (cm)": "height_cm",
    "Weight (kg)": "weight_kg",
    "BMI": "bmi",
    "Family H/O": "family_history",
    "Hypertension": "hypertension",
    "Diabetes": "diabetes",
    "Total_Cholesterol(mg/dL)": "total_cholesterol",
    "BP(mmHg)": "bp_mmhg",
    "H/O ChestPain": "chest_pain",
    "RBS(mmol/L)": "random_blood_sugar",
    "HDL(mg/dL)": "hdl",
    "LDL(mg/dL)": "ldl",
    "Triglycerides(mg/dL)": "triglycerides",
    "MaxHR": "max_hr",
    "Himoglobin": "hemoglobin",
    "Creatinine(mg/dL)": "creatinine",
    "Platelets": "platelets",
    "Sodium(mmol/L)": "sodium",
    "Potassium": "potassium",
    "Chloride": "chloride",
    "Troponin-I": "troponin_i",
    "Troponin- I assay type": "troponin_assay",
    "Heart Disease": TARGET,
    "UNIT": "unit",
}

INDIA_COLUMNS = {
    "patientid": "record_id",
    "age": "age",
    "gender": "sex_male",               # 1 = male, 0 = female
    "chestpain": "chest_pain_type",     # 0 typical, 1 atypical, 2 non-anginal, 3 asymptomatic
    "restingBP": "bp_mmhg",
    "serumcholestrol": "total_cholesterol",
    "fastingbloodsugar": "fasting_blood_sugar_high",  # 1 = >120 mg/dL
    "restingrelectro": "resting_ecg",
    "maxheartrate": "max_heart_rate",
    "exerciseangia": "exercise_angina",
    "oldpeak": "oldpeak",
    "slope": "st_slope",
    "noofmajorvessels": "n_major_vessels",
    "target": TARGET,
}

DATASETS = {
    "bangladesh_cardiac": {"pattern": "*.xlsx", "columns": BANGLADESH_COLUMNS},
    "mendeley_india_cvd": {"pattern": "*.csv", "columns": INDIA_COLUMNS},
}

# ---------------------------------------------------------------------------
# Generic cleaning helpers
# ---------------------------------------------------------------------------


def to_number(value) -> float:
    """Convert messy spreadsheet entries to a float.

    Handles typos seen in the data: stray backticks ("`2.6"), decimal commas ("4,3"),
    spaces inside numbers ("106. 8"). Censored values like ">25000" or "<2.50" and
    anything else unparseable become NaN.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().replace("`", "").replace(" ", "").replace(",", ".")
    if not re.fullmatch(r"-?\d+(\.\d+)?", text):
        return np.nan
    return float(text)


def apply_plausible_ranges(df: pd.DataFrame, ranges: dict) -> pd.DataFrame:
    """Set values outside clinically plausible ranges to NaN."""
    out = df.copy()
    for col, (lo, hi) in ranges.items():
        if col in out.columns:
            bad = (out[col] < lo) | (out[col] > hi)
            out.loc[bad, col] = np.nan
    return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_raw(dataset_id: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read a raw dataset file from ``data/raw/<dataset_id>/`` exactly as downloaded."""
    if dataset_id not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_id}'. Known: {sorted(DATASETS)}")
    folder = raw_dir / dataset_id
    files = sorted(folder.glob(DATASETS[dataset_id]["pattern"]))
    if not files:
        raise FileNotFoundError(f"No data file in {folder}; see data/README.md")
    path = files[0]
    if path.suffix == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Dataset-specific harmonisation
# ---------------------------------------------------------------------------


def harmonise_bangladesh(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the Northern Bangladesh dataset into the shared schema.

    Keeps leaky / derived columns (unit, troponin, max_hr) so they can be inspected;
    they are removed later by ``select_features`` via ``excluded_columns``.
    """
    out = df.rename(columns=lambda c: c.strip()).rename(columns=BANGLADESH_COLUMNS)
    numeric_cols = [c for c in out.columns if c not in {"sex", "troponin_assay", "unit"}]
    for col in numeric_cols:
        out[col] = out[col].map(to_number)

    out["sex_male"] = out["sex"].str.strip().str.upper().map({"M": 1, "F": 0})
    out = out.drop(columns=["sex"])

    # Recompute BMI from height and weight where both exist (the given BMI
    # disagrees with height/weight in ~9% of rows).
    bmi = out["weight_kg"] / (out["height_cm"] / 100) ** 2
    out["bmi"] = bmi.where(bmi.notna(), out["bmi"])

    out[TARGET] = out[TARGET].astype(int)
    return out


def harmonise_india(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the Mendeley India dataset into the shared schema (pipeline testing only)."""
    out = df.rename(columns=INDIA_COLUMNS)
    # Missing cholesterol is coded as 0 (all 53 such rows are disease-positive: a leak).
    out["total_cholesterol"] = out["total_cholesterol"].replace(0, np.nan)
    # st_slope = 0 is not in the dataset description (documented values 1-3).
    out["st_slope"] = out["st_slope"].where(out["st_slope"].isin([1, 2, 3]))
    # Binary chest pain comparable with Bangladesh "history of chest pain":
    # any chest pain type except 3 (asymptomatic).
    out["chest_pain"] = (out["chest_pain_type"] != 3).astype(int)
    out[TARGET] = out[TARGET].astype(int)
    return out


HARMONISERS = {
    "bangladesh_cardiac": harmonise_bangladesh,
    "mendeley_india_cvd": harmonise_india,
}


def harmonise(df: pd.DataFrame, dataset_id: str, config: dict) -> pd.DataFrame:
    """Dataset-specific cleaning, then shared rules: plausible ranges and adults only."""
    out = HARMONISERS[dataset_id](df)
    data_cfg = config["data"]
    out = apply_plausible_ranges(out, data_cfg.get("plausible_ranges", {}))
    min_age = data_cfg.get("min_age")
    if min_age is not None:
        out = out[out["age"] >= min_age]
    out = out.dropna(subset=["age", TARGET])
    out.insert(0, "dataset_id", dataset_id)
    return out.reset_index(drop=True)


def select_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Keep only the requested shared-schema features; missing ones become all-NaN."""
    return df.reindex(columns=features)


def load_dataset(
    dataset_id: str, config: dict, features: list[str] | None = None, raw_dir: Path = RAW_DIR
) -> tuple[pd.DataFrame, pd.Series]:
    """Return features ``X`` and target ``y`` for one dataset in the shared schema.

    ``features`` defaults to the training features in the config (numeric + categorical).
    Pass ``config["data"]["shared_features"]`` for cross-dataset tests.
    """
    df = harmonise(load_raw(dataset_id, raw_dir), dataset_id, config)
    if features is None:
        f = config["data"]["features"]
        features = f["numeric"] + f["categorical"]
    excluded = set(config["data"].get("excluded_columns", {}))
    leaked = excluded.intersection(features)
    if leaked:
        raise ValueError(f"Excluded columns requested as features: {sorted(leaked)}")
    return select_features(df, features), df[TARGET]
