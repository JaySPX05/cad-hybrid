"""Tests for the tabular loader using small fake data (never real patient data)."""

import numpy as np
import pandas as pd
import pytest

from cad_hybrid.data.tabular_loader import (
    TARGET,
    apply_plausible_ranges,
    harmonise,
    load_dataset,
    to_number,
)

CONFIG = {
    "data": {
        "min_age": 18,
        "features": {"numeric": ["age", "bmi", "hemoglobin", "chloride"],
                     "categorical": ["sex_male", "chest_pain"]},
        "shared_features": ["age", "sex_male", "bp_mmhg", "total_cholesterol", "chest_pain"],
        "excluded_columns": {"unit": "leak", "troponin_i": "acute marker", "max_hr": "derived"},
        "plausible_ranges": {"age": [18, 110], "bp_mmhg": [60, 260], "hemoglobin": [3, 25]},
    }
}


def fake_bangladesh() -> pd.DataFrame:
    return pd.DataFrame({
        "SL": [1, 2, 3, 4],
        " Age": [73, 35, 12, 50],                 # note leading space; 12 = minor
        "Sex": ["M", "F", "F", " m"],
        "Height (cm)": [158, 163, 150, np.nan],
        "Weight (kg)": [75, 59, 40, 70],
        "BMI": [99.0, 22.2, 17.8, 25.0],          # 99 is wrong; recomputed from height/weight
        "Family H/O": [1, 1, 0, np.nan],
        "Hypertension": [1, np.nan, 0, 1],
        "Diabetes": [1, 0, 0, 0],
        "Total_Cholesterol(mg/dL)": [255, 194, 180, np.nan],
        "BP(mmHg)": [120, 130, 110, 999],          # 999 out of range
        "H/O ChestPain": [1, 0, 0, 1],
        "RBS(mmol/L)": [4.3, 5.3, 5.0, 6.1],
        "HDL(mg/dL)": [36, 46, 50, 40],
        "LDL(mg/dL)": [155, 107, 90, 130],
        "Triglycerides(mg/dL)": [220, 184, 100, 150],
        "MaxHR": [156.9, 175.2, 199.6, 173.0],
        "Himoglobin": ["12", "`2.6", 13.0, "11.5"],
        "Creatinine(mg/dL)": [1.5, 0.8, 0.7, 1.0],
        "Platelets": [180000, 174000, 250000, 300000],
        "Sodium(mmol/L)": [145, 141, 139, 138],
        "Potassium": [4.5, "4,3", 4.0, 4.1],
        "Chloride": ["106. 8", "106,4", 100, 101],
        "Troponin-I": [0.04, ">25000", "<2.50", 0.01],
        "Troponin- I assay type": ["Quantitative Troponin-I (ng/mL)"] * 4,
        "Heart Disease": [1, 0, 0, 1],
        "UNIT": ["CCU", "General", "General", "General"],
    })


def fake_india() -> pd.DataFrame:
    return pd.DataFrame({
        "patientid": [1, 2, 3], "age": [53, 40, 49], "gender": [1, 1, 0],
        "chestpain": [2, 0, 3], "restingBP": [171, 94, 133],
        "serumcholestrol": [0, 229, 142], "fastingbloodsugar": [0, 0, 1],
        "restingrelectro": [1, 1, 0], "maxheartrate": [147, 115, 202],
        "exerciseangia": [0, 0, 1], "oldpeak": [5.3, 3.7, 5.0],
        "slope": [3, 0, 1], "noofmajorvessels": [3, 1, 0], "target": [1, 0, 0],
    })


@pytest.mark.parametrize("raw, expected", [
    ("12", 12.0), ("`2.6", 2.6), ("4,3", 4.3), ("106. 8", 106.8), (5, 5.0),
    (">25000", None), ("<2.50", None), ("abc", None), (None, None),
])
def test_to_number(raw, expected):
    result = to_number(raw)
    if expected is None:
        assert np.isnan(result)
    else:
        assert result == pytest.approx(expected)


def test_plausible_ranges_blank_out_impossible_values():
    df = pd.DataFrame({"bp_mmhg": [120, 999, 50]})
    out = apply_plausible_ranges(df, {"bp_mmhg": [60, 260]})
    assert out["bp_mmhg"].isna().tolist() == [False, True, True]


def test_bangladesh_harmonise():
    out = harmonise(fake_bangladesh(), "bangladesh_cardiac", CONFIG)
    assert len(out) == 3                                   # minor (age 12) removed
    assert out["age"].min() >= 18
    assert out["sex_male"].tolist() == [1, 0, 1]           # " m" handled
    assert out.loc[0, "bmi"] == pytest.approx(75 / 1.58 ** 2)   # recomputed
    assert out.loc[2, "bmi"] == pytest.approx(25.0)        # kept when height missing
    assert np.isnan(out.loc[1, "hemoglobin"])              # "`2.6" parsed, then < 3 g/dL blanked
    assert out.loc[1, "potassium"] == pytest.approx(4.3)
    assert out.loc[0, "chloride"] == pytest.approx(106.8)
    assert np.isnan(out.loc[2, "bp_mmhg"])                 # 999 blanked
    assert np.isnan(out.loc[1, "troponin_i"])              # censored ">25000"
    assert out[TARGET].tolist() == [1, 0, 1]


def test_india_harmonise():
    out = harmonise(fake_india(), "mendeley_india_cvd", CONFIG)
    assert np.isnan(out.loc[0, "total_cholesterol"])       # 0 -> missing
    assert np.isnan(out.loc[1, "st_slope"])                # undocumented 0 -> missing
    assert out["chest_pain"].tolist() == [1, 1, 0]         # 3 = asymptomatic -> 0


def test_load_dataset_uses_config_features_and_blocks_leaks(tmp_path):
    folder = tmp_path / "bangladesh_cardiac"
    folder.mkdir()
    fake_bangladesh().to_excel(folder / "data.xlsx", index=False)

    X, y = load_dataset("bangladesh_cardiac", CONFIG, raw_dir=tmp_path)
    assert list(X.columns) == ["age", "bmi", "hemoglobin", "chloride", "sex_male", "chest_pain"]
    assert len(X) == len(y) == 3

    with pytest.raises(ValueError, match="Excluded"):
        load_dataset("bangladesh_cardiac", CONFIG, features=["age", "unit"], raw_dir=tmp_path)


def test_shared_features_available_in_both(tmp_path):
    for name, frame, ext in [("bangladesh_cardiac", fake_bangladesh(), "xlsx"),
                             ("mendeley_india_cvd", fake_india(), "csv")]:
        folder = tmp_path / name
        folder.mkdir()
        if ext == "xlsx":
            frame.to_excel(folder / f"d.{ext}", index=False)
        else:
            frame.to_csv(folder / f"d.{ext}", index=False)
        X, _ = load_dataset(name, CONFIG, features=CONFIG["data"]["shared_features"],
                            raw_dir=tmp_path)
        assert list(X.columns) == CONFIG["data"]["shared_features"]


def test_unknown_dataset_rejected():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("nope", CONFIG)
