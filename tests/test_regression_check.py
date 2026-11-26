"""The regression gate (scripts/check_regression.py) on fake reports."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_regression", ROOT / "scripts" / "check_regression.py")
check_regression = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_regression)


def setup(tmp_path, auroc, brier):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "ecg.json").write_text(json.dumps({"metrics": {"auroc": auroc}}))
    pd.DataFrame([{"model": "m", "metric": "brier", "value": brier}]).to_csv(
        tmp_path / "reports" / "lab.csv", index=False)
    config = {"checks": [
        {"agent": "ecg", "name": "auroc", "report": "reports/ecg.json", "json": "metrics.auroc",
         "baseline": 0.95, "better": "higher", "allowed_change": 0.03},
        {"agent": "lab", "name": "brier", "report": "reports/lab.csv",
         "csv": {"model": "m", "metric": "brier", "column": "value"},
         "baseline": 0.03, "better": "lower", "allowed_change": 0.01},
    ]}
    path = tmp_path / "baseline.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_results_within_tolerance_pass(tmp_path):
    assert check_regression.run(setup(tmp_path, 0.93, 0.035), tmp_path) == []


def test_worse_results_fail(tmp_path):
    failures = check_regression.run(setup(tmp_path, 0.90, 0.05), tmp_path)
    assert len(failures) == 2 and "worse than allowed" in failures[0]


def test_missing_report_fails(tmp_path):
    path = setup(tmp_path, 0.95, 0.03)
    (tmp_path / "reports" / "ecg.json").unlink()
    assert any("report missing" in f for f in check_regression.run(path, tmp_path))


def test_repo_reports_meet_their_baselines():
    assert check_regression.run() == []


def test_bad_direction_rejected():
    with pytest.raises(ValueError):
        check_regression.evaluate({"better": "sideways", "baseline": 1, "allowed_change": 0}, 1.0)
