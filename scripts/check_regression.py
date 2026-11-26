"""Fail if any agent's committed results are worse than its baseline.

    python scripts/check_regression.py

Reads configs/regression_baseline.yaml and the result reports it points to (written by
the training and evaluation scripts, and committed). Prints each check; exits with an
error if a value got worse than allowed, or a report or value is missing. Runs in CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "regression_baseline.yaml"


def read_value(check: dict, root: Path = ROOT) -> float:
    """The current value for one check, from its report file."""
    path = root / check["report"]
    if not path.exists():
        raise FileNotFoundError(f"report missing: {check['report']}")
    if "csv" in check:
        spec = check["csv"]
        df = pd.read_csv(path)
        rows = df[(df["model"] == spec["model"]) & (df["metric"] == spec["metric"])]
        if len(rows) != 1:
            raise KeyError(f"{check['report']}: no single row for {spec['model']} / {spec['metric']}")
        return float(rows.iloc[0][spec["column"]])
    value = json.loads(path.read_text(encoding="utf-8"))
    for part in check["json"].split("."):
        value = value[part]
    return float(value)


def evaluate(check: dict, value: float) -> tuple[bool, float]:
    """``(ok, limit)``: whether value is within the allowed change from the baseline."""
    if check["better"] == "higher":
        limit = check["baseline"] - check["allowed_change"]
        return value >= limit, limit
    if check["better"] == "lower":
        limit = check["baseline"] + check["allowed_change"]
        return value <= limit, limit
    raise ValueError(f"better must be 'higher' or 'lower', got {check['better']!r}")


def run(config_path: Path = CONFIG, root: Path = ROOT) -> list[str]:
    """Check every entry; print a line each and return the failures."""
    checks = (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}).get("checks", [])
    failures = []
    for check in checks:
        label = f"{check['agent']}: {check['name']}"
        try:
            value = read_value(check, root)
        except (FileNotFoundError, KeyError, ValueError, TypeError) as exc:
            failures.append(f"{label}: {exc}")
            print(f"FAIL {label}: {exc}")
            continue
        ok, limit = evaluate(check, value)
        sign = ">=" if check["better"] == "higher" else "<="
        print(f"{'ok  ' if ok else 'FAIL'} {label}: {value:.3f} (baseline {check['baseline']:.3f}, "
              f"must be {sign} {limit:.3f})")
        if not ok:
            failures.append(f"{label}: {value:.3f} is worse than allowed ({sign} {limit:.3f})")
    return failures


def main() -> None:
    failures = run()
    if failures:
        sys.exit("Regression check failed:\n- " + "\n- ".join(failures))
    print("All agents at or above their baselines.")


if __name__ == "__main__":
    main()
