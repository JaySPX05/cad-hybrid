"""Re-choose the lab model's decision threshold without retraining.

    python scripts/choose_threshold.py
    python scripts/choose_threshold.py --method min_sensitivity --target 0.95

Uses the held-out predictions saved by scripts/train_tabular.py
(data/processed/oof_tabular.csv) and the saved model's calibrator. By default the rule
comes from final.threshold_method / final.threshold_target in configs/tabular.yaml;
--method and --target override it. Updates the saved model and
reports/threshold_tabular.md.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cad_hybrid.evaluation.thresholds import METHODS, threshold_summary, write_threshold_report
from cad_hybrid.tabular.artifact import PROJECT_ROOT, load_artifact, save_artifact

OOF_PATH = PROJECT_ROOT / "data" / "processed" / "oof_tabular.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "threshold_tabular.md"


def rechoose(config: dict, model_path: Path, oof_path: Path, report_path: Path,
             method: str | None = None, target: float | None = None) -> dict:
    """Choose a new threshold for the saved model; update it and the report. Returns metadata."""
    artifact = load_artifact(str(model_path))
    meta = artifact["metadata"]
    name = meta["model"]

    if not oof_path.exists():
        raise FileNotFoundError(f"No held-out predictions at {oof_path}. "
                                "Run python scripts/train_tabular.py first.")
    oof = pd.read_csv(oof_path)
    if name not in oof.columns or len(oof) != meta.get("n_patients"):
        raise ValueError(f"{oof_path} doesn't match the saved model ('{name}', "
                         f"{meta.get('n_patients')} patients). Re-run scripts/train_tabular.py.")

    final_cfg = config["final"]
    if method is None:
        method = final_cfg.get("threshold_method", "youden")
        target = final_cfg.get("threshold_target") if target is None else target

    y = oof["y"].to_numpy()
    calibrated = np.clip(artifact["calibrator"].transform(oof[name].to_numpy()), 0.0, 1.0)
    summary = threshold_summary(y, calibrated, method, target, seed=config.get("seed", 42))

    artifact = dict(artifact)
    artifact["threshold"] = summary["threshold"]
    artifact["metadata"] = {**meta,
                            "threshold": summary["threshold"],
                            "threshold_method": summary["method"],
                            "threshold_target": summary["target"],
                            "threshold_options": summary["options"]}
    save_artifact(artifact, model_path)
    write_threshold_report(summary, name, meta["dataset_id"], meta["n_patients"],
                           report_path, meta["trained_on"])
    return artifact["metadata"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/tabular.yaml")
    parser.add_argument("--method", choices=METHODS, default=None)
    parser.add_argument("--target", type=float, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    model_path = PROJECT_ROOT / config["final"]["output_path"]
    try:
        meta = rechoose(config, model_path, OOF_PATH, REPORT_PATH, args.method, args.target)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(str(exc))

    print(f"Model '{meta['model']}': threshold {meta['threshold']:.3f} "
          f"({meta['threshold_method']}"
          + (f", target {meta['threshold_target']}" if meta["threshold_target"] is not None else "")
          + ")")
    for o in meta["threshold_options"]:
        m = o["cv_metrics"]
        mark = "  <- chosen" if o["chosen"] else ""
        print(f"  {o['rule']:20s} t={o['threshold']:.3f}  sens {m['sensitivity']:.3f}  "
              f"spec {m['specificity']:.3f}  flagged {m['flagged']:.0%}{mark}")
    print(f"Updated {model_path} and {REPORT_PATH}")


if __name__ == "__main__":
    main()
