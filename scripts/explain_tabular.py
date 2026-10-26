"""Explain the saved lab model with SHAP: which lab values it relies on most.

    python scripts/explain_tabular.py
    python scripts/explain_tabular.py --n-patients 20      # quicker

Explains a random sample of training patients (stratified by outcome) and writes
reports/shap_tabular.md and reports/figures/shap_importance_tabular.png. Also stores the
explanation background in the saved model, so models trained before explanations
existed work in the app without retraining.

Slow with TabICL on CPU: the time for the first patient is printed with an estimate
for the rest. Settings are in configs/tabular.yaml under ``explain``.
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from cad_hybrid.data.tabular_loader import load_dataset
from cad_hybrid.explain.shap_tabular import (global_importance, make_background,
                                             plot_importance, shap_values, write_report)
from cad_hybrid.tabular.artifact import PROJECT_ROOT, load_artifact, save_artifact

REPORT = PROJECT_ROOT / "reports" / "shap_tabular.md"
FIGURE = PROJECT_ROOT / "reports" / "figures" / "shap_importance_tabular.png"


def sample_patients(X: pd.DataFrame, y: pd.Series, n: int, seed: int) -> pd.DataFrame:
    if n >= len(X):
        return X
    sample, _ = train_test_split(X, train_size=n, stratify=y, random_state=seed)
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/tabular.yaml")
    parser.add_argument("--n-patients", type=int, default=None)
    parser.add_argument("--nsamples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    cfg = config.get("explain", {})
    seed = config.get("seed", 42)
    settings = {
        "background_size": cfg.get("background_size", 10),
        "nsamples": args.nsamples or cfg.get("nsamples", 200),
    }
    n_patients = args.n_patients or cfg.get("report_patients", 50)

    model_path = PROJECT_ROOT / config["final"]["output_path"]
    try:
        artifact = dict(load_artifact(str(model_path)))
    except FileNotFoundError as exc:
        sys.exit(str(exc))
    meta = artifact["metadata"]

    dataset_id = config["data"]["train_sources"][0]
    X, y = load_dataset(dataset_id, config)
    if list(X.columns) != artifact["features"] or len(y) != meta.get("n_patients"):
        sys.exit("The saved model doesn't match the current data/config. "
                 "Re-run scripts/train_tabular.py first.")

    # (Re)build the background from the current settings and store it with the model.
    artifact["explain_background"] = make_background(
        X, config["data"]["features"]["categorical"], settings["background_size"], seed)
    artifact["explain_nsamples"] = settings["nsamples"]
    save_artifact(artifact, model_path)
    print(f"Stored explanation background ({settings['background_size']} centres) in {model_path}")

    patients = sample_patients(X, y, n_patients, seed)
    print(f"Explaining {len(patients)} patients with '{meta['model']}' "
          f"({settings['nsamples']} samples each)...")

    start = time.perf_counter()
    first, baseline = shap_values(artifact, patients.iloc[:1], settings["nsamples"], seed)
    per_patient = time.perf_counter() - start
    print(f"  first patient: {per_patient:.1f} s; estimated total "
          f"{per_patient * len(patients) / 60:.1f} min")

    chunks = [first]
    for i in range(1, len(patients), 10):
        values, _ = shap_values(artifact, patients.iloc[i:i + 10], settings["nsamples"], seed)
        chunks.append(values)
        print(f"  {min(i + 10, len(patients))}/{len(patients)} done")
    values = np.vstack(chunks)

    importance = global_importance(values, patients)
    figure = plot_importance(importance, FIGURE)
    write_report(importance, REPORT, meta["model"], dataset_id, len(patients), settings,
                 f"figures/{FIGURE.name}", time.strftime("%Y-%m-%d"))

    print(f"\nBaseline (average predicted risk): {baseline:.3f}")
    print("Most influential lab values (average effect, percentage points):")
    for _, row in importance.head(10).iterrows():
        print(f"  {row['feature']:20s} {row['mean_abs_shap'] * 100:5.1f}")
    print(f"Saved:\n  {REPORT}\n  {figure}")


if __name__ == "__main__":
    main()
