"""Train and evaluate the lab (tabular) branch.

    python scripts/train_tabular.py --config configs/tabular.yaml
"""

import argparse

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tabular.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # TODO (weeks 1-3):
    # 1. Load + harmonise data (cad_hybrid.data.tabular_loader)
    # 2. Split (cad_hybrid.data.splits)
    # 3. Baselines: logistic regression, XGBoost, LightGBM (cad_hybrid.tabular.baseline)
    # 4. Deep tabular (TabICL) + stack (cad_hybrid.tabular.deep_tabular / stack)
    # 5. Calibrate on validation set (cad_hybrid.fusion.calibrate)
    # 6. Metrics with bootstrap CIs (cad_hybrid.evaluation.metrics)
    # 7. Save model + calibrator to models/ (gitignored)
    raise SystemExit(f"Not implemented yet (config seed={config['seed']})")


if __name__ == "__main__":
    main()
