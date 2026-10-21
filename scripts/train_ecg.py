"""Train and evaluate the ECG (image) branch.

    python scripts/train_ecg.py --config configs/ecg.yaml
"""

import argparse

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ecg.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # TODO (weeks 1-3):
    # 1. Index images + apply label map (cad_hybrid.data.ecg_loader)
    # 2. Near-duplicate groups (cad_hybrid.data.dedupe), then group-aware split
    # 3. Two-stage CNN fine-tuning (cad_hybrid.ecg.cnn)
    # 4. Embeddings -> XGBoost/SVM head; compare with plain CNN (cad_hybrid.ecg.embed_gbm)
    # 5. Calibrate, evaluate, cross-country test Pakistan -> India
    # 6. Save weights to models/ (gitignored)
    raise SystemExit(f"Not implemented yet (backbone={config['model']['backbone']})")


if __name__ == "__main__":
    main()
