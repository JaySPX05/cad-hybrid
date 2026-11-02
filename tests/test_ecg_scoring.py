"""Tests for the ECG CAD score, its calibration and the ECG prior (fake data only)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from cad_hybrid.ecg.save_metadata import build_metadata
from cad_hybrid.ecg.scoring import (
    ARRHYTHMIA, CAD, NORMAL, apply_binary_calibrator, cad_vs_normal, cad_vs_normal_score,
    fit_binary_calibrator, one_per_group, positive_rate_per_ecg, score_logit)

ROOT = Path(__file__).resolve().parents[1]


def test_score_ignores_arrhythmia():
    probs = np.array([[0.2, 0.2, 0.6], [0.6, 0.2, 0.2]])
    assert cad_vs_normal_score(probs, [CAD, NORMAL, ARRHYTHMIA]) == pytest.approx([0.5, 0.75])


def test_score_follows_class_order():
    # classes listed as normal, arrhythmia, CAD
    probs = np.array([[0.1, 0.6, 0.3]])
    assert cad_vs_normal_score(probs, [NORMAL, ARRHYTHMIA, CAD]) == pytest.approx([0.75])


def test_score_rejects_zero_cad_and_normal():
    with pytest.raises(ValueError, match="sum to zero"):
        cad_vs_normal_score([[0.0, 0.0, 1.0]], [CAD, NORMAL, ARRHYTHMIA])


def test_logit_is_finite_at_the_edges():
    assert np.isfinite(score_logit([0.0, 0.5, 1.0])).all()
    assert score_logit([0.5])[0, 0] == pytest.approx(0.0)


def fake_scores(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    score = 1 / (1 + np.exp(-(rng.normal(0, 1, n) + 2.0 * (y - 0.5))))
    return score, y


def test_calibration_gives_probabilities_in_score_order():
    score, y = fake_scores()
    calibrator = fit_binary_calibrator(score, y)
    p = apply_binary_calibrator(calibrator, score)
    assert ((p > 0) & (p < 1)).all()
    order = np.argsort(score)
    assert np.all(np.diff(p[order]) >= -1e-12)  # monotone: ranking (AUROC) unchanged


def test_calibration_matches_the_cad_rate_on_average():
    score, y = fake_scores()
    p = apply_binary_calibrator(fit_binary_calibrator(score, y), score)
    assert p.mean() == pytest.approx(y.mean(), abs=0.02)


def test_calibration_needs_both_classes():
    with pytest.raises(ValueError, match="both CAD and normal"):
        fit_binary_calibrator([0.2, 0.7, 0.9], [1, 1, 1])


def split_table():
    """Two CAD ECGs (one copied 8 times), three normal ECGs, two arrhythmia ECGs."""
    rows = [("mi_a_%d.jpg" % i, "cad_positive", "g1") for i in range(8)]
    rows += [("mi_b.jpg", "cad_positive", "g2")]
    rows += [("n%d.jpg" % i, "negative", "g%d" % (3 + i)) for i in range(3)]
    rows += [("ar%d.jpg" % i, "arrhythmia", "g%d" % (6 + i)) for i in range(2)]
    return pd.DataFrame(rows, columns=["path", "label", "dup_group"])


def test_one_image_per_duplicate_group():
    df = split_table()
    assert len(one_per_group(df)) == df["dup_group"].nunique() == 7


def test_cad_vs_normal_drops_arrhythmia_and_copies():
    kept, y = cad_vs_normal(split_table())
    assert set(kept["label"]) == {"cad_positive", "negative"}
    assert list(np.sort(y)) == [0, 0, 0, 1, 1]


def test_positive_rate_counts_ecgs_not_images():
    rate = positive_rate_per_ecg(split_table())
    assert (rate["cad_ecgs"], rate["normal_ecgs"]) == (2, 3)
    assert rate["positive_rate"] == pytest.approx(0.4)  # per image it would be 9/12 = 0.75


def test_fusion_prior_is_the_calibration_set_rate():
    train, validation = split_table(), split_table().iloc[8:]  # validation without copies
    meta = build_metadata(train, validation)
    assert meta["fusion_prior"] == pytest.approx(positive_rate_per_ecg(validation)["positive_rate"],
                                                 abs=1e-4)
    assert meta["training_images"]["cad_positive"] == 9


def test_fusion_config_uses_the_saved_ecg_prior():
    with open(ROOT / "configs" / "fusion.yaml", encoding="utf-8") as f:
        fusion = yaml.safe_load(f)
    metadata = json.loads((ROOT / "reports" / "ecg_metadata.json").read_text(encoding="utf-8"))
    assert fusion["branch_priors"]["ecg"] == pytest.approx(metadata["fusion_prior"])
