"""Tests for choosing and evaluating the decision threshold (fake data only)."""

import numpy as np
import pytest

from cad_hybrid.evaluation.thresholds import (
    STANDARD_RULES, choose_threshold, cross_validated_metrics, metrics_at, threshold_summary,
    write_threshold_report)


def fake_scores(n: int = 400, seed: int = 0):
    """Overlapping scores: sick patients score higher on average."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = 1 / (1 + np.exp(-(rng.normal(0, 1, n) + 1.5 * (y - 0.5))))
    return y, p


def test_metrics_at_counts_correctly():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([.9, .8, .2, .7, .3, .2, .1, .1])
    m = metrics_at(y, p, 0.5)  # flagged: .9, .8 (sick) and .7 (healthy)
    assert m["sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(4 / 5)
    assert m["ppv"] == pytest.approx(2 / 3)
    assert m["npv"] == pytest.approx(4 / 5)
    assert m["flagged"] == pytest.approx(3 / 8)


def test_youden_separates_perfectly_separable_scores():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([.1, .2, .3, .6, .7, .8])
    t = choose_threshold(y, p, "youden")
    assert 0.3 < t <= 0.6
    assert metrics_at(y, p, t)["sensitivity"] == 1 and metrics_at(y, p, t)["specificity"] == 1


def test_min_sensitivity_reaches_target_with_highest_threshold():
    y, p = fake_scores()
    t = choose_threshold(y, p, "min_sensitivity", 0.9)
    assert metrics_at(y, p, t)["sensitivity"] >= 0.9
    # any higher score used as threshold would miss the target
    higher = np.unique(p[p > t])
    assert all(metrics_at(y, p, h)["sensitivity"] < 0.9 for h in higher[:5])


def test_stricter_sensitivity_target_gives_lower_threshold():
    y, p = fake_scores()
    assert choose_threshold(y, p, "min_sensitivity", 0.95) <= choose_threshold(
        y, p, "min_sensitivity", 0.80)


def test_fixed_returns_target():
    y, p = fake_scores()
    assert choose_threshold(y, p, "fixed", 0.37) == 0.37


@pytest.mark.parametrize("method,target", [("nope", None), ("fixed", None),
                                           ("min_sensitivity", None), ("min_sensitivity", 1.5)])
def test_bad_rules_rejected(method, target):
    y, p = fake_scores()
    with pytest.raises(ValueError):
        choose_threshold(y, p, method, target)


def test_cross_validated_metrics_are_valid():
    y, p = fake_scores()
    m = cross_validated_metrics(y, p, "youden")
    assert set(m) == {"sensitivity", "specificity", "ppv", "npv", "flagged"}
    assert all(0 <= v <= 1 for v in m.values())


def test_cross_validated_fixed_rule_equals_plain_metrics():
    y, p = fake_scores()
    assert cross_validated_metrics(y, p, "fixed", 0.5) == pytest.approx(metrics_at(y, p, 0.5))


def test_summary_marks_chosen_rule_and_adds_custom_ones():
    y, p = fake_scores()
    s = threshold_summary(y, p, "min_sensitivity", 0.85)
    assert len(s["options"]) == len(STANDARD_RULES) + 1
    chosen = [o for o in s["options"] if o["chosen"]]
    assert len(chosen) == 1 and chosen[0]["threshold"] == s["threshold"]
    assert s["prevalence"] == pytest.approx(y.mean(), abs=1e-3)


def test_report_written(tmp_path):
    y, p = fake_scores()
    path = tmp_path / "threshold.md"
    write_threshold_report(threshold_summary(y, p), "fake", "fake_ds", len(y), path, "2026-01-01")
    text = path.read_text(encoding="utf-8")
    assert "**youden (chosen)**" in text and "overstate" in text
