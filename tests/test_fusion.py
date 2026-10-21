import pytest

from cad_hybrid.fusion.fuse import fuse, risk_category


def test_no_branches_returns_prior():
    assert fuse({"tabular": None, "ecg": None}, prior=0.3) == pytest.approx(0.3)


def test_single_branch_returns_its_probability():
    assert fuse({"tabular": 0.7, "ecg": None}, prior=0.3) == pytest.approx(0.7)


def test_agreeing_branches_strengthen_evidence():
    both = fuse({"tabular": 0.7, "ecg": 0.7}, prior=0.3)
    assert both > 0.7


def test_zero_weight_ignores_branch():
    p = fuse({"tabular": 0.7, "ecg": 0.95}, prior=0.3, weights={"ecg": 0.0})
    assert p == pytest.approx(0.7)


def test_risk_category_bounds():
    cats = {"low": [0.0, 0.1], "moderate": [0.1, 0.2], "high": [0.2, 1.0]}
    assert risk_category(0.05, cats) == "low"
    assert risk_category(0.1, cats) == "moderate"
    assert risk_category(1.0, cats) == "high"
