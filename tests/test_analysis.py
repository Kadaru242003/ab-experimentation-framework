"""Tests for the existing analysis functions in src/."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as t_dist
from statsmodels.stats.proportion import proportions_ztest

from decision import make_decision
from metrics import compute_metrics
from peeking_bias import simulate_peeking
from stat_tests import conversion_z_test, revenue_t_test
from time_analysis import conversion_over_time
from validation import check_srm


def binary(successes, n):
    return pd.Series([1] * successes + [0] * (n - successes))


# --- Sample ratio mismatch -----------------------------------------------------

def make_split(n_control, n_treatment):
    return pd.DataFrame({"group": ["control"] * n_control + ["treatment"] * n_treatment})


def test_srm_balanced_split_passes():
    result = check_srm(make_split(50_000, 50_000))
    assert result["p_value"] == pytest.approx(1.0)
    assert not result["srm_detected"]


def test_srm_broken_split_detected():
    # chi2 = 2 * 1000^2 / 50000 = 40 on 1 df -> p ~ 2.5e-10
    result = check_srm(make_split(51_000, 49_000))
    assert result["p_value"] == pytest.approx(2.539e-10, rel=1e-3)
    assert result["srm_detected"]


def test_srm_small_imbalance_within_noise():
    result = check_srm(make_split(5_030, 4_970))
    assert not result["srm_detected"]


def test_srm_default_threshold_is_strict():
    # 50,500 / 49,500: chi2 = 2 * 500^2 / 50000 = 10, p ~ 0.0016
    df = make_split(50_500, 49_500)
    assert 0.001 < check_srm(df)["p_value"] < 0.05
    assert not check_srm(df)["srm_detected"]
    assert check_srm(df, threshold=0.05)["srm_detected"]


def test_generated_split_passes_srm():
    # Counts from the committed simulation (seed 42): p ~ 0.22.
    result = check_srm(make_split(149_666, 150_334))
    assert result["p_value"] == pytest.approx(0.2226, abs=1e-4)
    assert not result["srm_detected"]


# --- Two-proportion z-test ---------------------------------------------------

@pytest.mark.parametrize("x1, n1, x2, n2", [
    (1000, 10_000, 1100, 10_000),
    (1100, 10_000, 1000, 10_000),
    (50, 400, 70, 450),
])
def test_conversion_z_test_matches_statsmodels(x1, n1, x2, n2):
    control, treatment = binary(x1, n1), binary(x2, n2)
    z, p = proportions_ztest([x2, x1], [n2, n1])  # pooled, two-sided

    result = conversion_z_test(control, treatment)
    assert result["z_stat"] == pytest.approx(z)
    assert result["p_value"] == pytest.approx(p)
    assert result["lift"] == pytest.approx(x2 / n2 - x1 / n1)


def test_conversion_z_test_is_two_sided():
    # A significantly worse treatment must give a small p-value, not ~1.
    worse = conversion_z_test(binary(1100, 10_000), binary(1000, 10_000))
    better = conversion_z_test(binary(1000, 10_000), binary(1100, 10_000))
    assert worse["p_value"] == pytest.approx(better["p_value"])
    assert worse["p_value"] < 0.05
    assert worse["lift"] < 0


# --- Welch's t-test -----------------------------------------------------------

def test_revenue_t_test_matches_welch_formula():
    rng = np.random.default_rng(0)
    control = pd.Series(rng.normal(10, 2, size=200))
    treatment = pd.Series(rng.normal(11, 6, size=50))

    v1, v2 = control.var(ddof=1) / len(control), treatment.var(ddof=1) / len(treatment)
    t = (treatment.mean() - control.mean()) / np.sqrt(v1 + v2)
    dof = (v1 + v2) ** 2 / (v1**2 / (len(control) - 1) + v2**2 / (len(treatment) - 1))
    p = 2 * t_dist.sf(abs(t), dof)

    result = revenue_t_test(control, treatment)
    assert result["t_stat"] == pytest.approx(t)
    assert result["p_value"] == pytest.approx(p)
    assert result["delta"] == pytest.approx(treatment.mean() - control.mean())


# --- Metrics -------------------------------------------------------------------

def test_compute_metrics():
    df = pd.DataFrame({
        "group": ["control", "control", "treatment", "treatment"],
        "converted": [0, 1, 1, 1],
        "revenue": [0.0, 10.0, 20.0, 30.0],
        "latency_ms": [400, 410, 420, 430],
    })
    result = compute_metrics(df)
    assert result["control"] == {
        "users": 2, "conversion_rate": 0.5, "avg_revenue": 5.0, "avg_latency_ms": 405,
    }
    assert result["treatment"]["avg_revenue"] == 25.0


# --- Decision engine ----------------------------------------------------------

SIG = {"p_value": 0.001, "lift": 0.01}
REV_UP = {"delta": 1.0}


@pytest.mark.parametrize("srm, conv, rev, latency, expected", [
    (True, SIG, REV_UP, 0, "INVALID EXPERIMENT"),
    (False, {"p_value": 0.20, "lift": 0.05}, REV_UP, 0, "CONTINUE EXPERIMENT"),
    (False, {"p_value": 0.001, "lift": 0.001}, REV_UP, 0, "NO PRACTICAL IMPACT"),
    (False, SIG, {"delta": -0.5}, 0, "DO NOT SHIP: REVENUE RISK"),
    (False, SIG, REV_UP, 25, "ROLLBACK DUE TO LATENCY"),
    (False, SIG, REV_UP, 15, "SHIP CHANGE"),
])
def test_make_decision(srm, conv, rev, latency, expected):
    assert make_decision(srm, conv, rev, latency) == expected


def test_srm_overrides_significant_result():
    assert make_decision(True, {"p_value": 1e-30, "lift": 0.5}, REV_UP, 0) == \
        "INVALID EXPERIMENT"


# --- Peeking and time analysis -------------------------------------------------

def make_timeline(n, control_rate, treatment_rate):
    group = np.where(np.arange(n) % 2 == 0, "control", "treatment")
    idx = np.arange(n) // 2
    converted = np.where(
        group == "control",
        (idx % 10) < control_rate * 10,
        (idx % 10) < treatment_rate * 10,
    ).astype(int)
    return pd.DataFrame({
        "group": group,
        "converted": converted,
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="min"),
    })


def test_peeking_stops_at_first_significant_look():
    df = make_timeline(20_000, 0.10, 0.50)
    result = simulate_peeking(df, daily_sample_size=1000)
    assert len(result) == 1
    assert result[0]["users_seen"] == 1000
    assert result[0]["p_value"] < 0.05


def test_peeking_no_effect_never_stops():
    df = make_timeline(20_000, 0.10, 0.10)
    assert simulate_peeking(df, daily_sample_size=1000) == []


def test_conversion_over_time_daily():
    df = make_timeline(2 * 24 * 60, 0.10, 0.20)  # two days of minutes
    trend = conversion_over_time(df)
    assert len(trend) == 4
    treatment = trend[trend["group"] == "treatment"]["converted"]
    assert treatment.tolist() == pytest.approx([0.20, 0.20])
