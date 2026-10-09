import math

import pytest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    power_proportions_2indep,
    samplesize_proportions_2indep_onetail,
)

from power import (
    experiment_duration_days,
    power_means,
    power_two_proportions,
    sample_size_means,
    sample_size_two_proportions,
)


# --- Known textbook values -------------------------------------------------

def test_two_proportions_textbook_value():
    # 10% -> 12%, alpha 0.05 two-sided, 80% power: 3,841 per group
    # (Fleiss formula without continuity correction).
    assert sample_size_two_proportions(0.10, 0.02) == 3841


def test_means_textbook_value():
    # Cohen's d = 0.5, alpha 0.05 two-sided, 80% power: 63 per group under
    # the normal approximation (2 * (1.96 + 0.8416)^2 / 0.25 = 62.8).
    assert sample_size_means(sd=1.0, mde=0.5) == 63


def test_means_textbook_value_90_power():
    # d = 0.5, 90% power: 2 * (1.96 + 1.2816)^2 / 0.25 = 84.1 -> 85.
    assert sample_size_means(sd=1.0, mde=0.5, power=0.9) == 85


# --- Agreement with statsmodels ---------------------------------------------

@pytest.mark.parametrize("baseline, diff, alpha, power, alternative", [
    (0.10, 0.02, 0.05, 0.80, "two-sided"),
    (0.10, 0.01, 0.05, 0.90, "two-sided"),
    (0.05, 0.005, 0.01, 0.80, "two-sided"),
    (0.30, -0.03, 0.05, 0.80, "two-sided"),
    (0.10, 0.02, 0.05, 0.80, "one-sided"),
    (0.50, 0.05, 0.10, 0.95, "one-sided"),
])
def test_two_proportions_matches_statsmodels(
    baseline, diff, alpha, power, alternative,
):
    sm_alternative = "two-sided"
    if alternative == "one-sided":
        sm_alternative = "larger" if diff > 0 else "smaller"
    expected = samplesize_proportions_2indep_onetail(
        diff, baseline, power, alpha=alpha, alternative=sm_alternative,
    )
    result = sample_size_two_proportions(
        baseline, diff, alpha=alpha, power=power, alternative=alternative,
    )
    assert result == math.ceil(expected)


def test_relative_mde_equals_absolute_mde():
    # +10% relative on a 10% baseline is +1pp absolute.
    assert sample_size_two_proportions(0.10, 0.10, relative=True) == \
        sample_size_two_proportions(0.10, 0.01)


@pytest.mark.parametrize("sd, mde, alpha, power, alternative", [
    (1.0, 0.5, 0.05, 0.80, "two-sided"),
    (25.0, 1.0, 0.05, 0.80, "two-sided"),
    (25.0, -1.0, 0.01, 0.90, "two-sided"),
    (10.0, 0.5, 0.05, 0.80, "one-sided"),
])
def test_means_matches_statsmodels(sd, mde, alpha, power, alternative):
    sm_alternative = "two-sided" if alternative == "two-sided" else "larger"
    expected = NormalIndPower().solve_power(
        effect_size=abs(mde) / sd, alpha=alpha, power=power,
        alternative=sm_alternative,
    )
    result = sample_size_means(
        sd, mde, alpha=alpha, power=power, alternative=alternative,
    )
    assert result == math.ceil(expected)


@pytest.mark.parametrize("baseline, diff, n, alternative", [
    (0.10, 0.02, 3841, "two-sided"),
    (0.10, -0.02, 3000, "two-sided"),
    (0.10, 0.001, 500, "two-sided"),
    (0.10, 0.02, 3000, "one-sided"),
])
def test_power_two_proportions_matches_statsmodels(baseline, diff, n, alternative):
    sm_alternative = "two-sided"
    if alternative == "one-sided":
        sm_alternative = "larger" if diff > 0 else "smaller"
    expected = power_proportions_2indep(
        diff, baseline, n, alternative=sm_alternative,
    ).power
    result = power_two_proportions(baseline, diff, n, alternative=alternative)
    assert result == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("d, n, alternative", [
    (0.5, 63, "two-sided"),
    (0.2, 100, "two-sided"),
    (0.01, 50, "two-sided"),
    (0.3, 80, "one-sided"),
])
def test_power_means_matches_statsmodels(d, n, alternative):
    sm_alternative = "two-sided" if alternative == "two-sided" else "larger"
    expected = NormalIndPower().power(d, n, 0.05, alternative=sm_alternative)
    result = power_means(1.0, d, n, alternative=alternative)
    assert result == pytest.approx(expected, rel=1e-9)


# --- Consistency between sample size and power -----------------------------

@pytest.mark.parametrize("target", [0.5, 0.8, 0.9, 0.99])
def test_sample_size_achieves_target_power(target):
    n = sample_size_two_proportions(0.10, 0.02, power=target)
    assert power_two_proportions(0.10, 0.02, n) >= target
    # One fewer user per group should not be comfortably above target.
    assert power_two_proportions(0.10, 0.02, n - 1) < target + 1e-4

    n = sample_size_means(30.0, 2.0, power=target)
    assert power_means(30.0, 2.0, n) >= target
    assert power_means(30.0, 2.0, n - 1) < target + 1e-4


def test_one_sided_needs_fewer_users():
    assert sample_size_two_proportions(0.10, 0.02, alternative="one-sided") < \
        sample_size_two_proportions(0.10, 0.02)
    assert sample_size_means(1, 0.2, alternative="one-sided") < \
        sample_size_means(1, 0.2)


def test_power_increases_with_sample_size():
    powers = [power_two_proportions(0.10, 0.01, n) for n in (100, 1000, 10000)]
    assert powers == sorted(powers)


# --- Edge cases --------------------------------------------------------------

def test_tiny_effect_needs_huge_sample():
    # Halving the effect roughly quadruples the sample size (n ~ 1 / mde^2).
    n_small = sample_size_means(1.0, 0.001)
    n_half = sample_size_means(1.0, 0.0005)
    assert n_small > 15_000_000
    assert n_half / n_small == pytest.approx(4, rel=1e-3)

    n = sample_size_two_proportions(0.10, 0.0001)
    assert n > 100_000_000
    assert math.isfinite(n)


def test_tiny_effect_power_approaches_alpha():
    assert power_two_proportions(0.10, 1e-7, 1000) == pytest.approx(0.05, abs=1e-4)
    assert power_means(1.0, 1e-9, 1000, alternative="one-sided") == \
        pytest.approx(0.05, abs=1e-4)


def test_power_near_one():
    n_99 = sample_size_two_proportions(0.10, 0.02, power=0.99)
    n_9999 = sample_size_two_proportions(0.10, 0.02, power=0.9999)
    assert n_9999 > n_99 > sample_size_two_proportions(0.10, 0.02)
    assert power_two_proportions(0.10, 0.02, n_9999) >= 0.9999

    assert power_means(1.0, 0.5, 10_000) == pytest.approx(1.0)
    assert power_means(1.0, 0.5, 10_000) <= 1.0


@pytest.mark.parametrize("kwargs", [
    dict(baseline=0.0, mde=0.01),
    dict(baseline=1.0, mde=0.01),
    dict(baseline=0.10, mde=0.0),
    dict(baseline=0.95, mde=0.10),                  # treatment rate > 1
    dict(baseline=0.10, mde=-1.5, relative=True),   # treatment rate < 0
    dict(baseline=0.10, mde=0.01, power=1.0),
    dict(baseline=0.10, mde=0.01, power=0.04),      # power <= alpha
    dict(baseline=0.10, mde=0.01, alpha=0.0),
    dict(baseline=0.10, mde=0.01, alternative="larger"),
])
def test_two_proportions_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        sample_size_two_proportions(**kwargs)


@pytest.mark.parametrize("kwargs", [
    dict(sd=0, mde=1),
    dict(sd=-1, mde=1),
    dict(sd=1, mde=0),
    dict(sd=1, mde=1, power=1.0),
])
def test_means_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        sample_size_means(**kwargs)


# --- Duration ----------------------------------------------------------------

def test_duration_full_traffic_even_split():
    # 10,000 visitors/day split 50/50 -> 5,000 per group per day.
    assert experiment_duration_days(15_000, 10_000) == 3
    assert experiment_duration_days(15_001, 10_000) == 4


def test_duration_partial_allocation():
    # 2,000/day * 50% enrolled * 50% per group = 500 per group per day.
    assert experiment_duration_days(3841, 2000, traffic_allocation=0.5) == 8


def test_duration_uneven_split_uses_smaller_group():
    # 90/10 split: control fills fast, treatment gets 1,000 users/day.
    assert experiment_duration_days(5000, 10_000, treatment_share=0.1) == 5
    assert experiment_duration_days(5000, 10_000, treatment_share=0.9) == 5


@pytest.mark.parametrize("kwargs", [
    dict(n_per_group=0, daily_traffic=100),
    dict(n_per_group=100, daily_traffic=0),
    dict(n_per_group=100, daily_traffic=100, traffic_allocation=0),
    dict(n_per_group=100, daily_traffic=100, traffic_allocation=1.5),
    dict(n_per_group=100, daily_traffic=100, treatment_share=1.0),
])
def test_duration_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        experiment_duration_days(**kwargs)
