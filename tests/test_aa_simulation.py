import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.proportion import proportion_confint

from aa_simulation import (
    obrien_fleming_constant,
    pooled_z,
    simulate_aa,
    wilson_interval,
)
from stat_tests import conversion_z_test


@pytest.fixture(scope="module")
def results():
    return simulate_aa(n_experiments=2000, seed=42)


def test_single_test_false_positive_rate_close_to_alpha(results):
    single = results["single_test"]
    # 2,000 experiments -> standard error ~0.5pp around the true 5%.
    assert single["rate"] == pytest.approx(0.05, abs=0.015)
    assert single["ci_low"] <= 0.05 <= single["ci_high"]


def test_peeking_inflates_false_positive_rate(results):
    peeking = results["peeking"]
    # Ten looks at alpha = 0.05 give roughly a 19-20% false positive rate.
    assert 0.15 < peeking["rate"] < 0.25
    assert peeking["ci_low"] > results["single_test"]["ci_high"]


def test_bonferroni_controls_false_positive_rate(results):
    assert results["bonferroni"]["rate"] <= 0.05
    assert results["bonferroni"]["ci_high"] < results["peeking"]["ci_low"]


def test_obrien_fleming_restores_alpha(results):
    obf = results["obrien_fleming"]
    assert obf["rate"] == pytest.approx(0.05, abs=0.015)
    assert obf["ci_low"] <= 0.05 <= obf["ci_high"]


def test_simulation_is_reproducible():
    a = simulate_aa(n_experiments=200, seed=7)
    b = simulate_aa(n_experiments=200, seed=7)
    assert a == b


def test_single_look_methods_agree():
    # With one look there is nothing to peek at: every method is the same test.
    r = simulate_aa(n_experiments=500, n_per_group=5000, peek_every=5000)
    assert r["n_looks"] == 1
    rates = {r[m]["rate"] for m in ("single_test", "peeking", "bonferroni", "obrien_fleming")}
    assert len(rates) == 1


def test_peek_every_must_divide_sample_size():
    with pytest.raises(ValueError):
        simulate_aa(n_per_group=10_000, peek_every=3_000)


@pytest.mark.parametrize("n_looks, expected", [
    (1, 1.960), (2, 1.977), (5, 2.040), (10, 2.087), (20, 2.126),
])
def test_obrien_fleming_constant_matches_published_table(n_looks, expected):
    # Jennison & Turnbull, Group Sequential Methods, Table 2.3 (alpha = 0.05).
    assert obrien_fleming_constant(n_looks) == pytest.approx(expected, abs=1e-3)


def test_obrien_fleming_constant_by_monte_carlo():
    # Brownian-motion check: the boundary should be crossed ~5% of the time.
    n_looks, n_sims = 5, 400_000
    c = obrien_fleming_constant(n_looks)
    rng = np.random.default_rng(0)
    looks = np.arange(1, n_looks + 1)
    z = rng.standard_normal((n_sims, n_looks)).cumsum(axis=1) / np.sqrt(looks)
    crossed = (np.abs(z) >= c * np.sqrt(n_looks / looks)).any(axis=1).mean()
    assert crossed == pytest.approx(0.05, abs=0.002)


def test_pooled_z_matches_conversion_z_test():
    control = pd.Series([1] * 120 + [0] * 880)
    treatment = pd.Series([1] * 150 + [0] * 850)
    expected = conversion_z_test(control, treatment)["z_stat"]
    assert pooled_z(120, 150, 1000, 1000) == pytest.approx(expected)


def test_pooled_z_handles_no_conversions():
    assert pooled_z(np.array([0]), np.array([0]), 100, 100)[0] == 0


@pytest.mark.parametrize("successes, trials", [(103, 2000), (0, 50), (381, 2000), (50, 50)])
def test_wilson_interval_matches_statsmodels(successes, trials):
    expected = proportion_confint(successes, trials, method="wilson")
    assert wilson_interval(successes, trials) == pytest.approx(expected)
