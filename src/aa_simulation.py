"""A/A simulation: how much does peeking inflate the false positive rate?

Runs many simulated experiments in which treatment and control have the
*same* conversion rate, so every "significant" result is a false positive.
Each experiment is analyzed four ways:

* ``single_test``     - one two-sided z-test at the planned sample size
* ``peeking``         - a z-test after every ``peek_every`` users per group,
  stopping at the first p < alpha
* ``bonferroni``      - peeking, but each look uses alpha / number_of_looks
* ``obrien_fleming``  - peeking with an O'Brien-Fleming boundary: reject at
  look k of K if |z| >= c * sqrt(K / k), with c chosen so the overall
  false positive rate is alpha

Run ``python aa_simulation.py`` from ``src/`` to print the results.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

METHODS = ("single_test", "peeking", "bonferroni", "obrien_fleming")


def pooled_z(conv_control, conv_treatment, n_control, n_treatment):
    """Vectorized pooled two-proportion z statistic.

    Same formula as ``stat_tests.conversion_z_test``, applied to arrays of
    conversion counts so thousands of experiments run at once.
    """
    p1 = conv_control / n_control
    p2 = conv_treatment / n_treatment
    p_pool = (conv_control + conv_treatment) / (n_control + n_treatment)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_control + 1 / n_treatment))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (p2 - p1) / se
    return np.nan_to_num(z)  # no conversions yet -> no evidence


def _crossing_probability(bounds, step=0.05):
    """P(|Z_k| >= bounds[k-1] at some look k) under the null hypothesis.

    Looks are equally spaced. Uses recursive numerical integration
    (Armitage, McPherson & Rowe, 1969) with the trapezoid rule on the
    partial sums S_k = sqrt(k) Z_k, which have independent N(0, 1)
    increments: the density of S_k inside the continuation region is the
    density of S_{k-1} convolved with a standard normal.
    """
    total = 0.0
    for k, bound in enumerate(bounds, start=1):
        limit = bound * np.sqrt(k)
        n_steps = int(np.ceil(2 * limit / step))
        grid = np.linspace(-limit, limit, n_steps + 1)
        trapezoid = np.full(n_steps + 1, grid[1] - grid[0])
        trapezoid[[0, -1]] /= 2
        if k == 1:
            total += 2 * norm.sf(limit)
            density = norm.pdf(grid)
        else:
            weights = prev_density * prev_trapezoid
            total += np.sum(weights * (norm.sf(limit - prev_grid)
                                       + norm.cdf(-limit - prev_grid)))
            density = norm.pdf(grid[:, None] - prev_grid[None, :]) @ weights
        prev_grid, prev_density, prev_trapezoid = grid, density, trapezoid
    return total


def obrien_fleming_constant(n_looks, alpha=0.05):
    """Constant c for a two-sided O'Brien-Fleming boundary.

    Rejects at look k of K when |z| >= c * sqrt(K / k), with c solved so the
    overall type I error is ``alpha``. For alpha = 0.05 this gives about
    1.96, 1.98, 2.04 and 2.09 for 1, 2, 5 and 10 looks (Jennison & Turnbull,
    *Group Sequential Methods*, Table 2.3).
    """
    if n_looks == 1:
        return norm.ppf(1 - alpha / 2)
    looks = np.arange(1, n_looks + 1)

    def excess(c):
        return _crossing_probability(c * np.sqrt(n_looks / looks)) - alpha

    return brentq(excess, norm.ppf(1 - alpha / 2), 5.0, xtol=1e-6)


def wilson_interval(successes, trials, confidence=0.95):
    """Wilson score confidence interval for a binomial proportion."""
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = successes / trials
    centre = (p + z**2 / (2 * trials)) / (1 + z**2 / trials)
    half = (z / (1 + z**2 / trials)) * np.sqrt(
        p * (1 - p) / trials + z**2 / (4 * trials**2)
    )
    return centre - half, centre + half


def simulate_aa(
    n_experiments=2000, n_per_group=10_000, peek_every=1_000,
    baseline=0.10, alpha=0.05, seed=42,
):
    """Simulate A/A experiments and measure each method's false positive rate.

    Args:
        n_experiments: number of simulated experiments.
        n_per_group: planned users per group (the final look).
        peek_every: users per group added between looks. Must divide
            ``n_per_group``.
        baseline: true conversion rate in both groups.
        alpha: significance level.
        seed: random seed, so results are reproducible.

    Returns:
        dict mapping each method in ``METHODS`` to a dict with
        ``false_positives``, ``rate``, ``ci_low`` and ``ci_high`` (95% Wilson
        interval), plus ``n_looks`` and ``obf_constant``.
    """
    if n_per_group % peek_every:
        raise ValueError("peek_every must divide n_per_group")
    n_looks = n_per_group // peek_every
    rng = np.random.default_rng(seed)

    # Conversions added in each block of peek_every users, then running totals.
    shape = (n_experiments, n_looks)
    conv_control = rng.binomial(peek_every, baseline, size=shape).cumsum(axis=1)
    conv_treatment = rng.binomial(peek_every, baseline, size=shape).cumsum(axis=1)
    n_seen = peek_every * np.arange(1, n_looks + 1)

    abs_z = np.abs(pooled_z(conv_control, conv_treatment, n_seen, n_seen))

    z_crit = norm.ppf(1 - alpha / 2)
    z_bonferroni = norm.ppf(1 - alpha / (2 * n_looks))
    obf_constant = obrien_fleming_constant(n_looks, alpha)
    obf_bounds = obf_constant * np.sqrt(n_looks / np.arange(1, n_looks + 1))

    rejected = {
        "single_test": abs_z[:, -1] >= z_crit,
        "peeking": (abs_z >= z_crit).any(axis=1),
        "bonferroni": (abs_z >= z_bonferroni).any(axis=1),
        "obrien_fleming": (abs_z >= obf_bounds).any(axis=1),
    }

    results = {"n_looks": n_looks, "obf_constant": obf_constant}
    for method in METHODS:
        false_positives = int(rejected[method].sum())
        ci_low, ci_high = wilson_interval(false_positives, n_experiments)
        results[method] = {
            "false_positives": false_positives,
            "rate": false_positives / n_experiments,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    return results


if __name__ == "__main__":
    results = simulate_aa()
    print(f"A/A simulation: 2,000 experiments, 10,000 users per group, "
          f"{results['n_looks']} looks, alpha = 0.05")
    for method in METHODS:
        r = results[method]
        print(f"{method:>15}: {r['rate']:.1%} "
              f"(95% CI {r['ci_low']:.1%} - {r['ci_high']:.1%})")
