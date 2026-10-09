"""Power analysis and sample size planning for A/B tests.

All functions use the standard large-sample normal approximation and assume
two independent groups of equal size (``n`` users per group).

Notation used throughout:

* ``alpha``  - significance level (type I error rate)
* ``power``  - 1 - beta, the probability of detecting the effect if it is real
* ``z_a``    - critical value: ``Phi^-1(1 - alpha/2)`` for a two-sided test,
  ``Phi^-1(1 - alpha)`` for a one-sided test
* ``z_b``    - ``Phi^-1(power)``
* ``Phi``    - standard normal CDF

``alternative="one-sided"`` means a one-sided test in the direction of the
minimum detectable effect (MDE).
"""

import math

from scipy.stats import norm

_ALTERNATIVES = ("two-sided", "one-sided")


def _check_alpha(alpha):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")


def _check_alpha_power(alpha, power):
    _check_alpha(alpha)
    if not alpha < power < 1:
        raise ValueError("power must be greater than alpha and less than 1")


def _z_alpha(alpha, alternative):
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_ALTERNATIVES}")
    tail = alpha / 2 if alternative == "two-sided" else alpha
    return norm.ppf(1 - tail)


def _treatment_rate(baseline, mde, relative):
    if not 0 < baseline < 1:
        raise ValueError("baseline must be between 0 and 1")
    if mde == 0:
        raise ValueError("mde must be non-zero")

    p2 = baseline * (1 + mde) if relative else baseline + mde
    if not 0 < p2 < 1:
        raise ValueError(
            f"baseline and mde imply a treatment rate of {p2}, "
            "which is outside (0, 1)"
        )
    return p2


def sample_size_two_proportions(
    baseline, mde, alpha=0.05, power=0.8, relative=False,
    alternative="two-sided",
):
    """Required users per group to detect a change in a conversion rate.

    Uses the classic normal-approximation formula for a two-proportion
    z-test with a pooled variance under the null hypothesis and unpooled
    variance under the alternative (Fleiss, Levin & Paik, *Statistical
    Methods for Rates and Proportions*, without continuity correction)::

        p1    = baseline
        p2    = baseline + mde              (absolute MDE)
              = baseline * (1 + mde)        (relative MDE)
        p_bar = (p1 + p2) / 2

        n = ( z_a * sqrt(2 * p_bar * (1 - p_bar))
              + z_b * sqrt(p1 * (1 - p1) + p2 * (1 - p2)) )^2 / (p2 - p1)^2

    The result is rounded up to the next whole user. This matches
    ``statsmodels.stats.proportion.samplesize_proportions_2indep_onetail``.

    Example: baseline 10%, absolute MDE 2pp, alpha 0.05 two-sided,
    power 0.8 -> 3,841 users per group.

    Args:
        baseline: control conversion rate, in (0, 1).
        mde: minimum detectable effect. An absolute difference in rate
            (0.02 = +2 percentage points) unless ``relative`` is True, in
            which case a fraction of the baseline (0.10 = +10%).
        alpha: significance level.
        power: desired power, in (alpha, 1).
        relative: interpret ``mde`` as relative to the baseline.
        alternative: "two-sided" or "one-sided".

    Returns:
        Required number of users in each group (int).
    """
    _check_alpha_power(alpha, power)
    p1 = baseline
    p2 = _treatment_rate(baseline, mde, relative)
    p_bar = (p1 + p2) / 2

    z_a = _z_alpha(alpha, alternative)
    z_b = norm.ppf(power)

    numerator = (
        z_a * math.sqrt(2 * p_bar * (1 - p_bar))
        + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return math.ceil(numerator / (p2 - p1) ** 2)


def sample_size_means(sd, mde, alpha=0.05, power=0.8, alternative="two-sided"):
    """Required users per group to detect a difference in means.

    Normal-approximation formula for comparing two means with a common
    standard deviation ``sd`` (e.g. revenue per user)::

        n = 2 * sd^2 * (z_a + z_b)^2 / mde^2

    The result is rounded up to the next whole user. This matches
    ``statsmodels.stats.power.NormalIndPower`` with
    ``effect_size = mde / sd``. For small samples a t-test needs slightly
    more users than this.

    Example: sd 1, MDE 0.5 (Cohen's d = 0.5), alpha 0.05 two-sided,
    power 0.8 -> 63 users per group.

    Args:
        sd: standard deviation of the metric per user, > 0.
        mde: minimum detectable difference in means, in the metric's units.
        alpha: significance level.
        power: desired power, in (alpha, 1).
        alternative: "two-sided" or "one-sided".

    Returns:
        Required number of users in each group (int).
    """
    _check_alpha_power(alpha, power)
    if sd <= 0:
        raise ValueError("sd must be positive")
    if mde == 0:
        raise ValueError("mde must be non-zero")

    z_a = _z_alpha(alpha, alternative)
    z_b = norm.ppf(power)
    return math.ceil(2 * sd**2 * (z_a + z_b) ** 2 / mde**2)


def power_two_proportions(
    baseline, mde, n_per_group, alpha=0.05, relative=False,
    alternative="two-sided",
):
    """Achieved power of a two-proportion z-test with ``n_per_group`` users.

    With ``delta = |p2 - p1|``::

        se_null = sqrt(2 * p_bar * (1 - p_bar) / n)
        se_alt  = sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)

        power = Phi((delta - z_a * se_null) / se_alt)
              + Phi((-delta - z_a * se_null) / se_alt)   # two-sided only

    The second term is the (usually negligible) chance of rejecting in the
    wrong direction, so with tiny effects a two-sided test has power
    ``alpha`` rather than ``alpha / 2``. This matches
    ``statsmodels.stats.proportion.power_proportions_2indep``.

    Args:
        baseline, mde, alpha, relative, alternative: as in
            :func:`sample_size_two_proportions`.
        n_per_group: users in each group, > 0.

    Returns:
        Power as a float in (0, 1).
    """
    _check_alpha(alpha)
    if n_per_group <= 0:
        raise ValueError("n_per_group must be positive")
    p1 = baseline
    p2 = _treatment_rate(baseline, mde, relative)
    p_bar = (p1 + p2) / 2
    delta = abs(p2 - p1)

    z_a = _z_alpha(alpha, alternative)
    se_null = math.sqrt(2 * p_bar * (1 - p_bar) / n_per_group)
    se_alt = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n_per_group)

    power = norm.cdf((delta - z_a * se_null) / se_alt)
    if alternative == "two-sided":
        power += norm.cdf((-delta - z_a * se_null) / se_alt)
    return power


def power_means(sd, mde, n_per_group, alpha=0.05, alternative="two-sided"):
    """Achieved power of a two-sample test of means with ``n_per_group`` users.

    With ``d = |mde| / sd`` (Cohen's d)::

        power = Phi(d * sqrt(n / 2) - z_a)
              + Phi(-d * sqrt(n / 2) - z_a)   # two-sided only

    This matches ``statsmodels.stats.power.NormalIndPower().power``.

    Args:
        sd, mde, alpha, alternative: as in :func:`sample_size_means`.
        n_per_group: users in each group, > 0.

    Returns:
        Power as a float in (0, 1).
    """
    _check_alpha(alpha)
    if sd <= 0:
        raise ValueError("sd must be positive")
    if n_per_group <= 0:
        raise ValueError("n_per_group must be positive")

    z_a = _z_alpha(alpha, alternative)
    noncentrality = abs(mde) / sd * math.sqrt(n_per_group / 2)

    power = norm.cdf(noncentrality - z_a)
    if alternative == "two-sided":
        power += norm.cdf(-noncentrality - z_a)
    return power


def experiment_duration_days(
    n_per_group, daily_traffic, traffic_allocation=1.0, treatment_share=0.5,
):
    """Days needed for every group to reach ``n_per_group`` users.

    ``traffic_allocation`` is the fraction of daily traffic enrolled in the
    experiment, and ``treatment_share`` is the fraction of enrolled users
    sent to treatment (the rest go to control). The smaller group fills
    last, so::

        users_per_day_smallest_group = daily_traffic * traffic_allocation
                                       * min(treatment_share,
                                             1 - treatment_share)
        days = ceil(n_per_group / users_per_day_smallest_group)

    Example: 3,841 users per group, 2,000 visitors a day, 50% of traffic
    in the test, split 50/50 -> 500 users per group per day -> 8 days.

    Args:
        n_per_group: required users per group, e.g. from
            :func:`sample_size_two_proportions`.
        daily_traffic: eligible users per day, > 0.
        traffic_allocation: fraction of traffic in the experiment, in (0, 1].
        treatment_share: fraction of enrolled users in treatment, in (0, 1).

    Returns:
        Number of whole days (int).
    """
    if n_per_group <= 0:
        raise ValueError("n_per_group must be positive")
    if daily_traffic <= 0:
        raise ValueError("daily_traffic must be positive")
    if not 0 < traffic_allocation <= 1:
        raise ValueError("traffic_allocation must be in (0, 1]")
    if not 0 < treatment_share < 1:
        raise ValueError("treatment_share must be in (0, 1)")

    smallest_share = min(treatment_share, 1 - treatment_share)
    per_group_per_day = daily_traffic * traffic_allocation * smallest_share
    # Round first so float noise (e.g. 1 - 0.9 = 0.0999...) can't add a day.
    return math.ceil(round(n_per_group / per_group_per_day, 9))
