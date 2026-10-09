from scipy.stats import chisquare

def check_srm(df, threshold=0.001):
    """Chi-square test for Sample Ratio Mismatch against a 50/50 split.

    SRM is flagged when the p-value is below ``threshold``. The default of
    0.001 is the common industry choice: the check runs on every experiment,
    so a strict threshold keeps false alarms rare while real assignment bugs
    (which usually give tiny p-values) are still caught.
    """
    counts = df["group"].value_counts()

    control_count = counts.get("control", 0)
    treatment_count = counts.get("treatment", 0)

    total = control_count + treatment_count

    expected = [total / 2, total / 2]
    observed = [control_count, treatment_count]

    chi_stat, p_value = chisquare(observed, expected)

    return {
        "control_users": control_count,
        "treatment_users": treatment_count,
        "p_value": p_value,
        "srm_detected": p_value < threshold
    }

