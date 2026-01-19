from scipy.stats import chisquare

def check_srm(df):
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
        "srm_detected": p_value < 0.05
    }

