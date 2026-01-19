import pandas as pd
from stat_tests import conversion_z_test

def simulate_peeking(df, daily_sample_size=5000, alpha=0.05):
    df = df.sort_values("timestamp")

    decisions = []

    for i in range(daily_sample_size, len(df), daily_sample_size):
        subset = df.iloc[:i]

        control = subset[subset["group"] == "control"]["converted"]
        treatment = subset[subset["group"] == "treatment"]["converted"]

        result = conversion_z_test(control, treatment)

        if result["p_value"] < alpha:
            decisions.append({
                "users_seen": i,
                "p_value": result["p_value"],
                "lift": result["lift"]
            })
            break

    return decisions

