def compute_metrics(df):
    results = {}

    for group in ["control", "treatment"]:
        subset = df[df["group"] == group]

        results[group] = {
            "users": len(subset),
            "conversion_rate": subset["converted"].mean(),
            "avg_revenue": subset["revenue"].mean(),
            "avg_latency_ms": subset["latency_ms"].mean()
        }

    return results

