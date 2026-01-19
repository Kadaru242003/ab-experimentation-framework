import numpy as np
import pandas as pd

np.random.seed(42)

N = 300_000  # keep smaller for now; we’ll scale later

user_id = np.arange(N)

group = np.random.choice(["control", "treatment"], size=N, p=[0.5, 0.5])

base_conversion = 0.10
treatment_lift = 0.012

time_effect = np.linspace(1.5, 1.0, N)

conversion_prob = np.where(
    group == "treatment",
    base_conversion + treatment_lift * time_effect,
    base_conversion
)


converted = np.random.binomial(1, conversion_prob)

revenue = converted * np.random.gamma(shape=2.0, scale=30.0, size=N)

latency_ms = np.random.normal(
    loc=400,
    scale=50,
    size=N
) + np.where(group == "treatment", 15, 0)

timestamp = pd.date_range(
    start="2025-01-01",
    periods=N,
    freq="s"
)

df = pd.DataFrame({
    "user_id": user_id,
    "group": group,
    "converted": converted,
    "revenue": revenue,
    "latency_ms": latency_ms,
    "timestamp": timestamp
})

df.to_csv("../data/experiment_data.csv", index=False)

