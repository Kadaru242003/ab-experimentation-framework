import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/experiment_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"] = df["timestamp"].dt.date

trend = (
    df.groupby(["date", "group"])["converted"]
    .mean()
    .reset_index()
)

control = trend[trend["group"] == "control"]
treatment = trend[trend["group"] == "treatment"]

plt.figure()
plt.plot(control["date"], control["converted"], label="Control")
plt.plot(treatment["date"], treatment["converted"], label="Treatment")

plt.title("Conversion Rate Over Time (Novelty Effect Visible)")
plt.xlabel("Date")
plt.ylabel("Conversion Rate")
plt.legend()

plt.savefig(
    "../reports/conversion_over_time.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

