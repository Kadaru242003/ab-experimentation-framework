import pandas as pd
from metrics import compute_metrics

df = pd.read_csv("../data/experiment_data.csv")

results = compute_metrics(df)

print(results)

