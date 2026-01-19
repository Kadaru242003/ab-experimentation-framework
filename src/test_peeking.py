import pandas as pd
from peeking_bias import simulate_peeking

df = pd.read_csv("../data/experiment_data.csv")

result = simulate_peeking(df)

print(result)

