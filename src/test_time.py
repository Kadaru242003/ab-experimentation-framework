import pandas as pd
from time_analysis import conversion_over_time

df = pd.read_csv("../data/experiment_data.csv")
trend = conversion_over_time(df)

print(trend.head())

