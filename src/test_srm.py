import pandas as pd
from validation import check_srm

df = pd.read_csv("../data/experiment_data.csv")

result = check_srm(df)

print(result)

