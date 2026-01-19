import pandas as pd
from stat_tests import conversion_z_test, revenue_t_test

df = pd.read_csv("../data/experiment_data.csv")

control_conv = df[df["group"] == "control"]["converted"]
treatment_conv = df[df["group"] == "treatment"]["converted"]

control_rev = df[df["group"] == "control"]["revenue"]
treatment_rev = df[df["group"] == "treatment"]["revenue"]

conv_results = conversion_z_test(control_conv, treatment_conv)
rev_results = revenue_t_test(control_rev, treatment_rev)

print("Conversion test:", conv_results)
print("Revenue test:", rev_results)

