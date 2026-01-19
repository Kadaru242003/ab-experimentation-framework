import pandas as pd

from validation import check_srm
from metrics import compute_metrics
from stat_tests import conversion_z_test, revenue_t_test
from decision import make_decision

df = pd.read_csv("../data/experiment_data.csv")

srm_result = check_srm(df)
metrics = compute_metrics(df)

control_conv = df[df["group"] == "control"]["converted"]
treatment_conv = df[df["group"] == "treatment"]["converted"]

control_rev = df[df["group"] == "control"]["revenue"]
treatment_rev = df[df["group"] == "treatment"]["revenue"]

conv_result = conversion_z_test(control_conv, treatment_conv)
rev_result = revenue_t_test(control_rev, treatment_rev)

latency_diff = (
    metrics["treatment"]["avg_latency_ms"]
    - metrics["control"]["avg_latency_ms"]
)

decision = make_decision(
    srm_detected=srm_result["srm_detected"],
    conv_result=conv_result,
    rev_result=rev_result,
    latency_diff=latency_diff
)

print("SRM:", srm_result)
print("Metrics:", metrics)
print("Conversion Test:", conv_result)
print("Revenue Test:", rev_result)
print("Latency Diff (ms):", latency_diff)
print("FINAL DECISION:", decision)

