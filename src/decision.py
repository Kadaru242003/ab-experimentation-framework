def make_decision(srm_detected, conv_result, rev_result, latency_diff):
    if srm_detected:
        return "INVALID EXPERIMENT"

    if conv_result["p_value"] >= 0.05:
        return "CONTINUE EXPERIMENT"

    if conv_result["lift"] < 0.005:
        return "NO PRACTICAL IMPACT"

    if rev_result["delta"] < 0:
        return "DO NOT SHIP: REVENUE RISK"

    if latency_diff > 20:
        return "ROLLBACK DUE TO LATENCY"

    return "SHIP CHANGE"

