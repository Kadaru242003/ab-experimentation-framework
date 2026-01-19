import numpy as np
from scipy.stats import ttest_ind, norm

def conversion_z_test(control, treatment):
    p1 = control.mean()
    p2 = treatment.mean()

    n1 = len(control)
    n2 = len(treatment)

    p_pool = (control.sum() + treatment.sum()) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

    z = (p2 - p1) / se
    p_value = 1 - norm.cdf(z)

    return {
        "z_stat": z,
        "p_value": p_value,
        "lift": p2 - p1
    }

def revenue_t_test(control, treatment):
    stat, p_value = ttest_ind(treatment, control, equal_var=False)
    return {
        "t_stat": stat,
        "p_value": p_value,
        "delta": treatment.mean() - control.mean()
    }

