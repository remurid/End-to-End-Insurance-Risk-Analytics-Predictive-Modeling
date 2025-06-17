import numpy as np
from scipy.stats import chi2_contingency, ttest_ind
import pandas as pd

def run_chi2_test(group_a, group_b, metric: str) -> dict:
    """Run chi-squared test for categorical data (e.g., claim frequency)."""
    contingency = pd.crosstab(group_a[metric], group_b[metric])
    chi2, p, dof, expected = chi2_contingency(contingency)
    return {'test': 'chi2', 'statistic': chi2, 'p_value': p}

def run_ttest(group_a, group_b, metric: str) -> dict:
    """Run t-test for numerical data (e.g., claim severity, margin)."""
    stat, p = ttest_ind(group_a[metric], group_b[metric], nan_policy='omit')
    return {'test': 't-test', 'statistic': stat, 'p_value': p}
