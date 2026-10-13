"""Example 5 - an analysis that holds up.

Scenario: reaction times (ms) under two lighting conditions, 45 participants
per group, roughly normal with similar spread. The assistant used Welch's
t-test and reports the effect size. stataudit should find nothing to change.

Run:  stataudit run examples/05_good_analysis.py
"""

import numpy as np
from scipy import stats

rng = np.random.default_rng(5)

dim = rng.normal(loc=412.0, scale=35.0, size=45)
bright = rng.normal(loc=385.0, scale=33.0, size=45)

res = stats.ttest_ind(dim, bright, equal_var=False)
pooled_sd = np.sqrt((dim.var(ddof=1) + bright.var(ddof=1)) / 2)
d = (dim.mean() - bright.mean()) / pooled_sd
print(f"Welch t = {res.statistic:.2f}, p = {res.pvalue:.2g}, Cohen's d = {d:.2f}")
