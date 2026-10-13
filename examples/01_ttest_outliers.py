"""Example 1 - a two-sample t-test that an assistant might write.

Scenario: a new onboarding flow ("treated") vs the current one ("control").
Outcome: minutes to complete setup. The assistant ran Student's t-test and
reported a significant improvement... but three treated users left the tab
open for ages, and the treated group is small and far more variable.

Run:  stataudit run examples/01_ttest_outliers.py
"""

import numpy as np
from scipy import stats

rng = np.random.default_rng(7)

control = rng.normal(loc=12.0, scale=1.5, size=60).round(1)
treated = np.concatenate([
    rng.normal(loc=12.2, scale=1.5, size=11),  # typical users
    [21.5, 24.0, 26.5],                         # three users who left the tab open
]).round(1)

print(f"control mean = {control.mean():.2f} (n={control.size})")
print(f"treated mean = {treated.mean():.2f} (n={treated.size})")

t_stat, p_value = stats.ttest_ind(treated, control)
print(f"t = {t_stat:.2f}, p = {p_value:.4f}")
if p_value < 0.05:
    print("Conclusion: the new flow changes completion time (p < 0.05).")
