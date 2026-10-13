"""Example 2 - a correlation driven by two high-leverage points.

Scenario: weekly marketing spend vs sign-ups for 25 regions. The assistant
computed Pearson's r, found a "strong, significant" correlation and wrote it
up. Two regions ran one-off national campaigns and sit far from the rest;
they alone create the correlation.

Run:  stataudit run examples/02_pearson_leverage.py
"""

import numpy as np
from scipy.stats import pearsonr

rng = np.random.default_rng(11)

spend = rng.normal(loc=10.0, scale=2.0, size=23)          # thousand USD
signups = rng.normal(loc=200.0, scale=25.0, size=23)      # unrelated to spend
spend = np.append(spend, [31.0, 34.0])                    # two national campaigns
signups = np.append(signups, [290.0, 305.0])

r, p = pearsonr(spend, signups)
print(f"Pearson r = {r:.2f}, p = {p:.2g}")
print("Conclusion: marketing spend strongly predicts sign-ups." if p < 0.05 else "No association.")
